//! Actix-Style Share-Nothing Per-Core Worker Server (`serve_sharded_per_core`).
//!
//! Combines two OS-optimal accept strategies with a tuned per-core `hyper_util` connection engine:
//! 1. **Linux `SO_REUSEPORT` Kernel Multi-Queue Listeners (`#[cfg(target_os = "linux")]`)**: When
//!    bound with `SO_REUSEPORT`, each worker core owns an independent kernel accept queue hashed
//!    in-kernel by 4-tuple (`0` cross-thread channel hops). Falls back automatically to the
//!    round-robin `Accept` distributor if the caller passed a non-`SO_REUSEPORT` socket.
//! 2. **Round-Robin `Accept` Distributor (macOS `kqueue` & Ephemeral Test Sockets)**: Accepts TCP
//!    sockets, enforces `TCP_NODELAY = true` + non-blocking mode immediately, and dispatches each
//!    raw `std::net::TcpStream` strictly round-robin (`conn_idx % workers`) across all `N` cores.
//! 3. **Tuned Per-Core `hyper_util` Connection Engine**: Each `quixigate-worker-{idx}` OS thread
//!    drives `hyper_util::server::conn::auto::Builder` (`http1().max_buf_size(64 KiB).writev(true)
//!    .keep_alive(true)`, `http2().adaptive_window(true)`, WebSocket upgrades enabled) on its
//!    private `current_thread` reactor with a core-local `reqwest::Client` pool (`64 KiB` cap vs
//!    `400 KiB` default = `6.25×` lower peak connection buffer footprint while fitting full `32 KiB`
//!    vectored SSE writes in a single syscall, plus zero per-request `Extensions` heap allocation).

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::Body;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoConnBuilder;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, mpsc, watch};
use tower::Service;

use crate::auth::PeerTrustService;
use crate::build_worker_service;
use crate::state::AppState;

/// Binds a `tokio::net::TcpListener` with `SO_REUSEADDR` (and `SO_REUSEPORT` on Linux when `port != 0`)
/// and a `4096`-entry listen backlog so `serve_sharded_per_core` can attach per-core kernel accept queues.
pub fn bind_reuseport_or_standard_listener(addr: SocketAddr) -> std::io::Result<TcpListener> {
    let domain = socket2::Domain::for_address(addr);
    let sock = socket2::Socket::new(domain, socket2::Type::STREAM, Some(socket2::Protocol::TCP))?;
    let _ = sock.set_reuse_address(true);
    #[cfg(target_os = "linux")]
    if addr.port() != 0 {
        let _ = sock.set_reuse_port(true);
    }
    sock.set_nonblocking(true)?;
    sock.bind(&addr.into())?;
    sock.listen(4096)?;
    let std_listener: std::net::TcpListener = sock.into();
    TcpListener::from_std(std_listener)
}

#[cfg(target_os = "linux")]
fn try_bind_linux_reuseport_std(addr: SocketAddr) -> std::io::Result<std::net::TcpListener> {
    let domain = socket2::Domain::for_address(addr);
    let sock = socket2::Socket::new(domain, socket2::Type::STREAM, Some(socket2::Protocol::TCP))?;
    sock.set_reuse_address(true)?;
    sock.set_reuse_port(true)?;
    sock.set_nonblocking(true)?;
    sock.bind(&addr.into())?;
    sock.listen(4096)?;
    Ok(sock.into())
}

enum WorkerIngress {
    Channel(mpsc::UnboundedReceiver<(std::net::TcpStream, SocketAddr)>),
    #[cfg(target_os = "linux")]
    ReusePort(std::net::TcpListener),
}

fn build_tuned_conn_builder() -> Arc<AutoConnBuilder<TokioExecutor>> {
    let mut builder = AutoConnBuilder::new(TokioExecutor::new());
    builder
        .http1()
        .max_buf_size(64 * 1024)
        .writev(true)
        .keep_alive(true);
    builder.http2().adaptive_window(true);
    Arc::new(builder)
}

#[inline]
fn spawn_hyper_connection(
    tokio_stream: TcpStream,
    peer_addr: SocketAddr,
    app: PeerTrustService<axum::Router>,
    conn_builder: Arc<AutoConnBuilder<TokioExecutor>>,
    mut local_shutdown_rx: watch::Receiver<bool>,
    in_flight: Arc<AtomicUsize>,
    drained_notify: Arc<Notify>,
) {
    in_flight.fetch_add(1, Ordering::Relaxed);
    let conn_app = app.with_peer_ip(peer_addr.ip().to_canonical());
    tokio::spawn(async move {
        let io = TokioIo::new(tokio_stream);
        let svc = service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
            let mut svc_clone = conn_app.clone();
            svc_clone.call(req.map(Body::new))
        });
        let conn = conn_builder.serve_connection_with_upgrades(io, svc);
        tokio::pin!(conn);

        tokio::select! {
            biased;
            res = &mut conn => {
                let _ = res;
            }
            _ = local_shutdown_rx.changed() => {
                conn.as_mut().graceful_shutdown();
                let _ = conn.await;
            }
        }

        if in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
            drained_notify.notify_waiters();
        }
    });
}

async fn run_worker_event_loop(
    ingress: WorkerIngress,
    worker_state: AppState,
    mut shutdown_rx: watch::Receiver<bool>,
) -> std::io::Result<()> {
    let core_state = worker_state.fork_for_worker_core();
    let app = build_worker_service(core_state);
    let conn_builder = build_tuned_conn_builder();
    let in_flight = Arc::new(AtomicUsize::new(0));
    let drained_notify = Arc::new(Notify::new());
    let (local_shutdown_tx, local_shutdown_rx) = watch::channel(false);

    match ingress {
        WorkerIngress::Channel(mut rx) => loop {
            tokio::select! {
                biased;
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        break;
                    }
                }
                msg = rx.recv() => {
                    match msg {
                        Some((std_stream, peer_addr)) => {
                            if let Ok(tokio_stream) = TcpStream::from_std(std_stream) {
                                spawn_hyper_connection(
                                    tokio_stream,
                                    peer_addr,
                                    app.clone(),
                                    conn_builder.clone(),
                                    local_shutdown_rx.clone(),
                                    in_flight.clone(),
                                    drained_notify.clone(),
                                );
                            }
                        }
                        None => break,
                    }
                }
            }
        },
        #[cfg(target_os = "linux")]
        WorkerIngress::ReusePort(std_listener) => {
            let listener = TcpListener::from_std(std_listener)?;
            loop {
                tokio::select! {
                    biased;
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    accepted = listener.accept() => {
                        if let Ok((tokio_stream, peer_addr)) = accepted {
                            let _ = tokio_stream.set_nodelay(true);
                            spawn_hyper_connection(
                                tokio_stream,
                                peer_addr,
                                app.clone(),
                                conn_builder.clone(),
                                local_shutdown_rx.clone(),
                                in_flight.clone(),
                                drained_notify.clone(),
                            );
                        }
                    }
                }
            }
        }
    }

    let _ = local_shutdown_tx.send(true);
    while in_flight.load(Ordering::Acquire) > 0 {
        let notified = drained_notify.notified();
        if in_flight.load(Ordering::Acquire) == 0 {
            break;
        }
        notified.await;
    }
    Ok(())
}

/// Resolves the default worker thread count from `explicit`, `QUIXIGATE_WORKERS`, or
/// `std::thread::available_parallelism()`.
pub fn resolve_worker_count(explicit: Option<usize>) -> usize {
    if let Some(n) = explicit.filter(|&n| n > 0) {
        return n;
    }
    if let Some(n) = std::env::var("QUIXIGATE_WORKERS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1)
}

/// Serves QuixiGate across `N` share-nothing single-threaded Tokio worker runtimes (`1` per core)
/// driven by tuned `64 KiB` + `writev` `hyper_util` connections (with Linux `SO_REUSEPORT`
/// multi-queue listeners when supported, or Actix-style round-robin `Accept` distribution).
pub async fn serve_sharded_per_core<F>(
    listener: TcpListener,
    state: AppState,
    worker_count: Option<usize>,
    shutdown: F,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let workers = resolve_worker_count(worker_count);
    #[cfg(target_os = "linux")]
    let local_addr = listener.local_addr()?;

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let mut handles = Vec::with_capacity(workers);

    // On Linux, probe whether `listener` was bound with `SO_REUSEPORT` so each worker core can own
    // a dedicated kernel multi-queue listener without an intermediate distributor hop.
    #[cfg(target_os = "linux")]
    let mut extra_reuseport_listeners: Option<Vec<std::net::TcpListener>> =
        if workers > 1 && local_addr.port() != 0 {
            let mut bound = Vec::with_capacity(workers - 1);
            let mut ok = true;
            for _ in 1..workers {
                match try_bind_linux_reuseport_std(local_addr) {
                    Ok(l) => bound.push(l),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            ok.then_some(bound)
        } else {
            None
        };

    #[cfg(target_os = "linux")]
    if let Some(mut extras) = extra_reuseport_listeners.take() {
        let primary_std = listener.into_std()?;
        primary_std.set_nonblocking(true)?;
        let mut all_listeners = Vec::with_capacity(workers);
        all_listeners.push(primary_std);
        all_listeners.append(&mut extras);

        for (idx, std_l) in all_listeners.into_iter().enumerate() {
            let worker_state = state.clone();
            let s_rx = shutdown_rx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("quixigate-worker-{idx}"))
                .spawn(move || -> std::io::Result<()> {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()?;
                    rt.block_on(run_worker_event_loop(
                        WorkerIngress::ReusePort(std_l),
                        worker_state,
                        s_rx,
                    ))
                })?;
            handles.push(handle);
        }

        shutdown.await;
        let _ = shutdown_tx.send(true);
        return join_worker_handles(handles).await;
    }

    let mut worker_txs = Vec::with_capacity(workers);
    for idx in 0..workers {
        let (tx, rx) = mpsc::unbounded_channel::<(std::net::TcpStream, SocketAddr)>();
        worker_txs.push(tx);

        let worker_state = state.clone();
        let s_rx = shutdown_rx.clone();

        let handle = std::thread::Builder::new()
            .name(format!("quixigate-worker-{idx}"))
            .spawn(move || -> std::io::Result<()> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                rt.block_on(run_worker_event_loop(
                    WorkerIngress::Channel(rx),
                    worker_state,
                    s_rx,
                ))
            })?;
        handles.push(handle);
    }

    tokio::pin!(shutdown);
    let mut rr_idx = 0usize;

    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                break;
            }
            accepted = listener.accept() => {
                if let Ok((tokio_stream, peer_addr)) = accepted {
                    let _ = tokio_stream.set_nodelay(true);
                    if let Ok(std_stream) = tokio_stream.into_std() {
                        let _ = std_stream.set_nonblocking(true);
                        let target = rr_idx % workers;
                        rr_idx = rr_idx.wrapping_add(1);
                        let _ = worker_txs[target].send((std_stream, peer_addr));
                    }
                }
            }
        }
    }

    drop(worker_txs);
    drop(listener);
    let _ = shutdown_tx.send(true);

    join_worker_handles(handles).await
}

async fn join_worker_handles(
    handles: Vec<std::thread::JoinHandle<std::io::Result<()>>>,
) -> std::io::Result<()> {
    tokio::task::spawn_blocking(move || {
        let mut first_err: Option<std::io::Error> = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(std::io::Error::other("quixigate worker thread panicked"));
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    })
    .await
    .map_err(std::io::Error::other)?
}
