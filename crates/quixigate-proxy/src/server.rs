//! Actix-Style Share-Nothing Per-Core Worker Server (`serve_sharded_per_core`).
//!
//! Mirrors `actix-server`'s exact two-stage architecture:
//! 1. **Round-Robin `Accept` Distributor**: Accepts TCP sockets from the bound listener, enforces
//!    `TCP_NODELAY = true` + non-blocking mode immediately, and dispatches each raw
//!    `std::net::TcpStream` strictly round-robin (`conn_idx % workers`) across all `N` worker cores
//!    — eliminating `kqueue`/`epoll` shared-FD accept skew on long-lived HTTP keep-alive & SSE pools.
//! 2. **Share-Nothing Per-Core `current_thread` Workers**: Each of the `N` OS worker threads runs
//!    its own `tokio::runtime::Builder::new_current_thread()` event loop, registers its assigned
//!    `TcpStream`s onto that core's private `kqueue`/`epoll` reactor, and drives a worker-local
//!    `reqwest::Client` outbound pool (`state.fork_for_worker_core()`) — keeping inbound socket,
//!    `CutThroughSseStream`, and outbound socket pinned to one CPU core's L1/L2 cache.

use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Request};
use axum::serve::IncomingStream;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tower::ServiceBuilder;

use crate::build_router;
use crate::state::AppState;

struct WorkerChannelListener {
    rx: mpsc::UnboundedReceiver<(std::net::TcpStream, SocketAddr)>,
    local_addr: SocketAddr,
}

impl axum::serve::Listener for WorkerChannelListener {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.rx.recv().await {
                Some((std_stream, peer_addr)) => {
                    if let Ok(tokio_stream) = TcpStream::from_std(std_stream) {
                        return (tokio_stream, peer_addr);
                    }
                }
                None => std::future::pending().await,
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        Ok(self.local_addr)
    }
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
/// fed by an Actix-style round-robin connection distributor.
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
    let local_addr = listener.local_addr()?;

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut worker_txs = Vec::with_capacity(workers);
    let mut handles = Vec::with_capacity(workers);

    for idx in 0..workers {
        let (tx, rx) = mpsc::unbounded_channel::<(std::net::TcpStream, SocketAddr)>();
        worker_txs.push(tx);

        let worker_state = state.clone();
        let mut s_rx = shutdown_rx.clone();

        let handle = std::thread::Builder::new()
            .name(format!("quixigate-worker-{idx}"))
            .spawn(move || -> std::io::Result<()> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                rt.block_on(async move {
                    let worker_listener = WorkerChannelListener { rx, local_addr };
                    let core_state = worker_state.fork_for_worker_core();
                    let app = build_router(core_state);
                    let make_svc = tower::service_fn(
                        move |stream: IncomingStream<'_, WorkerChannelListener>| {
                            let peer = *stream.remote_addr();
                            let svc = ServiceBuilder::new()
                                .map_request(move |mut req: Request| {
                                    req.extensions_mut().insert(ConnectInfo(peer));
                                    req
                                })
                                .service(app.clone());
                            std::future::ready(Ok::<_, Infallible>(svc))
                        },
                    );
                    axum::serve(worker_listener, make_svc)
                        .with_graceful_shutdown(async move {
                            while !*s_rx.borrow_and_update() {
                                if s_rx.changed().await.is_err() {
                                    break;
                                }
                            }
                        })
                        .await
                })
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
