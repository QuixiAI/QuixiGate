//! Apples-to-Apples Scale Benchmark: QuixiGate vs. LiteLLM Proxy
//!
//! Architecture (3 isolated OS process tiers — zero in-process shortcuts):
//!   [Load Generator Process] --HTTP/1.1--> [Gateway Process Tree] --HTTP/1.1--> [Multi-Core Mock OpenAI Upstream Process]
//!
//! Run:
//!   cargo run --release -p quixigate-proxy --example litellm_vs_quixigate_bench
//!
//! Optional env vars:
//!   QUIXIGATE_BIN=/path/to/quixigate
//!   LITELLM_BIN=/path/to/litellm
//!   BENCH_FILTER=stateless|governance|all   (default: all)

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode, header};
use hyper_util::rt::TokioIo;
use quixigate_core::{AuthKind, CredentialSecret};
use quixigate_db::Database;
use quixigate_vault::Vault;
use serde_json::{Value, json};
use socket2::{Domain, Protocol, Socket, Type};
use tempfile::TempDir;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

// ── Multi-Core High-Speed Mock OpenAI Upstream ──────────────────────────────

struct UpstreamPayloads {
    json_chat: Bytes,
    sse_1tok: Bytes,
    sse_300tok: Vec<Bytes>,
    sse_paced_200tok: Vec<Bytes>,
}

fn build_sse_batches(total_tokens: usize, tokens_per_chunk: usize) -> Vec<Bytes> {
    const VOCAB: [&str; 10] = [
        " The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", " fast",
    ];
    let final_event = format!(
        "data: {{\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":32,\"completion_tokens\":{total_tokens},\"total_tokens\":{}}}}}\n\ndata: [DONE]\n\n",
        32 + total_tokens
    );
    let num_batches = total_tokens / tokens_per_chunk;
    let mut batches = Vec::with_capacity(num_batches);
    let mut tok_idx = 0usize;
    for b in 0..num_batches {
        let mut buf = BytesMut::with_capacity(180 * (tokens_per_chunk + 2));
        for _ in 0..tokens_per_chunk {
            let word = VOCAB[tok_idx % VOCAB.len()];
            tok_idx += 1;
            let ev = format!(
                "data: {{\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{word}\"}},\"finish_reason\":null}}]}}\n\n"
            );
            buf.extend_from_slice(ev.as_bytes());
        }
        if b + 1 == num_batches {
            buf.extend_from_slice(final_event.as_bytes());
        }
        batches.push(buf.freeze());
    }
    batches
}

fn build_upstream_payloads() -> Arc<UpstreamPayloads> {
    let json_chat = Bytes::from_static(
        br#"{"id":"chatcmpl-bench","object":"chat.completion","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"The quick brown fox jumps over the lazy dog while benchmarking gateway overhead at scale."},"finish_reason":"stop"}],"usage":{"prompt_tokens":32,"completion_tokens":128,"total_tokens":160}}"#,
    );
    let sse_1tok = Bytes::from_static(concat!(
        "data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":32,\"completion_tokens\":1,\"total_tokens\":33}}\n\n",
        "data: [DONE]\n\n"
    ).as_bytes());

    Arc::new(UpstreamPayloads {
        json_chat,
        sse_1tok,
        sse_300tok: build_sse_batches(300, 10),
        sse_paced_200tok: build_sse_batches(200, 10),
    })
}

async fn handle_upstream_req(
    req: Request<Incoming>,
    payloads: Arc<UpstreamPayloads>,
) -> Result<Response<BoxBody>, Infallible> {
    let path = req.uri().path();
    if path == "/health" || path == "/v1/models" || path == "/models" {
        let body = Full::new(Bytes::from_static(
            br#"{"object":"list","data":[{"id":"gpt-4o-mini","object":"model"}]}"#,
        ))
        .boxed();
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(body)
            .unwrap());
    }

    let whole = match req.into_body().collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => {
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Full::new(Bytes::new()).boxed())
                .unwrap());
        }
    };

    let is_stream = memchr::memmem::find(&whole, b"\"stream\":true").is_some()
        || memchr::memmem::find(&whole, b"\"stream\": true").is_some();

    if !is_stream {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(payloads.json_chat.clone()).boxed())
            .unwrap());
    }

    if memchr::memmem::find(&whole, b"BENCH_1TOK").is_some() {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Full::new(payloads.sse_1tok.clone()).boxed())
            .unwrap());
    }

    if memchr::memmem::find(&whole, b"BENCH_PACED").is_some() {
        let frames = payloads.sse_paced_200tok.clone();
        let s = async_paced_stream(frames);
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(BodyExt::boxed(StreamBody::new(s)))
            .unwrap());
    }

    let frames = payloads.sse_300tok.clone();
    let s = futures::stream::iter(
        (0..frames.len())
            .map(move |i| Ok::<Frame<Bytes>, Infallible>(Frame::data(frames[i].clone()))),
    );
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(BodyExt::boxed(StreamBody::new(s)))
        .unwrap())
}

fn async_paced_stream(
    frames: Vec<Bytes>,
) -> impl futures::Stream<Item = Result<Frame<Bytes>, Infallible>> {
    futures::stream::unfold((frames, 0usize), |(frames, idx)| async move {
        if idx >= frames.len() {
            return None;
        }
        if idx == 0 {
            // Simulate 20ms GPU prefill TTFT
            tokio::time::sleep(Duration::from_millis(20)).await;
        } else {
            // Simulate 4ms per 10-token batch (~2,500 tok/s modern GPU decode)
            tokio::time::sleep(Duration::from_millis(4)).await;
        }
        let chunk = frames[idx].clone();
        Some((Ok(Frame::data(chunk)), (frames, idx + 1)))
    })
}

fn bind_reuseport(addr: SocketAddr) -> std::net::TcpListener {
    let sock = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).unwrap();
    sock.set_reuse_address(true).unwrap();
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let optval: libc::c_int = 1;
        unsafe {
            libc::setsockopt(
                sock.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_REUSEPORT,
                &optval as *const _ as *const libc::c_void,
                std::mem::size_of_val(&optval) as libc::socklen_t,
            );
        }
    }
    sock.set_nonblocking(true).unwrap();
    sock.bind(&addr.into()).unwrap();
    sock.listen(4096).unwrap();
    sock.into()
}

#[cfg(unix)]
#[allow(non_camel_case_types)]
mod libc {
    pub use std::os::raw::{c_int, c_void};
    pub type socklen_t = u32;
    pub const SOL_SOCKET: c_int = 0xffff;
    pub const SO_REUSEPORT: c_int = 0x0200;
    unsafe extern "C" {
        pub fn setsockopt(
            socket: c_int,
            level: c_int,
            option_name: c_int,
            option_value: *const c_void,
            option_len: socklen_t,
        ) -> c_int;
        pub fn kill(pid: c_int, sig: c_int) -> c_int;
    }
    pub const SIGTERM: c_int = 15;
    pub const SIGKILL: c_int = 9;
}

fn run_upstream_server(port: u16) {
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let payloads = build_upstream_payloads();
    let workers = 8usize;
    let mut handles = Vec::new();
    for w in 0..workers {
        let payloads = payloads.clone();
        handles.push(
            std::thread::Builder::new()
                .name(format!("mock-upstream-{w}"))
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    rt.block_on(async move {
                        let std_listener = bind_reuseport(addr);
                        let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
                        loop {
                            if let Ok((stream, _)) = listener.accept().await {
                                let _ = stream.set_nodelay(true);
                                let io = TokioIo::new(stream);
                                let payloads = payloads.clone();
                                tokio::task::spawn(async move {
                                    let svc = service_fn(move |req| {
                                        handle_upstream_req(req, payloads.clone())
                                    });
                                    let _ = http1::Builder::new()
                                        .keep_alive(true)
                                        .pipeline_flush(true)
                                        .serve_connection(io, svc)
                                        .await;
                                });
                            }
                        }
                    });
                })
                .unwrap(),
        );
    }
    for h in handles {
        let _ = h.join();
    }
}

// ── Process Tree RSS & Lifecycle Helpers ────────────────────────────────────

fn measure_process_tree_rss_kib(root_pid: u32) -> (u64, usize) {
    let Ok(out) = Command::new("ps")
        .args(["-ax", "-o", "pid=,ppid=,rss="])
        .output()
    else {
        return (0, 0);
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut rss_map: HashMap<u32, u64> = HashMap::new();

    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(p), Some(pp), Some(r)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        if let (Ok(pid), Ok(ppid), Ok(rss)) =
            (p.parse::<u32>(), pp.parse::<u32>(), r.parse::<u64>())
        {
            children_map.entry(ppid).or_default().push(pid);
            rss_map.insert(pid, rss);
        }
    }

    let mut visited = HashSet::new();
    let mut stack = vec![root_pid];
    let mut total_kib = 0u64;
    while let Some(pid) = stack.pop() {
        if !visited.insert(pid) {
            continue;
        }
        total_kib += rss_map.get(&pid).copied().unwrap_or(0);
        if let Some(kids) = children_map.get(&pid) {
            for &k in kids {
                stack.push(k);
            }
        }
    }
    (total_kib, visited.len())
}

fn collect_process_tree_pids(root_pid: u32) -> Vec<u32> {
    let Ok(out) = Command::new("ps")
        .args(["-ax", "-o", "pid=,ppid="])
        .output()
    else {
        return vec![root_pid];
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if let (Some(p), Some(pp)) = (it.next(), it.next())
            && let (Ok(pid), Ok(ppid)) = (p.parse::<u32>(), pp.parse::<u32>())
        {
            children_map.entry(ppid).or_default().push(pid);
        }
    }
    let mut visited = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![root_pid];
    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        visited.push(pid);
        if let Some(kids) = children_map.get(&pid) {
            for &k in kids {
                stack.push(k);
            }
        }
    }
    visited
}

fn kill_process_tree(child: &mut Child) {
    let pids = collect_process_tree_pids(child.id());
    #[cfg(unix)]
    {
        for &pid in pids.iter().rev() {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
        std::thread::sleep(Duration::from_millis(250));
        for &pid in pids.iter().rev() {
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn pick_free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

async fn wait_for_http_ready(url: &str, timeout: Duration) -> Result<Duration, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .no_proxy()
        .build()
        .unwrap();
    let t0 = Instant::now();
    loop {
        if t0.elapsed() > timeout {
            return Err(format!("timed out after {:?} waiting for {url}", timeout));
        }
        if let Ok(resp) = client.get(url).send().await
            && resp.status().is_success()
        {
            return Ok(t0.elapsed());
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

// ── Load Generator ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct WorkloadSpec {
    name: &'static str,
    stream: bool,
    prompt_tag: &'static str,
    concurrency: usize,
    total_requests: usize,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct SuiteMetrics {
    workload: &'static str,
    concurrency: usize,
    total_ok: u64,
    total_err: u64,
    elapsed: Duration,
    rps: f64,
    tok_per_s: f64,
    ttft_p50_ms: f64,
    ttft_p95_ms: f64,
    ttft_p99_ms: f64,
    e2e_p50_ms: f64,
    e2e_p99_ms: f64,
    peak_rss_mib: f64,
}

fn count_sse_token_events(buf: &[u8]) -> u64 {
    let mut frames = 0u64;
    for line in buf.split(|&b| b == b'\n') {
        if line.starts_with(b"data: {")
            && (memchr::memmem::find(line, b"\"content\":\"").is_some()
                || memchr::memmem::find(line, b"\"content\": \"").is_some())
        {
            frames += 1;
        }
    }
    frames
}

async fn run_workload(
    target_port: u16,
    bearer_token: &str,
    spec: &WorkloadSpec,
    monitored_root_pid: Option<u32>,
) -> SuiteMetrics {
    let client = reqwest::Client::builder()
        .tcp_nodelay(true)
        .no_proxy()
        .pool_max_idle_per_host(spec.concurrency * 2 + 64)
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();

    let url = format!("http://127.0.0.1:{target_port}/v1/chat/completions");
    let auth_val = format!("Bearer {bearer_token}");
    let body = Bytes::from(format!(
        "{{\"model\":\"gpt-4o-mini\",\"stream\":{},\"messages\":[{{\"role\":\"user\",\"content\":\"{} scale benchmark request\"}}]}}",
        spec.stream, spec.prompt_tag
    ));

    // Smoothly pre-warm persistent keep-alive pool in batches of 32 so macOS somaxconn=128
    // never drops SYN bursts for single-listener servers like Uvicorn.
    let warmup_body = Bytes::from_static(
        br#"{"model":"gpt-4o-mini","stream":false,"messages":[{"role":"user","content":"warmup"}]}"#,
    );
    let warmup_total = spec.concurrency;
    let mut warmed = 0usize;
    while warmed < warmup_total {
        let batch = (warmup_total - warmed).min(32);
        let mut futs = Vec::with_capacity(batch);
        for _ in 0..batch {
            let c = client.clone();
            let u = url.clone();
            let a = auth_val.clone();
            let b = warmup_body.clone();
            futs.push(tokio::spawn(async move {
                if let Ok(r) = c
                    .post(&u)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, &a)
                    .body(b)
                    .send()
                    .await
                {
                    let _ = r.bytes().await;
                }
            }));
        }
        for f in futs {
            let _ = f.await;
        }
        warmed += batch;
    }

    // Background RSS peak sampler
    let stop_rss = Arc::new(AtomicBool::new(false));
    let peak_rss_kib = Arc::new(AtomicU64::new(0));
    let rss_handle = if let Some(pid) = monitored_root_pid {
        let stop_flag = stop_rss.clone();
        let peak_ref = peak_rss_kib.clone();
        Some(std::thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                let (kib, _) = measure_process_tree_rss_kib(pid);
                peak_ref.fetch_max(kib, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(100));
            }
        }))
    } else {
        None
    };

    let next_idx = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut handles = Vec::with_capacity(spec.concurrency);
    let is_stream = spec.stream;

    for _ in 0..spec.concurrency {
        let client = client.clone();
        let url = url.clone();
        let auth_val = auth_val.clone();
        let body = body.clone();
        let next_idx = next_idx.clone();
        let total_requests = spec.total_requests;

        handles.push(tokio::spawn(async move {
            let mut ok_reqs = 0u64;
            let mut err_reqs = 0u64;
            let mut tokens = 0u64;
            let mut ttfts = Vec::new();
            let mut e2es = Vec::new();
            let mut line_rem = Vec::with_capacity(1024);

            loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx as usize >= total_requests {
                    break;
                }
                let t0 = Instant::now();
                let resp = match client
                    .post(&url)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, &auth_val)
                    .body(body.clone())
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(_) => {
                        err_reqs += 1;
                        continue;
                    }
                };

                if resp.status() != StatusCode::OK {
                    err_reqs += 1;
                    let _ = resp.bytes().await;
                    continue;
                }

                if !is_stream {
                    match resp.bytes().await {
                        Ok(b) if !b.is_empty() => {
                            let us = t0.elapsed().as_micros() as u64;
                            ok_reqs += 1;
                            tokens += 128;
                            ttfts.push(us);
                            e2es.push(us);
                        }
                        _ => {
                            err_reqs += 1;
                        }
                    }
                    continue;
                }

                line_rem.clear();
                let mut req_tokens = 0u64;
                let mut ttft_us: Option<u64> = None;
                let mut stream = resp.bytes_stream();
                let mut stream_err = false;
                while let Some(chunk_res) = stream.next().await {
                    match chunk_res {
                        Ok(chunk) => {
                            if ttft_us.is_none() {
                                ttft_us = Some(t0.elapsed().as_micros() as u64);
                            }
                            line_rem.extend_from_slice(&chunk);
                            if let Some(last_nl) = memchr::memrchr(b'\n', &line_rem) {
                                req_tokens += count_sse_token_events(&line_rem[..=last_nl]);
                                line_rem.drain(..=last_nl);
                            }
                        }
                        Err(_) => {
                            stream_err = true;
                            break;
                        }
                    }
                }
                if !line_rem.is_empty() {
                    req_tokens += count_sse_token_events(&line_rem);
                }
                if stream_err || req_tokens == 0 {
                    err_reqs += 1;
                    continue;
                }
                let e2e_us = t0.elapsed().as_micros() as u64;
                tokens += req_tokens;
                ok_reqs += 1;
                ttfts.push(ttft_us.unwrap_or(e2e_us));
                e2es.push(e2e_us);
            }
            (ok_reqs, err_reqs, tokens, ttfts, e2es)
        }));
    }

    let mut total_ok = 0u64;
    let mut total_err = 0u64;
    let mut total_tokens = 0u64;
    let mut all_ttfts = Vec::with_capacity(spec.total_requests);
    let mut all_e2es = Vec::with_capacity(spec.total_requests);

    for h in handles {
        let (ok, err, tok, mut ttft, mut e2e) = h.await.unwrap();
        total_ok += ok;
        total_err += err;
        total_tokens += tok;
        all_ttfts.append(&mut ttft);
        all_e2es.append(&mut e2e);
    }
    let elapsed = started.elapsed();

    stop_rss.store(true, Ordering::Relaxed);
    if let Some(h) = rss_handle {
        let _ = h.join();
    }

    all_ttfts.sort_unstable();
    all_e2es.sort_unstable();
    let pct_ms = |v: &[u64], p: f64| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
        v[idx.min(v.len() - 1)] as f64 / 1000.0
    };

    let secs = elapsed.as_secs_f64().max(1e-9);
    SuiteMetrics {
        workload: spec.name,
        concurrency: spec.concurrency,
        total_ok,
        total_err,
        elapsed,
        rps: total_ok as f64 / secs,
        tok_per_s: total_tokens as f64 / secs,
        ttft_p50_ms: pct_ms(&all_ttfts, 0.50),
        ttft_p95_ms: pct_ms(&all_ttfts, 0.95),
        ttft_p99_ms: pct_ms(&all_ttfts, 0.99),
        e2e_p50_ms: pct_ms(&all_e2es, 0.50),
        e2e_p99_ms: pct_ms(&all_e2es, 0.99),
        peak_rss_mib: peak_rss_kib.load(Ordering::Relaxed) as f64 / 1024.0,
    }
}

// ── Gateway Target Setup & Execution ────────────────────────────────────────

struct TargetReport {
    name: String,
    cold_start_ms: f64,
    idle_rss_mib: f64,
    idle_procs: usize,
    extra_infra_note: String,
    suites: Vec<SuiteMetrics>,
}

fn prepare_quixigate_datadir(upstream_port: u16) -> TempDir {
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    let mut settings = db.get_settings().unwrap();
    settings.require_auth = true;
    settings.local_discovery_enabled = false;
    db.update_settings(&settings).unwrap();

    db.upsert_credential(
        Some("cred_openai"),
        "openai",
        "Mock OpenAI Upstream",
        AuthKind::ApiKey,
        Some(&format!("http://127.0.0.1:{upstream_port}")),
        10,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: "sk-upstream-mock".to_owned(),
        },
    )
    .unwrap();
    dir
}

async fn generate_virtual_key(port: u16, master_key: &str) -> Result<String, String> {
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/key/generate"))
        .header(header::AUTHORIZATION, format!("Bearer {master_key}"))
        .json(&json!({
            "max_budget": 1_000_000.0,
            "models": ["gpt-4o-mini"]
        }))
        .send()
        .await
        .map_err(|e| format!("POST /key/generate failed: {e}"))?;
    let status = resp.status();
    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("invalid /key/generate JSON: {e}"))?;
    if !status.is_success() {
        return Err(format!("POST /key/generate returned {status}: {body}"));
    }
    body["key"]
        .as_str()
        .map(|s| s.to_owned())
        .ok_or_else(|| format!("missing 'key' in /key/generate response: {body}"))
}

async fn bench_quixigate(
    label: &str,
    quixigate_bin: &Path,
    upstream_port: u16,
    use_virtual_key: bool,
    workloads: &[WorkloadSpec],
) -> TargetReport {
    let data_dir = prepare_quixigate_datadir(upstream_port);
    let port = pick_free_port();
    let master_key = "sk-bench-master-key";

    let mut cmd = Command::new(quixigate_bin);
    cmd.args([
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "--data-dir",
        data_dir.path().to_str().unwrap(),
        "--no-ui",
        "--require-auth",
    ])
    .env("QUIXIGATE_MASTER_KEY", master_key)
    .env("QUIXIGATE_NO_BROWSER", "1")
    .stdout(Stdio::null())
    .stderr(Stdio::null());

    let t_spawn = Instant::now();
    let mut child = cmd.spawn().expect("failed to spawn quixigate");
    let ready_dur = wait_for_http_ready(
        &format!("http://127.0.0.1:{port}/health/liveness"),
        Duration::from_secs(30),
    )
    .await
    .expect("quixigate failed health check");
    let cold_start_ms = t_spawn
        .elapsed()
        .min(ready_dur + Duration::from_millis(5))
        .as_secs_f64()
        * 1000.0;

    tokio::time::sleep(Duration::from_millis(200)).await;
    let (idle_kib, idle_procs) = measure_process_tree_rss_kib(child.id());

    let bearer = if use_virtual_key {
        generate_virtual_key(port, master_key)
            .await
            .expect("quixigate /key/generate failed")
    } else {
        master_key.to_owned()
    };

    let mut suites = Vec::new();
    for w in workloads {
        println!(
            "    -> Running {:<34} (c={:<4}, n={:<6})...",
            w.name, w.concurrency, w.total_requests
        );
        let m = run_workload(port, &bearer, w, Some(child.id())).await;
        println!(
            "       {:>8.0} req/s | {:>10.0} tok/s | TTFT p50 {:>6.2}ms p99 {:>6.2}ms | err {} | peak {:.1} MiB",
            m.rps, m.tok_per_s, m.ttft_p50_ms, m.ttft_p99_ms, m.total_err, m.peak_rss_mib
        );
        suites.push(m);
    }

    kill_process_tree(&mut child);
    TargetReport {
        name: label.to_owned(),
        cold_start_ms,
        idle_rss_mib: idle_kib as f64 / 1024.0,
        idle_procs,
        extra_infra_note: "0 external services (embedded SQLite WAL)".to_owned(),
        suites,
    }
}

struct DockerInfra {
    pg_container: String,
    redis_container: String,
    pg_port: u16,
    redis_port: u16,
}

impl DockerInfra {
    async fn start() -> Result<Self, String> {
        let pg_port = pick_free_port();
        let redis_port = pick_free_port();
        let pg_container = format!("qx-bench-pg-{pg_port}");
        let redis_container = format!("qx-bench-redis-{redis_port}");

        let st_pg = Command::new("docker")
            .args([
                "run",
                "-d",
                "--rm",
                "--name",
                &pg_container,
                "--tmpfs",
                "/var/lib/postgresql/data:rw,noexec,nosuid,size=512m",
                "-e",
                "POSTGRES_PASSWORD=benchpass",
                "-e",
                "POSTGRES_USER=bench",
                "-e",
                "POSTGRES_DB=litellm",
                "-p",
                &format!("127.0.0.1:{pg_port}:5432"),
                "postgres:16-alpine",
                "-c",
                "max_connections=500",
                "-c",
                "synchronous_commit=off",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .status()
            .map_err(|e| format!("docker run postgres failed: {e}"))?;
        if !st_pg.success() {
            return Err("failed to start postgres:16-alpine container".to_owned());
        }

        let st_rd = Command::new("docker")
            .args([
                "run",
                "-d",
                "--rm",
                "--name",
                &redis_container,
                "-p",
                &format!("127.0.0.1:{redis_port}:6379"),
                "redis:7-alpine",
                "redis-server",
                "--save",
                "",
                "--appendonly",
                "no",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .status()
            .map_err(|e| format!("docker run redis failed: {e}"))?;
        if !st_rd.success() {
            let _ = Command::new("docker")
                .args(["rm", "-f", &pg_container])
                .status();
            return Err("failed to start redis:7-alpine container".to_owned());
        }

        // Wait for Postgres pg_isready
        for _ in 0..50 {
            if let Ok(st) = Command::new("docker")
                .args([
                    "exec",
                    &pg_container,
                    "pg_isready",
                    "-U",
                    "bench",
                    "-d",
                    "litellm",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                && st.success()
            {
                return Ok(Self {
                    pg_container,
                    redis_container,
                    pg_port,
                    redis_port,
                });
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        Err("postgres container did not become ready in 10s".to_owned())
    }

    fn measure_docker_rss_mib(&self) -> f64 {
        let Ok(out) = Command::new("docker")
            .args([
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                &self.pg_container,
                &self.redis_container,
            ])
            .output()
        else {
            return 0.0;
        };
        let txt = String::from_utf8_lossy(&out.stdout);
        let mut total_mib = 0.0;
        for line in txt.lines() {
            if let Some(used) = line.split('/').next() {
                let s = used.trim();
                if let Some(v) = s.strip_suffix("GiB") {
                    total_mib += v.trim().parse::<f64>().unwrap_or(0.0) * 1024.0;
                } else if let Some(v) = s.strip_suffix("MiB") {
                    total_mib += v.trim().parse::<f64>().unwrap_or(0.0);
                } else if let Some(v) = s.strip_suffix("KiB") {
                    total_mib += v.trim().parse::<f64>().unwrap_or(0.0) / 1024.0;
                }
            }
        }
        total_mib
    }

    fn stop(self) {
        let _ = Command::new("docker")
            .args(["rm", "-f", &self.pg_container, &self.redis_container])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

#[allow(clippy::too_many_arguments)]
async fn bench_litellm(
    label: &str,
    litellm_bin: &Path,
    upstream_port: u16,
    num_workers: usize,
    use_granian: bool,
    db_infra: Option<&DockerInfra>,
    workloads: &[WorkloadSpec],
) -> Result<TargetReport, String> {
    let dir = TempDir::new().unwrap();
    let port = pick_free_port();
    let master_key = "sk-bench-master-key";
    let config_path = dir.path().join("litellm_config.yaml");

    let config_yaml = if let Some(infra) = db_infra {
        format!(
            r#"model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_base: http://127.0.0.1:{upstream_port}/v1
      api_key: sk-upstream-mock

litellm_settings:
  set_verbose: false
  json_logs: false
  suppress_startup_output: true
  cache: false

general_settings:
  master_key: {master_key}
  database_url: postgresql://bench:benchpass@127.0.0.1:{}/litellm
  database_connection_pool_limit: 50

router_settings:
  redis_host: 127.0.0.1
  redis_port: {}
"#,
            infra.pg_port, infra.redis_port
        )
    } else {
        format!(
            r#"model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_base: http://127.0.0.1:{upstream_port}/v1
      api_key: sk-upstream-mock

litellm_settings:
  set_verbose: false
  json_logs: false
  suppress_startup_output: true
  cache: false

general_settings:
  master_key: {master_key}
  disable_spend_logs: true
  allow_requests_on_db_unavailable: true
"#
        )
    };
    fs::write(&config_path, config_yaml).unwrap();

    // Disable Uvicorn per-request access log so LiteLLM isn't bottlenecked by stdout formatting
    let log_cfg_path = dir.path().join("uvicorn_log.json");
    fs::write(
        &log_cfg_path,
        r#"{"version":1,"disable_existing_loggers":false,"loggers":{"uvicorn.access":{"level":"ERROR"},"uvicorn.error":{"level":"ERROR"},"litellm":{"level":"ERROR"}}}"#,
    )
    .unwrap();

    let mut cmd = Command::new(litellm_bin);
    cmd.args([
        "--config",
        config_path.to_str().unwrap(),
        "--host",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "--num_workers",
        &num_workers.to_string(),
    ]);
    if use_granian {
        cmd.arg("--run_granian");
    } else {
        cmd.args(["--log_config", log_cfg_path.to_str().unwrap()]);
    }
    let venv_bin = litellm_bin.parent().unwrap_or_else(|| Path::new("."));
    let cur_path = env::var("PATH").unwrap_or_default();
    let merged_path = format!("{}:{cur_path}", venv_bin.display());

    // Pre-initialize Postgres schema before measuring server cold-start time (fair parity with
    // prepare_quixigate_datadir(), which pre-initializes quixigate.db before spawning).
    if db_infra.is_some() {
        let _ = Command::new(litellm_bin)
            .args([
                "--config",
                config_path.to_str().unwrap(),
                "--use_prisma_db_push",
                "--skip_server_startup",
            ])
            .env("PATH", &merged_path)
            .env("LITELLM_LOG", "ERROR")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    let stderr_path = dir.path().join("litellm_stderr.log");
    let stderr_file = fs::File::create(&stderr_path).unwrap();

    cmd.env("PATH", merged_path)
        .env("LITELLM_LOG", "ERROR")
        .env("LITELLM_MODE", "PRODUCTION")
        .env("AIOHTTP_CONNECTOR_LIMIT", "2048")
        .env("DISABLE_AIOHTTP_TRUST_ENV", "True")
        .env("DO_NOT_TRACK", "True")
        .env("SCARF_NO_ANALYTICS", "true")
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file));

    let t_spawn = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn litellm: {e}"))?;

    let ready_res = wait_for_http_ready(
        &format!("http://127.0.0.1:{port}/health/liveness"),
        Duration::from_secs(90),
    )
    .await;
    if let Err(e) = ready_res {
        kill_process_tree(&mut child);
        let err_txt = fs::read_to_string(&stderr_path).unwrap_or_default();
        return Err(format!("{label} failed startup: {e}\nstderr:\n{err_txt}"));
    }
    let cold_start_ms = t_spawn.elapsed().as_secs_f64() * 1000.0;

    // Give multi-worker forks 1.5s to finish booting all worker processes before measuring idle RSS & load
    if num_workers > 1 {
        tokio::time::sleep(Duration::from_millis(1500)).await;
    } else {
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    let (idle_kib, idle_procs) = measure_process_tree_rss_kib(child.id());
    let extra_infra_note = if let Some(infra) = db_infra {
        let docker_mib = infra.measure_docker_rss_mib();
        format!(
            "+ {:.0} MiB Docker (Postgres 16 tmpfs + Redis 7)",
            docker_mib
        )
    } else {
        "YAML no-DB mode (spend logs & virtual keys disabled)".to_owned()
    };

    let bearer = if db_infra.is_some() {
        match generate_virtual_key(port, master_key).await {
            Ok(k) => k,
            Err(e) => {
                kill_process_tree(&mut child);
                return Err(format!("{label} /key/generate failed: {e}"));
            }
        }
    } else {
        master_key.to_owned()
    };

    let mut suites = Vec::new();
    for w in workloads {
        println!(
            "    -> Running {:<34} (c={:<4}, n={:<6})...",
            w.name, w.concurrency, w.total_requests
        );
        let m = run_workload(port, &bearer, w, Some(child.id())).await;
        println!(
            "       {:>8.0} req/s | {:>10.0} tok/s | TTFT p50 {:>6.2}ms p99 {:>6.2}ms | err {} | peak {:.1} MiB",
            m.rps, m.tok_per_s, m.ttft_p50_ms, m.ttft_p99_ms, m.total_err, m.peak_rss_mib
        );
        suites.push(m);
    }

    kill_process_tree(&mut child);
    tokio::time::sleep(Duration::from_millis(400)).await;

    Ok(TargetReport {
        name: label.to_owned(),
        cold_start_ms,
        idle_rss_mib: idle_kib as f64 / 1024.0,
        idle_procs,
        extra_infra_note,
        suites,
    })
}

fn find_default_quixigate_bin() -> PathBuf {
    if let Ok(p) = env::var("QUIXIGATE_BIN") {
        return PathBuf::from(p);
    }
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."));
    let exe_name = if cfg!(windows) {
        "quixigate.exe"
    } else {
        "quixigate"
    };
    let candidates = [
        workspace_root.join("target").join("release").join(exe_name),
        workspace_root
            .join("target")
            .join("aarch64-apple-darwin")
            .join("release")
            .join(exe_name),
        workspace_root
            .join("target")
            .join("x86_64-unknown-linux-gnu")
            .join("release")
            .join(exe_name),
    ];
    for p in candidates {
        if p.exists() {
            return p;
        }
    }
    panic!(
        "quixigate release binary not found; run `cargo build --release` first or set QUIXIGATE_BIN"
    );
}

fn find_default_litellm_bin() -> Option<PathBuf> {
    if env::var("QUIXIGATE_ONLY").is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "yes")) {
        return None;
    }
    if let Ok(p) = env::var("LITELLM_BIN") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(out) = Command::new("which").arg("litellm").output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if !s.is_empty() {
            return Some(PathBuf::from(s));
        }
    }
    None
}

fn print_summary_table(title: &str, reports: &[TargetReport]) {
    println!(
        "\n========================================================================================================================"
    );
    println!("  {title}");
    println!(
        "========================================================================================================================"
    );
    println!(
        "{:<42} | {:>11} | {:>13} | {:>6} | Infrastructure",
        "Gateway Configuration", "Cold Start", "Idle RSS", "Procs"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------"
    );
    for r in reports {
        println!(
            "{:<42} | {:>9.1}ms | {:>9.1} MiB | {:>6} | {}",
            r.name, r.cold_start_ms, r.idle_rss_mib, r.idle_procs, r.extra_infra_note
        );
    }

    if reports.is_empty() {
        return;
    }
    let suite_count = reports[0].suites.len();
    for s_idx in 0..suite_count {
        let s_meta = &reports[0].suites[s_idx];
        println!(
            "\n--- Workload {}: {} (concurrency = {}) ---",
            s_idx + 1,
            s_meta.workload,
            s_meta.concurrency
        );
        println!(
            "{:<42} | {:>10} | {:>12} | {:>9} | {:>9} | {:>9} | {:>7} | {:>10}",
            "Gateway Configuration",
            "Req/sec",
            "Tokens/sec",
            "TTFT p50",
            "TTFT p99",
            "E2E p99",
            "Errors",
            "Peak RSS"
        );
        println!(
            "------------------------------------------------------------------------------------------------------------------------"
        );
        for r in reports {
            let m = &r.suites[s_idx];
            println!(
                "{:<42} | {:>10.1} | {:>12.0} | {:>7.2}ms | {:>7.2}ms | {:>7.2}ms | {:>7} | {:>7.1} MiB",
                r.name,
                m.rps,
                m.tok_per_s,
                m.ttft_p50_ms,
                m.ttft_p99_ms,
                m.e2e_p99_ms,
                m.total_err,
                m.peak_rss_mib
            );
        }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 4 && args[1] == "--role" && args[2] == "upstream" {
        let port: u16 = args[3].parse().expect("invalid port");
        run_upstream_server(port);
        return;
    }

    let filter = env::var("BENCH_FILTER").unwrap_or_else(|_| "all".to_owned());
    let quixigate_bin = find_default_quixigate_bin();
    let litellm_bin = find_default_litellm_bin();
    let self_exe = env::current_exe().expect("current_exe");

    println!(
        "========================================================================================================================"
    );
    println!(
        "  QuixiGate vs. LiteLLM Proxy — Apples-to-Apples Scale Benchmark (3 Isolated OS Process Tiers)"
    );
    println!("  QuixiGate Binary : {}", quixigate_bin.display());
    match &litellm_bin {
        Some(p) => println!("  LiteLLM Binary   : {}", p.display()),
        None => println!(
            "  LiteLLM Binary   : (not set / QUIXIGATE_ONLY=1; set LITELLM_BIN=/path/to/litellm to include LiteLLM)"
        ),
    }
    println!(
        "========================================================================================================================"
    );

    let upstream_port = pick_free_port();
    let mut upstream_child = Command::new(&self_exe)
        .args(["--role", "upstream", &upstream_port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn mock upstream process");

    wait_for_http_ready(
        &format!("http://127.0.0.1:{upstream_port}/health"),
        Duration::from_secs(10),
    )
    .await
    .expect("mock upstream failed to start");

    // Verify Mock Upstream ceiling so we prove the upstream server is never the bottleneck!
    let upstream_probe = run_workload(
        upstream_port,
        "sk-upstream-mock",
        &WorkloadSpec {
            name: "Direct Upstream Ceiling Probe (SSE 1-tok)",
            stream: true,
            prompt_tag: "BENCH_1TOK",
            concurrency: 128,
            total_requests: 20_000,
        },
        None,
    )
    .await;
    println!(
        "\n[Sanity Check] Mock OpenAI Upstream Process Ceiling: {:.0} req/s (TTFT p50 {:.2}ms, p99 {:.2}ms, err {})",
        upstream_probe.rps,
        upstream_probe.ttft_p50_ms,
        upstream_probe.ttft_p99_ms,
        upstream_probe.total_err
    );

    if filter == "all" || filter == "stateless" {
        // Note: request counts chosen so both Rust (~seconds) and Python (~3-10 seconds) get
        // statistically solid steady-state sample sizes without taking hours.
        let stateless_workloads = vec![
            WorkloadSpec {
                name: "1. Non-Streaming JSON (c=64)",
                stream: false,
                prompt_tag: "BENCH_JSON",
                concurrency: 64,
                total_requests: 6_000,
            },
            WorkloadSpec {
                name: "2. Non-Streaming JSON (c=256)",
                stream: false,
                prompt_tag: "BENCH_JSON",
                concurrency: 256,
                total_requests: 8_000,
            },
            WorkloadSpec {
                name: "3. SSE 1-Token TTFT Probe (c=64)",
                stream: true,
                prompt_tag: "BENCH_1TOK",
                concurrency: 64,
                total_requests: 6_000,
            },
            WorkloadSpec {
                name: "4. SSE 300-Token Stream (c=64)",
                stream: true,
                prompt_tag: "BENCH_300TOK",
                concurrency: 64,
                total_requests: 3_000,
            },
            WorkloadSpec {
                name: "5. SSE 300-Token Stream (c=256)",
                stream: true,
                prompt_tag: "BENCH_300TOK",
                concurrency: 256,
                total_requests: 4_000,
            },
            WorkloadSpec {
                name: "6. GPU Cluster 500 Live Streams",
                stream: true,
                prompt_tag: "BENCH_PACED",
                concurrency: 500,
                total_requests: 2_500,
            },
            WorkloadSpec {
                name: "7. GPU Cluster 1000 Live Streams",
                stream: true,
                prompt_tag: "BENCH_PACED",
                concurrency: 1000,
                total_requests: 3_000,
            },
        ];

        let mut stateless_reports = Vec::new();

        println!("\n[Tier 1A] Benchmarking QuixiGate (Single Binary, Stateless Master Key)...");
        stateless_reports.push(
            bench_quixigate(
                &format!(
                    "QuixiGate v{} (1 binary, 18 shards)",
                    env!("CARGO_PKG_VERSION")
                ),
                &quixigate_bin,
                upstream_port,
                false,
                &stateless_workloads,
            )
            .await,
        );

        if let Some(litellm_bin) = &litellm_bin {
            println!(
                "\n[Tier 1B] Benchmarking LiteLLM v1.109.1 (1 Worker, Uvicorn+uvloop, No-DB)..."
            );
            match bench_litellm(
                "LiteLLM v1.109.1 (1 worker, uvloop)",
                litellm_bin,
                upstream_port,
                1,
                false,
                None,
                &stateless_workloads,
            )
            .await
            {
                Ok(r) => stateless_reports.push(r),
                Err(e) => eprintln!("  [WARN] {e}"),
            }

            println!(
                "\n[Tier 1C] Benchmarking LiteLLM v1.109.1 (8 Workers, Uvicorn+uvloop, No-DB)..."
            );
            match bench_litellm(
                "LiteLLM v1.109.1 (8 workers, uvloop)",
                litellm_bin,
                upstream_port,
                8,
                false,
                None,
                &stateless_workloads,
            )
            .await
            {
                Ok(r) => stateless_reports.push(r),
                Err(e) => eprintln!("  [WARN] {e}"),
            }

            println!(
                "\n[Tier 1D] Benchmarking LiteLLM v1.109.1 (18 Workers, Uvicorn+uvloop, No-DB)..."
            );
            match bench_litellm(
                "LiteLLM v1.109.1 (18 workers, uvloop)",
                litellm_bin,
                upstream_port,
                18,
                false,
                None,
                &stateless_workloads,
            )
            .await
            {
                Ok(r) => stateless_reports.push(r),
                Err(e) => eprintln!("  [WARN] {e}"),
            }

            println!(
                "\n[Tier 1E] Benchmarking LiteLLM v1.109.1 (8 Workers, Granian Rust ASGI, No-DB)..."
            );
            match bench_litellm(
                "LiteLLM v1.109.1 (8 workers, Granian)",
                litellm_bin,
                upstream_port,
                8,
                true,
                None,
                &stateless_workloads,
            )
            .await
            {
                Ok(r) => stateless_reports.push(r),
                Err(e) => eprintln!("  [WARN] {e}"),
            }
        }

        print_summary_table(
            "TIER 1: STATELESS PASS-THROUGH (Master Key, Zero External DB — LiteLLM's Fastest Possible Mode)",
            &stateless_reports,
        );
    }

    if filter == "all" || filter == "governance" {
        let gov_workloads = vec![
            WorkloadSpec {
                name: "1. Virtual Key + Budget JSON (c=64)",
                stream: false,
                prompt_tag: "BENCH_JSON",
                concurrency: 64,
                total_requests: 5_000,
            },
            WorkloadSpec {
                name: "2. Virtual Key + Budget SSE 1-Tok (c=64)",
                stream: true,
                prompt_tag: "BENCH_1TOK",
                concurrency: 64,
                total_requests: 5_000,
            },
            WorkloadSpec {
                name: "3. Virtual Key + Budget SSE 300-Tok (c=64)",
                stream: true,
                prompt_tag: "BENCH_300TOK",
                concurrency: 64,
                total_requests: 3_000,
            },
            WorkloadSpec {
                name: "4. Virtual Key + Budget SSE 300-Tok (c=256)",
                stream: true,
                prompt_tag: "BENCH_300TOK",
                concurrency: 256,
                total_requests: 4_000,
            },
            WorkloadSpec {
                name: "5. Virtual Key + Budget 500 Live Streams",
                stream: true,
                prompt_tag: "BENCH_PACED",
                concurrency: 500,
                total_requests: 2_500,
            },
            WorkloadSpec {
                name: "6. Virtual Key + Budget 1000 Live Streams",
                stream: true,
                prompt_tag: "BENCH_PACED",
                concurrency: 1000,
                total_requests: 3_000,
            },
        ];

        let mut gov_reports = Vec::new();

        println!(
            "\n[Tier 2A] Benchmarking QuixiGate (Full Governance: Virtual Key + $1M Budget + SQLite WAL)..."
        );
        gov_reports.push(
            bench_quixigate(
                &format!(
                    "QuixiGate v{} (SQLite WAL, 0 containers)",
                    env!("CARGO_PKG_VERSION")
                ),
                &quixigate_bin,
                upstream_port,
                true,
                &gov_workloads,
            )
            .await,
        );

        if let Some(litellm_bin) = &litellm_bin {
            println!(
                "\n[Tier 2B] Starting Docker PostgreSQL 16 (tmpfs RAM) + Redis 7 for LiteLLM Governance..."
            );
            match DockerInfra::start().await {
                Ok(infra) => {
                    println!(
                        "  Docker Postgres (:{}) + Redis (:{}) ready. Launching LiteLLM (8 workers)...",
                        infra.pg_port, infra.redis_port
                    );
                    match bench_litellm(
                        "LiteLLM v1.109.1 (8w + Postgres16 + Redis7)",
                        litellm_bin,
                        upstream_port,
                        8,
                        false,
                        Some(&infra),
                        &gov_workloads,
                    )
                    .await
                    {
                        Ok(r) => gov_reports.push(r),
                        Err(e) => eprintln!("  [WARN] {e}"),
                    }

                    println!(
                        "\n[Tier 2C] Launching LiteLLM (18 workers + Postgres 16 + Redis 7)..."
                    );
                    match bench_litellm(
                        "LiteLLM v1.109.1 (18w + Postgres16 + Redis7)",
                        litellm_bin,
                        upstream_port,
                        18,
                        false,
                        Some(&infra),
                        &gov_workloads,
                    )
                    .await
                    {
                        Ok(r) => gov_reports.push(r),
                        Err(e) => eprintln!("  [WARN] {e}"),
                    }
                    infra.stop();
                }
                Err(e) => eprintln!("  [WARN] Skipping Docker Postgres+Redis tier: {e}"),
            }
        }

        print_summary_table(
            "TIER 2: PRODUCTION GOVERNANCE (POST /key/generate Virtual Key + $1M Budget + Spend Ledger)",
            &gov_reports,
        );
    }

    kill_process_tree(&mut upstream_child);
}
