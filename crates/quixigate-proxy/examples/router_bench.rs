//! Repeatable Release-Mode Router & Streaming Benchmark for QuixiGate.
//!
//! Measures 4 distinct dimensions on the exact same machine and Tokio runtime, comparing
//! QuixiGate today against a "Speed-of-Light" zero-copy cut-through Rust HTTP byte-pipe:
//!   Suite 1: Per-Request Control/Auth/SQLite Ceiling (1 token/stream, c=64) -> max `req/s` & TTFT
//!   Suite 2: Realistic Chat Turn Streaming (300 tokens/stream, c=64) -> mixed setup + streaming
//!   Suite 3: High-Concurrency Contention (300 tokens/stream, c=256) -> tail-latency under load
//!   Suite 4: Pure Streaming Bit-Shuffling Ceiling (10,000 tokens/stream, c=64, 128 streams total)
//!            -> amortizes per-request SQLite setup to isolate pure per-token CPU & memory cost!

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderValue, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use quixigate_auth::OAuthWizardManager;
use quixigate_auth::quota::QuotaEndpoints;
use quixigate_auth::refresh::{CredentialRefresher, RefreshEndpoints};
use quixigate_core::{
    AuthKind, BudgetWindow, CredentialSecret, ModelRoute, NanoUsd, PoolStrategy, RouteTarget,
    UserRole,
};
use quixigate_db::Database;
use quixigate_providers::local_discovery::LocalRuntimeScanner;
use quixigate_proxy::state::AppState;
use quixigate_router::GatewayRouter;
use quixigate_router::catalog::ModelCatalog;
use quixigate_telemetry::TelemetryHub;
use quixigate_vault::Vault;
use tempfile::TempDir;
use tokio::net::TcpListener;

#[derive(Clone)]
struct MockUpstreamState {
    chat_1_tok: Bytes,
    chat_300_batches: Arc<Vec<Bytes>>,
    chat_10k_batches: Arc<Vec<Bytes>>,
    resp_1_tok: Bytes,
    resp_300_batches: Arc<Vec<Bytes>>,
}

fn detect_mode(body: &[u8]) -> u32 {
    if body.windows(10).any(|w| w == b"BENCH_1TOK") {
        1
    } else if body.windows(11).any(|w| w == b"BENCH_10000") {
        10_000
    } else {
        300
    }
}

async fn mock_chat_completions(State(st): State<MockUpstreamState>, body: Bytes) -> Response {
    match detect_mode(&body) {
        1 => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from(st.chat_1_tok.clone()),
        )
            .into_response(),
        10_000 => {
            let frames = st.chat_10k_batches.clone();
            let stream = futures::stream::iter(
                (0..frames.len()).map(move |i| Ok::<Bytes, Infallible>(frames[i].clone())),
            );
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response()
        }
        _ => {
            let frames = st.chat_300_batches.clone();
            let stream = futures::stream::iter(
                (0..frames.len()).map(move |i| Ok::<Bytes, Infallible>(frames[i].clone())),
            );
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response()
        }
    }
}

async fn mock_openai_responses(State(st): State<MockUpstreamState>, body: Bytes) -> Response {
    if detect_mode(&body) == 1 {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from(st.resp_1_tok.clone()),
        )
            .into_response();
    }
    let frames = st.resp_300_batches.clone();
    let stream = futures::stream::iter(
        (0..frames.len()).map(move |i| Ok::<Bytes, Infallible>(frames[i].clone())),
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(stream),
    )
        .into_response()
}

fn build_chat_batches(total_tokens: usize, tokens_per_chunk: usize) -> Vec<Bytes> {
    let tok_event = "data: {\"id\":\"chatcmpl-b\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" token\"},\"finish_reason\":null}]}\n\n";
    let final_event = format!(
        "data: {{\"id\":\"chatcmpl-b\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":16,\"completion_tokens\":{total_tokens},\"total_tokens\":{}}}}}\n\ndata: [DONE]\n\n",
        16 + total_tokens
    );
    let num_batches = total_tokens / tokens_per_chunk;
    let mut batches = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        let mut buf = BytesMut::with_capacity(tok_event.len() * (tokens_per_chunk + 2));
        for _ in 0..tokens_per_chunk {
            buf.extend_from_slice(tok_event.as_bytes());
        }
        if b + 1 == num_batches {
            buf.extend_from_slice(final_event.as_bytes());
        }
        batches.push(buf.freeze());
    }
    batches
}

fn build_responses_batches(total_tokens: usize, tokens_per_chunk: usize) -> Vec<Bytes> {
    let start_ev = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o-mini\"}}\n\n";
    let tok_ev = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\" token\"}\n\n";
    let end_ev = format!(
        "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-4o-mini\",\"usage\":{{\"input_tokens\":16,\"output_tokens\":{total_tokens},\"total_tokens\":{}}}}}}}\n\n",
        16 + total_tokens
    );
    let num_batches = total_tokens / tokens_per_chunk;
    let mut batches = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        let mut buf = BytesMut::with_capacity(tok_ev.len() * (tokens_per_chunk + 2));
        if b == 0 {
            buf.extend_from_slice(start_ev.as_bytes());
        }
        for _ in 0..tokens_per_chunk {
            buf.extend_from_slice(tok_ev.as_bytes());
        }
        if b + 1 == num_batches {
            buf.extend_from_slice(end_ev.as_bytes());
        }
        batches.push(buf.freeze());
    }
    batches
}

async fn spawn_mock_upstream() -> SocketAddr {
    let single_chat = concat!(
        "data: {\"id\":\"chatcmpl-b\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-b\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":16,\"completion_tokens\":1,\"total_tokens\":17}}\n\n",
        "data: [DONE]\n\n"
    );
    let single_resp = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o-mini\"}}\n\n",
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\"}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o-mini\",\"usage\":{\"input_tokens\":16,\"output_tokens\":1,\"total_tokens\":17}}}\n\n"
    );

    let state = MockUpstreamState {
        chat_1_tok: Bytes::from_static(single_chat.as_bytes()),
        chat_300_batches: Arc::new(build_chat_batches(300, 10)),
        chat_10k_batches: Arc::new(build_chat_batches(10_000, 25)),
        resp_1_tok: Bytes::from_static(single_resp.as_bytes()),
        resp_300_batches: Arc::new(build_responses_batches(300, 10)),
    };

    let app = Router::new()
        .route("/chat/completions", post(mock_chat_completions))
        .route("/v1/chat/completions", post(mock_chat_completions))
        .route("/responses", post(mock_openai_responses))
        .route("/v1/responses", post(mock_openai_responses))
        .with_state(state);

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    addr
}

// ── Speed-of-Light Zero-Copy Reference Proxy ─────────────────────────────────

#[derive(Clone)]
struct ZeroCopyProxyState {
    http: reqwest::Client,
    upstream_url: String,
    expected_auth: HeaderValue,
    requests_served: Arc<AtomicU64>,
}

async fn zero_copy_proxy_handler(
    State(st): State<ZeroCopyProxyState>,
    req: Request<Body>,
) -> Response {
    if req.headers().get(header::AUTHORIZATION) != Some(&st.expected_auth) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    st.requests_served.fetch_add(1, Ordering::Relaxed);

    let body_bytes = match axum::body::to_bytes(req.into_body(), 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let upstream_resp = match st
        .http
        .post(&st.upstream_url)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer sk-upstream")
        .body(body_bytes)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
    };

    let byte_stream = upstream_resp
        .bytes_stream()
        .map(|res| res.map_err(std::io::Error::other));

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(byte_stream),
    )
        .into_response()
}

async fn spawn_zero_copy_proxy(upstream_addr: SocketAddr, token: &str) -> SocketAddr {
    let http = reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(1024)
        .build()
        .unwrap();
    let st = ZeroCopyProxyState {
        http,
        upstream_url: format!("http://{upstream_addr}/v1/chat/completions"),
        expected_auth: HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        requests_served: Arc::new(AtomicU64::new(0)),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(zero_copy_proxy_handler))
        .with_state(st);
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    addr
}

// ── Live QuixiGate Harness ───────────────────────────────────────────────────

struct QuixiHarness {
    _dir: TempDir,
    _shutdown_tx: tokio::sync::oneshot::Sender<()>,
    addr: SocketAddr,
    virtual_key: String,
}

async fn spawn_quixigate_instance(upstream_addr: SocketAddr, provider_id: &str) -> QuixiHarness {
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();

    let mut settings = db.get_settings().unwrap();
    settings.require_auth = true;
    db.update_settings(&settings).unwrap();

    let base_url = if provider_id == "openai" {
        format!("http://{upstream_addr}")
    } else {
        format!("http://{upstream_addr}/v1")
    };

    let (auth_kind, secret) = if provider_id == "ollama" {
        (AuthKind::LocalNoauth, CredentialSecret::LocalNoauth)
    } else {
        (
            AuthKind::ApiKey,
            CredentialSecret::ApiKey {
                key: "sk-bench-upstream".to_owned(),
            },
        )
    };

    db.upsert_credential(
        Some(&format!("cred_{provider_id}")),
        provider_id,
        "Bench Seat",
        auth_kind,
        Some(&base_url),
        10,
        true,
        true,
        &secret,
    )
    .unwrap();

    if provider_id == "ollama" {
        db.upsert_model_route(&ModelRoute {
            alias: "gpt-4o-mini".to_owned(),
            targets: vec![RouteTarget {
                model: "ollama/gpt-4o-mini".to_owned(),
                provider_id: Some("ollama".to_owned()),
                credential_id: Some("cred_ollama".to_owned()),
            }],
            pool_strategy: PoolStrategy::FillFirst,
            allow_paid_overflow: true,
            blocked: false,
            input_nanos_per_m: None,
            output_nanos_per_m: None,
            cache_read_nanos_per_m: None,
            team_id: None,
            created_by: None,
            updated_by: None,
            created_at_ms: Some(0),
            updated_at_ms: 0,
        })
        .unwrap();
    }

    let user = db
        .create_user(
            "bench_user",
            "Bench User",
            UserRole::Member,
            None,
            Some(NanoUsd::from_nanos(1_000_000_000_000_000)),
            BudgetWindow::Monthly,
        )
        .unwrap();

    let (_vk, raw_key) = db
        .create_virtual_key(
            "bench_key",
            &user.id,
            None,
            Some(NanoUsd::from_nanos(1_000_000_000_000_000)),
            BudgetWindow::Monthly,
            None,
            None,
            None,
            &[],
            Some(false),
            None,
        )
        .unwrap();

    let refresher = CredentialRefresher::with_endpoints(db.clone(), RefreshEndpoints::default());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    let router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner);
    let telemetry = TelemetryHub::new();
    let wizard = OAuthWizardManager::new(db.clone(), refresher)
        .with_quota_endpoints(QuotaEndpoints::default());
    let state = AppState::new(db, router, telemetry, wizard, true);

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = quixigate_proxy::serve_sharded_per_core(listener, state, None, async move {
            let _ = shutdown_rx.await;
        })
        .await;
    });

    QuixiHarness {
        _dir: dir,
        _shutdown_tx: shutdown_tx,
        addr,
        virtual_key: raw_key,
    }
}

// ── Load Runner ──────────────────────────────────────────────────────────────

struct BenchResult {
    label: String,
    elapsed: Duration,
    total_requests: u64,
    failed_requests: u64,
    total_tokens: u64,
    total_bytes: u64,
    ttft_p50_us: u64,
    ttft_p99_us: u64,
    e2e_p99_us: u64,
}

fn count_sse_token_events(buf: &[u8]) -> u64 {
    let mut data_json_frames = 0u64;
    for line in buf.split(|&b| b == b'\n') {
        if line.starts_with(b"data: {") && line.windows(10).any(|w| w == b"\"content\":") {
            data_json_frames += 1;
        }
    }
    data_json_frames
}

async fn run_load(
    label: &str,
    target_addr: SocketAddr,
    bearer_token: &str,
    prompt_tag: &str,
    concurrency: usize,
    total_requests: usize,
) -> BenchResult {
    let client = reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(concurrency * 2)
        .build()
        .unwrap();

    let url = format!("http://{target_addr}/v1/chat/completions");
    let auth_val = format!("Bearer {bearer_token}");
    let body_bytes = Bytes::from(format!(
        "{{\"model\":\"gpt-4o-mini\",\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"{prompt_tag} benchmark prompt\"}}]}}"
    ));

    for _ in 0..8 {
        if let Ok(resp) = client
            .post(&url)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::AUTHORIZATION, &auth_val)
            .body(body_bytes.clone())
            .send()
            .await
        {
            let _ = resp.bytes().await;
        }
    }

    let next_idx = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut handles = Vec::with_capacity(concurrency);

    for _ in 0..concurrency {
        let client = client.clone();
        let url = url.clone();
        let auth_val = auth_val.clone();
        let body_bytes = body_bytes.clone();
        let next_idx = next_idx.clone();

        handles.push(tokio::spawn(async move {
            let mut ok_reqs = 0u64;
            let mut err_reqs = 0u64;
            let mut tokens = 0u64;
            let mut bytes = 0u64;
            let mut ttfts = Vec::new();
            let mut e2es = Vec::new();
            let mut line_rem = Vec::with_capacity(512);

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
                    .body(body_bytes.clone())
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
                            bytes += chunk.len() as u64;
                            line_rem.extend_from_slice(&chunk);
                            if let Some(last_nl) = line_rem.iter().rposition(|&b| b == b'\n') {
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
                if stream_err {
                    err_reqs += 1;
                    continue;
                }
                let e2e_us = t0.elapsed().as_micros() as u64;
                tokens += req_tokens;
                ok_reqs += 1;
                ttfts.push(ttft_us.unwrap_or(e2e_us));
                e2es.push(e2e_us);
            }

            (ok_reqs, err_reqs, tokens, bytes, ttfts, e2es)
        }));
    }

    let mut total_ok = 0u64;
    let mut total_err = 0u64;
    let mut total_tokens = 0u64;
    let mut total_bytes = 0u64;
    let mut all_ttfts = Vec::with_capacity(total_requests);
    let mut all_e2es = Vec::with_capacity(total_requests);

    for h in handles {
        let (ok, err, tok, b, mut ttft, mut e2e) = h.await.unwrap();
        total_ok += ok;
        total_err += err;
        total_tokens += tok;
        total_bytes += b;
        all_ttfts.append(&mut ttft);
        all_e2es.append(&mut e2e);
    }
    let elapsed = started.elapsed();

    all_ttfts.sort_unstable();
    all_e2es.sort_unstable();
    let pct = |v: &[u64], p: f64| -> u64 {
        if v.is_empty() {
            return 0;
        }
        let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
        v[idx.min(v.len() - 1)]
    };

    BenchResult {
        label: label.to_owned(),
        elapsed,
        total_requests: total_ok,
        failed_requests: total_err,
        total_tokens,
        total_bytes,
        ttft_p50_us: pct(&all_ttfts, 0.50),
        ttft_p99_us: pct(&all_ttfts, 0.99),
        e2e_p99_us: pct(&all_e2es, 0.99),
    }
}

fn print_table(title: &str, results: &[BenchResult]) {
    println!(
        "\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("  {title}");
    println!(
        "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "{:<42} | {:>9} | {:>12} | {:>9} | {:>9} | {:>9} | {:>9} | {:>6}",
        "Scenario", "Req/sec", "Tokens/sec", "MiB/sec", "TTFT p50", "TTFT p99", "E2E p99", "Errors"
    );
    println!(
        "-------------------------------------------+-----------+--------------+-----------+-----------+-----------+-----------+-------"
    );
    for r in results {
        let secs = r.elapsed.as_secs_f64();
        let rps = r.total_requests as f64 / secs;
        let tps = r.total_tokens as f64 / secs;
        let mib_s = (r.total_bytes as f64 / (1024.0 * 1024.0)) / secs;
        println!(
            "{:<42} | {:>9.1} | {:>12.0} | {:>9.2} | {:>7.2}ms | {:>7.2}ms | {:>7.2}ms | {:>6}",
            r.label,
            rps,
            tps,
            mib_s,
            r.ttft_p50_us as f64 / 1000.0,
            r.ttft_p99_us as f64 / 1000.0,
            r.e2e_p99_us as f64 / 1000.0,
            r.failed_requests
        );
    }
}

#[tokio::main]
async fn main() {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("QuixiGate Router Release Benchmark — Logical CPU Cores: {cores}");

    let upstream_addr = spawn_mock_upstream().await;
    let zc_token = "sk-qx-speed-of-light-token";
    let zc_addr = spawn_zero_copy_proxy(upstream_addr, zc_token).await;

    let qg_openrouter = spawn_quixigate_instance(upstream_addr, "openrouter").await;
    let qg_ollama = spawn_quixigate_instance(upstream_addr, "ollama").await;
    let qg_openai = spawn_quixigate_instance(upstream_addr, "openai").await;

    // Suite 1: Per-Request Routing / Auth / Ledger Ceiling (1 token per stream, c=64)
    let c1 = 64;
    let zc_1 = run_load(
        "0. Zero-Copy Cut-Through Pipe (c=64)",
        zc_addr,
        zc_token,
        "BENCH_1TOK",
        c1,
        10_000,
    )
    .await;
    let qg_or_1 = run_load(
        "1. QuixiGate: Pooled Provider (c=64)",
        qg_openrouter.addr,
        &qg_openrouter.virtual_key,
        "BENCH_1TOK",
        c1,
        10_000,
    )
    .await;
    let qg_ol_1 = run_load(
        "2. QuixiGate: Local Provider (c=64)",
        qg_ollama.addr,
        &qg_ollama.virtual_key,
        "BENCH_1TOK",
        c1,
        10_000,
    )
    .await;
    let qg_oa_1 = run_load(
        "3. QuixiGate: Built-in OpenAI (c=64)",
        qg_openai.addr,
        &qg_openai.virtual_key,
        "BENCH_1TOK",
        c1,
        10_000,
    )
    .await;
    print_table(
        "SUITE 1 — PER-REQUEST CONTROL/AUTH/SQLITE CEILING (1 token/stream, c=64)",
        &[zc_1, qg_or_1, qg_ol_1, qg_oa_1],
    );

    // Suite 2: Realistic 300-Token Streaming (300 tokens per stream, c=64)
    let zc_300 = run_load(
        "0. Zero-Copy Cut-Through Pipe (c=64)",
        zc_addr,
        zc_token,
        "BENCH_300",
        c1,
        6_000,
    )
    .await;
    let qg_or_300 = run_load(
        "1. QuixiGate: Pooled Provider (c=64)",
        qg_openrouter.addr,
        &qg_openrouter.virtual_key,
        "BENCH_300",
        c1,
        6_000,
    )
    .await;
    let qg_ol_300 = run_load(
        "2. QuixiGate: Local Provider (c=64)",
        qg_ollama.addr,
        &qg_ollama.virtual_key,
        "BENCH_300",
        c1,
        6_000,
    )
    .await;
    let qg_oa_300 = run_load(
        "3. QuixiGate: Built-in OpenAI (c=64)",
        qg_openai.addr,
        &qg_openai.virtual_key,
        "BENCH_300",
        c1,
        6_000,
    )
    .await;
    print_table(
        "SUITE 2 — MIXED REQUEST + 300-TOKEN STREAMING (300 tokens/stream, c=64)",
        &[zc_300, qg_or_300, qg_ol_300, qg_oa_300],
    );

    // Suite 3: High Concurrency Contention Stress (300 tokens/stream, c=256)
    let c3 = 256;
    let zc_c256 = run_load(
        "0. Zero-Copy Cut-Through Pipe (c=256)",
        zc_addr,
        zc_token,
        "BENCH_300",
        c3,
        6_000,
    )
    .await;
    let qg_or_c256 = run_load(
        "1. QuixiGate: Pooled Provider (c=256)",
        qg_openrouter.addr,
        &qg_openrouter.virtual_key,
        "BENCH_300",
        c3,
        6_000,
    )
    .await;
    let qg_ol_c256 = run_load(
        "2. QuixiGate: Local Provider (c=256)",
        qg_ollama.addr,
        &qg_ollama.virtual_key,
        "BENCH_300",
        c3,
        6_000,
    )
    .await;
    print_table(
        "SUITE 3 — HIGH-CONCURRENCY CONTENTION (300 tokens/stream, c=256)",
        &[zc_c256, qg_or_c256, qg_ol_c256],
    );

    // Suite 4: Pure Bit-Shuffling Streaming Ceiling (10,000 tokens/stream, c=64)
    // Amortizes per-request SQLite overhead across 10,000 tokens to isolate pure token-pumping CPU!
    let zc_10k = run_load(
        "0. Zero-Copy Cut-Through Pipe (10k tok)",
        zc_addr,
        zc_token,
        "BENCH_10000",
        c1,
        512,
    )
    .await;
    let qg_or_10k = run_load(
        "1. QuixiGate: Pooled Provider (10k tok)",
        qg_openrouter.addr,
        &qg_openrouter.virtual_key,
        "BENCH_10000",
        c1,
        512,
    )
    .await;
    print_table(
        "SUITE 4 — PURE BIT-SHUFFLING CEILING (10,000 tokens/stream, c=64 — Isolates Per-Token Pipeline)",
        &[zc_10k, qg_or_10k],
    );
}
