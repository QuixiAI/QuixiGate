//! Comprehensive end-to-end integration tests for `quixigate-proxy` across all
//! polyglot ingress wire protocols, WebSocket `/v1/responses`, sticky multi-seat
//! failover, model-scoped cooldowns, single-flight OAuth token refresh, exact
//! `NanoUsd` budgets, RPM rate limits, BLAKE3 caching, `HttpOnly` auth sessions,
//! and Arize Phoenix OpenInference binary OTLP protobuf trace export.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Bytes;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value;
use quixigate_auth::OAuthWizardManager;
use quixigate_auth::quota::QuotaEndpoints;
use quixigate_auth::refresh::{CredentialRefresher, RefreshEndpoints};
use quixigate_core::{AuthKind, BudgetWindow, CredentialSecret, NanoUsd, now_ms};
use quixigate_db::Database;
use quixigate_providers::local_discovery::{LocalEndpointSpec, LocalRuntimeScanner};
use quixigate_proxy::build_router;
use quixigate_proxy::state::AppState;
use quixigate_router::GatewayRouter;
use quixigate_router::catalog::ModelCatalog;
use quixigate_telemetry::TelemetryHub;
use quixigate_vault::Vault;
use parking_lot::Mutex;
use prost::Message as ProstMessage;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as WsMessage;

struct TestGateway {
    _dir: TempDir,
    db: Database,
    base_url: String,
    ws_url: String,
}

async fn spawn_axum(app: Router) -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}

async fn spawn_test_gateway(
    refresh_endpoints: RefreshEndpoints,
    quota_endpoints: QuotaEndpoints,
    ollama_url: Option<String>,
    require_auth: bool,
) -> TestGateway {
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    if require_auth {
        let mut s = db.get_settings().unwrap();
        s.require_auth = true;
        db.update_settings(&s).unwrap();
    }

    let refresher = CredentialRefresher::with_endpoints(db.clone(), refresh_endpoints);
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    if let Some(ref u) = ollama_url {
        scanner.add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: "ollama".to_owned(),
            display_name: "Ollama".to_owned(),
            base_url: u.clone(),
        });
        scanner.scan_all().await;
    }

    let mut router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner);
    if let Some(ref u) = ollama_url {
        router = router.with_web_search_url(format!("{}/search", u.trim_end_matches("/v1")));
    }
    let telemetry = TelemetryHub::new();
    let wizard =
        OAuthWizardManager::new(db.clone(), refresher).with_quota_endpoints(quota_endpoints.clone());
    let state = AppState::new(db.clone(), router, telemetry, wizard, true)
        .with_quota_endpoints(quota_endpoints);

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state);
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });

    TestGateway {
        _dir: dir,
        db,
        base_url: format!("http://{addr}"),
        ws_url: format!("ws://{addr}"),
    }
}

#[tokio::test]
async fn e2e_all_ingress_protocols_and_websockets() {
    // Pure-Rust mock upstream speaking OpenAI Chat Completions (JSON + SSE + tool calls),
    // Anthropic Messages (JSON + SSE), Embeddings, and Ollama discovery.
    let chat_handler = |Json(body): Json<Value>| async move {
                let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
                let has_tools = body
                    .get("tools")
                    .and_then(Value::as_array)
                    .map(|a| !a.is_empty())
                    .unwrap_or(false);

                if has_tools && !stream {
                    return Json(json!({
                        "id": "chatcmpl-tool-1",
                        "object": "chat.completion",
                        "created": 1735689600,
                        "model": "gpt-5",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": null,
                                "tool_calls": [{
                                    "id": "call_weather_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "{\"city\":\"San Francisco\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": { "prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32 }
                    }))
                    .into_response();
                }

                if stream {
                    let has_tool_result = body["messages"]
                        .as_array()
                        .map(|arr| arr.iter().any(|m| m["role"].as_str() == Some("tool")))
                        .unwrap_or(false);
                    let offered_online = body["tools"]
                        .as_array()
                        .map(|arr| {
                            arr.iter().any(|t| {
                                t["function"]["name"].as_str() == Some("quixigate_web_search")
                            })
                        })
                        .unwrap_or(false);
                    if offered_online && !has_tool_result {
                        let tool_sse = concat!(
                            "data: {\"id\":\"c_tool\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_ws_1\",\"type\":\"function\",\"function\":{\"name\":\"quixigate_web_search\",\"arguments\":\"{\\\"query\\\":\\\"quixigate\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
                            "data: {\"id\":\"c_tool\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":9,\"total_tokens\":27}}\n\n",
                            "data: [DONE]\n\n"
                        );
                        return (StatusCode::OK, [("content-type", "text/event-stream")], tool_sse)
                            .into_response();
                    }
                    let sse = concat!(
                        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Thinking step.\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Polyglot!\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":8,\"total_tokens\":23}}\n\n",
                        "data: [DONE]\n\n"
                    );
                    return (StatusCode::OK, [("content-type", "text/event-stream")], sse)
                        .into_response();
                }

                Json(json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1735689600,
                    "model": "gpt-5",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "Quick check.",
                            "content": "Hello Polyglot!"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20 }
                }))
                .into_response()
            };

    let upstream = Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .route("/chat/completions", post(chat_handler))
        .route(
            "/v1/embeddings",
            post(|| async {
                Json(json!({
                    "object": "list",
                    "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }],
                    "usage": { "prompt_tokens": 5, "total_tokens": 5 }
                }))
            }),
        )
        .route(
            "/embeddings",
            post(|| async {
                Json(json!({
                    "object": "list",
                    "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }],
                    "usage": { "prompt_tokens": 5, "total_tokens": 5 }
                }))
            }),
        )
        .route(
            "/api/tags",
            get(|| async { Json(json!({ "models": [{ "name": "qwen2.5-coder:7b" }] })) }),
        )
        .route("/api/ps", get(|| async { Json(json!({ "models": [] })) }))
        .route(
            "/v1/models",
            get(|| async { Json(json!({ "data": [{ "id": "qwen2.5-coder:7b" }] })) }),
        )
        .route(
            "/search",
            get(|| async {
                Json(json!({
                    "results": [{
                        "title": "QuixiGate Docs",
                        "url": "https://quixigate.local/docs",
                        "snippet": "Exact NanoUsd accounting."
                    }]
                }))
            }),
        );

    let upstream_url = spawn_axum(upstream).await;
    let gw = spawn_test_gateway(
        RefreshEndpoints::default(),
        QuotaEndpoints::default(),
        Some(format!("{upstream_url}/v1")),
        false,
    )
    .await;

    // Seed an OpenAI-compatible seat (`openaichatcompletions`) + OpenAI seat (`openai` for embeddings)
    gw.db
        .upsert_credential(
            Some("seat_oai_compat"),
            "openaichatcompletions",
            "Mock OpenAI Chat",
            AuthKind::ApiKey,
            Some(&upstream_url),
            10,
            true,
            true,
            &CredentialSecret::ApiKey {
                key: "sk-mock".to_owned(),
            },
        )
        .unwrap();
    gw.db
        .upsert_credential(
            Some("seat_oai_emb"),
            "openai",
            "Mock OpenAI Embeddings",
            AuthKind::ApiKey,
            Some(&upstream_url),
            20,
            true,
            true,
            &CredentialSecret::ApiKey {
                key: "sk-mock".to_owned(),
            },
        )
        .unwrap();

    let http = reqwest::Client::new();

    // 1. OpenAI `/v1/chat/completions` (non-streaming + client tool pass-through)
    let tool_res: Value = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "messages": [{ "role": "user", "content": "Weather in SF?" }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetch weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        tool_res["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );

    // 2. OpenAI `/v1/chat/completions` (SSE streaming)
    let chat_sse = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "stream": true,
            "messages": [{ "role": "user", "content": "Say hello" }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(chat_sse.contains("Hello "));
    assert!(chat_sse.contains("Polyglot!"));
    assert!(chat_sse.contains("data: [DONE]"));

    // 3. OpenAI `/v1/responses` (POST SSE)
    let resp_sse = http
        .post(format!("{}/v1/responses", gw.base_url))
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "stream": true,
            "input": "Stream via Responses API"
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(resp_sse.contains("response.created"));
    assert!(resp_sse.contains("response.output_text.delta"));
    assert!(resp_sse.contains("response.completed"));

    // 4. OpenAI `/v1/responses` WebSocket (`GET /v1/responses`)
    let (mut ws, _) =
        tokio_tungstenite::connect_async(format!("{}/v1/responses", gw.ws_url))
            .await
            .unwrap();
    ws.send(WsMessage::Text(
        json!({
            "type": "response.create",
            "response": {
                "model": "openaichatcompletions/gpt-5",
                "input": "Hello over WebSocket"
            }
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    let mut saw_delta = false;
    let mut saw_completed = false;
    while let Some(Ok(msg)) = ws.next().await {
        if let WsMessage::Text(txt) = msg {
            let v: Value = serde_json::from_str(&txt).unwrap();
            if v["type"] == "response.output_text.delta" {
                saw_delta = true;
            }
            if v["type"] == "response.completed" {
                saw_completed = true;
                break;
            }
        }
    }
    assert!(saw_delta && saw_completed, "WebSocket /v1/responses did not complete");

    // 5. Anthropic `/v1/messages` (SSE) & `/v1/messages/count_tokens`
    let ant_sse = http
        .post(format!("{}/v1/messages", gw.base_url))
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "max_tokens": 128,
            "stream": true,
            "messages": [{ "role": "user", "content": "Translate to Anthropic SSE" }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(ant_sse.contains("event: message_start"));
    assert!(ant_sse.contains("thinking_delta"));
    assert!(ant_sse.contains("Hello "));
    assert!(ant_sse.contains("event: message_stop"));

    let tok_count: Value = http
        .post(format!("{}/v1/messages/count_tokens", gw.base_url))
        .json(&json!({
            "model": "claude-sonnet-4-5",
            "messages": [{ "role": "user", "content": "Estimate tokens cleanly" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(tok_count["input_tokens"].as_u64().unwrap() > 0);

    // 6. Google Gemini `/v1beta/models/openaichatcompletions/gpt-5:generateContent`
    let gem_json: Value = http
        .post(format!(
            "{}/v1beta/models/openaichatcompletions/gpt-5:generateContent",
            gw.base_url
        ))
        .json(&json!({
            "contents": [{ "role": "user", "parts": [{ "text": "Hello Gemini wire" }] }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        gem_json["candidates"][0]["content"]["parts"][0]["text"],
        "Hello Polyglot!"
    );

    // 7. Vercel AI SDK `UIMessageStream` (`POST /api/chat`)
    let vercel_res = http
        .post(format!("{}/api/chat", gw.base_url))
        .json(&json!({
            "id": "chat-ui-1",
            "model": "openaichatcompletions/gpt-5",
            "messages": [{
                "id": "u1",
                "role": "user",
                "parts": [{ "type": "text", "text": "Hello useChat" }]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        vercel_res
            .headers()
            .get("x-vercel-ai-ui-message-stream")
            .and_then(|v| v.to_str().ok()),
        Some("v1")
    );
    let vercel_body = vercel_res.text().await.unwrap();
    assert!(vercel_body.contains("\"type\":\"text-delta\""));
    assert!(vercel_body.contains("\"type\":\"message-metadata\""));
    assert!(vercel_body.contains("data: [DONE]"));

    // 7b. Vercel AI SDK `UIMessageStream` with `:online` multi-step server tool loop
    let vercel_online_body = http
        .post(format!("{}/api/chat", gw.base_url))
        .json(&json!({
            "id": "chat-ui-online",
            "model": "openaichatcompletions/gpt-5",
            "online": true,
            "messages": [{
                "id": "u2",
                "role": "user",
                "parts": [{ "type": "text", "text": "Search QuixiGate docs" }]
            }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!vercel_online_body.contains("\"type\":\"tool-input-delta\""));
    assert!(vercel_online_body.contains("\"type\":\"tool-input-available\""));
    assert!(vercel_online_body.contains("\"type\":\"tool-output-available\""));
    assert!(vercel_online_body.contains("\"providerExecuted\":true"));
    assert!(vercel_online_body.contains("Hello "));

    // 8. Embeddings, `/v1/models`, OpenRouter `/api/v1/key`, `/metrics`, and SPA asset serving
    let emb_res: Value = http
        .post(format!("{}/v1/embeddings", gw.base_url))
        .json(&json!({
            "model": "openai/text-embedding-3-small",
            "input": ["vectorize me"]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(emb_res["data"][0]["embedding"].as_array().unwrap().len(), 4);

    let models_res: Value = http
        .get(format!("{}/v1/models", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!models_res["data"].as_array().unwrap().is_empty());

    let prom_text = http
        .get(format!("{}/metrics", gw.base_url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(prom_text.contains("quixigate_requests_total"));

    let api_root: Value = http
        .get(format!("{}/", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(api_root["service"], json!("quixigate"));
    assert_eq!(api_root["openapi"], json!("/openapi.json"));

    let unknown_resp = http
        .get(format!("{}/chat", gw.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(unknown_resp.status(), StatusCode::NOT_FOUND);

    #[cfg(feature = "ui")]
    {
        let spa_html = http
            .get(format!("{}/ui/chat", gw.base_url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(spa_html.contains("<div id=\"root\"></div>"));
    }
}

#[tokio::test]
async fn e2e_sticky_pool_model_scoped_cooldown_and_single_flight_refresh() {
    let refresh_calls = Arc::new(AtomicUsize::new(0));
    let seat1_gpt5_429 = Arc::new(AtomicUsize::new(0));

    let rc = refresh_calls.clone();
    let s1_counter = seat1_gpt5_429.clone();

    let upstream = Router::new()
        .route(
            "/oauth/token",
            post(move || {
                let rc = rc.clone();
                async move {
                    rc.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(60)).await;
                    // Return JWT with future exp
                    Json(json!({
                        "access_token": "fresh_codex_access_token",
                        "refresh_token": "rotated_refresh_token",
                        "expires_in": 3600
                    }))
                }
            }),
        )
        .route(
            "/seat1/responses",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let s1_counter = s1_counter.clone();
                async move {
                    let auth = headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or_default();
                    assert!(
                        auth.contains("fresh_codex_access_token"),
                        "expected refreshed token, got {auth}"
                    );
                    let model = body["model"].as_str().unwrap_or_default();
                    if model == "gpt-5" {
                        s1_counter.fetch_add(1, Ordering::SeqCst);
                        return (
                            StatusCode::TOO_MANY_REQUESTS,
                            Json(json!({ "error": { "message": "gpt-5 model rate limit" } })),
                        )
                            .into_response();
                    }
                    let sse = concat!(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Seat1 OK\"}\n\n",
                        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n"
                    );
                    (StatusCode::OK, [("content-type", "text/event-stream")], sse).into_response()
                }
            }),
        )
        .route(
            "/seat2/responses",
            post(|| async move {
                let sse = concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Seat2 Failover OK\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":3}}}\n\n"
                );
                (StatusCode::OK, [("content-type", "text/event-stream")], sse).into_response()
            }),
        );

    let upstream_url = spawn_axum(upstream).await;
    let refresh_ep = RefreshEndpoints {
        codex_token_url: format!("{upstream_url}/oauth/token"),
        ..Default::default()
    };
    let gw = spawn_test_gateway(refresh_ep, QuotaEndpoints::default(), None, false).await;

    // Seat 1 has an expiring token (triggers single-flight refresh!)
    let expiring = now_ms() + 1_000;
    gw.db
        .upsert_credential(
            Some("codex_seat_1"),
            "openai-codex",
            "Codex Seat 1",
            AuthKind::Oauth,
            Some(&format!("{upstream_url}/seat1")),
            10,
            true,
            false,
            &CredentialSecret::Codex {
                access: "stale_token".to_owned(),
                refresh: "initial_refresh".to_owned(),
                expires_ms: expiring,
                account_id: "acct_1".to_owned(),
            },
        )
        .unwrap();

    gw.db
        .upsert_credential(
            Some("codex_seat_2"),
            "openai-codex",
            "Codex Seat 2",
            AuthKind::Oauth,
            Some(&format!("{upstream_url}/seat2")),
            20,
            true,
            false,
            &CredentialSecret::Codex {
                access: "seat2_valid_token".to_owned(),
                refresh: "seat2_refresh".to_owned(),
                expires_ms: now_ms() + 7_200_000,
                account_id: "acct_2".to_owned(),
            },
        )
        .unwrap();

    let http = reqwest::Client::new();

    // Burst 12 concurrent requests for `gpt-5-mini` (served by Seat 1 after refreshing once!)
    let mut tasks = Vec::new();
    for _ in 0..12 {
        let client = http.clone();
        let url = format!("{}/v1/chat/completions", gw.base_url);
        tasks.push(tokio::spawn(async move {
            let res: Value = client
                .post(url)
                .json(&json!({
                    "model": "openai/gpt-5-mini",
                    "messages": [{ "role": "user", "content": "burst" }]
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(res["choices"][0]["message"]["content"], "Seat1 OK");
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    assert_eq!(
        refresh_calls.load(Ordering::SeqCst),
        1,
        "Single-flight coalescer must refresh expiring OAuth token exactly once!"
    );

    // Now request `openai/gpt-5`: Seat 1 returns 429 -> router places model-scoped cooldown on
    // `(codex_seat_1, gpt-5)` and transparently fails over to `codex_seat_2`!
    let failover_res: Value = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openai/gpt-5",
            "messages": [{ "role": "user", "content": "test failover" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        failover_res["choices"][0]["message"]["content"],
        "Seat2 Failover OK"
    );

    // Verify Seat 1 was NOT marked dead or globally exhausted: only `gpt-5` has a model-scoped
    // cooldown, while `gpt-5-mini` on Seat 1 remains healthy!
    let seat1_rec = gw.db.get_credential("codex_seat_1").unwrap();
    assert_eq!(seat1_rec.status, quixigate_core::CredentialStatus::Ok);
    assert!(seat1_rec.model_cooldowns.contains_key("gpt-5"));
    assert!(!seat1_rec.model_cooldowns.contains_key("gpt-5-mini"));

    // Because `(codex_seat_1, gpt-5)` is in model-scoped cooldown, `openai/gpt-5-mini` STILL
    // drains Seat 1 (priority #10) while `openai/gpt-5` immediately routes to Seat 2 without
    // hitting Seat 1 again!
    let sibling_res: Value = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openai/gpt-5-mini",
            "messages": [{ "role": "user", "content": "sibling model check" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(sibling_res["choices"][0]["message"]["content"], "Seat1 OK");

    let second_gpt5: Value = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openai/gpt-5",
            "messages": [{ "role": "user", "content": "second gpt-5 check" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        second_gpt5["choices"][0]["message"]["content"],
        "Seat2 Failover OK"
    );
    assert_eq!(
        seat1_gpt5_429.load(Ordering::SeqCst),
        1,
        "Seat 1 must be skipped while its per-model cooldown for gpt-5 is active"
    );
}

#[tokio::test]
async fn e2e_nanousd_budgets_rate_limits_cache_key_rotation_auth_and_openinference() {
    let captured_spans: Arc<Mutex<Vec<HashMap<String, String>>>> = Arc::new(Mutex::new(Vec::new()));
    let spans_sink = captured_spans.clone();

    let paid_chat = || async {
                Json(json!({
                    "id": "chatcmpl-paid",
                    "object": "chat.completion",
                    "created": 1735689600,
                    "model": "gpt-5",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "Paid completion OK" },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000 }
                }))
            };
    let upstream = Router::new()
        .route("/v1/chat/completions", post(paid_chat))
        .route("/chat/completions", post(paid_chat))
        .route(
            "/v1/traces",
            post(move |body: Bytes| {
                let spans_sink = spans_sink.clone();
                async move {
                    let decoded = ExportTraceServiceRequest::decode(body.as_ref()).unwrap();
                    let mut guard = spans_sink.lock();
                    for rs in decoded.resource_spans {
                        for ss in rs.scope_spans {
                            for sp in ss.spans {
                                let mut map = HashMap::new();
                                for kv in sp.attributes {
                                    let val_str = match kv.value.and_then(|v| v.value) {
                                        Some(any_value::Value::StringValue(s)) => s,
                                        Some(any_value::Value::IntValue(i)) => i.to_string(),
                                        Some(any_value::Value::DoubleValue(d)) => format!("{d:.6}"),
                                        Some(any_value::Value::BoolValue(b)) => b.to_string(),
                                        other => format!("{other:?}"),
                                    };
                                    map.insert(kv.key, val_str);
                                }
                                guard.push(map);
                            }
                        }
                    }
                    (StatusCode::OK, Json(json!({ "partialSuccess": {} })))
                }
            }),
        );

    let upstream_url = spawn_axum(upstream).await;
    let gw = spawn_test_gateway(
        RefreshEndpoints::default(),
        QuotaEndpoints::default(),
        None,
        true, // strict `--require-auth` mode!
    )
    .await;

    gw.db
        .upsert_credential(
            Some("paid_seat"),
            "openaichatcompletions",
            "Paid API Seat",
            AuthKind::ApiKey,
            Some(&upstream_url),
            200,
            true,
            true,
            &CredentialSecret::ApiKey {
                key: "sk-paid".to_owned(),
            },
        )
        .unwrap();

    let http = reqwest::Client::new();

    // 1. Unauthenticated admin call must be rejected (`401`) when `require_auth` is enabled.
    let unauth = http
        .get(format!("{}/api/admin/overview", gw.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    // 2. Bootstrap Argon2id password and capture `HttpOnly` session cookie!
    let boot_res = http
        .post(format!("{}/api/admin/auth/bootstrap", gw.base_url))
        .json(&json!({ "username": "local", "password": "correct-horse-battery" }))
        .send()
        .await
        .unwrap();
    assert_eq!(boot_res.status(), StatusCode::OK);
    let set_cookie = boot_res
        .headers()
        .get("set-cookie")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    assert!(set_cookie.contains("HttpOnly"));
    let cookie_pair = set_cookie.split(';').next().unwrap().to_owned();

    // Authenticated `/api/admin/overview` now succeeds (`200`)!
    let auth_ok = http
        .get(format!("{}/api/admin/overview", gw.base_url))
        .header("cookie", &cookie_pair)
        .send()
        .await
        .unwrap();
    assert_eq!(auth_ok.status(), StatusCode::OK);

    // 3. Configure Arize Phoenix OTLP preset + enable response cache via Admin REST API
    let preset_res = http
        .post(format!("{}/api/admin/telemetry/phoenix-preset", gw.base_url))
        .header("cookie", &cookie_pair)
        .json(&json!({
            "phoenix_ui_url": upstream_url,
            "otlp_endpoint": format!("{upstream_url}/v1/traces"),
            "project_name": "quixigate-e2e"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(preset_res.status(), StatusCode::OK);
    let mut cur_settings: Value = preset_res.json().await.unwrap();
    cur_settings["cache_enabled_default"] = json!(true);
    http.put(format!("{}/api/admin/settings", gw.base_url))
        .header("cookie", &cookie_pair)
        .json(&cur_settings)
        .send()
        .await
        .unwrap();

    // Trigger "Send Test Span"
    let span_test: Value = http
        .post(format!("{}/api/admin/telemetry/test-span", gw.base_url))
        .header("cookie", &cookie_pair)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(span_test["ok"], json!(true));

    // 4. Mint a `sk-qx-...` key with a tight $0.02 budget (`20_000_000` nanos) and `rpm = 3`
    let (vk, raw_key) = gw
        .db
        .create_virtual_key(
            "tight-budget-key",
            "local",
            None,
            Some(NanoUsd::parse_usd("0.02").unwrap()),
            BudgetWindow::Monthly,
            Some(3),
            None,
            None,
            &[],
            None,
            None,
        )
        .unwrap();

    // 1st call: `MISS` (`max_tokens: 500` reserves ~$0.005 <= $0.02; actual 1k in + 1k out = $0.01125)
    let r1 = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .bearer_auth(&raw_key)
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "max_tokens": 500,
            "messages": [{ "role": "user", "content": "cacheable question" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), StatusCode::OK);
    assert_eq!(
        r1.headers().get("x-quixigate-cache").unwrap().to_str().unwrap(),
        "MISS"
    );

    // 2nd identical call: exact BLAKE3 `HIT` ($0 incremental spend!)
    let r2 = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .bearer_auth(&raw_key)
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "max_tokens": 500,
            "messages": [{ "role": "user", "content": "cacheable question" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), StatusCode::OK);
    assert_eq!(
        r2.headers().get("x-quixigate-cache").unwrap().to_str().unwrap(),
        "HIT"
    );

    // 3rd call with a DIFFERENT prompt (`max_tokens: 1500` reserves ~$0.015 > $0.00875 remaining)
    // -> `402 Payment Required`!
    let r3 = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .bearer_auth(&raw_key)
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "max_tokens": 1500,
            "messages": [{ "role": "user", "content": "uncached expensive question" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r3.status(), StatusCode::PAYMENT_REQUIRED);

    // 4th call within the same minute hits `rpm_limit = 3` -> `429 Too Many Requests` + `Retry-After`!
    let r4 = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .bearer_auth(&raw_key)
        .json(&json!({
            "model": "openaichatcompletions/gpt-5",
            "max_tokens": 500,
            "messages": [{ "role": "user", "content": "cacheable question" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r4.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(r4.headers().get("retry-after").is_some());

    // 5. Key rotation grace period + immediate revocation
    let (vk2, old_secret) = gw
        .db
        .create_virtual_key(
            "rotatable-key",
            "local",
            None,
            None,
            BudgetWindow::Monthly,
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .unwrap();
    let (_, new_secret) = gw.db.rotate_virtual_key(&vk2.id, 300).unwrap();
    // Both old and new secrets work during the 300s grace window
    assert!(gw.db.verify_virtual_key(&old_secret).is_ok());
    assert!(gw.db.verify_virtual_key(&new_secret).is_ok());
    // Revoking immediately invalidates both
    let _ = gw.db.revoke_virtual_key(&vk2.id).unwrap();
    assert!(gw.db.verify_virtual_key(&old_secret).is_err());
    assert!(gw.db.verify_virtual_key(&new_secret).is_err());
    let _ = gw.db.delete_virtual_key(&vk.id);

    // 6. Verify OpenInference binary protobuf spans arrived at the collector
    tokio::time::sleep(Duration::from_millis(80)).await;
    let spans = captured_spans.lock();
    assert!(
        spans.len() >= 2,
        "expected at least test span + live LLM span, got {}",
        spans.len()
    );
    assert!(spans.iter().any(|s| {
        s.get("openinference.span.kind").map(String::as_str) == Some("LLM")
            && s.get("openinference.project.name").map(String::as_str) == Some("quixigate-e2e")
            && s.contains_key("llm.token_count.prompt")
            && s.contains_key("llm.cost.total")
    }));
}

#[tokio::test]
async fn e2e_llamacpp_server_router_and_instance_discovery_aliases_load_control_and_zero_cost() {
    let model_b_loaded = Arc::new(Mutex::new(false));
    let model_b_get = model_b_loaded.clone();
    let model_b_load = model_b_loaded.clone();
    let model_b_unload = model_b_loaded.clone();
    let captured_bodies: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_sink = captured_bodies.clone();

    let check_llama_auth = |headers: &HeaderMap| -> bool {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            == Some("Bearer llama-key-99")
    };

    let llama_server = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({ "status": "ok" })) }),
        )
        .route(
            "/props",
            get(move |headers: HeaderMap| async move {
                if !check_llama_auth(&headers) {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(json!({ "error": { "message": "Invalid API Key", "code": 401 } })),
                    )
                        .into_response();
                }
                Json(json!({
                    "role": "router",
                    "max_instances": 2,
                    "models_autoload": true,
                    "build_info": "b6220-metal"
                }))
                .into_response()
            }),
        )
        .route(
            "/v1/models",
            get(move |headers: HeaderMap| {
                let b_loaded = *model_b_get.lock();
                async move {
                    if !check_llama_auth(&headers) {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(json!({ "error": { "message": "Invalid API Key", "code": 401 } })),
                        )
                            .into_response();
                    }
                    Json(json!({
                        "object": "list",
                        "data": [
                            {
                                "id": "ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M",
                                "aliases": ["qwen3-coder-fast"],
                                "owned_by": "llamacpp",
                                "status": { "value": "loaded", "args": [] },
                                "context_length": 131072,
                                "meta": { "n_ctx": 65536 },
                                "architecture": { "input_modalities": ["text", "image"] }
                            },
                            {
                                "id": "nomic-embed-text-v1.5.Q8_0.gguf",
                                "aliases": ["nomic-local"],
                                "owned_by": "llamacpp",
                                "status": {
                                    "value": if b_loaded { "loaded" } else { "unloaded" },
                                    "args": []
                                },
                                "context_length": 8192,
                                "architecture": { "input_modalities": ["text"] }
                            }
                        ]
                    }))
                    .into_response()
                }
            }),
        )
        .route(
            "/models/load",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let model_b_load = model_b_load.clone();
                async move {
                    assert!(check_llama_auth(&headers));
                    assert_eq!(body["model"], "nomic-embed-text-v1.5.Q8_0.gguf");
                    *model_b_load.lock() = true;
                    Json(json!({ "success": true }))
                }
            }),
        )
        .route(
            "/models/unload",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let model_b_unload = model_b_unload.clone();
                async move {
                    assert!(check_llama_auth(&headers));
                    assert_eq!(body["model"], "nomic-embed-text-v1.5.Q8_0.gguf");
                    *model_b_unload.lock() = false;
                    Json(json!({ "success": true }))
                }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let captured_sink = captured_sink.clone();
                async move {
                    if !check_llama_auth(&headers) {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(json!({ "error": { "message": "Invalid API Key" } })),
                        )
                            .into_response();
                    }
                    captured_sink.lock().push(body.clone());
                    let model = body["model"].as_str().unwrap_or_default().to_owned();
                    if body["stream"].as_bool() == Some(true) {
                        let sse = format!(
                            concat!(
                                "data: {{\"id\":\"chatcmpl-llama\",\"object\":\"chat.completion.chunk\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"reasoning_content\":\"Thinking locally...\"}}}}]}}\n\n",
                                "data: {{\"id\":\"chatcmpl-llama\",\"object\":\"chat.completion.chunk\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"llama.cpp stream OK\"}}}}]}}\n\n",
                                "data: {{\"id\":\"chatcmpl-llama\",\"object\":\"chat.completion.chunk\",\"model\":\"{}\",\"choices\":[],\"usage\":{{\"prompt_tokens\":120,\"completion_tokens\":16,\"total_tokens\":136}},\"timings\":{{\"cache_n\":80,\"prompt_n\":40,\"predicted_n\":16}}}}\n\n",
                                "data: [DONE]\n\n"
                            ),
                            model, model, model
                        );
                        return (
                            StatusCode::OK,
                            [("content-type", "text/event-stream")],
                            sse,
                        )
                            .into_response();
                    }
                    Json(json!({
                        "id": "chatcmpl-llama",
                        "object": "chat.completion",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "Local GGUF reasoning",
                                "content": "llama.cpp OK"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                            "prompt_tokens_details": { "cached_tokens": 60 }
                        }
                    }))
                    .into_response()
                }
            }),
        )
        .route(
            "/v1/embeddings",
            post(move |headers: HeaderMap, Json(body): Json<Value>| async move {
                assert!(check_llama_auth(&headers));
                assert_eq!(body["model"], "nomic-embed-text-v1.5.Q8_0.gguf");
                Json(json!({
                    "object": "list",
                    "data": [{ "object": "embedding", "index": 0, "embedding": [0.25, -0.5, 0.75] }],
                    "usage": { "prompt_tokens": 6, "total_tokens": 6 }
                }))
            }),
        );

    let llama_url = spawn_axum(llama_server).await;
    let gw = spawn_test_gateway(
        RefreshEndpoints::default(),
        QuotaEndpoints::default(),
        None,
        false,
    )
    .await;
    let http = reqwest::Client::new();

    // 1. Register llama.cpp endpoint WITHOUT api_key -> surfaces `auth_required: true`!
    let rts_unauth: Vec<Value> = http
        .post(format!("{}/api/admin/local-runtimes", gw.base_url))
        .json(&json!({
            "runtime_id": "llama.cpp",
            "display_name": "llama.cpp Server",
            "base_url": format!("{llama_url}/v1")
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let lc_unauth = rts_unauth
        .iter()
        .find(|r| r["runtime_id"] == "llamacpp")
        .unwrap();
    assert_eq!(lc_unauth["healthy"], json!(false));
    assert_eq!(lc_unauth["auth_required"], json!(true));

    // 2. Register WITH `api_key` -> vault stores encrypted key, scanner discovers Router mode!
    let rts_auth: Vec<Value> = http
        .post(format!("{}/api/admin/local-runtimes", gw.base_url))
        .json(&json!({
            "runtime_id": "llamacpp",
            "display_name": "llama.cpp Server",
            "base_url": format!("{llama_url}/v1"),
            "api_key": "llama-key-99"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let lc = rts_auth
        .iter()
        .find(|r| r["runtime_id"] == "llamacpp")
        .unwrap();
    assert_eq!(lc["healthy"], json!(true));
    assert_eq!(lc["mode"], json!("router"));
    assert_eq!(lc["total_slots"], json!(2));
    assert_eq!(lc["supports_load_control"], json!(true));

    // 3. `/v1/models` exposes `llamacpp/ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M` with 65536 context & vision!
    let models_list: Value = http
        .get(format!("{}/v1/models", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let qwen_entry = models_list["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "llamacpp/ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M")
        .expect("llama.cpp model must appear in /v1/models");
    assert_eq!(qwen_entry["context_length"], json!(65536));
    assert_eq!(qwen_entry["quixigate"]["supports_vision"], json!(true));
    assert_eq!(qwen_entry["quixigate"]["ready"], json!(true));

    // 4. Test Router load / unload endpoints via `/api/admin/local-runtimes/llamacpp/{load,unload}`
    let after_load: Vec<Value> = http
        .post(format!("{}/api/admin/local-runtimes/llamacpp/load", gw.base_url))
        .json(&json!({ "model": "nomic-embed-text-v1.5.Q8_0.gguf" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let lc_loaded = after_load
        .iter()
        .find(|r| r["runtime_id"] == "llamacpp")
        .unwrap();
    assert_eq!(lc_loaded["loaded_models"].as_array().unwrap().len(), 2);

    // 5. Non-streaming call via `llama.cpp/<alias>` + multi-turn `reasoning_content` history!
    let chat_resp = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "llama.cpp/qwen3-coder-fast",
            "top_k": 40,
            "messages": [
                { "role": "user", "content": "Step 1" },
                {
                    "role": "assistant",
                    "reasoning_content": "Prior turn thought",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": { "name": "read_status", "arguments": "{}" }
                    }]
                },
                { "role": "tool", "tool_call_id": "call_abc", "content": "all green" },
                { "role": "user", "content": "Summarize" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), StatusCode::OK);
    assert_eq!(
        chat_resp
            .headers()
            .get("x-quixigate-provider")
            .unwrap()
            .to_str()
            .unwrap(),
        "llamacpp"
    );
    assert_eq!(
        chat_resp
            .headers()
            .get("x-quixigate-cost-nanos")
            .unwrap()
            .to_str()
            .unwrap(),
        "0",
        "local llama.cpp completion must bill exact 0 NanoUsd"
    );
    let equiv_nanos: i64 = chat_resp
        .headers()
        .get("x-quixigate-api-equiv-nanos")
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(equiv_nanos > 0, "local savings equivalent must be tracked");
    let chat_json: Value = chat_resp.json().await.unwrap();
    assert_eq!(chat_json["choices"][0]["message"]["content"], "llama.cpp OK");
    assert_eq!(
        chat_json["choices"][0]["message"]["reasoning_content"],
        "Local GGUF reasoning"
    );

    // Verify alias resolved to canonical `ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M`, `top_k` forwarded,
    // and multi-turn `reasoning_content` + `tool_calls` coalesced onto one assistant message!
    {
        let bodies = captured_bodies.lock();
        let sent = bodies.last().unwrap();
        assert_eq!(sent["model"], "ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M");
        assert_eq!(sent["top_k"], json!(40));
        assert_eq!(sent["messages"][1]["role"], "assistant");
        assert_eq!(sent["messages"][1]["reasoning_content"], "Prior turn thought");
        assert_eq!(
            sent["messages"][1]["tool_calls"][0]["function"]["name"],
            "read_status"
        );
    }

    // 6. Streaming call using bare HuggingFace `org/repo:quant` ID (`ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M`)
    // verifying `org/` prefix is preserved and `timings.cache_n` populates cached input tokens!
    let stream_text = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "ggml-org/Qwen3-Coder-30B-GGUF:Q4_K_M",
            "stream": true,
            "messages": [{ "role": "user", "content": "stream test" }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(stream_text.contains("Thinking locally..."));
    assert!(stream_text.contains("llama.cpp stream OK"));

    // 7. Embeddings call via `llama.cpp/nomic-local` alias!
    let emb_res: Value = http
        .post(format!("{}/v1/embeddings", gw.base_url))
        .json(&json!({
            "model": "llama.cpp/nomic-local",
            "input": ["local GGUF embeddings"]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(emb_res["data"][0]["embedding"], json!([0.25, -0.5, 0.75]));

    // Unload model cleanly
    let after_unload: Vec<Value> = http
        .post(format!(
            "{}/api/admin/local-runtimes/llamacpp/unload",
            gw.base_url
        ))
        .json(&json!({ "model": "nomic-embed-text-v1.5.Q8_0.gguf" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let lc_unloaded = after_unload
        .iter()
        .find(|r| r["runtime_id"] == "llamacpp")
        .unwrap();
    assert_eq!(lc_unloaded["loaded_models"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn e2e_optional_ui_loopback_vs_remote_auth_and_litellm_management_api() {
    use axum::body::Body;
    use axum::extract::ConnectInfo;
    use axum::http::Request;
    use std::net::SocketAddr;
    use tower::ServiceExt;

    // Mock MCP JSON-RPC server + OpenAI upstream for headless Management API verification
    let mock_mcp_and_llm = Router::new()
        .route(
            "/mcp",
            post(|Json(body): Json<Value>| async move {
                assert_eq!(body["method"], json!("tools/list"));
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [{
                            "name": "lookup_ticket",
                            "description": "Looks up a support ticket by ID",
                            "inputSchema": {
                                "type": "object",
                                "properties": { "id": { "type": "string" } },
                                "required": ["id"]
                            }
                        }]
                    }
                }))
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|| async {
                Json(json!({
                    "id": "chatcmpl-headless",
                    "object": "chat.completion",
                    "created": 1735689600,
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "Headless OK" },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250 }
                }))
            }),
        );
    let upstream_url = spawn_axum(mock_mcp_and_llm).await;

    // Build a gateway bound as `0.0.0.0` (`bound_loopback = false`) with a generated Master Key!
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    let refresher = CredentialRefresher::new(db.clone());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    let router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner);
    let telemetry = TelemetryHub::new();
    let wizard = OAuthWizardManager::new(db.clone(), refresher);
    let master_key = quixigate_proxy::state::generate_master_key();
    assert!(master_key.starts_with("sk-qx-master-"));

    // Boot with `.with_ui_override(Some(false))` to simulate `quixigate serve --no-ui` / `DISABLE_ADMIN_UI=true`,
    // proving the Management API can dynamically switch `/ui` ON and OFF at runtime without restarting!
    let state = AppState::new(db.clone(), router, telemetry, wizard, false)
        .with_master_key(&master_key)
        .with_ui_override(Some(false));
    let app = build_router(state.clone());

    // ── Part A: Per-Connection Loopback vs Remote IP & Spoofing Guards ───────
    let local_peer: SocketAddr = "127.0.0.1:54321".parse().unwrap();
    let ipv6_loopback_peer: SocketAddr = "[::1]:54322".parse().unwrap();
    let remote_peer: SocketAddr = "192.168.1.77:54323".parse().unwrap();

    // A1. Connecting from 127.0.0.1 (including `Host: 0.0.0.0:4000` or `localhost:4000`) requires NO key!
    let mut req_local = Request::builder()
        .uri("/api/admin/overview")
        .header("host", "0.0.0.0:4000")
        .body(Body::empty())
        .unwrap();
    req_local.extensions_mut().insert(ConnectInfo(local_peer));
    let resp_local = app.clone().oneshot(req_local).await.unwrap();
    assert_eq!(resp_local.status(), StatusCode::OK);

    // A2. Connecting from IPv6 loopback `[::1]` also requires NO key!
    let mut req_v6 = Request::builder()
        .uri("/api/admin/auth/status")
        .header("host", "[::1]:4000")
        .body(Body::empty())
        .unwrap();
    req_v6
        .extensions_mut()
        .insert(ConnectInfo(ipv6_loopback_peer));
    let resp_v6 = app.clone().oneshot(req_v6).await.unwrap();
    assert_eq!(resp_v6.status(), StatusCode::OK);

    // A2b. Open WebUI / LibreChat running in Docker (`http://host.docker.internal:4000/v1`):
    // - macOS Docker Desktop / OrbStack (`127.0.0.1` peer + `Host: host.docker.internal:4000` + empty `Bearer `)
    // - Linux Docker bridge (`172.17.0.2` peer + `Host: host.docker.internal:4000` or `172.17.0.1:4000`)
    // - Browser Direct Connections (`Origin: http://localhost:3333`)
    let docker_bridge_peer: SocketAddr = "172.17.0.2:49152".parse().unwrap();
    for (peer, host_hdr, auth_hdr, origin_hdr) in [
        (local_peer, "host.docker.internal:4000", Some("Bearer "), None),
        (local_peer, "host.docker.internal:4000", Some("Bearer none"), None),
        (docker_bridge_peer, "host.docker.internal:4000", None, None),
        (docker_bridge_peer, "172.17.0.1:4000", None, None),
        (local_peer, "localhost:4000", None, Some("http://localhost:3333")),
    ] {
        let mut req_builder = Request::builder()
            .uri("/v1/models")
            .header("host", host_hdr);
        if let Some(a) = auth_hdr {
            req_builder = req_builder.header("authorization", a);
        }
        if let Some(o) = origin_hdr {
            req_builder = req_builder.header("origin", o);
        }
        let mut req_docker = req_builder.body(Body::empty()).unwrap();
        req_docker.extensions_mut().insert(ConnectInfo(peer));
        let resp_docker = app.clone().oneshot(req_docker).await.unwrap();
        assert_eq!(
            resp_docker.status(),
            StatusCode::OK,
            "expected Docker Open WebUI connection ({peer}, Host={host_hdr}) to succeed passwordlessly"
        );
    }

    // Cross-origin browser spoofing against `host.docker.internal` MUST still be rejected!
    let mut req_docker_csrf = Request::builder()
        .uri("/v1/models")
        .header("host", "host.docker.internal:4000")
        .header("origin", "https://evil.example")
        .body(Body::empty())
        .unwrap();
    req_docker_csrf
        .extensions_mut()
        .insert(ConnectInfo(local_peer));
    assert_eq!(
        app.clone().oneshot(req_docker_csrf).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    // A3. Connecting from remote LAN/WAN IP (`192.168.1.77`) WITHOUT a key -> `401 Unauthorized`!
    for protected_uri in ["/v1/models", "/api/admin/overview", "/key/list"] {
        let mut req_remote = Request::builder()
            .uri(protected_uri)
            .header("host", "192.168.1.10:4000")
            // Attempt to spoof internal loopback header — must be stripped by `peer_trust_middleware`!
            .header("x-quixigate-internal-direct-loopback", "1")
            .body(Body::empty())
            .unwrap();
        req_remote.extensions_mut().insert(ConnectInfo(remote_peer));
        let resp = app.clone().oneshot(req_remote).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "remote unauthenticated call to {protected_uri} must be rejected with 401"
        );
    }

    // A4. Local peer WITH reverse-proxy forwarding header (`X-Forwarded-For`), DNS-rebinding `Host`,
    // or cross-origin `Origin` must NOT get passwordless loopback trust!
    let mut req_xff = Request::builder()
        .uri("/api/admin/overview")
        .header("host", "127.0.0.1:4000")
        .header("x-forwarded-for", "203.0.113.9")
        .body(Body::empty())
        .unwrap();
    req_xff.extensions_mut().insert(ConnectInfo(local_peer));
    assert_eq!(
        app.clone().oneshot(req_xff).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    let mut req_rebind = Request::builder()
        .uri("/api/admin/overview")
        .header("host", "rebind.attacker.example")
        .body(Body::empty())
        .unwrap();
    req_rebind.extensions_mut().insert(ConnectInfo(local_peer));
    assert_eq!(
        app.clone().oneshot(req_rebind).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    let mut req_csrf = Request::builder()
        .uri("/api/admin/overview")
        .header("host", "localhost:4000")
        .header("origin", "https://evil.example")
        .body(Body::empty())
        .unwrap();
    req_csrf.extensions_mut().insert(ConnectInfo(local_peer));
    assert_eq!(
        app.clone().oneshot(req_csrf).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    // A5. Remote peer WITH Master Key (`Authorization: Bearer` OR `x-litellm-api-key`) -> `200 OK`!
    let mut req_remote_bearer = Request::builder()
        .uri("/api/admin/overview")
        .header("authorization", format!("Bearer {master_key}"))
        .body(Body::empty())
        .unwrap();
    req_remote_bearer
        .extensions_mut()
        .insert(ConnectInfo(remote_peer));
    assert_eq!(
        app.clone()
            .oneshot(req_remote_bearer)
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );

    let mut req_remote_litellm_hdr = Request::builder()
        .uri("/key/info")
        .header("x-litellm-api-key", &master_key)
        .body(Body::empty())
        .unwrap();
    req_remote_litellm_hdr
        .extensions_mut()
        .insert(ConnectInfo(remote_peer));
    assert_eq!(
        app.clone()
            .oneshot(req_remote_litellm_hdr)
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );

    // A6. Remote peer can also sign in at `POST /api/admin/auth/login` using the stdout Master Key!
    let mut req_remote_login = Request::builder()
        .method("POST")
        .uri("/api/admin/auth/login")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({ "username": "local", "password": master_key }).to_string(),
        ))
        .unwrap();
    req_remote_login
        .extensions_mut()
        .insert(ConnectInfo(remote_peer));
    let login_resp = app.clone().oneshot(req_remote_login).await.unwrap();
    assert_eq!(login_resp.status(), StatusCode::OK);
    assert!(
        login_resp
            .headers()
            .get("set-cookie")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("quixigate_session=qxs_")
    );

    // A7. Wrong `sk-qx-master-...` key must fail even on loopback (`401`)!
    let mut req_bad_master = Request::builder()
        .uri("/api/admin/overview")
        .header("authorization", "Bearer sk-qx-master-wrong-secret")
        .body(Body::empty())
        .unwrap();
    req_bad_master
        .extensions_mut()
        .insert(ConnectInfo(local_peer));
    assert_eq!(
        app.clone().oneshot(req_bad_master).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    // ── Part B: Bind Real TCP Listener & Exercise Full Headless Management API ─
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    let base = format!("http://{addr}");
    let http = reqwest::Client::new();

    // B1. OpenAPI 3.1.0 & Optional `/ui` runtime toggle
    let spec: Value = http
        .get(format!("{base}/openapi.json"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(spec["openapi"], json!("3.1.0"));
    assert!(spec["paths"]["/key/generate"].is_object());
    assert!(spec["paths"]["/api/admin/mcp"].is_object());
    assert!(spec["paths"]["/api/admin/ui"].is_object());
    assert!(spec["paths"]["/config/ui"].is_object());
    assert!(spec["paths"]["/api/admin/spend/export"].is_object());

    // Initially disabled by startup `--no-ui` override (`with_ui_override(Some(false))`)
    let ui_initial_status: Value = http
        .get(format!("{base}/api/admin/ui"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ui_initial_status["enabled"], json!(false));
    assert_eq!(ui_initial_status["runtime_override"], json!(false));

    let ui_boot_off = http.get(format!("{base}/ui")).send().await.unwrap();
    assert_eq!(ui_boot_off.status(), StatusCode::NOT_FOUND);

    #[cfg(feature = "ui")]
    {
        // Live switch ON via `POST /api/admin/ui/enable` — overrides `--no-ui` with zero restart or config file edit!
        let ui_enabled_resp = http
            .post(format!("{base}/api/admin/ui/enable"))
            .send()
            .await
            .unwrap();
        assert_eq!(ui_enabled_resp.status(), StatusCode::OK);
        let ui_enabled_json: Value = ui_enabled_resp.json().await.unwrap();
        assert_eq!(ui_enabled_json["enabled"], json!(true));
        assert_eq!(ui_enabled_json["ui_url"], json!("/ui"));
        assert_eq!(ui_enabled_json["runtime_override"], Value::Null);

        let ui_html = http.get(format!("{base}/ui")).send().await.unwrap();
        assert_eq!(ui_html.status(), StatusCode::OK);
        let html_body = ui_html.text().await.unwrap();
        assert!(html_body.contains("id=\"root\""));

        let root_with_ui: Value = http
            .get(format!("{base}/"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(root_with_ui["ui_enabled"], json!(true));
        assert_eq!(root_with_ui["ui_url"], json!("/ui"));

        // Live switch OFF via LiteLLM-compatible `POST /config/ui` -> instant `404 ui_disabled`!
        let ui_off_litellm: Value = http
            .post(format!("{base}/config/ui"))
            .json(&json!({ "enabled": false }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(ui_off_litellm["enabled"], json!(false));

        // Ephemeral in-memory switch ON (`persist: false`) via `POST /api/admin/ui`, then OFF via `POST /api/admin/ui/disable`
        let ui_ephemeral: Value = http
            .post(format!("{base}/api/admin/ui"))
            .json(&json!({ "enabled": true, "persist": false }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(ui_ephemeral["enabled"], json!(true));
        assert_eq!(ui_ephemeral["runtime_override"], json!(true));
        assert_eq!(ui_ephemeral["persisted_ui_enabled"], json!(false));
        assert_eq!(
            http.get(format!("{base}/ui")).send().await.unwrap().status(),
            StatusCode::OK
        );

        let ui_disabled_again: Value = http
            .post(format!("{base}/api/admin/ui/disable"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(ui_disabled_again["enabled"], json!(false));
    }

    #[cfg(not(feature = "ui"))]
    {
        // When compiled with `--no-default-features`, attempting to enable `/ui` returns a clean 400 validation error
        assert_eq!(ui_initial_status["compiled_in"], json!(false));
        let err_enable = http
            .post(format!("{base}/api/admin/ui/enable"))
            .send()
            .await
            .unwrap();
        assert_eq!(err_enable.status(), StatusCode::BAD_REQUEST);
    }

    // Confirm `PATCH /api/admin/settings` keeps UI disabled (`404` JSON) while API stays up!
    let patched_settings: Value = http
        .patch(format!("{base}/api/admin/settings"))
        .json(&json!({ "ui_enabled": false }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(patched_settings["ui_enabled"], json!(false));

    let ui_off = http.get(format!("{base}/ui")).send().await.unwrap();
    assert_eq!(ui_off.status(), StatusCode::NOT_FOUND);
    let ui_off_json: Value = ui_off.json().await.unwrap();
    assert_eq!(ui_off_json["error"]["code"], json!("ui_disabled"));

    // Root API discovery reports `ui_enabled: false` and remains `200 OK`
    let root_headless: Value = http
        .get(format!("{base}/"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(root_headless["ui_enabled"], json!(false));
    assert_eq!(root_headless["ui_url"], Value::Null);

    // B2. LiteLLM User + Team + Key + Model + Spend + Cache + YAML Config workflow (100% headless!)
    let new_user: Value = http
        .post(format!("{base}/user/new"))
        .json(&json!({
            "username": "alice",
            "user_alias": "Alice Ops",
            "user_role": "internal_user",
            "max_budget": 25.0,
            "budget_duration": "monthly",
            "auto_create_key": false
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let alice_id = new_user["user_id"].as_str().unwrap().to_owned();
    assert_eq!(new_user["max_budget_nanos"], json!(25_000_000_000_i64));

    let new_team: Value = http
        .post(format!("{base}/team/new"))
        .json(&json!({
            "team_alias": "Platform Core",
            "max_budget": "100.00",
            "budget_duration": "monthly"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let team_id = new_team["team_id"].as_str().unwrap().to_owned();

    let team_with_alice: Value = http
        .post(format!("{base}/team/member_add"))
        .json(&json!({
            "team_id": team_id,
            "member": [{ "user_id": "alice", "role": "user" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        team_with_alice["members"]
            .as_array()
            .unwrap()
            .contains(&json!(alice_id))
    );

    // Register model + encrypted credential via LiteLLM `/model/new`
    let model_created: Value = http
        .post(format!("{base}/model/new"))
        .json(&json!({
            "model_name": "headless-gpt",
            "litellm_params": {
                "model": "openaichatcompletions/gpt-4o-mini",
                "api_key": "sk-upstream-headless",
                "api_base": format!("{upstream_url}/v1")
            }
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(model_created["model_name"], json!("headless-gpt"));
    assert!(model_created["credential_id"].as_str().is_some());

    // Mint virtual key via LiteLLM `/key/generate`
    let minted: Value = http
        .post(format!("{base}/key/generate"))
        .json(&json!({
            "key_alias": "alice-ci-key",
            "user_id": alice_id,
            "team_id": team_id,
            "max_budget": 5.5,
            "rpm_limit": 60,
            "models": ["headless-gpt"]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let raw_vk = minted["key"].as_str().unwrap().to_owned();
    let vk_id = minted["token_id"].as_str().unwrap().to_owned();
    assert!(raw_vk.starts_with("sk-qx-"));
    assert_eq!(minted["max_budget_nanos"], json!(5_500_000_000_i64));

    // Execute real inference using the minted key!
    let chat_ok: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&raw_vk)
        .json(&json!({
            "model": "headless-gpt",
            "messages": [{ "role": "user", "content": "ping" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(chat_ok["choices"][0]["message"]["content"], "Headless OK");

    // Block key via `/key/block` -> inference returns `401`, then `/key/unblock` -> `200`!
    let blocked: Value = http
        .post(format!("{base}/key/block"))
        .json(&json!({ "key": raw_vk }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(blocked["blocked"], json!(true));

    let chat_blocked = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&raw_vk)
        .json(&json!({
            "model": "headless-gpt",
            "messages": [{ "role": "user", "content": "ping" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat_blocked.status(), StatusCode::UNAUTHORIZED);

    let unblocked: Value = http
        .post(format!("{base}/key/unblock"))
        .json(&json!({ "key": vk_id }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(unblocked["blocked"], json!(false));

    // Update key via `/key/update` & rotate via `/key/regenerate`
    let updated_vk: Value = http
        .post(format!("{base}/key/update"))
        .json(&json!({
            "key": vk_id,
            "max_budget": "12.25",
            "rpm_limit": 120
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(updated_vk["max_budget_nanos"], json!(12_250_000_000_i64));
    assert_eq!(updated_vk["rpm_limit"], json!(120));

    let rotated_vk: Value = http
        .post(format!("{base}/key/regenerate"))
        .json(&json!({ "key": vk_id, "grace_seconds": 60 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let new_raw_vk = rotated_vk["key"].as_str().unwrap();
    assert_ne!(new_raw_vk, raw_vk);

    // Verify `/spend/calculate`, `/spend/logs`, `/global/spend/report`, `/cache/ping`, `/config/yaml`, `/health/liveliness`
    let calc: Value = http
        .post(format!("{base}/spend/calculate"))
        .json(&json!({
            "model": "openai/gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 500
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(calc["cost_nanos"].as_i64().unwrap() > 0);

    let spend_logs: Vec<Value> = http
        .get(format!("{base}/spend/logs?user_id={alice_id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(spend_logs.len(), 1);
    assert_eq!(spend_logs[0]["model"], json!("gpt-4o-mini"));

    let report: Value = http
        .get(format!("{base}/global/spend/report"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(report["users"].is_array());
    assert!(report["keys"].is_array());

    let cache_ping: Value = http
        .get(format!("{base}/cache/ping"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cache_ping["status"], json!("healthy"));

    let cfg_yaml: Value = http
        .get(format!("{base}/config/yaml"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(cfg_yaml["yaml"].as_str().unwrap().contains("headless-gpt"));

    let liveliness: Value = http
        .get(format!("{base}/health/liveliness"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(liveliness["ok"], json!(true));

    // Verify `/api/admin/doctor` and `/api/admin/mcp` + `/api/admin/mcp/probe`
    let doctor: Value = http
        .get(format!("{base}/api/admin/doctor"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(doctor["ok"], json!(true));

    let mcp_url = format!("{upstream_url}/mcp");
    let mcp_added: Value = http
        .post(format!("{base}/api/admin/mcp"))
        .json(&json!({ "url": mcp_url }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(mcp_added["mcp_server_urls"], json!([mcp_url]));

    let mcp_probe: Value = http
        .post(format!("{base}/api/admin/mcp/probe"))
        .json(&json!({ "url": mcp_url }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(mcp_probe["tool_count"], json!(1));
    assert_eq!(mcp_probe["tools"][0]["name"], json!("lookup_ticket"));

    // Zero-cost `mock_response` & `x-litellm-response-cost` header verification
    let mock_resp = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "ping"}],
            "mock_response": "Instant zero-cost test"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(mock_resp.status(), StatusCode::OK);
    assert_eq!(
        mock_resp
            .headers()
            .get("x-litellm-response-cost")
            .unwrap()
            .to_str()
            .unwrap(),
        "0.00"
    );
    let mock_json: Value = mock_resp.json().await.unwrap();
    assert_eq!(
        mock_json["choices"][0]["message"]["content"],
        json!("Instant zero-cost test")
    );

    // Spend CSV & JSON export (`/api/admin/spend/export` & `/spend/export`)
    let csv_resp = http
        .get(format!("{base}/api/admin/spend/export?format=csv"))
        .send()
        .await
        .unwrap();
    assert_eq!(csv_resp.status(), StatusCode::OK);
    assert!(
        csv_resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/csv")
    );
    let csv_body = csv_resp.text().await.unwrap();
    assert!(csv_body.starts_with(
        "id,timestamp_ms,ingress_protocol,requested_model,resolved_model,provider_id,"
    ));

    let json_export: Value = http
        .get(format!("{base}/spend/export?format=json"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(json_export["logs"].is_array());

    // Clean up key via `/key/delete`
    let del_res: Value = http
        .post(format!("{base}/key/delete"))
        .json(&json!({ "keys": [vk_id] }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(del_res["deleted_keys"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn e2e_three_coequal_pillars_and_optional_openrouter_oidc_presets_exact_cost_and_disconnect() {
    let or_captured: Arc<Mutex<Vec<(HeaderMap, Value)>>> = Arc::new(Mutex::new(Vec::new()));
    let or_sink = or_captured.clone();

    let mock_all_pillars = Router::new()
        // Pillar 1: Local Ollama runtime
        .route(
            "/ollama/api/tags",
            get(|| async { Json(json!({ "models": [{ "name": "qwen2.5-coder:7b" }] })) }),
        )
        .route(
            "/ollama/api/ps",
            get(|| async { Json(json!({ "models": [] })) }),
        )
        .route(
            "/ollama/v1/models",
            get(|| async { Json(json!({ "data": [{ "id": "qwen2.5-coder:7b" }] })) }),
        )
        .route(
            "/ollama/v1/chat/completions",
            post(|| async {
                Json(json!({
                    "id": "chatcmpl-local-ollama",
                    "object": "chat.completion",
                    "created": 1735689600,
                    "model": "qwen2.5-coder:7b",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "Local Pillar OK" },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16 }
                }))
            }),
        )
        // Pillar 2: OAuth Subscription (`openai-codex` SSE)
        .route(
            "/codex/responses",
            post(|| async {
                let sse = concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"OAuth Pillar OK\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":14,\"output_tokens\":4}}}\n\n"
                );
                (StatusCode::OK, [("content-type", "text/event-stream")], sse)
            }),
        )
        // Pillar 3: OpenRouter (RFC 8693 OIDC + `/key` + `/credits` + `/models` + `/presets` + `/chat/completions`)
        .route(
            "/openrouter/api/v1/oauth/token",
            post(|body: Bytes| async move {
                let form_str = std::str::from_utf8(&body).unwrap_or_default();
                assert!(
                    form_str.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Atoken-exchange"),
                    "expected RFC 8693 token-exchange grant_type, got {form_str}"
                );
                assert!(form_str.contains("subject_token=jwt-workload-token-123"));
                assert!(form_str.contains("federation_policy_id=pol_eng_team"));
                Json(json!({
                    "access_token": "sk-or-oidc-access-1",
                    "token_type": "Bearer",
                    "expires_in": 900
                }))
            }),
        )
        .route(
            "/openrouter/api/v1/key",
            get(|headers: HeaderMap| async move {
                assert_eq!(
                    headers.get("authorization").and_then(|v| v.to_str().ok()),
                    Some("Bearer sk-or-oidc-access-1")
                );
                Json(json!({
                    "data": {
                        "label": "eng-oidc",
                        "limit": 100.0,
                        "limit_remaining": 92.5,
                        "usage": 7.5,
                        "is_free_tier": false
                    }
                }))
            }),
        )
        .route(
            "/openrouter/api/v1/credits",
            get(|headers: HeaderMap| async move {
                assert_eq!(
                    headers.get("authorization").and_then(|v| v.to_str().ok()),
                    Some("Bearer sk-or-oidc-access-1")
                );
                Json(json!({
                    "data": {
                        "total_credits": 50.0,
                        "total_usage": 7.5
                    }
                }))
            }),
        )
        .route(
            "/openrouter/api/v1/models",
            get(|| async {
                Json(json!({
                    "data": [
                        {
                            "id": "openrouter-lab/aurora-coder-v1",
                            "name": "Aurora Coder v1",
                            "context_length": 163840,
                            "pricing": {
                                "prompt": "0.00000055",
                                "completion": "0.00000219"
                            },
                            "supported_parameters": ["tools", "reasoning"],
                            "architecture": { "input_modalities": ["text"] }
                        },
                        {
                            "id": "anthropic/claude-opus-4-6",
                            "name": "Claude Opus 4.6",
                            "context_length": 200000,
                            "pricing": {
                                "prompt": "0.000015",
                                "completion": "0.000075"
                            },
                            "supported_parameters": ["tools", "reasoning"],
                            "architecture": { "input_modalities": ["text", "image"] }
                        }
                    ]
                }))
            }),
        )
        .route(
            "/openrouter/api/v1/presets",
            get(|headers: HeaderMap| async move {
                assert_eq!(
                    headers.get("authorization").and_then(|v| v.to_str().ok()),
                    Some("Bearer sk-or-oidc-access-1")
                );
                Json(json!({
                    "data": [
                        {
                            "slug": "code-reviewer",
                            "name": "Code Reviewer",
                            "description": "Strict Rust security & correctness reviewer",
                            "model": "anthropic/claude-opus-4-6"
                        },
                        {
                            "slug": "inbound-classifier",
                            "name": "Inbound Classifier",
                            "description": "Fast triage router",
                            "model": "openai/gpt-5-mini"
                        }
                    ]
                }))
            }),
        )
        .route(
            "/openrouter/api/v1/chat/completions",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let or_sink = or_sink.clone();
                async move {
                    assert_eq!(
                        headers.get("authorization").and_then(|v| v.to_str().ok()),
                        Some("Bearer sk-or-oidc-access-1")
                    );
                    assert!(
                        headers.get("http-referer").is_some(),
                        "OpenRouter attribution header HTTP-Referer must be set"
                    );
                    assert!(
                        headers.get("x-title").is_some(),
                        "OpenRouter attribution header X-Title must be set"
                    );
                    assert_eq!(
                        body["usage"]["include"],
                        json!(true),
                        "OpenRouter requests must request exact usage.cost accounting"
                    );
                    or_sink.lock().push((headers, body.clone()));

                    if body["stream"].as_bool() == Some(true) {
                        let sse = concat!(
                            "data: {\"id\":\"or-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1735689600,\"model\":\"anthropic/claude-opus-4-6\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Routing via :nitro...\"},\"finish_reason\":null}]}\n\n",
                            "data: {\"id\":\"or-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1735689600,\"model\":\"anthropic/claude-opus-4-6\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OpenRouter Nitro Stream OK\"},\"finish_reason\":null}]}\n\n",
                            "data: {\"id\":\"or-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1735689600,\"model\":\"anthropic/claude-opus-4-6\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":8,\"total_tokens\":23,\"cost\":0.0015}}\n\n",
                            "data: [DONE]\n\n"
                        );
                        return (StatusCode::OK, [("content-type", "text/event-stream")], sse)
                            .into_response();
                    }

                    Json(json!({
                        "id": "or-chat-1",
                        "object": "chat.completion",
                        "created": 1735689600,
                        "model": "anthropic/claude-opus-4-6",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Preset Code Review OK"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                            "total_tokens": 30,
                            "cost": 0.00425
                        }
                    }))
                    .into_response()
                }
            }),
        );

    let mock_url = spawn_axum(mock_all_pillars).await;
    let gw = spawn_test_gateway(
        RefreshEndpoints::default(),
        QuotaEndpoints::default(),
        Some(format!("{mock_url}/ollama/v1")),
        false,
    )
    .await;

    // Seed Pillar 2 (`openai-codex` OAuth subscription, priority = 50 -> $0 marginal cost)
    gw.db
        .upsert_credential(
            Some("codex_oauth_pillar"),
            "openai-codex",
            "ChatGPT Plus Seat",
            AuthKind::Oauth,
            Some(&format!("{mock_url}/codex")),
            50,
            true,
            false,
            &CredentialSecret::Codex {
                access: "codex-valid-token".to_owned(),
                refresh: "codex-refresh".to_owned(),
                expires_ms: now_ms() + 3_600_000,
                account_id: "acct_codex_1".to_owned(),
            },
        )
        .unwrap();

    let http = reqwest::Client::new();

    // 1. BEFORE connecting OpenRouter: OpenRouter is strictly optional!
    // Zero `@preset/*`, zero `openrouter/auto`, and zero `~*` models may leak into `/v1/models`
    // or `/api/v1/presets`, while Local Ollama + OAuth Codex models ARE ready and routable.
    let or_status_before: Value = http
        .get(format!("{}/api/admin/openrouter/status", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(or_status_before["connected"], json!(false));
    assert_eq!(or_status_before["preset_count"], json!(0));

    let presets_before: Value = http
        .get(format!("{}/api/v1/presets", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(presets_before["data"], json!([]));

    let models_before: Value = http
        .get(format!("{}/v1/models", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids_before: Vec<String> = models_before["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect();
    assert!(ids_before.iter().any(|id| id == "ollama/qwen2.5-coder:7b"));
    assert!(ids_before.iter().any(|id| id == "openai/gpt-5-mini"));
    assert!(!ids_before.iter().any(|id| id.starts_with("@preset/")));
    assert!(!ids_before.iter().any(|id| id.starts_with('~')));
    assert!(!ids_before.iter().any(|id| id == "openrouter/auto"));
    assert!(!ids_before.iter().any(|id| id == "openrouter-lab/aurora-coder-v1"));

    // 2. Connect OpenRouter via RFC 8693 OIDC Workload Identity (`/api/admin/openrouter/connect`)
    let connect_res: Value = http
        .post(format!("{}/api/admin/openrouter/connect", gw.base_url))
        .json(&json!({
            "method": "oidc",
            "label": "OpenRouter OIDC Workload",
            "subject_token": "jwt-workload-token-123",
            "federation_policy_id": "pol_eng_team",
            "token_endpoint": format!("{mock_url}/openrouter/api/v1/oauth/token"),
            "base_url": format!("{mock_url}/openrouter/api/v1")
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(connect_res["mode"], json!("connected"));
    assert_eq!(connect_res["synced_presets"], json!(2));
    assert_eq!(connect_res["account"]["remaining_credits"], json!(42.5));
    let or_cred_id = connect_res["credential"]["id"]
        .as_str()
        .unwrap()
        .to_owned();

    // 3. `/api/v1/presets` and `/v1/models` now surface `@preset/*`, `openrouter/auto`, `~*-latest`, and synced OpenRouter models!
    let presets_after: Value = http
        .get(format!("{}/api/v1/presets", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(presets_after["data"].as_array().unwrap().len(), 2);

    let models_after: Value = http
        .get(format!("{}/v1/models", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids_after: Vec<String> = models_after["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect();
    assert!(ids_after.iter().any(|id| id == "@preset/code-reviewer"));
    assert!(ids_after.iter().any(|id| id == "@preset/inbound-classifier"));
    assert!(ids_after.iter().any(|id| id == "openrouter/auto"));
    assert!(ids_after.iter().any(|id| id == "~anthropic/claude-opus-latest"));
    assert!(ids_after.iter().any(|id| id == "openrouter-lab/aurora-coder-v1"));

    // Verify per-token pricing conversion (`0.00000055` per token -> `$0.55 / 1M` = `550_000_000` nanos/1M)
    let ds_entry = models_after["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "openrouter-lab/aurora-coder-v1")
        .unwrap();
    assert_eq!(ds_entry["pricing"]["input_nanos_per_m"], json!(550_000_000));
    assert_eq!(
        ds_entry["pricing"]["output_nanos_per_m"],
        json!(2_190_000_000_i64)
    );

    // 4. Co-equal First-Class Priority Proof:
    //    4a. Local Ollama request -> `$0` cost!
    let local_resp = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "ollama/qwen2.5-coder:7b",
            "messages": [{ "role": "user", "content": "local check" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        local_resp.headers().get("x-quixigate-provider").unwrap(),
        "ollama"
    );
    assert_eq!(
        local_resp.headers().get("x-quixigate-cost-nanos").unwrap(),
        "0"
    );

    //    4b. Unpinned `openai/gpt-5-mini` STILL drains the `$0` OAuth Codex seat (`priority = 50`)
    //        rather than billing OpenRouter (`priority = 200`), even though OpenRouter uses OIDC!
    let oauth_resp = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openai/gpt-5-mini",
            "messages": [{ "role": "user", "content": "oauth priority check" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        oauth_resp.headers().get("x-quixigate-provider").unwrap(),
        "openai-codex"
    );
    assert_eq!(
        oauth_resp.headers().get("x-quixigate-cost-nanos").unwrap(),
        "0"
    );
    let oauth_json: Value = oauth_resp.json().await.unwrap();
    assert_eq!(oauth_json["choices"][0]["message"]["content"], "OAuth Pillar OK");

    // 5. OpenRouter `@preset/code-reviewer` via `/api/v1/chat/completions` alias with `provider` routing prefs
    //    + exact `usage.cost` (`$0.00425` -> `4_250_000` NanoUsd) + concrete `resolved_model`!
    let preset_resp = http
        .post(format!("{}/api/v1/chat/completions", gw.base_url))
        .header("http-referer", "https://quixigate.test/app")
        .header("x-title", "QuixiGate Test Harness")
        .json(&json!({
            "model": "@preset/code-reviewer",
            "provider": { "order": ["Anthropic"], "allow_fallbacks": false },
            "messages": [{ "role": "user", "content": "review this patch" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(preset_resp.status(), StatusCode::OK);
    assert_eq!(
        preset_resp.headers().get("x-quixigate-provider").unwrap(),
        "openrouter"
    );
    assert_eq!(
        preset_resp.headers().get("x-quixigate-cost-nanos").unwrap(),
        "4250000"
    );
    let preset_json: Value = preset_resp.json().await.unwrap();
    assert_eq!(preset_json["model"], json!("anthropic/claude-opus-4-6"));
    assert_eq!(
        preset_json["choices"][0]["message"]["content"],
        json!("Preset Code Review OK")
    );

    // 6. Streaming `~anthropic/claude-opus-latest:nitro` -> forwards `~anthropic/claude-opus-latest:nitro`
    //    upstream and records exact streamed `usage.cost` (`$0.0015` -> `1_500_000` NanoUsd)!
    let nitro_stream = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "~anthropic/claude-opus-latest:nitro",
            "stream": true,
            "messages": [{ "role": "user", "content": "fast opus check" }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(nitro_stream.contains("OpenRouter Nitro Stream OK"));

    {
        let sent = or_captured.lock();
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].1["model"], json!("@preset/code-reviewer"));
        assert_eq!(
            sent[0].1["provider"],
            json!({ "order": ["Anthropic"], "allow_fallbacks": false })
        );
        assert_eq!(
            sent[0].0.get("http-referer").unwrap().to_str().unwrap(),
            "https://quixigate.test/app"
        );
        assert_eq!(
            sent[0].0.get("x-title").unwrap().to_str().unwrap(),
            "QuixiGate Test Harness"
        );
        assert_eq!(
            sent[1].1["model"],
            json!("~anthropic/claude-opus-latest:nitro")
        );
        // Default OpenRouter app-leaderboard attribution headers when caller omits them:
        assert_eq!(
            sent[1].0.get("http-referer").unwrap().to_str().unwrap(),
            "https://quixi.ai"
        );
        assert_eq!(
            sent[1].0.get("x-title").unwrap().to_str().unwrap(),
            "QuixiGate"
        );
    }

    // 6b. 1-Click Cloud Burst (`POST /api/admin/burst/enable`): when an OAuth seat (`allow_api_overflow = false`)
    //     hits a 429 cooldown, Cloud Burst seamlessly overflows onto OpenRouter; disabling Burst restores strict isolation!
    gw.db
        .record_credential_cooldown("codex_oauth_pillar", quixigate_core::now_ms() + 60_000, "429 quota")
        .unwrap();

    // Before enabling Burst: `codex_seat` has `allow_api_overflow = false`, so `openai/gpt-5-mini` does NOT spill to paid cloud
    let blocked_before_burst = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openai/gpt-5-mini",
            "messages": [{ "role": "user", "content": "strict before burst" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(blocked_before_burst.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Enable Cloud Burst in one API call -> same request now bursts seamlessly to OpenRouter!
    let burst_on: Value = http
        .post(format!("{}/api/admin/burst/enable", gw.base_url))
        .json(&json!({ "model": "openrouter/auto" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(burst_on["enabled"], json!(true));
    assert_eq!(burst_on["app_title"], json!("QuixiGate"));

    let burst_resp = http
        .post(format!("{}/v1/chat/completions", gw.base_url))
        .json(&json!({
            "model": "openai/gpt-5-mini",
            "messages": [{ "role": "user", "content": "burst to openrouter!" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(burst_resp.status(), StatusCode::OK);
    assert_eq!(
        burst_resp.headers().get("x-quixigate-provider").unwrap(),
        "openrouter"
    );

    // Clear cooldown & disable Cloud Burst before step 7
    let _ = http
        .post(format!(
            "{}/api/admin/credentials/codex_oauth_pillar/clear-cooldown",
            gw.base_url
        ))
        .send()
        .await
        .unwrap();
    let _ = http
        .post(format!("{}/api/admin/burst/disable", gw.base_url))
        .send()
        .await
        .unwrap();

    // Verify SQLite request_logs via `/api/admin/traces` captured both concrete `resolved_model`
    // and exact streamed cost (`1_500_000` nanos)
    tokio::time::sleep(Duration::from_millis(40)).await;
    let logs: Vec<Value> = http
        .get(format!("{}/api/admin/traces", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let stream_log = logs
        .iter()
        .find(|l| l["requested_model"] == json!("~anthropic/claude-opus-latest:nitro"))
        .expect("streamed OpenRouter request must be logged");
    assert_eq!(stream_log["resolved_model"], json!("anthropic/claude-opus-4-6"));
    assert_eq!(stream_log["cost_nanos"], json!(1_500_000));

    // 7. Disconnect OpenRouter -> `@preset/*` and OpenRouter-only models immediately disappear,
    //    while Local Ollama + OAuth Codex remain routable!
    let del_or = http
        .delete(format!("{}/api/admin/credentials/{or_cred_id}", gw.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(del_or.status(), StatusCode::OK);

    let presets_after_del: Value = http
        .get(format!("{}/api/v1/presets", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(presets_after_del["data"], json!([]));

    let models_after_del: Value = http
        .get(format!("{}/v1/models", gw.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids_after_del: Vec<String> = models_after_del["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect();
    assert!(!ids_after_del.iter().any(|id| id.starts_with("@preset/")));
    assert!(!ids_after_del.iter().any(|id| id == "openrouter/auto"));
    assert!(ids_after_del.iter().any(|id| id == "ollama/qwen2.5-coder:7b"));
    assert!(ids_after_del.iter().any(|id| id == "openai/gpt-5-mini"));
}

#[tokio::test]
async fn e2e_ollama_reasoning_token_reconciliation_inline_think_and_504_timeout_diagnostics() {
    let mock_ollama = Router::new()
        .route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "object": "list",
                    "data": [{ "id": "lfm2.5:24b", "object": "model", "owned_by": "library" }]
                }))
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                let prompt = body["messages"]
                    .as_array()
                    .and_then(|m| m.last())
                    .and_then(|m| m["content"].as_str())
                    .unwrap_or_default();

                if prompt == "Hello!" {
                    // Simulate Ollama >100ms generation (longer than a 60ms OAuth refresher timeout!)
                    // returning `message.reasoning` WITHOUT `completion_tokens_details.reasoning_tokens`
                    tokio::time::sleep(Duration::from_millis(140)).await;
                    Json(json!({
                        "id": "chatcmpl-ollama-hello",
                        "object": "chat.completion",
                        "model": "lfm2.5:24b",
                        "choices": [{
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Hello! 👋 How can I help you today?",
                                "reasoning": "The user says \"Hello!\" This is a simple greeting. The assistant should respond politely, perhaps ask how they can help. There's no policy violation. So just greet back and ask what they need.\n\nWe must follow the style guidelines: be friendly, helpful. Possibly ask about their day or what they'd like assistance with.\n\nThus answer: \"Hello! How can I help you today?\" That's it."
                            }
                        }],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 100,
                            "total_tokens": 112
                        }
                    }))
                } else if prompt == "inline think length" {
                    Json(json!({
                        "id": "chatcmpl-ollama-think",
                        "object": "chat.completion",
                        "model": "lfm2.5:24b",
                        "choices": [{
                            "index": 0,
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": "<think>\nExploring graph neighbors using BinaryHeap...\n</think>\npub fn astar("
                            }
                        }],
                        "usage": {
                            "prompt_tokens": 18,
                            "completion_tokens": 20,
                            "total_tokens": 38
                        }
                    }))
                } else {
                    // Simulate hung upstream exceeding custom inference read_timeout
                    tokio::time::sleep(Duration::from_millis(450)).await;
                    Json(json!({ "choices": [] }))
                }
            }),
        );
    let ollama_base = spawn_axum(mock_ollama).await;

    // Build a gateway whose `GatewayRouter` has a 300ms inference read_timeout:
    // - Fast enough that 140ms `"Hello!"` succeeds cleanly
    // - Short enough that 450ms `"hang"` triggers a deterministic 504 Gateway Timeout
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    let refresher = CredentialRefresher::with_endpoints(db.clone(), RefreshEndpoints::default());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    scanner.add_or_update_endpoint(LocalEndpointSpec {
        runtime_id: "ollama".to_owned(),
        display_name: "Ollama".to_owned(),
        base_url: format!("{ollama_base}/v1"),
    });
    scanner.scan_all().await;

    let inference_http = reqwest::Client::builder()
        .read_timeout(Duration::from_millis(300))
        .build()
        .unwrap();
    let router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner)
        .with_inference_http(inference_http);
    let state = AppState::new(
        db.clone(),
        router,
        TelemetryHub::new(),
        OAuthWizardManager::new(db.clone(), refresher),
        true,
    );
    let gw_url = spawn_axum(build_router(state)).await;
    let http = reqwest::Client::new();

    // 1. `"Hello!"` -> reconciles `reasoning_tokens` (> 0) both in response JSON and SQLite trace!
    let r_hello = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .json(&json!({
            "model": "ollama/lfm2.5:24b",
            "messages": [{ "role": "user", "content": "Hello!" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r_hello.status(), StatusCode::OK);
    let hello_json: Value = r_hello.json().await.unwrap();
    assert_eq!(
        hello_json["choices"][0]["message"]["content"],
        json!("Hello! 👋 How can I help you today?")
    );
    assert!(
        hello_json["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .unwrap()
            .starts_with("The user says \"Hello!\"")
    );
    let rt = hello_json["usage"]["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap();
    assert!(
        (85..=95).contains(&rt),
        "expected reconciled reasoning_tokens in 85..=95, got {rt}"
    );

    // 2. Inline `<think>...</think>` + `finish_reason: "length"` preserved
    let r_think: Value = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .json(&json!({
            "model": "ollama/lfm2.5:24b",
            "messages": [{ "role": "user", "content": "inline think length" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r_think["choices"][0]["finish_reason"], json!("length"));
    assert_eq!(
        r_think["choices"][0]["message"]["reasoning_content"],
        json!("Exploring graph neighbors using BinaryHeap...")
    );
    assert_eq!(
        r_think["choices"][0]["message"]["content"],
        json!("pub fn astar(")
    );
    assert!(
        r_think["usage"]["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    // 3. Hung upstream -> returns HTTP 504 `upstream_timeout` with full causal chain & logs `provider_id = "ollama"`
    let r_timeout = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .json(&json!({
            "model": "ollama/lfm2.5:24b",
            "messages": [{ "role": "user", "content": "hang" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r_timeout.status(), StatusCode::GATEWAY_TIMEOUT);
    let err_json: Value = r_timeout.json().await.unwrap();
    assert_eq!(err_json["error"]["code"], json!("upstream_timeout"));
    assert_eq!(err_json["error"]["provider"], json!("ollama"));
    let err_msg = err_json["error"]["message"].as_str().unwrap();
    assert!(
        err_msg.to_ascii_lowercase().contains("timed out"),
        "expected error message to include root cause 'timed out', got: {err_msg}"
    );

    tokio::time::sleep(Duration::from_millis(40)).await;
    let traces: Vec<Value> = http
        .get(format!("{gw_url}/api/admin/traces"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let failed_trace = traces
        .iter()
        .find(|t| t["status_code"] == json!(504))
        .expect("504 timeout trace should be logged");
    assert_eq!(failed_trace["provider_id"], json!("ollama"));
    let ok_trace = traces
        .iter()
        .find(|t| t["status_code"] == json!(200) && t["output_tokens"] == json!(100))
        .expect("Hello! trace should be logged");
    assert_eq!(ok_trace["reasoning_tokens"], json!(rt));
}

#[tokio::test]
async fn e2e_high_roi_enterprise_capabilities_tags_audit_guardrails_mcp_rbac_and_jwt() {
    use axum::body::Body;
    use axum::extract::ConnectInfo;
    use axum::http::Request;
    use quixigate_core::SecretScannerMode;
    use std::net::SocketAddr;
    use tower::ServiceExt;

    let captured_prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_prompts_ref = captured_prompts.clone();

    let mock_chat_fn = move |Json(body): Json<Value>| {
        let captured = captured_prompts_ref.clone();
        async move {
            let prompt = body["messages"]
                .as_array()
                .and_then(|m| m.last())
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .to_owned();
            captured.lock().push(prompt.clone());
            Json(json!({
                "id": "chatcmpl_ent_1",
                "object": "chat.completion",
                "created": 1_794_460_000,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": format!("ack:{prompt}") },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200
                }
            }))
        }
    };
    let mock_upstream = Router::new()
        .route("/v1/chat/completions", post(mock_chat_fn.clone()))
        .route("/chat/completions", post(mock_chat_fn));
    let upstream_url = spawn_axum(mock_upstream).await;

    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    db.upsert_credential(
        None,
        "openaichatcompletions",
        "OpenAI Enterprise Seat",
        AuthKind::ApiKey,
        Some(&upstream_url),
        100,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: "sk-upstream-test".to_owned(),
        },
    )
    .unwrap();

    let master_key = "sk-qx-master-enterprise-test-key";
    let jwt_secret = b"enterprise-hs256-jwt-signing-secret-32b";

    let refresher = CredentialRefresher::with_endpoints(db.clone(), RefreshEndpoints::default());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    let router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner);
    let state = AppState::new(
        db.clone(),
        router,
        TelemetryHub::new(),
        OAuthWizardManager::new(db.clone(), refresher),
        false,
    )
    .with_master_key(master_key)
    .with_jwt_secret(jwt_secret);

    let mut init_settings = db.get_settings().unwrap();
    init_settings.require_auth = true;
    init_settings.capture_payloads_default = true;
    init_settings.docs_title = Some("Acme Enterprise AI Gateway".to_owned());
    init_settings.docs_description = Some("Zero-dependency Rust AI Gateway".to_owned());
    init_settings.docs_filtered = true;
    init_settings
        .tag_budgets_nanos
        .insert("feature:capped".to_owned(), NanoUsd(5_000_000)); // $0.005 cap
    db.update_settings(&init_settings).unwrap();

    let app = build_router(state.clone());
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app_for_server = app.clone();
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app_for_server.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    let gw_url = format!("http://{addr}");
    let http = reqwest::Client::new();

    // =========================================================================
    // 1. Register priced route (`$2/M` in + `$10/M` out -> 1000 in + 200 out = $0.004 = 4_000_000 nanos)
    // =========================================================================
    let route_resp = http
        .post(format!("{gw_url}/api/admin/routes"))
        .bearer_auth(master_key)
        .json(&json!({
            "alias": "gpt-4o",
            "targets": [{ "provider_id": "openaichatcompletions", "model": "gpt-4o" }],
            "input_nanos_per_m": 2_000_000_000_i64,
            "output_nanos_per_m": 10_000_000_000_i64
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(route_resp.status(), StatusCode::OK);

    // Verify `/model/info` exposes `created_by`, `updated_by`, and `created_at_ms` (Bundle 2.7)
    let model_info: Value = http
        .get(format!("{gw_url}/model/info"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let gpt_entry = model_info["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model_name"] == json!("gpt-4o"))
        .expect("gpt-4o route in /model/info");
    assert_eq!(gpt_entry["model_info"]["created_by"], json!("local"));
    assert_eq!(gpt_entry["model_info"]["updated_by"], json!("local"));
    assert!(gpt_entry["model_info"]["created_at_ms"].as_i64().unwrap() > 0);

    // =========================================================================
    // 2. Bundle 2.1 — Key & Request `tags` + Per-Model & Per-Tag `NanoUsd` Budgets + `/spend/report`
    // =========================================================================
    let dev_user_resp: Value = http
        .post(format!("{gw_url}/api/admin/users"))
        .bearer_auth(master_key)
        .json(&json!({ "username": "dev-member", "display_name": "Developer Member", "role": "member" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let dev_user_id = dev_user_resp["id"].as_str().unwrap().to_owned();

    let key_gen: Value = http
        .post(format!("{gw_url}/key/generate"))
        .bearer_auth(master_key)
        .json(&json!({
            "key_alias": "prod-billing-key",
            "user_id": dev_user_id,
            "tags": ["env:prod"],
            "model_max_budget": { "gpt-4o": 0.005 },
            "permissions": { "get_spend_routes": true }
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let prod_key = key_gen["key"].as_str().unwrap().to_owned();
    assert_eq!(key_gen["tags"], json!(["env:prod"]));
    assert_eq!(key_gen["allow_spend_read"], json!(true));

    // Call #1 with `x-quixigate-tags: feature:billing` succeeds (pre-flight ~$0.00203 <= $0.005 cap) and commits $0.004 (4_000_000 nanos)
    let call1 = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&prod_key)
        .header("x-quixigate-tags", "feature:billing")
        .json(&json!({
            "model": "gpt-4o",
            "max_tokens": 200,
            "messages": [{ "role": "user", "content": "invoice summary" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(call1.status(), StatusCode::OK);

    // Call #2 on `gpt-4o` with the same key must fail pre-flight because $0.004 + ~$0.00203 > model_max_budget ($0.005)!
    let call2 = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&prod_key)
        .json(&json!({
            "model": "gpt-4o",
            "max_tokens": 200,
            "messages": [{ "role": "user", "content": "second invoice" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(call2.status(), StatusCode::PAYMENT_REQUIRED);
    let call2_err: Value = call2.json().await.unwrap();
    assert!(
        call2_err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("model 'gpt-4o' budget limit"),
        "unexpected error message: {:?}",
        call2_err
    );

    // Grouped `/spend/report?group_by=tag` & `/spend/tags` reflect both `env:prod` and `feature:billing`!
    let tag_report: Value = http
        .get(format!("{gw_url}/spend/report?group_by=tag"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let tag_rows = tag_report["data"].as_array().unwrap();
    let prod_row = tag_rows
        .iter()
        .find(|r| r["group_key"] == json!("env:prod"))
        .expect("env:prod tag spend row");
    assert_eq!(prod_row["spend_nanos"], json!(4_000_000));
    let billing_row = tag_rows
        .iter()
        .find(|r| r["group_key"] == json!("feature:billing"))
        .expect("feature:billing tag spend row");
    assert_eq!(billing_row["spend_nanos"], json!(4_000_000));

    // Self-service key spend inspection (`allow_spend_read = true`) works on `/spend/report` & `/spend/logs`!
    let self_report = http
        .get(format!("{gw_url}/spend/report?group_by=model"))
        .bearer_auth(&prod_key)
        .send()
        .await
        .unwrap();
    assert_eq!(self_report.status(), StatusCode::OK);
    let self_logs = http
        .get(format!("{gw_url}/spend/logs"))
        .bearer_auth(&prod_key)
        .send()
        .await
        .unwrap();
    assert_eq!(self_logs.status(), StatusCode::OK);

    // Global per-tag budget (`feature:capped` = 3_000_000 nanos): mint an uncapped key, run 1 request tagged `feature:capped` (spends 4_000_000), second request is blocked!
    let uncapped_gen: Value = http
        .post(format!("{gw_url}/key/generate"))
        .bearer_auth(master_key)
        .json(&json!({
            "key_alias": "uncapped-dev-key",
            "user_id": dev_user_id,
            "allow_spend_read": false
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let uncapped_key = uncapped_gen["key"].as_str().unwrap().to_owned();

    let cap_call1 = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .json(&json!({
            "model": "gpt-4o",
            "max_tokens": 200,
            "metadata": { "tags": ["feature:capped"] },
            "messages": [{ "role": "user", "content": "first capped call" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(cap_call1.status(), StatusCode::OK);

    let cap_call2 = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .json(&json!({
            "model": "gpt-4o",
            "max_tokens": 200,
            "metadata": { "tags": ["feature:capped"] },
            "messages": [{ "role": "user", "content": "second capped call" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(cap_call2.status(), StatusCode::PAYMENT_REQUIRED);
    let cap_err: Value = cap_call2.json().await.unwrap();
    assert!(
        cap_err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("tag 'feature:capped' budget limit"),
        "unexpected cap_err: {:?}",
        cap_err
    );

    // Key without `allow_spend_read` gets 403 on `/spend/report`!
    let denied_spend = http
        .get(format!("{gw_url}/spend/report"))
        .bearer_auth(&uncapped_key)
        .send()
        .await
        .unwrap();
    assert_eq!(denied_spend.status(), StatusCode::FORBIDDEN);

    // =========================================================================
    // 3. Bundle 2.2 — Immutable SQLite Management Audit Logs (with Secret Redaction)
    // =========================================================================
    let audit_resp: Value = http
        .get(format!("{gw_url}/audit/logs"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let audits = audit_resp["audit_logs"].as_array().unwrap();
    assert!(
        audits.len() >= 3,
        "expected >= 3 audit entries, got {}",
        audits.len()
    );
    let serialized_audits = serde_json::to_string(&audits).unwrap();
    assert!(
        !serialized_audits.contains(&prod_key),
        "audit logs must never leak raw virtual key material"
    );
    let first_audit_id = audits[0]["id"].as_str().unwrap();
    let single_audit = http
        .get(format!("{gw_url}/audit/{first_audit_id}"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap();
    assert_eq!(single_audit.status(), StatusCode::OK);

    // =========================================================================
    // 4. Bundle 2.3 — Network, Split-Plane, `/robots.txt`, `max_request_size_mb`, & `enforced_params`
    // =========================================================================
    let robots_txt = http
        .get(format!("{gw_url}/robots.txt"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(robots_txt, "User-agent: *\nDisallow: /\n");

    // `enforced_params`: require `"user"` and `"header:x-session-id"`
    http.patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({
            "enforced_params": ["user", "header:x-session-id"],
            "max_request_size_mb": 1
        }))
        .send()
        .await
        .unwrap();

    let missing_user = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .header("x-session-id", "sess-42")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(missing_user.status(), StatusCode::BAD_REQUEST);

    let missing_hdr = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .json(&json!({
            "model": "gpt-4o",
            "user": "end-user-1",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(missing_hdr.status(), StatusCode::BAD_REQUEST);

    let satisfied_params = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .header("x-session-id", "sess-42")
        .json(&json!({
            "model": "gpt-4o",
            "user": "end-user-1",
            "messages": [{ "role": "user", "content": "params ok" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(satisfied_params.status(), StatusCode::OK);

    // Oversized payload (> 1 MiB) -> 413 Payload Too Large
    // (Under HTTP/1.1 loopback, immediate 413 rejection before draining >1 MiB can either return
    // `Ok(413)` or reset the write side mid-upload; verify status when received and also verify
    // `Content-Length` rejection deterministically over TCP.)
    let big_string = "x".repeat(1_150_000);
    match http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .header("x-session-id", "sess-42")
        .json(&json!({
            "model": "gpt-4o",
            "user": "end-user-1",
            "messages": [{ "role": "user", "content": big_string }]
        }))
        .send()
        .await
    {
        Ok(resp) => assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE),
        Err(err) => assert!(
            err.is_request(),
            "unexpected non-request error on oversized body: {err:?}"
        ),
    }

    // Clear `enforced_params` for subsequent checks
    http.patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({ "enforced_params": [], "max_request_size_mb": 32 }))
        .send()
        .await
        .unwrap();

    // Per-key `allowed_ips` & global `allowed_cidrs` via `ConnectInfo(SocketAddr)`
    let cidr_key_gen: Value = http
        .post(format!("{gw_url}/key/generate"))
        .bearer_auth(master_key)
        .json(&json!({
            "key_alias": "cidr-locked-key",
            "allowed_ips": ["203.0.113.0/24"]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let cidr_key = cidr_key_gen["key"].as_str().unwrap().to_owned();

    let mut bad_ip_req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", format!("Bearer {cidr_key}"))
        .body(Body::empty())
        .unwrap();
    bad_ip_req
        .extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([198, 51, 100, 77], 50000))));
    assert_eq!(
        app.clone().oneshot(bad_ip_req).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );

    let mut good_ip_req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", format!("Bearer {cidr_key}"))
        .body(Body::empty())
        .unwrap();
    good_ip_req
        .extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([203, 0, 113, 19], 50000))));
    assert_eq!(
        app.clone().oneshot(good_ip_req).await.unwrap().status(),
        StatusCode::OK
    );

    // Split-plane: `disable_inference_endpoints: true` blocks `/v1/models` with 403 while `/api/admin/settings` remains accessible
    http.patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({ "disable_inference_endpoints": true }))
        .send()
        .await
        .unwrap();
    let inf_disabled = http
        .get(format!("{gw_url}/v1/models"))
        .bearer_auth(&uncapped_key)
        .send()
        .await
        .unwrap();
    assert_eq!(inf_disabled.status(), StatusCode::FORBIDDEN);
    http.patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({ "disable_inference_endpoints": false }))
        .send()
        .await
        .unwrap();

    // Split-plane: `disable_admin_endpoints: true` blocks `/api/admin/*` and `/key/*` while `/v1/models` keeps serving
    {
        let mut s = db.get_settings().unwrap();
        s.disable_admin_endpoints = true;
        db.update_settings(&s).unwrap();
        let adm_blocked = http
            .get(format!("{gw_url}/api/admin/overview"))
            .bearer_auth(master_key)
            .send()
            .await
            .unwrap();
        assert_eq!(adm_blocked.status(), StatusCode::FORBIDDEN);
        let inf_still_ok = http
            .get(format!("{gw_url}/v1/models"))
            .bearer_auth(&uncapped_key)
            .send()
            .await
            .unwrap();
        assert_eq!(inf_still_ok.status(), StatusCode::OK);
        s.disable_admin_endpoints = false;
        db.update_settings(&s).unwrap();
    }

    // =========================================================================
    // 5. Bundle 2.4 — Local Guardrails: `banned_keywords`, `SecretScannerMode`, `/user/block`, & `x-quixigate-no-log`
    // =========================================================================
    http.patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({
            "banned_keywords": ["project-chimera-classified"],
            "secret_scanner_mode": SecretScannerMode::Block
        }))
        .send()
        .await
        .unwrap();

    let kw_blocked = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "Explain PROJECT-CHIMERA-CLASSIFIED details" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(kw_blocked.status(), StatusCode::BAD_REQUEST);

    let fake_openai_secret = format!("sk-proj-{}", "A1b2C3d4E5f6G7h8I9j0K1l2M3n4");
    let secret_blocked = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": format!("debug token {fake_openai_secret}") }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(secret_blocked.status(), StatusCode::BAD_REQUEST);

    // Switch scanner to `Redact`: upstream & trace receive `[REDACTED:OPENAI_API_KEY]`, never raw key!
    http.patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({
            "banned_keywords": [],
            "secret_scanner_mode": SecretScannerMode::Redact
        }))
        .send()
        .await
        .unwrap();

    let redact_ok: Value = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": format!("here is {fake_openai_secret}") }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        redact_ok["choices"][0]["message"]["content"],
        json!("ack:here is [REDACTED:OPENAI_API_KEY]")
    );
    assert!(
        !captured_prompts
            .lock()
            .iter()
            .any(|p| p.contains(&fake_openai_secret)),
        "upstream must never receive unredacted secret"
    );

    // Privacy header `x-quixigate-no-log: true`: trace count unchanged, yet key spend increments by 4_000_000 nanos!
    let traces_before: Vec<Value> = http
        .get(format!("{gw_url}/api/admin/traces"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let key_info_before: Value = http
        .get(format!("{gw_url}/key/info?key={uncapped_key}"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let spend_before = key_info_before["info"]["spend_nanos"].as_i64().unwrap();

    let nolog_resp = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&uncapped_key)
        .header("x-quixigate-no-log", "true")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "sensitive medical note" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(nolog_resp.status(), StatusCode::OK);

    let traces_after: Vec<Value> = http
        .get(format!("{gw_url}/api/admin/traces"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        traces_before.len(),
        traces_after.len(),
        "x-quixigate-no-log must suppress SQLite trace persistence"
    );
    let key_info_after: Value = http
        .get(format!("{gw_url}/key/info?key={uncapped_key}"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        key_info_after["info"]["spend_nanos"].as_i64().unwrap(),
        spend_before + 4_000_000,
        "x-quixigate-no-log must STILL commit exact NanoUsd spend"
    );

    // =========================================================================
    // 6. Bundle 2.6 & `/user/block` — Team-Scoped Models + Team-Admin RBAC + User Block/Unblock
    // =========================================================================
    let alice_resp: Value = http
        .post(format!("{gw_url}/api/admin/users"))
        .bearer_auth(master_key)
        .json(&json!({ "username": "alice", "display_name": "Alice Team Lead", "role": "member" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let alice_id = alice_resp["id"].as_str().unwrap().to_owned();

    let bob_resp: Value = http
        .post(format!("{gw_url}/api/admin/users"))
        .bearer_auth(master_key)
        .json(&json!({ "username": "bob", "display_name": "Bob Outsider", "role": "member" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let bob_id = bob_resp["id"].as_str().unwrap().to_owned();

    let team_resp: Value = http
        .post(format!("{gw_url}/team/new"))
        .bearer_auth(master_key)
        .json(&json!({
            "team_alias": "ml-platform",
            "members_with_roles": [{ "user_id": alice_id, "role": "admin" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ml_team_id = team_resp["team_id"].as_str().unwrap().to_owned();

    let alice_key_gen: Value = http
        .post(format!("{gw_url}/key/generate"))
        .bearer_auth(master_key)
        .json(&json!({ "key_alias": "alice-personal", "user_id": alice_id }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let alice_key = alice_key_gen["key"].as_str().unwrap().to_owned();

    let bob_key_gen: Value = http
        .post(format!("{gw_url}/key/generate"))
        .bearer_auth(master_key)
        .json(&json!({ "key_alias": "bob-personal", "user_id": bob_id }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let bob_key = bob_key_gen["key"].as_str().unwrap().to_owned();

    // Alice (team admin, NOT global admin) creates a team-scoped model route `ml-secret-gpt` AND mints a team key!
    let team_route_create = http
        .post(format!("{gw_url}/api/admin/routes"))
        .bearer_auth(&alice_key)
        .json(&json!({
            "alias": "ml-secret-gpt",
            "team_id": ml_team_id,
            "targets": [{ "provider_id": "openaichatcompletions", "model": "gpt-4o" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(team_route_create.status(), StatusCode::OK);

    // Alice cannot create a global route (no `team_id`) because she is not global admin
    let global_route_denied = http
        .post(format!("{gw_url}/api/admin/routes"))
        .bearer_auth(&alice_key)
        .json(&json!({
            "alias": "unauthorized-global",
            "targets": [{ "provider_id": "openaichatcompletions", "model": "gpt-4o" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(global_route_denied.status(), StatusCode::FORBIDDEN);

    let ml_team_key_resp = http
        .post(format!("{gw_url}/api/admin/keys"))
        .bearer_auth(&alice_key)
        .json(&json!({
            "name": "ml-team-ci-key",
            "team_id": ml_team_id
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ml_team_key_resp.status(), StatusCode::OK);
    let ml_team_key_json: Value = ml_team_key_resp.json().await.unwrap();
    let ml_team_key = ml_team_key_json["raw_key"].as_str().unwrap().to_owned();

    // Outsider Bob does NOT see `ml-secret-gpt` in `/v1/models` and gets 403 calling it!
    let bob_models: Value = http
        .get(format!("{gw_url}/v1/models"))
        .bearer_auth(&bob_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        !bob_models["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == json!("ml-secret-gpt")),
        "outsider key must not see team-scoped model in /v1/models"
    );
    let bob_call_team_model = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&bob_key)
        .json(&json!({
            "model": "ml-secret-gpt",
            "messages": [{ "role": "user", "content": "sneak in" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(bob_call_team_model.status(), StatusCode::FORBIDDEN);

    // Team key DOES see `ml-secret-gpt` and succeeds calling it!
    let team_call_ok = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(&ml_team_key)
        .json(&json!({
            "model": "ml-secret-gpt",
            "messages": [{ "role": "user", "content": "team authorized" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(team_call_ok.status(), StatusCode::OK);

    // `/user/block` + `/user/unblock` immediately revokes and restores Bob's inference access!
    let block_bob = http
        .post(format!("{gw_url}/user/block"))
        .bearer_auth(master_key)
        .json(&json!({ "user_ids": [bob_id] }))
        .send()
        .await
        .unwrap();
    assert_eq!(block_bob.status(), StatusCode::OK);
    let bob_blocked_call = http
        .get(format!("{gw_url}/v1/models"))
        .bearer_auth(&bob_key)
        .send()
        .await
        .unwrap();
    assert_eq!(bob_blocked_call.status(), StatusCode::FORBIDDEN);

    let unblock_bob = http
        .post(format!("{gw_url}/user/unblock"))
        .bearer_auth(master_key)
        .json(&json!({ "user_ids": [bob_id] }))
        .send()
        .await
        .unwrap();
    assert_eq!(unblock_bob.status(), StatusCode::OK);
    let bob_unblocked_call = http
        .get(format!("{gw_url}/v1/models"))
        .bearer_auth(&bob_key)
        .send()
        .await
        .unwrap();
    assert_eq!(bob_unblocked_call.status(), StatusCode::OK);

    // =========================================================================
    // 7. Bundle 2.5 — Native Hosted Gateway MCP Server (`POST /mcp`)
    // =========================================================================
    let mcp_init_resp = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(&prod_key)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": { "name": "claude-code", "version": "1.0.0" }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(mcp_init_resp.status(), StatusCode::OK);
    assert!(mcp_init_resp.headers().contains_key("mcp-session-id"));
    let mcp_init_json: Value = mcp_init_resp.json().await.unwrap();
    assert_eq!(
        mcp_init_json["result"]["serverInfo"]["name"],
        json!("quixigate")
    );

    // Non-admin key sees 4 developer tools; Master Key sees all 7 tools (including admin tools)
    let dev_tools: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(&prod_key)
        .json(&json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(dev_tools["result"]["tools"].as_array().unwrap().len(), 4);

    let admin_tools: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(master_key)
        .json(&json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/list" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(admin_tools["result"]["tools"].as_array().unwrap().len(), 7);

    // Call `quixigate_spend_report` with self-service `prod_key` and `quixigate_gateway_health` with `master_key`
    let mcp_spend: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(&prod_key)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": { "name": "quixigate_spend_report", "arguments": { "group_by": "tag" } }
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(mcp_spend["result"]["isError"], json!(false));
    assert!(
        !mcp_spend["result"]["structuredContent"]["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let mcp_health: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(master_key)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": { "name": "quixigate_gateway_health", "arguments": {} }
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        mcp_health["result"]["structuredContent"]["status"],
        json!("healthy")
    );

    // =========================================================================
    // 8. Bundle 2.7 — Custom/Filtered OpenAPI & Inbound `HS256` JWT Bearer Auth
    // =========================================================================
    let public_spec: Value = http
        .get(format!("{gw_url}/openapi.json"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        public_spec["info"]["title"],
        json!("Acme Enterprise AI Gateway")
    );
    assert!(public_spec["paths"].get("/v1/chat/completions").is_some());
    assert!(public_spec["paths"].get("/mcp").is_some());
    assert!(
        public_spec["paths"].get("/api/admin/keys").is_none(),
        "filtered /openapi.json must hide /api/admin/* from unauthenticated callers"
    );
    assert!(
        public_spec["paths"].get("/key/generate").is_none(),
        "filtered /openapi.json must hide LiteLLM management paths from unauthenticated callers"
    );

    let admin_spec: Value = http
        .get(format!("{gw_url}/openapi.json"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(admin_spec["paths"].get("/api/admin/keys").is_some());
    assert!(admin_spec["paths"].get("/key/generate").is_some());

    // Valid `HS256` JWT bearer authenticates `alice`!
    let now_sec = now_ms() / 1000;
    let valid_jwt = quixigate_proxy::auth::sign_hs256_jwt(
        &json!({
            "sub": alice_id,
            "iat": now_sec,
            "exp": now_sec + 3600
        }),
        jwt_secret,
    );
    let jwt_ok = http
        .get(format!("{gw_url}/v1/models"))
        .bearer_auth(&valid_jwt)
        .send()
        .await
        .unwrap();
    assert_eq!(jwt_ok.status(), StatusCode::OK);

    // Expired JWT and wrong-secret JWT both fail with 401!
    let expired_jwt = quixigate_proxy::auth::sign_hs256_jwt(
        &json!({
            "sub": alice_id,
            "exp": now_sec - 300
        }),
        jwt_secret,
    );
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&expired_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let bad_sig_jwt = quixigate_proxy::auth::sign_hs256_jwt(
        &json!({
            "sub": alice_id,
            "exp": now_sec + 3600
        }),
        b"wrong-signing-secret",
    );
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&bad_sig_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn e2e_inbound_oidc_jwks_discovery_es256_eddsa_rotation_team_mapping_and_mcp() {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use ring::rand::SystemRandom;
    use ring::signature::{
        ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, Ed25519KeyPair, KeyPair,
    };

    let rng = SystemRandom::new();

    // 1. Generate ES256 (NIST P-256) keypair (`kid = "dex-es256-v1"`) + EdDSA (Ed25519) rotated keypair (`kid = "cf-eddsa-v2"`)
    let es256_pkcs8 = EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &rng).unwrap();
    let es256_kp =
        EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, es256_pkcs8.as_ref(), &rng)
            .unwrap();
    let es256_pub = es256_kp.public_key().as_ref(); // 65 bytes: 0x04 || X(32) || Y(32)
    assert_eq!(es256_pub.len(), 65);
    let es256_jwk = json!({
        "kty": "EC",
        "crv": "P-256",
        "use": "sig",
        "alg": "ES256",
        "kid": "dex-es256-v1",
        "x": URL_SAFE_NO_PAD.encode(&es256_pub[1..33]),
        "y": URL_SAFE_NO_PAD.encode(&es256_pub[33..65]),
    });

    let ed25519_pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
    let ed25519_kp = Ed25519KeyPair::from_pkcs8(ed25519_pkcs8.as_ref()).unwrap();
    let ed25519_jwk = json!({
        "kty": "OKP",
        "crv": "Ed25519",
        "use": "sig",
        "alg": "EdDSA",
        "kid": "cf-eddsa-v2",
        "x": URL_SAFE_NO_PAD.encode(ed25519_kp.public_key().as_ref()),
    });

    let sign_es256 = |kid: &str, claims: &Value| -> String {
        let hdr = URL_SAFE_NO_PAD.encode(
            json!({ "alg": "ES256", "typ": "JWT", "kid": kid })
                .to_string()
                .as_bytes(),
        );
        let pay = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
        let input = format!("{hdr}.{pay}");
        let sig = es256_kp.sign(&rng, input.as_bytes()).unwrap();
        format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig.as_ref()))
    };

    let sign_eddsa = |kid: &str, claims: &Value| -> String {
        let hdr = URL_SAFE_NO_PAD.encode(
            json!({ "alg": "EdDSA", "typ": "JWT", "kid": kid })
                .to_string()
                .as_bytes(),
        );
        let pay = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
        let input = format!("{hdr}.{pay}");
        let sig = ed25519_kp.sign(input.as_bytes());
        format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig.as_ref()))
    };

    // 2. Spin up mock OIDC Identity Provider (`/.well-known/openid-configuration` + `/jwks.json`)
    let jwks_keys: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(vec![es256_jwk]));
    let jwks_hits = Arc::new(AtomicUsize::new(0));
    let idp_listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let idp_addr = idp_listener.local_addr().unwrap();
    let idp_base = format!("http://{idp_addr}");

    let idp_base_disc = idp_base.clone();
    let jwks_keys_route = jwks_keys.clone();
    let jwks_hits_route = jwks_hits.clone();
    let mock_idp = Router::new()
        .route(
            "/.well-known/openid-configuration",
            get(move || {
                let issuer = idp_base_disc.clone();
                async move {
                    Json(json!({
                        "issuer": issuer,
                        "jwks_uri": format!("{issuer}/jwks.json")
                    }))
                }
            }),
        )
        .route(
            "/jwks.json",
            get(move || {
                jwks_hits_route.fetch_add(1, Ordering::SeqCst);
                let keys = jwks_keys_route.lock().clone();
                async move { Json(json!({ "keys": keys })) }
            }),
        );
    tokio::spawn(async move {
        let _ = axum::serve(idp_listener, mock_idp).await;
    });

    // 3. Spin up mock OpenAI upstream + QuixiGate gateway
    let mock_chat = |Json(_body): Json<Value>| async move {
        Json(json!({
            "id": "chatcmpl_oidc_1",
            "object": "chat.completion",
            "created": 1_794_461_000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "oidc-authenticated-ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200 }
        }))
    };
    let upstream_url = spawn_axum(
        Router::new()
            .route("/v1/chat/completions", post(mock_chat))
            .route("/chat/completions", post(mock_chat)),
    )
    .await;

    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    db.upsert_credential(
        None,
        "openaichatcompletions",
        "Upstream Seat",
        AuthKind::ApiKey,
        Some(&upstream_url),
        100,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: "sk-upstream".to_owned(),
        },
    )
    .unwrap();

    let master_key = "sk-qx-master-oidc-e2e";
    let refresher = CredentialRefresher::with_endpoints(db.clone(), RefreshEndpoints::default());
    let router = GatewayRouter::new(
        db.clone(),
        refresher.clone(),
        ModelCatalog::embedded(),
        LocalRuntimeScanner::new(),
    );
    let state = AppState::new(
        db.clone(),
        router,
        TelemetryHub::new(),
        OAuthWizardManager::new(db.clone(), refresher),
        false,
    )
    .with_master_key(master_key);
    let gw_url = spawn_axum(build_router(state)).await;
    let http = reqwest::Client::new();

    // Configure OIDC live via `PATCH /api/admin/settings` (using `oidc_issuer_url` alias)
    let patch_resp = http
        .patch(format!("{gw_url}/api/admin/settings"))
        .bearer_auth(master_key)
        .json(&json!({
            "require_auth": true,
            "oidc_issuer_url": idp_base,
            "jwt_audience": "quixigate-gateway",
            "jwt_user_claim": "email",
            "jwt_roles_claim": "groups",
            "jwt_admin_roles": ["quixigate-admins"],
            "jwt_team_claim": "groups"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(patch_resp.status(), StatusCode::OK);

    // Create team `oidc-ml-team` (with team tag `dept:research`) and private team-scoped route `oidc-team-gpt`
    let team_create: Value = http
        .post(format!("{gw_url}/team/new"))
        .bearer_auth(master_key)
        .json(&json!({
            "team_alias": "oidc-ml-team",
            "tags": ["dept:research"]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let oidc_team_id = team_create["team_id"].as_str().unwrap().to_owned();

    let route_create = http
        .post(format!("{gw_url}/api/admin/routes"))
        .bearer_auth(master_key)
        .json(&json!({
            "alias": "oidc-team-gpt",
            "team_id": oidc_team_id,
            "targets": [{ "provider_id": "openaichatcompletions", "model": "gpt-4o" }],
            "input_nanos_per_m": 2_000_000_000_i64,
            "output_nanos_per_m": 10_000_000_000_i64
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(route_create.status(), StatusCode::OK);

    // 4. Authenticate inference & `/v1/models` with ES256 OIDC JWT — ZERO static keys!
    let now_sec = now_ms() / 1000;
    let researcher_jwt = sign_es256(
        "dex-es256-v1",
        &json!({
            "iss": idp_base,
            "sub": "oidc|99182",
            "email": "researcher@acme.dev",
            "name": "Dr. Researcher",
            "aud": ["quixigate-gateway", "dex-webui"],
            "groups": ["eng-all", "oidc-ml-team"],
            "exp": now_sec + 3600
        }),
    );

    let models_json: Value = http
        .get(format!("{gw_url}/v1/models"))
        .bearer_auth(&researcher_jwt)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        models_json["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == json!("oidc-team-gpt")),
        "OIDC user mapped into oidc-ml-team must see team-scoped model oidc-team-gpt"
    );

    // Run 2 inference calls ($0.004 each = 8_000_000 nanos total) and verify JWKS was cached (1 HTTP fetch)!
    for _ in 0..2 {
        let chat_res = http
            .post(format!("{gw_url}/v1/chat/completions"))
            .bearer_auth(&researcher_jwt)
            .json(&json!({
                "model": "oidc-team-gpt",
                "max_tokens": 200,
                "messages": [{ "role": "user", "content": "hello via OIDC" }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(chat_res.status(), StatusCode::OK);
    }
    assert_eq!(
        jwks_hits.load(Ordering::SeqCst),
        1,
        "JWKS must be served from in-memory cache across repeated requests"
    );

    // Verify exact NanoUsd spend attributed to BOTH the JIT-provisioned user, `oidc-ml-team`, and inherited team tag `dept:research`!
    let team_report: Value = http
        .get(format!("{gw_url}/spend/report?group_by=team"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let team_row = team_report["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["group_key"] == json!(oidc_team_id))
        .expect("team spend row for oidc-ml-team");
    assert_eq!(team_row["spend_nanos"], json!(8_000_000));

    let tag_report: Value = http
        .get(format!("{gw_url}/spend/report?group_by=tag"))
        .bearer_auth(master_key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let dept_tag_row = tag_report["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["group_key"] == json!("dept:research"))
        .expect("inherited team tag dept:research spend row");
    assert_eq!(dept_tag_row["spend_nanos"], json!(8_000_000));

    // 5. Authenticate Hosted MCP (`POST /mcp`) using the same OIDC Bearer JWT!
    let mcp_spend: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(&researcher_jwt)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "quixigate_spend_report", "arguments": { "group_by": "model" } }
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(mcp_spend["result"]["isError"], json!(false));
    assert_eq!(
        mcp_spend["result"]["structuredContent"]["rows"][0]["spend_nanos"],
        json!(8_000_000)
    );

    // 6. Key rotation (`kid` cache miss -> live JWKS re-fetch) + `quixigate-admins` group -> `UserRole::Admin`!
    tokio::time::sleep(Duration::from_millis(550)).await;
    jwks_keys.lock().push(ed25519_jwk);

    let rotated_admin_jwt = sign_eddsa(
        "cf-eddsa-v2",
        &json!({
            "iss": idp_base,
            "email": "sre-lead@acme.dev",
            "aud": "quixigate-gateway",
            "groups": ["quixigate-admins"],
            "exp": now_sec + 3600
        }),
    );
    let admin_overview = http
        .get(format!("{gw_url}/api/admin/overview"))
        .bearer_auth(&rotated_admin_jwt)
        .send()
        .await
        .unwrap();
    assert_eq!(admin_overview.status(), StatusCode::OK);
    assert_eq!(
        jwks_hits.load(Ordering::SeqCst),
        2,
        "unknown kid must trigger a single-flight JWKS refresh"
    );

    // 7. Reject wrong `aud`, wrong `iss`, expired `exp`, tampered payload, and blocked OIDC user!
    let wrong_aud_jwt = sign_es256(
        "dex-es256-v1",
        &json!({
            "iss": idp_base,
            "email": "researcher@acme.dev",
            "aud": "other-api",
            "exp": now_sec + 3600
        }),
    );
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&wrong_aud_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let wrong_iss_jwt = sign_es256(
        "dex-es256-v1",
        &json!({
            "iss": "https://evil-idp.example",
            "email": "researcher@acme.dev",
            "aud": "quixigate-gateway",
            "exp": now_sec + 3600
        }),
    );
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&wrong_iss_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let expired_oidc_jwt = sign_es256(
        "dex-es256-v1",
        &json!({
            "iss": idp_base,
            "email": "researcher@acme.dev",
            "aud": "quixigate-gateway",
            "exp": now_sec - 300
        }),
    );
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&expired_oidc_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    // Tampered payload with original signature -> 401
    let mut segs: Vec<&str> = researcher_jwt.split('.').collect();
    let tampered_claims = URL_SAFE_NO_PAD.encode(
        json!({
            "iss": idp_base,
            "email": "admin@acme.dev",
            "aud": "quixigate-gateway",
            "groups": ["quixigate-admins"],
            "exp": now_sec + 3600
        })
        .to_string()
        .as_bytes(),
    );
    segs[1] = &tampered_claims;
    let tampered_jwt = segs.join(".");
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&tampered_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    // Blocking `researcher@acme.dev` via `/user/block` immediately denies their valid OIDC JWT with 403!
    let researcher_user = db.get_user("researcher@acme.dev").unwrap();
    http.post(format!("{gw_url}/user/block"))
        .bearer_auth(master_key)
        .json(&json!({ "user_ids": [researcher_user.id] }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        http.get(format!("{gw_url}/v1/models"))
            .bearer_auth(&researcher_jwt)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
}

#[tokio::test]
async fn e2e_custom_sk_qx_key_generation_tags_no_log_and_mcp_tools() {
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();
    let mut s = db.get_settings().unwrap();
    s.require_auth = true;
    db.update_settings(&s).unwrap();

    let refresher = CredentialRefresher::new(db.clone());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    let router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner);
    let telemetry = TelemetryHub::new();
    let wizard = OAuthWizardManager::new(db.clone(), refresher);
    let master_key = "sk-qx-master-custom-e2e-999";
    let state = AppState::new(db.clone(), router, telemetry, wizard, true)
        .with_master_key(master_key);
    let gw_url = spawn_axum(build_router(state)).await;
    let http = reqwest::Client::new();

    // 1. Newly minted keys default to `sk-qx-`
    let minted_qx: Value = http
        .post(format!("{gw_url}/key/generate"))
        .header("x-quixigate-admin-key", master_key)
        .json(&json!({ "key_alias": "new-quixigate-key" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let qx_token = minted_qx["key"].as_str().unwrap();
    assert!(qx_token.starts_with("sk-qx-"), "expected sk-qx- prefix, got {qx_token}");

    // 2. Custom `sk-qx-...` token on `POST /key/generate` authenticates cleanly under strict auth
    let custom_token = "sk-qx-custom-team-token-abcdef123456";
    let minted_custom = http
        .post(format!("{gw_url}/key/generate"))
        .header("x-quixigate-admin-key", master_key)
        .json(&json!({
            "key_alias": "custom-quixigate-key",
            "key": custom_token
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(minted_custom.status(), StatusCode::OK);

    // 3. `x-quixigate-tags` & `x-quixigate-no-log` headers work on inference with `sk-qx-...` bearer auth
    let chat_resp = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(custom_token)
        .header("x-quixigate-tags", "env:staging,ci-runner")
        .header("x-quixigate-no-log", "true")
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "ephemeral prompt"}],
            "mock_response": "qx-ok"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), StatusCode::OK);

    // 4. Hosted `/mcp` executes `quixigate_list_models`
    let mcp_res = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(qx_token)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "quixigate_list_models",
                "arguments": { "limit": 3 }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(mcp_res.status(), StatusCode::OK);
    let mcp_body: Value = mcp_res.json().await.unwrap();
    assert_eq!(mcp_body["result"]["isError"], json!(false));
}
