//! End-to-End Verification & 997-Route Exhaustiveness Matrix for 100% LiteLLM Compatibility:
//! 1. Live functional tests for all newly added Data-Plane, Multimodal, Search, Rerank,
//!    Developer Utility, and Control-Plane endpoints.
//! 2. Complete 997-route sweep across every route decorator in LiteLLM Core + Enterprise,
//!    proving zero routes hit a generic `route '...' not found` 404 or `405 Method Not Allowed`,
//!    and every intentionally unsupported route returns structured `400` DIY migration guidance.

use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use quixigate_auth::OAuthWizardManager;
use quixigate_auth::quota::QuotaEndpoints;
use quixigate_auth::refresh::{CredentialRefresher, RefreshEndpoints};
use quixigate_core::{AuthKind, CredentialSecret};
use quixigate_db::Database;
use quixigate_providers::local_discovery::LocalRuntimeScanner;
use quixigate_proxy::build_router;
use quixigate_proxy::state::AppState;
use quixigate_router::GatewayRouter;
use quixigate_router::catalog::ModelCatalog;
use quixigate_telemetry::TelemetryHub;
use quixigate_vault::Vault;
use serde::Deserialize;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;

struct TestHarness {
    _dir: TempDir,
    db: Database,
    base_url: String,
    upstream_url: String,
}

async fn spawn_mock_upstream() -> String {
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                let is_stream = body["stream"].as_bool().unwrap_or(false);
                if is_stream {
                    let sse = concat!(
                        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello stream\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n",
                        "data: [DONE]\n\n"
                    );
                    return (
                        StatusCode::OK,
                        [("content-type", "text/event-stream")],
                        sse.to_owned(),
                    );
                }
                (
                    StatusCode::OK,
                    [("content-type", "application/json")],
                    json!({
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-4o-mini",
                        "choices": [{
                            "index": 0,
                            "message": { "role": "assistant", "content": "Mock reply OK" },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 4,
                            "total_tokens": 16
                        }
                    })
                    .to_string(),
                )
            }),
        )
        .route(
            "/v1/images/generations",
            post(|| async {
                Json(json!({
                    "created": 1700000000,
                    "data": [{ "b64_json": "aGVsbG8=", "revised_prompt": "A crisp diagram" }]
                }))
            }),
        )
        .route(
            "/v1/audio/speech",
            post(|| async {
                (
                    StatusCode::OK,
                    [("content-type", "audio/mpeg")],
                    vec![0xFFu8, 0xFB, 0x90, 0x64],
                )
            }),
        )
        .route(
            "/v1/audio/transcriptions",
            post(|| async { Json(json!({ "text": "transcribed audio text" })) }),
        )
        .route(
            "/search",
            get(|| async {
                Json(json!({
                    "results": [
                        {
                            "title": "Result for quixigate",
                            "url": "https://quixi.ai/docs",
                            "snippet": "QuixiGate single-binary Rust LLM gateway."
                        }
                    ]
                }))
            }),
        )
        .route(
            "/mcp",
            post(|Json(body): Json<Value>| async move {
                let method = body["method"].as_str().unwrap_or("");
                if method == "tools/list" {
                    Json(json!({
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "tools": [{
                                "name": "echo_mcp",
                                "description": "Echoes input text",
                                "inputSchema": { "type": "object", "properties": { "msg": { "type": "string" } } }
                            }]
                        }
                    }))
                } else {
                    Json(json!({
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "content": [{ "type": "text", "text": "mcp-echo-ok" }]
                        }
                    }))
                }
            }),
        )
        .route("/health", get(|| async { "ok" }));

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}

async fn spawn_harness() -> TestHarness {
    let upstream_url = spawn_mock_upstream().await;
    let dir = TempDir::new().unwrap();
    let vault = Vault::load_or_init(dir.path()).unwrap();
    let db = Database::open(&dir.path().join("quixigate.db"), vault).unwrap();

    db.upsert_credential(
        Some("cred_openrouter_mock"),
        "openrouter",
        "Mock OpenRouter",
        AuthKind::ApiKey,
        Some(&format!("{upstream_url}/v1")),
        10,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: "sk-or-mock-upstream".to_owned(),
        },
    )
    .unwrap();

    let refresher = CredentialRefresher::with_endpoints(db.clone(), RefreshEndpoints::default());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    let router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner)
        .with_web_search_url(format!("{upstream_url}/search"));
    let telemetry = TelemetryHub::new();
    let wizard = OAuthWizardManager::new(db.clone(), refresher)
        .with_quota_endpoints(QuotaEndpoints::default());
    let state = AppState::new(db.clone(), router, telemetry, wizard, true);

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

    TestHarness {
        _dir: dir,
        db,
        base_url: format!("http://{addr}"),
        upstream_url,
    }
}

#[tokio::test]
async fn litellm_full_data_and_control_plane_e2e() {
    let h = spawn_harness().await;
    let client = reqwest::Client::new();

    // 1. Model list & slash-namespaced retrieve aliases (`/models`, `/v1/models/{*id}`, `/cursor/v1/models`)
    let models_res: Value = client
        .get(format!("{}/models", h.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models_res["object"], "list");

    let single_model: Value = client
        .get(format!("{}/v1/models/openai/gpt-4o-mini", h.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(single_model["object"], "model");
    assert_eq!(single_model["id"], "openai/gpt-4o-mini");

    let cursor_models = client
        .get(format!("{}/cursor/v1/models", h.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(cursor_models.status(), StatusCode::OK);

    // 2. Legacy OpenAI Text Completions (`POST /v1/completions` JSON + echo + SSE)
    let comp_json: Value = client
        .post(format!("{}/v1/completions", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "prompt": "Prefix: ",
            "echo": true,
            "suffix": " [end]"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(comp_json["object"], "text_completion");
    let comp_text = comp_json["choices"][0]["text"].as_str().unwrap();
    assert!(comp_text.starts_with("Prefix: "));
    assert!(comp_text.ends_with(" [end]"));

    let comp_sse = client
        .post(format!("{}/completions", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "prompt": "Stream this",
            "stream": true
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(comp_sse.contains("\"text_completion\""));
    assert!(comp_sse.contains("data: [DONE]"));

    // 3. Access Groups + Azure OpenAI `api-key` header + `/openai/deployments/{model}/chat/completions`
    let ag_resp = client
        .post(format!("{}/access_group/new", h.base_url))
        .json(&json!({
            "access_group": "ag_mini_only",
            "model_names": ["openai/gpt-4o-mini"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ag_resp.status(), StatusCode::OK);

    let key_gen: Value = client
        .post(format!("{}/key/generate", h.base_url))
        .json(&json!({
            "key_alias": "azure-sdk-key",
            "models": ["access_group:ag_mini_only"],
            "max_budget": 5.0
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let vk = key_gen["key"].as_str().unwrap().to_owned();

    // Allowed via `ag_mini_only` using Azure `api-key` header!
    let azure_ok = client
        .post(format!(
            "{}/openai/deployments/openai%2Fgpt-4o-mini/chat/completions",
            h.base_url
        ))
        .header("api-key", &vk)
        .json(&json!({
            "messages": [{ "role": "user", "content": "Hello Azure" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(azure_ok.status(), StatusCode::OK);

    // Disallowed model outside `ag_mini_only` is rejected with 403!
    let azure_denied = client
        .post(format!("{}/v1/chat/completions", h.base_url))
        .header("api-key", &vk)
        .json(&json!({
            "model": "openai/gpt-4o",
            "messages": [{ "role": "user", "content": "Should fail allowlist" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(azure_denied.status(), StatusCode::FORBIDDEN);

    // 4. Responses helpers (`input_tokens` & `compact`)
    let in_tok: Value = client
        .post(format!("{}/v1/responses/input_tokens", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "input": "Count the tokens in this stateless input."
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(in_tok["input_tokens"].as_u64().unwrap() > 0);

    let compact_res: Value = client
        .post(format!("{}/v1/responses/compact", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "input": [
                { "role": "user", "content": "Turn 1" },
                { "role": "assistant", "content": "Reply 1" }
            ]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(compact_res["object"], "response.compaction");

    // 5. Multimodal Images, Audio TTS & Audio Transcriptions
    let img_res: Value = client
        .post(format!("{}/v1/images/generations", h.base_url))
        .json(&json!({
            "model": "openai/dall-e-3",
            "prompt": "A clean Rust Crab"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(img_res["data"][0]["b64_json"], "aGVsbG8=");

    let tts_resp = client
        .post(format!("{}/v1/audio/speech", h.base_url))
        .json(&json!({
            "model": "openai/tts-1",
            "input": "Hello world",
            "voice": "alloy"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(tts_resp.status(), StatusCode::OK);
    assert_eq!(tts_resp.bytes().await.unwrap().len(), 4);

    let boundary = "quixigate-test-boundary";
    let multipart_body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nopenai/whisper-1\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n--{boundary}--\r\n"
    );
    let stt_res: Value = client
        .post(format!("{}/v1/audio/transcriptions", h.base_url))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(multipart_body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stt_res["text"], "transcribed audio text");

    // 6. Moderations (local secret detection), Rerank & Search
    let mod_res: Value = client
        .post(format!("{}/v1/moderations", h.base_url))
        .json(&json!({
            "input": "Here is my key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(mod_res["results"][0]["flagged"], true);

    let rerank_res: Value = client
        .post(format!("{}/v1/rerank", h.base_url))
        .json(&json!({
            "query": "rust sqlite gateway",
            "documents": [
                "Python proxy with external Postgres and Redis",
                "Single-binary Rust gateway backed by SQLite WAL"
            ],
            "top_n": 1
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(rerank_res["results"][0]["index"], 1);

    let search_res: Value = client
        .post(format!("{}/v1/search", h.base_url))
        .json(&json!({ "query": "quixigate" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!search_res["results"].as_array().unwrap().is_empty());

    // 7. Developer Utilities (`/utils/*`, `/cost/estimate`, `/apply_guardrail`)
    let tok_cnt: Value = client
        .post(format!("{}/utils/token_counter", h.base_url))
        .json(&json!({ "model": "openai/gpt-4o-mini", "text": "Hello token counter" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(tok_cnt["total_tokens"].as_u64().unwrap() > 0);

    let cost_est: Value = client
        .post(format!("{}/cost/estimate", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "input_tokens": 1000,
            "output_tokens": 500
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(cost_est["total_cost_nanos"].as_i64().unwrap() > 0);

    let gr_res: Value = client
        .post(format!("{}/apply_guardrail", h.base_url))
        .json(&json!({
            "text": "leaked sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 secret"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(gr_res["triggered"], true);
    assert!(
        gr_res["response_text"]
            .as_str()
            .unwrap()
            .contains("[REDACTED")
    );

    // 8. Customers / End-Users + Budgets + Tags + Fallbacks + MCP Registry
    let bdg: Value = client
        .post(format!("{}/budget/new", h.base_url))
        .json(&json!({ "budget_id": "zero_tier", "max_budget": 0.0 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(bdg["budget_id"], "zero_tier");

    let cust: Value = client
        .post(format!("{}/customer/new", h.base_url))
        .json(&json!({ "user_id": "end_cust_1", "blocked": true }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cust["blocked"], true);

    // Blocked end-user rejected on inference!
    let blocked_eu_resp = client
        .post(format!("{}/v1/chat/completions", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "user": "end_cust_1",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(blocked_eu_resp.status(), StatusCode::FORBIDDEN);

    // Unblock end-user -> succeeds!
    client
        .post(format!("{}/end_user/unblock", h.base_url))
        .json(&json!({ "user_id": "end_cust_1" }))
        .send()
        .await
        .unwrap();
    let unblocked_eu_resp = client
        .post(format!("{}/v1/chat/completions", h.base_url))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "user": "end_cust_1",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(unblocked_eu_resp.status(), StatusCode::OK);

    // Fallbacks CRUD
    let fb_created: Value = client
        .post(format!("{}/fallback", h.base_url))
        .json(&json!({
            "model": "smart-chat",
            "fallback_models": ["openai/gpt-4o-mini", "openai/gpt-4o"]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(fb_created["fallback_models"].as_array().unwrap().len(), 2);

    let fb_get: Value = client
        .get(format!("{}/fallback/smart-chat", h.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(fb_get["fallback_models"].as_array().unwrap().len(), 2);

    // MCP Registry & REST Bridge
    let mcp_reg: Value = client
        .post(format!("{}/v1/mcp/server", h.base_url))
        .json(&json!({ "url": format!("{}/mcp", h.upstream_url), "server_name": "mock-mcp" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(mcp_reg["server_id"].as_str().unwrap().starts_with("mcp_"));

    let mcp_tools: Value = client
        .get(format!("{}/v1/mcp/tools", h.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        mcp_tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "echo_mcp")
    );

    let mcp_call: Value = client
        .post(format!("{}/mcp-rest/tools/call", h.base_url))
        .json(&json!({ "name": "echo_mcp", "arguments": { "msg": "hi" } }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(mcp_call["content"][0]["text"], "mcp-echo-ok");

    // Health OPTIONS preflight & truly unknown non-LiteLLM 404 preservation
    let opt_resp = client
        .request(
            reqwest::Method::OPTIONS,
            format!("{}/health/liveliness", h.base_url),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(opt_resp.status(), StatusCode::NO_CONTENT);

    let true_404 = client
        .get(format!(
            "{}/definitely-not-a-litellm-endpoint-xyz",
            h.base_url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(true_404.status(), StatusCode::NOT_FOUND);
    let true_404_body: Value = true_404.json().await.unwrap();
    assert_eq!(true_404_body["error"]["code"], "not_found");

    // Sanity check DB recorded end-user spend > 0
    let eu_record = h.db.get_end_user("end_cust_1").unwrap();
    assert!(eu_record.spend_nanos.nanos() > 0);
}

#[derive(Debug, Deserialize)]
struct RouteFixture {
    index: usize,
    file: String,
    line: usize,
    methods: Vec<String>,
    template_path: String,
    sample_path: String,
}

#[tokio::test]
async fn all_997_litellm_route_decorators_are_supported_or_return_actionable_400_guidance() {
    let h = spawn_harness().await;
    let client = reqwest::Client::new();

    let raw_fixture = include_str!("fixtures/litellm_997_routes.json");
    let fixtures: Vec<RouteFixture> = serde_json::from_str(raw_fixture).unwrap();
    assert_eq!(
        fixtures.len(),
        997,
        "expected all 997 LiteLLM Core + Enterprise route decorators"
    );

    let mut supported_hits = 0usize;
    let mut guided_400_hits = 0usize;
    let mut failures = Vec::new();

    for route in &fixtures {
        for m_str in &route.methods {
            let method = reqwest::Method::from_bytes(m_str.as_bytes()).unwrap();
            let url = format!("{}{}", h.base_url, route.sample_path);
            let mut req = client.request(method.clone(), &url);
            if matches!(
                method,
                reqwest::Method::POST | reqwest::Method::PUT | reqwest::Method::PATCH
            ) {
                req = req.header("content-type", "application/json").body("{}");
            }
            let resp = req.send().await.unwrap();
            let status = resp.status();
            let err_type_hdr = resp
                .headers()
                .get("x-quixigate-error-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            let err_code_hdr = resp
                .headers()
                .get("x-quixigate-error-code")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            let text = resp.text().await.unwrap_or_default();

            if status == StatusCode::NOT_IMPLEMENTED {
                failures.push(format!(
                    "#{} {} {} ({}): returned 501 instead of 400 Bad Request!",
                    route.index, m_str, route.sample_path, route.template_path
                ));
                continue;
            }

            if status == StatusCode::METHOD_NOT_ALLOWED {
                failures.push(format!(
                    "#{} {} {} ({} -> {}:{}): returned 405 Method Not Allowed: {}",
                    route.index,
                    m_str,
                    route.sample_path,
                    route.template_path,
                    route.file,
                    route.line,
                    text
                ));
                continue;
            }

            if status == StatusCode::NOT_FOUND {
                let parsed: Value = serde_json::from_str(&text).unwrap_or_default();
                let msg = parsed["error"]["message"].as_str().unwrap_or(&text);
                if parsed["error"]["code"] == "not_found" && msg.starts_with("route '") {
                    failures.push(format!(
                        "#{} {} {} ({} -> {}:{}): UNHANDLED 404 route not found!",
                        route.index,
                        m_str,
                        route.sample_path,
                        route.template_path,
                        route.file,
                        route.line
                    ));
                    continue;
                }
            }

            if err_type_hdr == "unsupported_litellm_endpoint" {
                if status != StatusCode::BAD_REQUEST {
                    failures.push(format!(
                        "#{} {} {} ({}): expected 400 Bad Request for unsupported guidance, got {status}",
                        route.index, m_str, route.sample_path, route.template_path
                    ));
                    continue;
                }
                let valid = if method == reqwest::Method::HEAD {
                    !err_code_hdr.is_empty()
                } else {
                    let parsed: Value = serde_json::from_str(&text).unwrap_or_default();
                    let err = &parsed["error"];
                    err["type"] == "unsupported_litellm_endpoint"
                        && err["code"].as_str().is_some_and(|s| !s.is_empty())
                        && err["reason"].as_str().is_some_and(|s| !s.is_empty())
                        && err["quixigate_alternative"]
                            .as_str()
                            .is_some_and(|s| !s.is_empty())
                        && err["how_to_proceed"]
                            .as_array()
                            .is_some_and(|steps| steps.len() >= 2)
                };
                if !valid {
                    failures.push(format!(
                        "#{} {} {} ({}): malformed 400 guidance payload: {}",
                        route.index, m_str, route.sample_path, route.template_path, text
                    ));
                } else {
                    guided_400_hits += 1;
                }
            } else {
                supported_hits += 1;
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Found {} unhandled LiteLLM routes out of 997:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        supported_hits > 200,
        "expected >200 live supported method/route hits, got {supported_hits}"
    );
    assert!(
        guided_400_hits > 500,
        "expected >500 guided 400 method/route hits, got {guided_400_hits}"
    );
}
