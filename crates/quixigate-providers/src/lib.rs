//! Unified `aisdk` provider layer: all built-in `aisdk` providers, custom
//! OAuth `LanguageModel` implementations (`ChatGptCodexModel`, `AnthropicOAuthModel`,
//! `GoogleCodeAssistModel`, `GitHubCopilotModel`), embeddings, and local runtime auto-discovery.

#[cfg(feature = "subscription-oauth")]
pub mod anthropic_oauth;
pub mod builtin;
#[cfg(feature = "subscription-oauth")]
pub mod codex;
pub mod copilot;
#[cfg(feature = "subscription-oauth")]
pub mod google_code_assist;
pub mod local_discovery;
pub mod sse;

#[cfg(feature = "subscription-oauth")]
pub use anthropic_oauth::AnthropicOAuthModel;
pub use builtin::{
    AISDK_BUILTIN_PROVIDERS, AnyLanguageModel, EmbeddingOutput, ModelBuildRequest,
    build_aisdk_builtin_provider, embed_texts, instantiate_model,
    openai_compatible_default_base_url,
};
#[cfg(feature = "subscription-oauth")]
pub use codex::ChatGptCodexModel;
pub use copilot::{GitHubCopilotModel, UpstreamResponseMeta, parse_exact_cost_nanos};
#[cfg(feature = "subscription-oauth")]
pub use google_code_assist::GoogleCodeAssistModel;
pub use local_discovery::{LocalEndpointSpec, LocalRuntimeScanner, normalize_local_runtime_id};
pub use sse::{FrameTooLarge, SseDecoder, SseFrame};

/// Returns a process-wide shared `rustls::ClientConfig` (`Arc<RootCertStore>` + `ring` crypto +
/// HTTP/2 & HTTP/1.1 ALPN) so all `reqwest::Client` instances across worker shards and background
/// services share a single ~2.5 MiB TLS root store instead of duplicating it per client.
pub fn shared_rustls_client_config() -> rustls::ClientConfig {
    static TLS_CFG: std::sync::OnceLock<rustls::ClientConfig> = std::sync::OnceLock::new();
    TLS_CFG
        .get_or_init(|| {
            let mut roots = rustls::RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let provider = std::sync::Arc::new(rustls::crypto::ring::default_provider());
            let mut cfg = rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .expect("valid TLS protocol versions")
                .with_root_certificates(roots)
                .with_no_client_auth();
            cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
            cfg
        })
        .clone()
}

/// Returns a `reqwest::ClientBuilder` preconfigured with the shared `rustls::ClientConfig`.
#[inline]
pub fn shared_tls_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder().use_preconfigured_tls(shared_rustls_client_config())
}

thread_local! {
    static WORKER_CORE_INDEX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Marks the calling OS thread (`quixigate-worker-{idx}`) with its deterministic core index so
/// [`ShardedHttpPool::get`] binds a dedicated 1:1 lazy `reqwest::Client` to this worker's reactor.
pub fn set_current_worker_core_index(idx: usize) {
    WORKER_CORE_INDEX.with(|c| c.set(Some(idx)));
}

/// Builds a `reqwest::Client` tuned for ultra-low-latency streaming (`TCP_NODELAY`, high idle pool
/// capacity, HTTP/2 adaptive window) as well as long-running local (`ollama`, `llama.cpp`, `lmstudio`,
/// `vllm`) and cloud reasoning inference:
/// - Fast 10s TCP connect timeout so dead ports fail over quickly
/// - `TCP_NODELAY = true` so small SSE token frames flush immediately without Nagle's 40ms delay
/// - 30s TCP keepalive + 30s idle pool expiry (`pool_max_idle_per_host = 2048`)
/// - Per-read timeout (`QUIXIGATE_UPSTREAM_READ_TIMEOUT_SECS`, default 1800s = 30 min) rather than a
///   short total wall-clock `.timeout(...)`, so neither 24B/70B non-streaming thoughts nor multi-minute
///   SSE streams get killed prematurely at 30 seconds.
pub fn build_inference_http_client() -> reqwest::Client {
    let read_secs = quixigate_core::brand::env_opt("QUIXIGATE_UPSTREAM_READ_TIMEOUT_SECS")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(1_800);
    shared_tls_client_builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(2_048)
        .http2_adaptive_window(true)
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(read_secs))
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default()
}

/// Returns a process-wide warm `reqwest::Client` so model constructors never re-parse system TLS
/// root stores when instantiated without an explicit worker-local client.
#[inline]
pub fn shared_inference_http_client() -> reqwest::Client {
    static SHARED_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    SHARED_CLIENT
        .get_or_init(build_inference_http_client)
        .clone()
}

/// Lazy sharded pool of `reqwest::Client` instances indexed by worker core (or caller `ThreadId`),
/// eliminating cross-core Hyper `PoolInner` mutex contention under multi-threaded concurrency while
/// allocating zero HTTP pools or TLS state at startup until traffic actually arrives on a shard.
#[derive(Clone)]
pub struct ShardedHttpPool {
    shards: std::sync::Arc<Vec<std::sync::OnceLock<reqwest::Client>>>,
}

impl ShardedHttpPool {
    pub fn new(shard_count: usize) -> Self {
        let count = shard_count.clamp(1, 128);
        let shards = (0..count).map(|_| std::sync::OnceLock::new()).collect();
        Self {
            shards: std::sync::Arc::new(shards),
        }
    }

    pub fn from_single(client: reqwest::Client) -> Self {
        let cell = std::sync::OnceLock::new();
        let _ = cell.set(client);
        Self {
            shards: std::sync::Arc::new(vec![cell]),
        }
    }

    #[inline]
    pub fn get(&self) -> &reqwest::Client {
        if self.shards.len() == 1 {
            return self.shards[0].get_or_init(build_inference_http_client);
        }
        let idx = match WORKER_CORE_INDEX.with(|c| c.get()) {
            Some(worker_idx) => worker_idx % self.shards.len(),
            None => {
                thread_local! {
                    static THREAD_SHARD_HASH: usize = {
                        use std::hash::{Hash, Hasher};
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        std::thread::current().id().hash(&mut h);
                        h.finish() as usize
                    };
                }
                THREAD_SHARD_HASH.with(|&h| h % self.shards.len())
            }
        };
        self.shards[idx].get_or_init(build_inference_http_client)
    }
}

/// Walks the full `std::error::Error::source()` chain of a `reqwest::Error` so root causes like
/// `operation timed out`, `Connection refused`, or `connection closed before message completed`
/// are never hidden behind `error sending request for url (...)`, and maps timeouts to HTTP `504`.
pub fn describe_reqwest_error(err: &reqwest::Error) -> (Option<reqwest::StatusCode>, String) {
    use std::error::Error as _;

    let status = err.status().or_else(|| {
        if err.is_timeout() {
            Some(reqwest::StatusCode::GATEWAY_TIMEOUT)
        } else if err.is_connect() {
            Some(reqwest::StatusCode::BAD_GATEWAY)
        } else {
            None
        }
    });

    let mut msg = err.to_string();
    let mut cur = err.source();
    while let Some(cause) = cur {
        let piece = cause.to_string();
        if !piece.is_empty() && !msg.contains(&piece) {
            msg.push_str(": ");
            msg.push_str(&piece);
        }
        cur = cause.source();
    }
    (status, msg)
}

/// Returns true when a pre-header `reqwest::Error` came from an idle pooled HTTP/1.1 keep-alive
/// socket closing right as the request was written (safe for a single immediate retry).
pub fn is_retryable_stale_socket_error(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.status().is_some() {
        return false;
    }
    let (_, full) = describe_reqwest_error(err);
    let lower = full.to_ascii_lowercase();
    lower.contains("connection closed before message completed")
        || lower.contains("broken pipe")
        || lower.contains("connection reset by peer")
}

#[cfg(all(test, feature = "subscription-oauth"))]
mod tests {
    use super::*;
    use aisdk::core::LanguageModelRequest;
    use axum::Router;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use quixigate_core::CredentialSecret;
    use serde_json::{Value, json};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn codex_anthropic_code_assist_copilot_and_ollama_work_through_aisdk_request() {
        let app = Router::new()
            .route(
                "/responses",
                post(|headers: HeaderMap, axum::Json(body): axum::Json<Value>| async move {
                    assert_eq!(
                        headers.get("ChatGPT-Account-ID").and_then(|v| v.to_str().ok()),
                        Some("acct_123")
                    );
                    assert_eq!(body["store"], json!(false));
                    assert_eq!(body["stream"], json!(true));
                    let sse = concat!(
                        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\"}}\n\n",
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Codex \"}\n\n",
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}\n\n",
                        "data: {\"type\":\"response.output_text.done\"}\n\n",
                        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Codex OK\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n",
                    );
                    (StatusCode::OK, [("content-type", "text/event-stream")], sse).into_response()
                }),
            )
            .route(
                "/v1/messages",
                post(|headers: HeaderMap| async move {
                    assert_eq!(
                        headers.get("anthropic-beta").and_then(|v| v.to_str().ok()),
                        Some("oauth-2025-04-20")
                    );
                    axum::Json(json!({
                        "id": "msg_1",
                        "content": [{ "type": "text", "text": "Claude OAuth OK" }],
                        "usage": { "input_tokens": 12, "output_tokens": 4, "cache_read_input_tokens": 3 }
                    }))
                }),
            )
            .route(
                "/v1internal/gemini-2.5-pro:generateContent",
                post(|headers: HeaderMap| async move {
                    assert_eq!(
                        headers.get("X-Goog-User-Project").and_then(|v| v.to_str().ok()),
                        Some("proj_abc")
                    );
                    axum::Json(json!({
                        "response": {
                            "candidates": [{ "content": { "parts": [{ "text": "Gemini CLI OK" }] } }],
                            "usageMetadata": { "promptTokenCount": 8, "candidatesTokenCount": 3 }
                        }
                    }))
                }),
            )
            .route(
                "/chat/completions",
                post(|headers: HeaderMap| async move {
                    let ua = headers
                        .get("User-Agent")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or_default()
                        .to_owned();
                    let reply = if ua.starts_with("GitHubCopilotChat/") {
                        "Copilot OK"
                    } else {
                        "Local OK"
                    };
                    axum::Json(json!({
                        "choices": [{ "message": { "role": "assistant", "content": reply } }],
                        "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
                    }))
                }),
            )
            .route(
                "/api/tags",
                get(|| async {
                    axum::Json(json!({ "models": [{ "name": "qwen3:32b" }] }))
                }),
            )
            .route(
                "/api/ps",
                get(|| async {
                    axum::Json(json!({ "models": [{ "name": "qwen3:32b" }] }))
                }),
            );

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}");

        // 1. Codex LanguageModel via AnyLanguageModel
        let codex = instantiate_model(ModelBuildRequest {
            provider_id: "openai-codex".to_owned(),
            model_name: "gpt-5.4".to_owned(),
            secret: Some(CredentialSecret::Codex {
                access: "tok".to_owned(),
                refresh: "ref".to_owned(),
                expires_ms: 9999999999000,
                account_id: "acct_123".to_owned(),
            }),
            base_url: Some(base.clone()),
            headers: None,
            body: None,
            tools: Vec::new(),
            http: None,
        })
        .unwrap();
        let resp = LanguageModelRequest::builder()
            .model(codex)
            .prompt("hi")
            .build()
            .generate_text()
            .await
            .unwrap();
        assert_eq!(resp.text().as_deref(), Some("Codex OK"));

        // 2. Anthropic Setup-Token LanguageModel
        let claude = instantiate_model(ModelBuildRequest {
            provider_id: "anthropic-token".to_owned(),
            model_name: "claude-sonnet-4-5".to_owned(),
            secret: Some(CredentialSecret::AnthropicSetupToken {
                token: "sk-ant-oat01-test".to_owned(),
            }),
            base_url: Some(base.clone()),
            headers: None,
            body: None,
            tools: Vec::new(),
            http: None,
        })
        .unwrap();
        let resp = LanguageModelRequest::builder()
            .model(claude)
            .prompt("hi")
            .build()
            .generate_text()
            .await
            .unwrap();
        assert_eq!(resp.text().as_deref(), Some("Claude OAuth OK"));

        // 3. Google Code Assist LanguageModel
        let gemini = instantiate_model(ModelBuildRequest {
            provider_id: "google-gemini-cli".to_owned(),
            model_name: "gemini-2.5-pro".to_owned(),
            secret: Some(CredentialSecret::GeminiCli {
                access: "ya29.test".to_owned(),
                refresh: "1//ref".to_owned(),
                expires_ms: 9999999999000,
                project_id: "proj_abc".to_owned(),
                email: None,
                client_id: "cid".to_owned(),
                client_secret: "csec".to_owned(),
            }),
            base_url: Some(base.clone()),
            headers: None,
            body: None,
            tools: Vec::new(),
            http: None,
        })
        .unwrap();
        let resp = LanguageModelRequest::builder()
            .model(gemini)
            .prompt("hi")
            .build()
            .generate_text()
            .await
            .unwrap();
        assert_eq!(resp.text().as_deref(), Some("Gemini CLI OK"));

        // 4. GitHub Copilot LanguageModel
        let copilot = instantiate_model(ModelBuildRequest {
            provider_id: "github-copilot".to_owned(),
            model_name: "gpt-4.1".to_owned(),
            secret: Some(CredentialSecret::GitHubCopilot {
                github_token: "ghu_x".to_owned(),
                access: "tid_tok".to_owned(),
                expires_ms: 9999999999000,
                base_url: base.clone(),
                enterprise_domain: None,
            }),
            base_url: None,
            headers: None,
            body: None,
            tools: Vec::new(),
            http: None,
        })
        .unwrap();
        let resp = LanguageModelRequest::builder()
            .model(copilot)
            .prompt("hi")
            .build()
            .generate_text()
            .await
            .unwrap();
        assert_eq!(resp.text().as_deref(), Some("Copilot OK"));

        // 5. Local Ollama discovery + VRAM residency
        let scanner = LocalRuntimeScanner::new();
        scanner.add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: "ollama".to_owned(),
            display_name: "Ollama".to_owned(),
            base_url: format!("{base}/v1"),
        });
        let statuses = scanner.scan_all().await;
        let ollama = statuses.iter().find(|s| s.runtime_id == "ollama").unwrap();
        assert!(ollama.healthy);
        assert_eq!(ollama.models, vec!["qwen3:32b".to_owned()]);
        assert_eq!(ollama.loaded_models, vec!["qwen3:32b".to_owned()]);

        // 6. Verify all 73 built-in aisdk providers are registered
        assert_eq!(AISDK_BUILTIN_PROVIDERS.len(), 73);
    }

    #[tokio::test]
    async fn llamacpp_instance_router_auth_and_multiturn_reasoning_coalescing() {
        use aisdk::core::language_model::{LanguageModelOptions, LanguageModelResponseContentType};
        use aisdk::core::tools::ToolDetails;
        use aisdk::core::{
            AssistantMessage, LanguageModelRequest, Message, ToolCallInfo, ToolResultInfo,
            UserMessage,
        };
        use parking_lot::Mutex;
        use std::sync::Arc;

        // 1. Verify multi-turn assistant reasoning + tool_calls coalesce into one OpenAI message
        // matching llama.cpp `server-chat.cpp` (`msg["reasoning_content"]` + `msg["tool_calls"]`).
        let msgs = vec![
            Message::User(UserMessage::new("Check disk usage")),
            Message::Assistant(AssistantMessage::new(
                LanguageModelResponseContentType::Reasoning {
                    content: "I should run df -h first.".to_owned(),
                    extensions: Default::default(),
                },
                None,
            )),
            Message::Assistant(AssistantMessage::new(
                LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                    tool: ToolDetails {
                        id: "call_1".to_owned(),
                        name: "run_df".to_owned(),
                    },
                    input: json!({ "human": true }),
                    extensions: Default::default(),
                }),
                None,
            )),
            Message::Tool(ToolResultInfo {
                tool: ToolDetails {
                    id: "call_1".to_owned(),
                    name: "run_df".to_owned(),
                },
                output: Ok(json!("42% used")),
            }),
        ];
        let dummy_model = instantiate_model(ModelBuildRequest {
            provider_id: "llamacpp".to_owned(),
            model_name: "qwen3".to_owned(),
            secret: None,
            base_url: Some("http://127.0.0.1:8080/v1".to_owned()),
            headers: None,
            body: None,
            tools: Vec::new(),
            http: None,
        })
        .unwrap();
        let built = LanguageModelRequest::builder()
            .model(dummy_model)
            .messages(msgs)
            .top_k(40_u32)
            .build();
        let opts: LanguageModelOptions = (*built).clone();
        let body = crate::copilot::build_chat_completions_body("qwen3", &opts, &[], false);
        assert_eq!(body["top_k"], 40);
        let encoded_msgs = body["messages"].as_array().unwrap();
        assert_eq!(encoded_msgs.len(), 3);
        assert_eq!(encoded_msgs[1]["role"], "assistant");
        assert_eq!(
            encoded_msgs[1]["reasoning_content"],
            "I should run df -h first."
        );
        assert!(encoded_msgs[1]["content"].is_null());
        assert_eq!(
            encoded_msgs[1]["tool_calls"][0]["function"]["name"],
            "run_df"
        );

        // 2. Mock a `llama-server --api-key secret-llama` in Router Mode (`--models-dir`)
        let second_loaded = Arc::new(Mutex::new(false));
        let second_loaded_get = second_loaded.clone();
        let second_loaded_load = second_loaded.clone();
        let second_loaded_unload = second_loaded.clone();

        let router_app = Router::new()
            .route(
                "/health",
                get(|| async { axum::Json(json!({ "status": "ok" })) }),
            )
            .route(
                "/props",
                get(|headers: HeaderMap| async move {
                    if headers.get("authorization").and_then(|v| v.to_str().ok())
                        != Some("Bearer secret-llama")
                    {
                        return (
                            StatusCode::UNAUTHORIZED,
                            axum::Json(json!({ "error": { "message": "Invalid API Key" } })),
                        )
                            .into_response();
                    }
                    axum::Json(json!({
                        "role": "router",
                        "max_instances": 3,
                        "models_autoload": true,
                        "build_info": "b6210-arm64-apple-metal"
                    }))
                    .into_response()
                }),
            )
            .route(
                "/v1/models",
                get(move |headers: HeaderMap| {
                    let is_loaded = *second_loaded_get.lock();
                    async move {
                        if headers.get("authorization").and_then(|v| v.to_str().ok())
                            != Some("Bearer secret-llama")
                        {
                            return (
                                StatusCode::UNAUTHORIZED,
                                axum::Json(json!({ "error": { "message": "Invalid API Key" } })),
                            )
                                .into_response();
                        }
                        axum::Json(json!({
                            "object": "list",
                            "data": [
                                {
                                    "id": "ggml-org/gemma-3-27b-it-GGUF:Q4_K_M",
                                    "aliases": ["gemma3-local"],
                                    "owned_by": "llamacpp",
                                    "status": { "value": "loaded", "args": [] },
                                    "context_length": 131072,
                                    "meta": { "n_ctx": 65536 },
                                    "architecture": { "input_modalities": ["text", "image"] }
                                },
                                {
                                    "id": "Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
                                    "aliases": ["qwen3-coder-local"],
                                    "owned_by": "llamacpp",
                                    "status": {
                                        "value": if is_loaded { "loaded" } else { "unloaded" },
                                        "args": []
                                    },
                                    "context_length": 32768,
                                    "architecture": { "input_modalities": ["text"] }
                                },
                                {
                                    "id": "unfinished-partial.gguf",
                                    "aliases": [],
                                    "owned_by": "llamacpp",
                                    "status": { "value": "incomplete", "args": [] }
                                }
                            ]
                        }))
                        .into_response()
                    }
                }),
            )
            .route(
                "/models/load",
                post(move |axum::Json(body): axum::Json<Value>| {
                    assert_eq!(body["model"], "Qwen3-Coder-30B-A3B-Q4_K_M.gguf");
                    *second_loaded_load.lock() = true;
                    async { axum::Json(json!({ "success": true })) }
                }),
            )
            .route(
                "/models/unload",
                post(move |axum::Json(body): axum::Json<Value>| {
                    assert_eq!(body["model"], "Qwen3-Coder-30B-A3B-Q4_K_M.gguf");
                    *second_loaded_unload.lock() = false;
                    async { axum::Json(json!({ "success": true })) }
                }),
            );

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router_app).await.unwrap();
        });

        let scanner = LocalRuntimeScanner::new();
        scanner.add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: "llama.cpp".to_owned(), // verifies alias normalization -> "llamacpp"
            display_name: "llama.cpp Server".to_owned(),
            base_url: format!("http://{addr}/v1"),
        });

        // Without key: detects public `/health` 200 + `/v1/models` 401 -> `auth_required = true`
        scanner.set_api_key("llamacpp", None);
        let unauth_scan = scanner.scan_all().await;
        let st_unauth = unauth_scan
            .iter()
            .find(|s| s.runtime_id == "llamacpp")
            .unwrap();
        assert!(!st_unauth.healthy);
        assert!(st_unauth.auth_required);

        // With key: discovers router mode, slots, vision, aliases, and skips `incomplete`
        scanner.set_api_key("llamacpp", Some("secret-llama".to_owned()));
        let auth_scan = scanner.scan_all().await;
        let st = auth_scan
            .iter()
            .find(|s| s.runtime_id == "llamacpp")
            .unwrap();
        assert!(st.healthy);
        assert!(!st.auth_required);
        assert_eq!(st.mode.as_deref(), Some("router"));
        assert_eq!(st.total_slots, Some(3));
        assert!(st.supports_load_control);
        assert_eq!(
            st.server_version.as_deref(),
            Some("b6210-arm64-apple-metal")
        );
        assert_eq!(
            st.models,
            vec![
                "ggml-org/gemma-3-27b-it-GGUF:Q4_K_M".to_owned(),
                "Qwen3-Coder-30B-A3B-Q4_K_M.gguf".to_owned(),
            ]
        );
        assert_eq!(
            st.loaded_models,
            vec!["ggml-org/gemma-3-27b-it-GGUF:Q4_K_M".to_owned()]
        );
        let gemma_info = &st.model_details["ggml-org/gemma-3-27b-it-GGUF:Q4_K_M"];
        assert_eq!(gemma_info.context_window, Some(65536));
        assert!(gemma_info.supports_vision);
        assert_eq!(gemma_info.aliases, vec!["gemma3-local".to_owned()]);

        // Dynamic load & unload via `/models/load` and `/models/unload`
        let after_load = scanner
            .load_model("llamacpp", "Qwen3-Coder-30B-A3B-Q4_K_M.gguf")
            .await
            .unwrap();
        assert_eq!(after_load.loaded_models.len(), 2);

        let after_unload = scanner
            .unload_model("llamacpp", "Qwen3-Coder-30B-A3B-Q4_K_M.gguf")
            .await
            .unwrap();
        assert_eq!(after_unload.loaded_models.len(), 1);
    }

    #[tokio::test]
    async fn reconciles_ollama_reasoning_tokens_inline_think_and_timeout_cause() {
        use crate::copilot::{reconcile_reasoning_usage, split_inline_think_block};
        use aisdk::core::language_model::Usage;

        // 1. Exact reproduction of Ollama `lfm2.5:24b` omitting `completion_tokens_details.reasoning_tokens`
        let reasoning = "The user says \"Hello!\" This is a simple greeting. The assistant should respond politely, perhaps ask how they can help. There's no policy violation. So just greet back and ask what they need.\n\nWe must follow the style guidelines: be friendly, helpful. Possibly ask about their day or what they'd like assistance with.\n\nThus answer: \"Hello! How can I help you today?\" That's it.";
        let content = "Hello! 👋 How can I help you today?";
        let reconciled = reconcile_reasoning_usage(
            Some(Usage {
                input_tokens: Some(12),
                output_tokens: Some(100),
                cached_tokens: None,
                reasoning_tokens: None,
            }),
            reasoning,
            content,
        )
        .unwrap();
        assert_eq!(reconciled.output_tokens, Some(100));
        let rt = reconciled.reasoning_tokens.unwrap();
        assert!(
            (85..=95).contains(&rt),
            "expected ~91 reasoning tokens out of 100, got {rt}"
        );

        // 2. Inline `<think>...</think>` block extraction
        let (r, c) = split_inline_think_block(
            "<think>\nPlan A* priority queue with BinaryHeap.\n</think>\n\n```rust\nfn astar() {}\n```",
        )
        .unwrap();
        assert_eq!(r, "Plan A* priority queue with BinaryHeap.");
        assert_eq!(c, "```rust\nfn astar() {}\n```");
        assert!(split_inline_think_block("Normal text mentioning <think> mid-line").is_none());

        // 3. `describe_reqwest_error` surfaces root cause (`timed out`) and maps to HTTP 504
        let slow_app = Router::new().route(
            "/slow",
            post(|| async {
                tokio::time::sleep(std::time::Duration::from_millis(350)).await;
                axum::Json(json!({"ok": true}))
            }),
        );
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, slow_app).await;
        });

        let fast_timeout_http = reqwest::Client::builder()
            .read_timeout(std::time::Duration::from_millis(80))
            .build()
            .unwrap();
        let err = fast_timeout_http
            .post(format!("http://{addr}/slow"))
            .json(&json!({}))
            .send()
            .await
            .unwrap_err();
        let (status, msg) = describe_reqwest_error(&err);
        assert_eq!(status, Some(reqwest::StatusCode::GATEWAY_TIMEOUT));
        assert!(
            msg.to_ascii_lowercase().contains("timed out"),
            "expected causal chain to include 'timed out', got: {msg}"
        );
    }
}
