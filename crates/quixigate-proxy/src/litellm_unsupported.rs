//! Actionable Migration Guidance Catcher for Unsupported LiteLLM Endpoints.
//!
//! When a LiteLLM client, SDK, or automation script calls a LiteLLM route that QuixiGate
//! intentionally excludes to stay a single-binary, zero-bloat Rust gateway, QuixiGate
//! intercepts the request and returns HTTP `400 Bad Request` (`type: "unsupported_litellm_endpoint"`)
//! with:
//! - `reason`: Why LiteLLM had this endpoint and why QuixiGate keeps it out of the proxy hot path
//! - `quixigate_alternative`: The native QuixiGate endpoint(s) or pattern to use instead
//! - `how_to_proceed`: Concrete, step-by-step instructions to implement equivalent functionality cleanly
//!
//! Returning `400 Bad Request` (rather than `5xx`) ensures production load balancers and Prometheus
//! rules never treat unsupported client calls as server crashes, and OpenAI/Anthropic/LiteLLM SDKs
//! fail fast (`BadRequestError`, zero retries) while printing `error.message` directly to the caller.
//! Genuinely unknown non-LiteLLM URLs continue to receive standard `404 Not Found` (`code: "not_found"`).

use axum::Json;
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, Clone, Serialize)]
pub struct UnsupportedGuidance {
    pub code: &'static str,
    pub summary: &'static str,
    pub reason: &'static str,
    pub quixigate_alternative: &'static str,
    pub how_to_proceed: &'static [&'static str],
}

/// Axum `.fallback(...)` handler: returns `400 Bad Request` with actionable migration guidance
/// if `uri.path()` belongs to an unsupported LiteLLM endpoint family, otherwise `404 Not Found`.
pub async fn fallback_handler(method: Method, uri: Uri) -> Response {
    if let Some(g) = classify_unsupported_litellm_route(&method, uri.path()) {
        return render_guidance_response(&method, uri.path(), g);
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "type": "error",
            "error": {
                "message": format!("route '{}' not found", uri.path()),
                "type": "not_found_error",
                "code": "not_found"
            }
        })),
    )
        .into_response()
}

/// Axum `.method_not_allowed_fallback(...)` handler: returns `400 Bad Request` with actionable
/// migration guidance when a LiteLLM-specific HTTP verb hits a shared path, otherwise `405`.
pub async fn method_not_allowed_handler(method: Method, uri: Uri) -> Response {
    if let Some(g) = classify_unsupported_litellm_route(&method, uri.path()) {
        return render_guidance_response(&method, uri.path(), g);
    }
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "type": "error",
            "error": {
                "message": format!("method {} not allowed on '{}'", method, uri.path()),
                "type": "invalid_request_error",
                "code": "method_not_allowed"
            }
        })),
    )
        .into_response()
}

fn render_guidance_response(method: &Method, path: &str, g: UnsupportedGuidance) -> Response {
    let endpoint = format!("{} {}", method.as_str(), path);
    let steps_inline = g
        .how_to_proceed
        .iter()
        .enumerate()
        .map(|(idx, step)| format!("({}) {}", idx + 1, step))
        .collect::<Vec<_>>()
        .join(" ");
    (
        StatusCode::BAD_REQUEST,
        [
            ("x-quixigate-error-type", "unsupported_litellm_endpoint"),
            ("x-quixigate-error-code", g.code),
            ("x-should-retry", "false"),
        ],
        Json(json!({
            "type": "error",
            "error": {
                "message": format!(
                    "{} ({endpoint}). Recommended alternative: {}. How to proceed: {}",
                    g.summary, g.quixigate_alternative, steps_inline
                ),
                "type": "unsupported_litellm_endpoint",
                "code": g.code,
                "litellm_endpoint": endpoint,
                "reason": g.reason,
                "quixigate_alternative": g.quixigate_alternative,
                "how_to_proceed": g.how_to_proceed,
            }
        })),
    )
        .into_response()
}

fn strip_first_segment(path: &str) -> Option<&str> {
    let trimmed = path.strip_prefix('/')?;
    let (_, rest) = trimmed.split_once('/')?;
    Some(rest)
}

fn starts_with_any(s: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| {
        if p.ends_with('/') {
            return s.starts_with(*p);
        }
        s == *p
            || s.strip_prefix(p).is_some_and(|rem| {
                rem.is_empty()
                    || rem.starts_with('/')
                    || rem.starts_with(':')
                    || rem.starts_with('{')
            })
    })
}

/// Classifies an unmounted `(Method, path)` against all intentionally unsupported LiteLLM route families.
pub fn classify_unsupported_litellm_route(
    method: &Method,
    raw_path: &str,
) -> Option<UnsupportedGuidance> {
    let path = if raw_path.len() > 1 {
        raw_path.trim_end_matches('/')
    } else {
        raw_path
    };
    let after_first = strip_first_segment(path);

    // 1. Organization & Project 4-Tier Hierarchy (`/organization/*`, `/v2/organization/*`, `/project/*`)
    if starts_with_any(path, &["/organization", "/v2/organization", "/project"]) {
        return Some(UnsupportedGuidance {
            code: "unsupported_org_project_hierarchy",
            summary: "QuixiGate intentionally omits LiteLLM's 4-tier Organization/Project RBAC tables in favor of a flat Team + Key + Tag budget model",
            reason: "Nested Organization -> Team -> Project -> Key join hierarchies add multi-table lock contention on the hot path when flat Teams, per-model budgets, and scoped spend tags express the same cost boundaries.",
            quixigate_alternative: "POST /team/new + POST /tag/new + POST /key/generate (with tags & model_max_budget)",
            how_to_proceed: &[
                "Map each LiteLLM Organization or Project to a QuixiGate Team (`POST /team/new`) or a managed cost tag (`POST /tag/new` with `max_budget`).",
                "Attach `tags: [\"org:<org_id>\", \"project:<project_id>\"]` when minting virtual keys (`POST /key/generate`) or teams (`POST /team/update`).",
                "Query rolled-up spend per organization or project at any time via `GET /global/spend/tags` or `GET /spend/export?format=csv`.",
            ],
        });
    }

    // 2. Stateful Files, Async Batch Queues & Fine-Tuning (`/{provider}/v1/{files,batches}*`, `/{v1/,}{files,batches,fine_tuning}*`, `DELETE /v1/models/{id}`)
    let is_provider_files_or_batches = after_first.is_some_and(|rest| {
        starts_with_any(
            rest,
            &[
                "v1/files",
                "files",
                "v1/batches",
                "batches",
                "v1/fine_tuning",
            ],
        )
    });
    let is_model_delete = *method == Method::DELETE
        && (path.starts_with("/v1/models/") || path.starts_with("/models/"));
    if is_provider_files_or_batches
        || is_model_delete
        || starts_with_any(
            path,
            &[
                "/v1/files",
                "/files",
                "/v1/batches",
                "/batches",
                "/v1/fine_tuning",
                "/fine_tuning",
            ],
        )
    {
        return Some(UnsupportedGuidance {
            code: "unsupported_stateful_files_batches_finetuning",
            summary: "Stateful provider blob storage (/v1/files), async 24h batch jobs (/v1/batches), and fine-tuning jobs (/v1/fine_tuning) are not proxied through QuixiGate",
            reason: "Provider-issued `file-*`, `batch_*`, and `ftjob-*` IDs are pinned to a single upstream vendor account and break multi-provider failover while buffering multi-hundred-MB JSONL uploads through the inference gateway.",
            quixigate_alternative: "Inline base64/URL content on POST /v1/chat/completions or POST /v1/responses, or direct provider SDK calls for batch/fine-tuning",
            how_to_proceed: &[
                "For PDFs, images, or documents in chat/responses: pass HTTPS presigned S3/R2 URLs or inline `data:<mime>;base64,...` content blocks directly to `POST /v1/chat/completions` or `POST /v1/responses`.",
                "For 24-hour offline `/v1/batches` or `/v1/fine_tuning/jobs`: instantiate a dedicated upstream provider client (`OpenAI(base_url=\"https://api.openai.com/v1\")`) in your background worker for file upload and job polling.",
                "Once a fine-tuned model completes, register its model alias in QuixiGate via `POST /model/new` (`{\"model_name\": \"my-ft\", \"litellm_params\": {\"model\": \"openai/ft:...\"}}`) so live inference still benefits from QuixiGate budgets, caching, and tracing.",
            ],
        });
    }

    // 3. Server-Stored Responses State (`/{v1/,openai/v1/,}responses/{response_id}*`)
    if starts_with_any(
        path,
        &["/v1/responses/", "/responses/", "/openai/v1/responses/"],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_stateful_response_storage",
            summary: "Server-side stored `/v1/responses/{response_id}` retrieval, deletion, and cancellation are intentionally disabled in QuixiGate's stateless Responses engine",
            reason: "Persisting every multi-turn conversation transcript on the gateway creates unbounded PII retention liability and prevents transparent mid-conversation failover across OpenAI, Anthropic, Gemini, and local models.",
            quixigate_alternative: "Stateless POST /v1/responses (client-managed `input` array) + POST /v1/responses/compact",
            how_to_proceed: &[
                "Set `store: false` in your client mental model and persist conversation turns (`input` items) in your own application database or session state.",
                "Pass the prior conversation messages in `input: [...]` on each `POST /v1/responses` call so QuixiGate can fail over seamlessly to any healthy provider.",
                "When conversation history grows long, call `POST /v1/responses/compact` on QuixiGate to summarize older turns into a compact stateless item list.",
            ],
        });
    }

    // 4. Realtime Voice WebRTC / Bi-Directional Audio WebSockets (`/v1/realtime*`, `/realtime*`, `/vertex_ai/live`, `/v1/live/sessions`, `/deepgram/*`)
    if starts_with_any(
        path,
        &[
            "/v1/realtime",
            "/realtime",
            "/openai/v1/realtime",
            "/v1/live/sessions",
            "/live/sessions",
            "/openai/v1/live/sessions",
            "/vertex_ai/live",
            "/deepgram",
        ],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_realtime_voice_transport",
            summary: "Bi-directional Realtime Voice audio WebSockets/WebRTC are not relayed through QuixiGate",
            reason: "Relaying raw PCM/Opus audio frames through an HTTP API gateway adds an extra network hop and jitter buffer; production voice agents connect peer-to-peer via WebRTC/WS using short-lived ephemeral session tokens.",
            quixigate_alternative: "Direct WebRTC/WebSocket with server-minted ephemeral tokens, or turn-based POST /v1/audio/transcriptions + POST /v1/chat/completions + POST /v1/audio/speech",
            how_to_proceed: &[
                "For sub-300ms live voice conversations: have your backend mint a short-lived ephemeral client secret (`POST https://api.openai.com/v1/realtime/sessions`) and connect the browser/mobile client directly via WebRTC.",
                "Record session usage in your backend or forward tool-call reasoning turns through QuixiGate's `POST /v1/responses` or `POST /v1/chat/completions`.",
                "For push-to-talk or server-side audio pipelines: call QuixiGate's `POST /v1/audio/transcriptions`, `POST /v1/chat/completions` (SSE), and `POST /v1/audio/speech`, which are fully supported with NanoUsd spend tracking.",
            ],
        });
    }

    // 5. RAG, Vector Stores, Hosted Memory & OCR (`/{v1/,}{rag,vector_stores,vector_store,indexes,memory,ocr}*`, `/milvus/*`, `/opensearch/*`)
    if starts_with_any(
        path,
        &[
            "/v1/rag",
            "/rag",
            "/v1/vector_stores",
            "/vector_stores",
            "/v1/vector_store",
            "/vector_store",
            "/v1/indexes",
            "/indexes",
            "/v1/memory",
            "/memory",
            "/v1/ocr",
            "/ocr",
            "/milvus",
            "/opensearch",
        ],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_rag_vector_store_ocr",
            summary: "Document chunking, vector database hosting, key-value agent memory, and dedicated `/v1/ocr` are decoupled from QuixiGate's inference plane",
            reason: "Embedding document parsers, chunkers, and stateful vector indexes inside the LLM proxy couples retrieval schema lifecycle to gateway upgrades; dedicated vector stores (`pgvector`, Qdrant, LanceDB, Milvus) outperform proxy-wrapped RAG.",
            quixigate_alternative: "POST /v1/embeddings + POST /v1/rerank + Vision POST /v1/chat/completions paired with your own vector store",
            how_to_proceed: &[
                "For embeddings & reranking: call QuixiGate's `POST /v1/embeddings` during ingestion/query and `POST /v1/rerank` to score top-K candidates with full NanoUsd budget & fallback protection.",
                "Store and query vectors directly in your application's `pgvector`, SQLite-vec, LanceDB, Qdrant, or Milvus client rather than routing vector CRUD through the LLM gateway.",
                "For OCR (`/v1/ocr`): send page images or PDF screenshots as image content blocks to a vision model (`google/gemini-2.5-flash`, `openai/gpt-4o`, or `qwen/qwen2.5-vl`) via `POST /v1/chat/completions` with structured JSON output.",
            ],
        });
    }

    // 6. Deprecated Assistants v1, Cloud Sandbox Containers, Video Generation & Evals
    let is_provider_videos =
        after_first.is_some_and(|rest| starts_with_any(rest, &["v1/videos", "videos"]));
    if is_provider_videos
        || starts_with_any(
            path,
            &[
                "/v1/assistants",
                "/assistants",
                "/v1/threads",
                "/threads",
                "/v1/containers",
                "/containers",
                "/v1/videos",
                "/videos",
                "/v1/evals",
                "/evals",
            ],
        )
    {
        return Some(UnsupportedGuidance {
            code: "unsupported_assistants_containers_videos_evals",
            summary: "Deprecated Assistants/Threads v1, remote code-interpreter containers, async video generation, and hosted eval runs are not hosted by QuixiGate",
            reason: "OpenAI deprecated `/v1/assistants` & `/v1/threads` in favor of `/v1/responses`, while cloud containers, multi-minute video jobs, and eval suites are asynchronous job runners outside stateless LLM routing.",
            quixigate_alternative: "POST /v1/responses + MCP tools (/mcp) + Promptfoo / Inspect AI / Arize Phoenix pointed at QuixiGate",
            how_to_proceed: &[
                "Migrate `/v1/assistants` and `/v1/threads/*/runs` loops to `POST /v1/responses` (or `POST /v1/chat/completions`) with standard function/MCP tool definitions.",
                "For `/v1/containers` code execution: run code locally in a sandboxed Docker/Firecracker/E2B container and expose it to models as a tool or MCP server (`POST /v1/mcp/server`).",
                "For `/v1/evals`: point Promptfoo, Braintrust, Inspect AI, or Arize Phoenix at `OPENAI_BASE_URL=http://<quixigate>/v1` (and optionally enable `POST /api/admin/telemetry/phoenix-preset`).",
                "For `/v1/videos`: submit long-running Sora/Veo/Runway render jobs directly via the vendor's async job SDK and store rendered MP4 assets in S3/R2.",
            ],
        });
    }

    // 7. SCIM v2 Directory Sync & Custom JWT Key-Mapping Tables (`/scim/v2*`, `/jwt/key/mapping*`)
    if starts_with_any(path, &["/scim", "/jwt/key/mapping"]) {
        return Some(UnsupportedGuidance {
            code: "unsupported_scim_and_jwt_mapping_crud",
            summary: "Push-based SCIM v2 user provisioning and database JWT-key mapping tables are replaced by QuixiGate's zero-sync OIDC/JWKS claim verification",
            reason: "SCIM v2 directory webhooks require exposing an inbound provisioning endpoint to your IdP and keeping duplicate user state in sync; stateless OIDC/JWKS verifies IdP tokens directly from group/role claims.",
            quixigate_alternative: "Built-in OIDC/JWKS verification (`oidc_issuer_url`, `jwt_roles_claim`, `jwt_team_claim`) via PATCH /api/admin/settings or POST /key/generate",
            how_to_proceed: &[
                "Configure `oidc_issuer_url` (e.g. Okta, Entra ID, Auth0, Keycloak, Google Workspace), `oidc_audience`, `jwt_roles_claim`, and `jwt_team_claim` via `PATCH /api/admin/settings` or `quixigate.toml`.",
                "Pass your IdP-issued JWT directly in `Authorization: Bearer <jwt>` — QuixiGate validates signatures against live JWKS and maps `jwt_team_claim` / `jwt_roles_claim` automatically without SCIM sync.",
                "If you prefer static service keys per IdP group, run a tiny CI/Terraform step calling `POST /team/new` and `POST /key/generate` on group changes.",
            ],
        });
    }

    // 8. SaaS FinOps Push Integrations, External Enterprise Vaults & Redis Cluster Sidecars
    if starts_with_any(
        path,
        &[
            "/cloudzero",
            "/vantage",
            "/config_overrides",
            "/credentials/migrate-encryption",
            "/coordination_redis",
            "/cache/redis",
            "/cache/settings",
            "/langfuse",
            "/moyai",
            "/lens",
        ],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_external_saas_sidecar_integration",
            summary: "Vendor-specific FinOps push connectors, external Vault/CyberArk config overrides, and Redis sidecar endpoints are omitted by design",
            reason: "QuixiGate runs as a self-contained single binary with embedded SQLite WAL, an in-process AES-256-GCM credential vault, lock-free DashMap rate limiters, and vendor-neutral OpenTelemetry/Prometheus exports.",
            quixigate_alternative: "GET /spend/export?format=csv, GET /metrics (Prometheus), OTLP export (PATCH /api/admin/settings), and `${ENV_VAR}` interpolation in POST /config/yaml",
            how_to_proceed: &[
                "For CloudZero / Vantage / FinOps ingestion: schedule a lightweight cron job to pull `GET /spend/export?format=csv` (or `GET /global/spend/report`) and upload FOCUS/CSV rows to your FinOps bucket.",
                "For HashiCorp Vault / CyberArk / AWS Secrets Manager: inject secrets into environment variables via Vault Agent or Kubernetes ExternalSecrets and reference `${OPENAI_API_KEY}` in `quixigate.toml` or `POST /config/yaml`.",
                "For Langfuse / Phoenix / Datadog observability: set `otlp_endpoint` and `otlp_headers` via `PATCH /api/admin/settings` — QuixiGate emits native OpenInference OTLP spans with zero Python callback overhead.",
                "For caching & rate-limit state: QuixiGate uses embedded SQLite WAL + in-memory DashMap atomics (`GET /cache/ping`, `POST /cache/flushall`) — no Redis server is needed.",
            ],
        });
    }

    // 9. Hosted Prompt Registry, Policy Engine & Arbitrary Custom-Code Guardrail CMS
    if starts_with_any(
        path,
        &[
            "/prompts",
            "/utils/dotprompt_json_converter",
            "/policies",
            "/policy",
            "/utils/test_policies_and_guardrails",
            "/guardrails",
            "/compliance",
        ],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_hosted_prompt_policy_cms",
            summary: "Database-hosted prompt templates, YAML policy attachment DSLs, and custom-code guardrail CRUD are not stored inside QuixiGate",
            reason: "Storing prompt templates and executing custom guardrail callbacks inside the gateway separates prompts from application Git history and adds arbitrary-code attack surface to the proxy.",
            quixigate_alternative: "Git-versioned prompts in your codebase + built-in secret/keyword guardrails (`PATCH /api/admin/settings` & `POST /apply_guardrail`)",
            how_to_proceed: &[
                "Keep `.prompt` / Jinja / TypeScript / Rust prompt templates version-controlled alongside your application code (or use OpenRouter `@preset/<slug>` presets via `GET /api/v1/presets`).",
                "Enable QuixiGate's zero-latency built-in secret scanner (`secret_scanner_mode: \"redact\"` or `\"block\"`) and keyword blocklist (`guardrail_blocked_keywords`) via `PATCH /api/admin/settings`.",
                "Test inputs or model outputs on demand with `POST /apply_guardrail` or `POST /v1/moderations`.",
                "If you need heavy ML PII/toxicity classifiers (e.g. Presidio or Llama-Guard), call that classifier in your application middleware before sending the prompt to QuixiGate.",
            ],
        });
    }

    // 10. Skill Zip Hosting, Hosted A2A/Agent/Workflow Registries & Multi-Tenant MCP OAuth Marketplace
    let is_dynamic_mcp_subroute = after_first
        .is_some_and(|rest| matches!(rest, "mcp" | "authorize" | "token" | "register"))
        && !path.starts_with("/api/")
        && !path.starts_with("/v1/");
    if is_dynamic_mcp_subroute
        || starts_with_any(
            path,
            &[
                "/claude-code",
                "/v1/skills",
                "/skill-hub",
                "/.well-known/agent-skills",
                "/.well-known/skills",
                "/api/plugins",
                "/plugin-proxy",
                "/a2a",
                "/v1/a2a",
                "/agents",
                "/v1/agents",
                "/v1beta/agents",
                "/agent",
                "/interactions",
                "/v1/interactions",
                "/v1beta/interactions",
                "/v1/workflows",
                "/workflows",
                "/v1/tool",
                "/toolset",
                "/search_tools",
                "/auto_router",
                "/v1/mcp",
                "/mcp/proxy",
                "/authorize",
                "/token",
                "/revoke",
                "/introspect",
                "/callback",
                "/register",
                "/.well-known/oauth-authorization-server",
                "/.well-known/oauth-protected-resource",
                "/.well-known/litellm-cli-auth",
            ],
        )
    {
        return Some(UnsupportedGuidance {
            code: "unsupported_agent_skill_mcp_marketplace",
            summary: "Hosted skill zip registries, server-side A2A/workflow state machines, and MCP OAuth marketplace brokering are outside QuixiGate's scope",
            reason: "QuixiGate focuses on fast, reliable model routing and Streamable-HTTP MCP bridging rather than acting as a plugin zip CDN, workflow orchestrator, or multi-tenant OAuth authorization server.",
            quixigate_alternative: "POST /mcp (Streamable-HTTP JSON-RPC), GET/POST /v1/mcp/server, POST /mcp-rest/tools/call, and POST /v1/search",
            how_to_proceed: &[
                "For MCP tools: register upstream Streamable-HTTP MCP servers via `POST /v1/mcp/server` (or `POST /api/admin/mcp`) and connect agents to QuixiGate's unified `POST /mcp` endpoint.",
                "For Claude Code / Cursor skills & plugins: distribute skills via Git repositories (`.claude/skills/` or `.agents/skills/`) rather than uploading zip archives to the LLM proxy.",
                "For A2A / multi-step workflows: run LangGraph, Temporal, PydanticAI, or OpenAI Agents SDK in your worker tier with `OPENAI_BASE_URL=http://<quixigate>/v1`.",
                "For search tools (`/search_tools/*`): configure search API keys in QuixiGate's vault or env (`TAVILY_API_KEY`, `EXA_API_KEY`, `BRAVE_API_KEY`, `SEARXNG_URL`) and call `POST /v1/search` or append `:online` to any model ID.",
            ],
        });
    }

    // 11. Raw Vendor-Specific Pass-Throughs (`/openai/*`, `/anthropic/*`, `/vertex_ai/*`, `/bedrock/*`, etc.)
    if starts_with_any(
        path,
        &[
            "/openai",
            "/openai_passthrough",
            "/anthropic",
            "/gemini",
            "/vertex_ai",
            "/vertex-ai",
            "/openrouter",
            "/vllm",
            "/mistral",
            "/cohere",
            "/bedrock",
            "/azure",
            "/azure_ai",
            "/azure_speech",
            "/{AZURE_SPEECH_PASS_THROUGH_ROUTE_PREFIX}",
            "/nvidia_nim",
            "/assemblyai",
            "/eu.assemblyai",
            "/fal_ai",
            "/typesafe",
            "/tinyfish",
            "/gigachat",
            "/watsonx",
            "/comprehendmedical",
            "/transcribe",
            "/backend-api/codex",
            "/config/pass_through",
            "/cursor/",
            "/v1/systemone",
            "/systemone",
            "/v1/decisions",
            "/decisions",
            "/v1/extraction",
            "/extraction",
            "/eu_jev",
            "/laya",
            "/bespoke",
        ],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_raw_vendor_passthrough",
            summary: "Opaque vendor-specific pass-through tunnels are disabled in favor of QuixiGate's normalized polyglot inference codecs",
            reason: "Blindly forwarding unparsed vendor HTTP paths bypasses unified token/NanoUsd budget reservation, secret-scanner guardrails, semantic caching, and cross-provider failover.",
            quixigate_alternative: "POST /v1/chat/completions, POST /v1/responses, POST /v1/messages, POST /v1beta/models/{model}:generateContent, POST /v1/embeddings, POST /v1/images/*, POST /v1/audio/*, POST /v1/rerank",
            how_to_proceed: &[
                "Point OpenAI, Anthropic, Google Gemini, Vercel AI, or Azure SDKs at QuixiGate's native polyglot endpoints (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1beta/models/...`, `/openai/deployments/{model}/...`).",
                "Select any provider or model using `<provider>/<model>` (e.g. `anthropic/claude-sonnet-4-5`, `bedrock/...`, `openrouter/...`, `ollama/...`) and pass vendor-specific parameters in extra JSON body fields, which QuixiGate preserves.",
                "For non-LLM domain APIs (such as AWS Comprehend Medical, AssemblyAI async webhooks, or custom internal microservices), call those endpoints directly or route them through a general-purpose reverse proxy (Envoy, Caddy, NGINX).",
            ],
        });
    }

    // 12. LiteLLM Internal Next.js Dashboard RPCs, SAML SSO, ROI Calculator, CPython Debug & OTLP Sink
    if starts_with_any(
        path,
        &[
            "/get/",
            "/update/",
            "/add/allowed_ip",
            "/delete/allowed_ip",
            "/upload/logo",
            "/get_logo_url",
            "/get_image",
            "/get_favicon",
            "/sso",
            "/login",
            "/v2/login",
            "/v3/login",
            "/fallback/login",
            "/oauth",
            "/onboarding",
            "/invitation",
            "/roi-calculator",
            "/telemetry",
            "/subscription_accounts",
            "/model-insights",
            "/cost/predict-cache",
            "/cost_optimization",
            "/debug",
            "/memory-usage",
            "/memory-usage-in-mem-cache",
            "/memory-usage-in-mem-cache-items",
            "/lazy/warm",
            "/schedule",
            "/reload/anthropic_beta_headers",
            "/queue",
            "/alerting",
            "/health/history",
            "/health/latest",
            "/health/shared-status",
            "/health/license",
            "/health/deployment_warnings",
            "/v1/traces",
            "/v1/logs",
            "/v1/metrics",
            "/v1/rust_control_plane",
            "/logs",
            "/api/event_logging",
            "/otel-spans",
            "/callbacks",
            "/active/callbacks",
            "/config/",
            "/team/",
            "/user/filter/ui",
            "/tag/",
            "/global/activity/",
            "/model/",
            "/model_group/make_public",
            "/model_hub/update_useful_links",
            "/public/",
            "/spend/",
            "/gateway/errors/activity",
            "/.well-known/litellm-ui-config",
            "/litellm/.well-known/litellm-ui-config",
            "/.well-known/openid-configuration",
            "/.well-known/jwks.json",
            "/managed/settings",
            "/credentials/",
        ],
    ) {
        return Some(UnsupportedGuidance {
            code: "unsupported_litellm_internal_ui_or_debug_rpc",
            summary: "This endpoint is an internal LiteLLM Next.js UI helper, CPython runtime diagnostic, or OTLP collector sink not needed by QuixiGate",
            reason: "QuixiGate ships its own lightweight embedded operator UI (`/ui`) and clean `/api/admin/*` REST control plane, has no Python GC/lazy-import runtime to debug, and exports OTLP traces outward to Phoenix/Jaeger rather than acting as a trace database.",
            quixigate_alternative: "QuixiGate `/api/admin/*` REST API, optional `/ui` dashboard, `GET /metrics` (Prometheus), and `GET /spend/logs`",
            how_to_proceed: &[
                "For operator authentication & UI management: use `POST /api/admin/auth/login`, `GET /api/admin/overview`, `GET /api/admin/doctor`, or enable OIDC SSO (`oidc_issuer_url` in `PATCH /api/admin/settings`; bridge SAML IdPs via Dex or Keycloak).",
                "For IP allowlisting (`/add/allowed_ip`): attach `allowed_ips` / `allowed_cidrs` directly per virtual key (`POST /key/generate` or `POST /key/update`) or enforce network ACLs at your ingress/firewall.",
                "For trace & error inspection (`/v1/traces`, `/logs`, `/gateway/errors/activity`): query `GET /api/admin/traces` (or live SSE `GET /api/admin/traces/stream`) or point `otlp_endpoint` at Arize Phoenix / Grafana Tempo.",
                "For cost & spend analytics (`/spend/*`, `/tag/*`, `/model/metrics*`): use `GET /spend/report`, `GET /global/spend/report`, `GET /spend/export?format=csv`, and Prometheus `GET /metrics`.",
            ],
        });
    }

    None
}
