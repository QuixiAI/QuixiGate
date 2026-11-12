//! OpenAPI 3.1.0 specification (`/openapi.json`, `/api/admin/openapi.json`) and
//! LiteLLM-style API root / documentation endpoints (`/` and `/docs`).
//!
//! Works identically whether the optional `/ui` convenience dashboard is enabled,
//! disabled at runtime (`--no-ui` / `DISABLE_ADMIN_UI=true`), or omitted at compile
//! time (`cargo build --no-default-features`).

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::state::AppState;

fn docs_disabled_by_env() -> bool {
    std::env::var("NO_DOCS")
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

fn prefers_html(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|accept| accept.contains("text/html"))
}

pub async fn handle_api_root(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let settings = state.db.get_settings().unwrap_or_default();
    if !docs_disabled_by_env() && prefers_html(&headers) {
        return render_docs_html(state.is_ui_enabled(), settings.docs_title.as_deref());
    }
    let ui_enabled = state.is_ui_enabled();
    Json(json!({
        "service": "quixigate",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "ok",
        "openapi": "/openapi.json",
        "docs": if docs_disabled_by_env() { Value::Null } else { json!("/docs") },
        "ui_enabled": ui_enabled,
        "ui_url": ui_enabled.then_some("/ui"),
        "endpoints": {
            "inference": [
                "POST /v1/chat/completions",
                "POST /v1/responses",
                "GET  /v1/responses (WebSocket)",
                "POST /v1/messages",
                "POST /v1/messages/count_tokens",
                "POST /v1beta/models/{model}:{action}",
                "POST /v1/embeddings",
                "GET  /v1/models",
                "POST /api/chat"
            ],
            "management_litellm": [
                "POST /key/generate",
                "POST /key/update",
                "POST /key/delete",
                "GET  /key/info",
                "GET  /key/list",
                "POST /key/regenerate",
                "POST /key/block",
                "POST /key/unblock",
                "POST /user/new",
                "POST /user/update",
                "POST /user/delete",
                "GET  /user/info",
                "GET  /user/list",
                "POST /team/new",
                "POST /team/update",
                "POST /team/delete",
                "GET  /team/info",
                "GET  /team/list",
                "POST /team/member_add",
                "POST /team/member_delete",
                "POST /model/new",
                "POST /model/update",
                "POST /model/delete",
                "GET  /model/info",
                "GET  /v1/model/info",
                "GET  /model_group/info",
                "GET  /credentials",
                "POST /credentials",
                "DELETE /credentials/{id}",
                "GET  /spend/logs",
                "GET  /spend/export",
                "POST /spend/calculate",
                "GET  /global/spend",
                "GET  /global/spend/report",
                "POST /global/spend/reset",
                "GET  /cache/ping",
                "POST /cache/flushall",
                "GET  /config/yaml",
                "POST /config/yaml",
                "GET  /config/ui",
                "POST /config/ui"
            ],
            "management_admin": "/api/admin/*",
            "ui_toggle": [
                "GET  /api/admin/ui",
                "POST /api/admin/ui/enable",
                "POST /api/admin/ui/disable",
                "POST /api/admin/ui",
                "POST /config/ui"
            ],
            "health": ["/health", "/health/liveness", "/health/liveliness", "/health/readiness", "/metrics"]
        }
    }))
    .into_response()
}

pub async fn handle_docs(State(state): State<AppState>) -> Response {
    if docs_disabled_by_env() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": "API docs disabled via NO_DOCS",
                    "type": "not_found_error"
                }
            })),
        )
            .into_response();
    }
    let settings = state.db.get_settings().unwrap_or_default();
    render_docs_html(state.is_ui_enabled(), settings.docs_title.as_deref())
}

type OperationEntry = (&'static str, &'static str, &'static str);
type PathEntry = (&'static str, &'static [OperationEntry]);

pub async fn handle_openapi_json(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<Value> {
    let ui_enabled = state.is_ui_enabled();
    let settings = state.db.get_settings().unwrap_or_default();
    let hide_mgmt = settings.docs_filtered
        && !crate::auth::authenticate_inference(&state, &headers, None)
            .is_ok_and(|p| p.user.role == quixigate_core::UserRole::Admin);
    let specs: &[PathEntry] = &[
        // Polyglot Inference
        (
            "/v1/chat/completions",
            &[(
                "post",
                "Inference",
                "OpenAI Chat Completions (streaming & non-streaming)",
            )],
        ),
        (
            "/v1/responses",
            &[
                ("post", "Inference", "OpenAI Responses API (HTTP/SSE)"),
                ("get", "Inference", "OpenAI Responses Realtime WebSocket"),
            ],
        ),
        (
            "/v1/messages",
            &[("post", "Inference", "Anthropic Messages API")],
        ),
        (
            "/v1/messages/count_tokens",
            &[("post", "Inference", "Anthropic Token Counting")],
        ),
        (
            "/v1beta/models/{model_action}",
            &[(
                "post",
                "Inference",
                "Google Gemini generateContent / streamGenerateContent",
            )],
        ),
        (
            "/v1/embeddings",
            &[("post", "Inference", "OpenAI Embeddings")],
        ),
        (
            "/v1/models",
            &[("get", "Inference", "List available models")],
        ),
        (
            "/api/v1/presets",
            &[(
                "get",
                "Inference",
                "OpenRouter-compatible @preset/* & Smart Route preset catalog",
            )],
        ),
        (
            "/api/v1/chat/completions",
            &[(
                "post",
                "Inference",
                "OpenRouter-compatible /api/v1 Chat Completions alias",
            )],
        ),
        (
            "/api/v1/responses",
            &[(
                "post",
                "Inference",
                "OpenRouter-compatible /api/v1 Responses alias",
            )],
        ),
        (
            "/api/v1/key",
            &[(
                "get",
                "Inference",
                "OpenRouter-compatible key usage & limit introspection",
            )],
        ),
        (
            "/api/v1/credits",
            &[(
                "get",
                "Inference",
                "OpenRouter-compatible credits introspection",
            )],
        ),
        (
            "/api/chat",
            &[("post", "Inference", "Vercel AI SDK UI Message Stream")],
        ),
        (
            "/mcp",
            &[
                (
                    "post",
                    "MCP",
                    "Streamable-HTTP JSON-RPC 2.0 MCP server (initialize, tools/list, tools/call)",
                ),
                ("get", "MCP", "Streamable-HTTP MCP SSE / metadata endpoint"),
                ("delete", "MCP", "Streamable-HTTP MCP session close"),
            ],
        ),
        // LiteLLM-Compatible Management API
        (
            "/key/generate",
            &[("post", "LiteLLM Keys", "Mint a new sk-qx-... virtual key")],
        ),
        (
            "/key/update",
            &[(
                "post",
                "LiteLLM Keys",
                "Update virtual key limits, models, alias, or block state",
            )],
        ),
        (
            "/key/delete",
            &[("post", "LiteLLM Keys", "Delete one or more virtual keys")],
        ),
        (
            "/key/info",
            &[(
                "get",
                "LiteLLM Keys",
                "Inspect a virtual key or current caller key",
            )],
        ),
        (
            "/key/list",
            &[("get", "LiteLLM Keys", "List virtual keys")],
        ),
        (
            "/key/regenerate",
            &[(
                "post",
                "LiteLLM Keys",
                "Rotate virtual key secret with optional grace window",
            )],
        ),
        (
            "/key/{key_id}/regenerate",
            &[("post", "LiteLLM Keys", "Rotate virtual key by path ID")],
        ),
        (
            "/key/block",
            &[("post", "LiteLLM Keys", "Block (revoke) a virtual key")],
        ),
        (
            "/key/unblock",
            &[("post", "LiteLLM Keys", "Unblock (unrevoke) a virtual key")],
        ),
        (
            "/key/health",
            &[("post", "LiteLLM Keys", "Check caller key health")],
        ),
        (
            "/user/new",
            &[(
                "post",
                "LiteLLM Users",
                "Create user (and optional default virtual key)",
            )],
        ),
        (
            "/user/update",
            &[(
                "post",
                "LiteLLM Users",
                "Update user role, display alias, password, or budget",
            )],
        ),
        (
            "/user/delete",
            &[("post", "LiteLLM Users", "Delete users")],
        ),
        (
            "/user/block",
            &[("post", "LiteLLM Users", "Block one or more users")],
        ),
        (
            "/user/unblock",
            &[("post", "LiteLLM Users", "Unblock one or more users")],
        ),
        (
            "/user/info",
            &[(
                "get",
                "LiteLLM Users",
                "Get user details, owned keys, and teams",
            )],
        ),
        ("/user/list", &[("get", "LiteLLM Users", "List users")]),
        (
            "/team/new",
            &[(
                "post",
                "LiteLLM Teams",
                "Create team with budget and members",
            )],
        ),
        (
            "/team/update",
            &[("post", "LiteLLM Teams", "Update team settings and budget")],
        ),
        (
            "/team/delete",
            &[("post", "LiteLLM Teams", "Delete teams")],
        ),
        (
            "/team/info",
            &[("get", "LiteLLM Teams", "Get team details and keys")],
        ),
        ("/team/list", &[("get", "LiteLLM Teams", "List teams")]),
        (
            "/team/member_add",
            &[("post", "LiteLLM Teams", "Add members to team")],
        ),
        (
            "/team/member_delete",
            &[("post", "LiteLLM Teams", "Remove member from team")],
        ),
        (
            "/model/new",
            &[(
                "post",
                "LiteLLM Models",
                "Register model route (and optional vault credential)",
            )],
        ),
        (
            "/model/update",
            &[("post", "LiteLLM Models", "Update model route")],
        ),
        (
            "/model/{model_id}/update",
            &[("patch", "LiteLLM Models", "Patch model route by ID")],
        ),
        (
            "/model/delete",
            &[("post", "LiteLLM Models", "Delete model route")],
        ),
        (
            "/model/info",
            &[(
                "get",
                "LiteLLM Models",
                "List configured routes and active catalog models",
            )],
        ),
        (
            "/v1/model/info",
            &[(
                "get",
                "LiteLLM Models",
                "List configured routes and active catalog models",
            )],
        ),
        (
            "/model_group/info",
            &[(
                "get",
                "LiteLLM Models",
                "List model groups, capabilities, and NanoUsd rates",
            )],
        ),
        (
            "/credentials",
            &[
                (
                    "get",
                    "LiteLLM Credentials",
                    "List vault credentials (redacted)",
                ),
                (
                    "post",
                    "LiteLLM Credentials",
                    "Create encrypted vault credential",
                ),
            ],
        ),
        (
            "/credentials/{id}",
            &[(
                "delete",
                "LiteLLM Credentials",
                "Delete vault credential",
            )],
        ),
        (
            "/spend/logs",
            &[("get", "LiteLLM Spend & Ops", "Query spend & request logs")],
        ),
        (
            "/spend/report",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Grouped NanoUsd spend & savings report (tag, model, provider, key, team, user, day)",
            )],
        ),
        (
            "/spend/tags",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Spend & savings report grouped by request/key/team tags",
            )],
        ),
        (
            "/audit/logs",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Query immutable management audit logs",
            )],
        ),
        (
            "/audit/{id}",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Fetch single management audit log entry",
            )],
        ),
        (
            "/spend/calculate",
            &[(
                "post",
                "LiteLLM Spend & Ops",
                "Calculate exact NanoUsd cost for model usage",
            )],
        ),
        (
            "/global/spend",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Get global paid spend & OAuth/local savings",
            )],
        ),
        (
            "/global/spend/report",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Get spend breakdown by model, provider, user, team, and key",
            )],
        ),
        (
            "/global/spend/reset",
            &[("post", "LiteLLM Spend & Ops", "Reset spend counters")],
        ),
        (
            "/cache/ping",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Check response cache health and entry counts",
            )],
        ),
        (
            "/cache/flushall",
            &[("post", "LiteLLM Spend & Ops", "Flush all cached responses")],
        ),
        (
            "/cache/delete",
            &[("post", "LiteLLM Spend & Ops", "Delete cached responses")],
        ),
        (
            "/config/yaml",
            &[
                (
                    "get",
                    "LiteLLM Spend & Ops",
                    "Export gateway config as YAML",
                ),
                (
                    "post",
                    "LiteLLM Spend & Ops",
                    "Import gateway config from YAML",
                ),
            ],
        ),
        (
            "/config/ui",
            &[
                (
                    "get",
                    "LiteLLM Spend & Ops",
                    "Inspect optional Web UI runtime status",
                ),
                (
                    "post",
                    "LiteLLM Spend & Ops",
                    "Enable or disable optional Web UI live at runtime",
                ),
            ],
        ),
        (
            "/spend/export",
            &[(
                "get",
                "LiteLLM Spend & Ops",
                "Export NanoUsd spend & request ledger as CSV or JSON",
            )],
        ),
        // Native QuixiGate Admin REST API
        (
            "/api/admin/auth/status",
            &[(
                "get",
                "QuixiGate Admin",
                "Check auth policy, loopback trust, and session principal",
            )],
        ),
        (
            "/api/admin/auth/bootstrap",
            &[(
                "post",
                "QuixiGate Admin",
                "Bootstrap initial admin password & session",
            )],
        ),
        (
            "/api/admin/auth/login",
            &[("post", "QuixiGate Admin", "Sign in and mint qxs_... session")],
        ),
        (
            "/api/admin/auth/logout",
            &[("post", "QuixiGate Admin", "Revoke current session")],
        ),
        (
            "/api/admin/auth/password",
            &[("post", "QuixiGate Admin", "Rotate operator password")],
        ),
        (
            "/api/admin/overview",
            &[(
                "get",
                "QuixiGate Admin",
                "Dashboard telemetry, seat pool, and local runtime overview",
            )],
        ),
        (
            "/api/admin/doctor",
            &[(
                "get",
                "QuixiGate Admin",
                "Run diagnostic health checks via API",
            )],
        ),
        (
            "/api/admin/spend/summary",
            &[(
                "get",
                "QuixiGate Admin",
                "Detailed NanoUsd spend & savings breakdown",
            )],
        ),
        (
            "/api/admin/spend/export",
            &[(
                "get",
                "QuixiGate Admin",
                "Export exact NanoUsd request & spend ledger as CSV or JSON",
            )],
        ),
        (
            "/api/admin/spend/reset",
            &[(
                "post",
                "QuixiGate Admin",
                "Reset spend for user, team, key, or all",
            )],
        ),
        (
            "/api/admin/providers",
            &[(
                "get",
                "QuixiGate Admin",
                "List supported OAuth, local, and API-key providers",
            )],
        ),
        (
            "/api/admin/openrouter/status",
            &[(
                "get",
                "QuixiGate Admin",
                "OpenRouter connection status, env detection, credits, and @preset/* summary",
            )],
        ),
        (
            "/api/admin/openrouter/connect",
            &[(
                "post",
                "QuixiGate Admin",
                "Connect OpenRouter via Browser PKCE, OIDC Workload Identity, API key, or env var",
            )],
        ),
        (
            "/api/admin/openrouter/sync",
            &[(
                "post",
                "QuixiGate Admin",
                "Sync live OpenRouter credits, /models pricing, and @preset/* presets",
            )],
        ),
        (
            "/api/admin/burst",
            &[
                (
                    "get",
                    "QuixiGate Admin",
                    "Inspect Local/Subscription -> OpenRouter Cloud Burst status and attribution headers",
                ),
                (
                    "post",
                    "QuixiGate Admin",
                    "Configure Local/Subscription -> OpenRouter Cloud Burst mode live",
                ),
            ],
        ),
        (
            "/api/admin/burst/enable",
            &[(
                "post",
                "QuixiGate Admin",
                "Enable automatic OpenRouter Cloud Burst when local GPUs or subscription seats saturate",
            )],
        ),
        (
            "/api/admin/burst/disable",
            &[(
                "post",
                "QuixiGate Admin",
                "Disable automatic OpenRouter Cloud Burst",
            )],
        ),
        (
            "/api/admin/credentials",
            &[
                ("get", "QuixiGate Admin", "List credentials"),
                (
                    "post",
                    "QuixiGate Admin",
                    "Add API-key or setup-token credential",
                ),
            ],
        ),
        (
            "/api/admin/credentials/{id}",
            &[
                ("get", "QuixiGate Admin", "Get credential by ID"),
                (
                    "patch",
                    "QuixiGate Admin",
                    "Update credential priority, base_url, status, or secret",
                ),
                ("delete", "QuixiGate Admin", "Delete credential"),
            ],
        ),
        (
            "/api/admin/credentials/{id}/reset-health",
            &[(
                "post",
                "QuixiGate Admin",
                "Clear credential cooldowns and errors",
            )],
        ),
        (
            "/api/admin/credentials/{id}/refresh-quota",
            &[(
                "post",
                "QuixiGate Admin",
                "Poll live subscription quota windows",
            )],
        ),
        (
            "/api/admin/credentials/{id}/refresh-token",
            &[("post", "QuixiGate Admin", "Force OAuth token refresh")],
        ),
        (
            "/api/admin/credentials/{id}/test",
            &[(
                "post",
                "QuixiGate Admin",
                "Probe credential with a live test request",
            )],
        ),
        (
            "/api/admin/oauth/start",
            &[(
                "post",
                "QuixiGate Admin",
                "Start PKCE or device-code OAuth flow",
            )],
        ),
        (
            "/api/admin/oauth/sessions/{id}",
            &[("get", "QuixiGate Admin", "Poll OAuth session status")],
        ),
        (
            "/api/admin/oauth/sessions/{id}/complete",
            &[("post", "QuixiGate Admin", "Complete OAuth callback exchange")],
        ),
        (
            "/api/admin/models",
            &[("get", "QuixiGate Admin", "List models (active_only=true|false)")],
        ),
        (
            "/api/admin/models/sync",
            &[("post", "QuixiGate Admin", "Sync models.dev catalog")],
        ),
        (
            "/api/admin/routes",
            &[
                ("get", "QuixiGate Admin", "List routing aliases"),
                ("post", "QuixiGate Admin", "Create or update routing alias"),
            ],
        ),
        (
            "/api/admin/routes/{alias}",
            &[
                ("get", "QuixiGate Admin", "Get routing alias"),
                ("put", "QuixiGate Admin", "Upsert routing alias"),
                ("patch", "QuixiGate Admin", "Patch routing alias"),
                ("delete", "QuixiGate Admin", "Delete routing alias"),
            ],
        ),
        (
            "/api/admin/local-runtimes",
            &[
                (
                    "get",
                    "QuixiGate Admin",
                    "List local runtimes (Ollama, LM Studio, llama.cpp, MLX, vLLM)",
                ),
                (
                    "post",
                    "QuixiGate Admin",
                    "Register custom local runtime endpoint",
                ),
            ],
        ),
        (
            "/api/admin/local-runtimes/scan",
            &[(
                "post",
                "QuixiGate Admin",
                "Trigger immediate local runtime discovery scan",
            )],
        ),
        (
            "/api/admin/local-runtimes/{id}",
            &[
                ("get", "QuixiGate Admin", "Get local runtime status"),
                (
                    "delete",
                    "QuixiGate Admin",
                    "Remove custom local runtime endpoint",
                ),
            ],
        ),
        (
            "/api/admin/local-runtimes/{id}/load",
            &[("post", "QuixiGate Admin", "Warm/load model on local runtime")],
        ),
        (
            "/api/admin/local-runtimes/{id}/unload",
            &[(
                "post",
                "QuixiGate Admin",
                "Evict/unload model on local runtime",
            )],
        ),
        (
            "/api/admin/users",
            &[
                ("get", "QuixiGate Admin", "List users"),
                ("post", "QuixiGate Admin", "Create user"),
            ],
        ),
        (
            "/api/admin/users/{id}",
            &[
                ("get", "QuixiGate Admin", "Get user by ID or username"),
                ("patch", "QuixiGate Admin", "Update user"),
                ("delete", "QuixiGate Admin", "Delete user"),
            ],
        ),
        (
            "/api/admin/users/{id}/reset-spend",
            &[("post", "QuixiGate Admin", "Reset user spend")],
        ),
        (
            "/api/admin/teams",
            &[
                ("get", "QuixiGate Admin", "List teams"),
                ("post", "QuixiGate Admin", "Create team"),
            ],
        ),
        (
            "/api/admin/teams/{id}",
            &[
                ("get", "QuixiGate Admin", "Get team"),
                ("patch", "QuixiGate Admin", "Update team"),
                ("delete", "QuixiGate Admin", "Delete team"),
            ],
        ),
        (
            "/api/admin/teams/{id}/members",
            &[("post", "QuixiGate Admin", "Add members to team")],
        ),
        (
            "/api/admin/teams/{id}/members/{user_id}",
            &[("delete", "QuixiGate Admin", "Remove member from team")],
        ),
        (
            "/api/admin/teams/{id}/reset-spend",
            &[("post", "QuixiGate Admin", "Reset team spend")],
        ),
        (
            "/api/admin/keys",
            &[
                ("get", "QuixiGate Admin", "List virtual keys"),
                ("post", "QuixiGate Admin", "Create virtual key"),
            ],
        ),
        (
            "/api/admin/keys/{id}",
            &[
                ("get", "QuixiGate Admin", "Get virtual key"),
                ("patch", "QuixiGate Admin", "Update virtual key"),
                ("delete", "QuixiGate Admin", "Delete virtual key"),
            ],
        ),
        (
            "/api/admin/keys/{id}/rotate",
            &[("post", "QuixiGate Admin", "Rotate virtual key")],
        ),
        (
            "/api/admin/keys/{id}/revoke",
            &[("post", "QuixiGate Admin", "Revoke virtual key")],
        ),
        (
            "/api/admin/keys/{id}/unrevoke",
            &[("post", "QuixiGate Admin", "Unrevoke virtual key")],
        ),
        (
            "/api/admin/keys/{id}/reset-spend",
            &[("post", "QuixiGate Admin", "Reset virtual key spend")],
        ),
        (
            "/api/admin/traces",
            &[
                ("get", "QuixiGate Admin", "Query request traces"),
                ("delete", "QuixiGate Admin", "Purge request traces"),
            ],
        ),
        (
            "/api/admin/traces/stream",
            &[("get", "QuixiGate Admin", "Live SSE trace stream")],
        ),
        (
            "/api/admin/traces/{id}",
            &[("get", "QuixiGate Admin", "Get trace by ID")],
        ),
        (
            "/api/admin/settings",
            &[
                ("get", "QuixiGate Admin", "Get gateway settings"),
                ("put", "QuixiGate Admin", "Replace gateway settings"),
                (
                    "patch",
                    "QuixiGate Admin",
                    "Partially update gateway settings",
                ),
            ],
        ),
        (
            "/api/admin/ui",
            &[
                (
                    "get",
                    "QuixiGate Admin",
                    "Get live Web UI compile-time & runtime status",
                ),
                (
                    "post",
                    "QuixiGate Admin",
                    "Switch optional Web UI (/ui) on or off live at runtime",
                ),
                (
                    "put",
                    "QuixiGate Admin",
                    "Switch optional Web UI (/ui) on or off live at runtime",
                ),
                (
                    "patch",
                    "QuixiGate Admin",
                    "Switch optional Web UI (/ui) on or off live at runtime",
                ),
            ],
        ),
        (
            "/api/admin/ui/enable",
            &[(
                "post",
                "QuixiGate Admin",
                "Switch optional Web UI (/ui) ON immediately without restart",
            )],
        ),
        (
            "/api/admin/ui/disable",
            &[(
                "post",
                "QuixiGate Admin",
                "Switch optional Web UI (/ui) OFF immediately without restart",
            )],
        ),
        (
            "/api/admin/mcp",
            &[
                ("get", "QuixiGate Admin", "List configured MCP server URLs"),
                ("post", "QuixiGate Admin", "Add MCP server URL"),
                ("delete", "QuixiGate Admin", "Remove MCP server URL"),
            ],
        ),
        (
            "/api/admin/mcp/probe",
            &[("post", "QuixiGate Admin", "Probe MCP server tools/list")],
        ),
        (
            "/api/admin/telemetry/phoenix-preset",
            &[(
                "post",
                "QuixiGate Admin",
                "Configure Arize Phoenix OpenInference preset",
            )],
        ),
        (
            "/api/admin/telemetry/test-span",
            &[(
                "post",
                "QuixiGate Admin",
                "Emit OpenInference OTLP test span",
            )],
        ),
        (
            "/api/admin/cache",
            &[("get", "QuixiGate Admin", "Get cache status & entry counts")],
        ),
        (
            "/api/admin/cache/clear",
            &[("post", "QuixiGate Admin", "Clear response cache")],
        ),
        (
            "/api/admin/config/export",
            &[("get", "QuixiGate Admin", "Export config as TOML or YAML")],
        ),
        (
            "/api/admin/config/import",
            &[("post", "QuixiGate Admin", "Import config from TOML or YAML")],
        ),
        (
            "/api/admin/spend/report",
            &[(
                "get",
                "QuixiGate Admin",
                "Grouped NanoUsd spend & savings report (model, provider, tag, key, team, user, day)",
            )],
        ),
        (
            "/api/admin/audit-logs",
            &[(
                "get",
                "QuixiGate Admin",
                "Query immutable management audit logs",
            )],
        ),
        (
            "/api/admin/audit-logs/{id}",
            &[(
                "get",
                "QuixiGate Admin",
                "Get management audit log entry by ID",
            )],
        ),
        // Health & Metrics
        (
            "/robots.txt",
            &[("get", "Health", "Search engine & web crawler Disallow policy")],
        ),
        ("/health", &[("get", "Health", "Gateway liveness check")]),
        (
            "/health/liveness",
            &[("get", "Health", "Gateway liveness check")],
        ),
        (
            "/health/liveliness",
            &[("get", "Health", "LiteLLM-compatible liveliness check")],
        ),
        (
            "/health/readiness",
            &[("get", "Health", "SQLite & router readiness check")],
        ),
        (
            "/metrics",
            &[("get", "Health", "Prometheus text exposition metrics")],
        ),
    ];

    let mut paths = serde_json::Map::with_capacity(specs.len());
    for (path, ops) in specs {
        let mut method_map = serde_json::Map::with_capacity(ops.len());
        for (method, tag, summary) in *ops {
            if hide_mgmt && (*tag == "QuixiGate Admin" || tag.starts_with("LiteLLM ")) {
                continue;
            }
            method_map.insert(
                (*method).to_owned(),
                json!({
                    "tags": [tag],
                    "summary": summary,
                    "responses": { "200": { "description": "Success" } }
                }),
            );
        }
        if !method_map.is_empty() {
            paths.insert((*path).to_owned(), Value::Object(method_map));
        }
    }

    let info_title = settings
        .docs_title
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "QuixiGate Polyglot AI Gateway & Management API".to_owned());
    let info_description = settings
        .docs_description
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            format!(
                "Pure-Rust polyglot LLM gateway, AES-256-GCM credential vault, exact NanoUsd(i64) ledger, hosted MCP server, and LiteLLM-compatible Management API. Optional Web UI at /ui is currently {}.",
                if ui_enabled { "enabled" } else { "disabled" }
            )
        });

    Json(json!({
        "openapi": "3.1.0",
        "info": {
            "title": info_title,
            "version": env!("CARGO_PKG_VERSION"),
            "description": info_description
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "Master Key (sk-qx-master-...), Virtual Key (sk-qx-...), or HS256 JWT"
                },
                "LiteLlmApiKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "x-litellm-api-key"
                },
                "QuixigateAdminKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "x-quixigate-admin-key"
                }
            }
        },
        "security": [
            { "BearerAuth": [] },
            { "LiteLlmApiKey": [] },
            { "QuixigateAdminKey": [] }
        ],
        "paths": paths
    }))
}

fn escape_html_text(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn render_docs_html(ui_enabled: bool, custom_title: Option<&str>) -> Response {
    let heading = custom_title
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(escape_html_text)
        .unwrap_or_else(|| "QuixiGate Gateway &amp; Management API".to_owned());
    let ui_pill = if ui_enabled {
        r#"<a href="/ui" class="pill accent">Open Convenience Web UI (/ui) &rarr;</a>"#
    } else {
        r#"<span class="pill muted">Web UI Disabled (Headless API Mode)</span>"#
    };
    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{heading}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #090d16; color: #e2e8f0; }}
    header {{ padding: 24px 32px; border-bottom: 1px solid #1e293b; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; background: #0f172a; }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 650; letter-spacing: -0.01em; }}
    .sub {{ margin-top: 4px; font-size: 13px; color: #94a3b8; }}
    .links {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .pill {{ padding: 7px 13px; border-radius: 8px; font-size: 12.5px; font-weight: 600; text-decoration: none; border: 1px solid #334155; color: #cbd5e1; background: #1e293b; }}
    .pill.accent {{ background: #4f46e5; border-color: #6366f1; color: #fff; }}
    .pill.muted {{ color: #64748b; background: #0f172a; }}
    main {{ max-width: 1120px; margin: 24px auto; padding: 0 24px 64px; }}
    .group {{ margin-top: 28px; }}
    .group h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
    .op {{ display: flex; align-items: center; gap: 12px; padding: 9px 12px; border: 1px solid #1e293b; border-radius: 8px; margin-top: 8px; background: #0f172a; font-size: 13px; }}
    .method {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 5px; min-width: 54px; text-align: center; text-transform: uppercase; }}
    .GET {{ background: #0c4a6e; color: #7dd3fc; }}
    .POST {{ background: #064e3b; color: #6ee7b7; }}
    .PUT, .PATCH {{ background: #78350f; color: #fcd34d; }}
    .DELETE {{ background: #7f1d1d; color: #fca5a5; }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #f8fafc; font-weight: 500; }}
    .desc {{ color: #94a3b8; margin-left: auto; text-align: right; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{heading}</h1>
      <div class="sub">OpenAPI 3.1.0 &middot; Polyglot Inference + LiteLLM-Compatible &amp; Native Control Plane</div>
    </div>
    <div class="links">
      <a href="/openapi.json" class="pill">OpenAPI Spec (/openapi.json)</a>
      {ui_pill}
    </div>
  </header>
  <main id="spec-root"><p style="color:#94a3b8">Loading /openapi.json&hellip;</p></main>
  <script>
    fetch('/openapi.json').then(r => r.json()).then(spec => {{
      const groups = {{}};
      for (const [path, methods] of Object.entries(spec.paths || {{}})) {{
        for (const [method, info] of Object.entries(methods)) {{
          const tag = (info.tags && info.tags[0]) || 'General';
          (groups[tag] = groups[tag] || []).push({{ method: method.toUpperCase(), path, summary: info.summary || '' }});
        }}
      }}
      const root = document.getElementById('spec-root');
      root.innerHTML = '';
      for (const [tag, items] of Object.entries(groups)) {{
        const sec = document.createElement('section');
        sec.className = 'group';
        const h2 = document.createElement('h2');
        h2.textContent = tag + ' (' + items.length + ')';
        sec.appendChild(h2);
        for (const it of items) {{
          const row = document.createElement('div');
          row.className = 'op';
          row.innerHTML = '<span class="method ' + it.method + '">' + it.method + '</span><code class="path"></code><span class="desc"></span>';
          row.querySelector('.path').textContent = it.path;
          row.querySelector('.desc').textContent = it.summary;
          sec.appendChild(row);
        }}
        root.appendChild(sec);
      }}
    }});
  </script>
</body>
</html>"#
    );

    let mut resp = (StatusCode::OK, html).into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    resp
}
