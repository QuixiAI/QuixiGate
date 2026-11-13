//! Native Hosted Gateway MCP Server (`POST /mcp`, `GET /mcp`, `DELETE /mcp`)
//! implementing Model Context Protocol Streamable-HTTP JSON-RPC 2.0 (`2025-03-26` / `2024-11-05`)
//! so Claude Code, Cursor, OpenCode, Windsurf, and MCP clients can connect directly to QuixiGate
//! for SSRF-hardened web search/fetch, model discovery, spend reporting, and gateway inspection.

use std::convert::Infallible;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use quixigate_core::{IngressProtocol, NanoUsd, QuixigateError, RequestLog, UserRole, now_ms};
use quixigate_db::{AuthenticatedPrincipal, LogFilter};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::{
    extract_request_tags, is_catalog_model_routable, should_suppress_request_log,
};
use crate::state::{ApiError, ApiResult, AppState};

const DEFAULT_PROTOCOL_VERSION: &str = "2025-03-26";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

fn ensure_mcp_enabled(state: &AppState) -> quixigate_core::Result<()> {
    let settings = state.db.get_settings()?;
    if !settings.mcp_server_enabled {
        return Err(QuixigateError::Forbidden(
            "hosted MCP server endpoint (/mcp) is disabled in gateway settings".to_owned(),
        ));
    }
    Ok(())
}

fn can_read_spend(_state: &AppState, principal: &AuthenticatedPrincipal) -> bool {
    if principal.user.role == UserRole::Admin {
        return true;
    }
    if let Some(vk) = &principal.virtual_key {
        return vk.allow_spend_read;
    }
    true
}

fn is_full_admin(principal: &AuthenticatedPrincipal) -> bool {
    principal.user.role == UserRole::Admin
}

fn resolve_session_id(headers: &HeaderMap) -> String {
    if let Some(existing) = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty() && s.len() <= 128 && s.chars().all(|c| c.is_ascii_graphic()))
    {
        return existing.to_owned();
    }
    format!("mcp_{}", Uuid::new_v4().simple())
}

/// `GET /mcp` — Streamable-HTTP SSE listener (when `Accept: text/event-stream`) or server metadata JSON.
pub async fn handle_mcp_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Response> {
    ensure_mcp_enabled(&state)?;
    let _principal = authenticate_inference(&state, &headers, None)?;
    let session_id = resolve_session_id(&headers);

    let wants_sse = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/event-stream"));

    if wants_sse {
        let initial = futures::stream::once(async {
            Ok::<Event, Infallible>(Event::default().comment("quixigate-mcp-ready"))
        });
        let mut resp = Sse::new(initial)
            .keep_alive(KeepAlive::default())
            .into_response();
        if let Ok(hv) = HeaderValue::from_str(&session_id) {
            resp.headers_mut().insert("mcp-session-id", hv);
        }
        return Ok(resp);
    }

    let mut resp = Json(json!({
        "name": quixigate_core::brand::BRAND_SLUG,
        "title": format!("{} Hosted MCP Server", quixigate_core::brand::BRAND_NAME),
        "version": env!("CARGO_PKG_VERSION"),
        "protocol_versions": SUPPORTED_PROTOCOL_VERSIONS,
        "transport": "streamable-http",
        "endpoint": "/mcp",
    }))
    .into_response();
    if let Ok(hv) = HeaderValue::from_str(&session_id) {
        resp.headers_mut().insert("mcp-session-id", hv);
    }
    Ok(resp)
}

/// `DELETE /mcp` — Streamable-HTTP session close acknowledgement (`204 No Content`).
pub async fn handle_mcp_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Response> {
    ensure_mcp_enabled(&state)?;
    let _principal = authenticate_inference(&state, &headers, None)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// `POST /mcp` — JSON-RPC 2.0 Streamable-HTTP dispatcher (single message or batch array).
pub async fn handle_mcp_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> ApiResult<Response> {
    ensure_mcp_enabled(&state)?;
    let principal = authenticate_inference(&state, &headers, None)?;
    let session_id = resolve_session_id(&headers);

    let mut resp = match payload {
        Value::Array(items) => {
            if items.is_empty() {
                return Err(ApiError(QuixigateError::Validation(
                    "JSON-RPC batch array cannot be empty".to_owned(),
                )));
            }
            let mut responses = Vec::new();
            for item in items {
                if let Some(r) = dispatch_jsonrpc_item(&state, &principal, &headers, item).await {
                    responses.push(r);
                }
            }
            if responses.is_empty() {
                StatusCode::ACCEPTED.into_response()
            } else {
                Json(Value::Array(responses)).into_response()
            }
        }
        single => match dispatch_jsonrpc_item(&state, &principal, &headers, single).await {
            Some(reply) => Json(reply).into_response(),
            None => StatusCode::ACCEPTED.into_response(),
        },
    };

    if let Ok(hv) = HeaderValue::from_str(&session_id) {
        resp.headers_mut().insert("mcp-session-id", hv);
    }
    Ok(resp)
}

fn jsonrpc_ok(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn jsonrpc_err(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        }
    })
}

async fn dispatch_jsonrpc_item(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    headers: &HeaderMap,
    msg: Value,
) -> Option<Value> {
    let Some(obj) = msg.as_object() else {
        return Some(jsonrpc_err(
            Value::Null,
            -32600,
            "invalid JSON-RPC request object",
        ));
    };
    let id = obj.get("id").cloned();
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        return Some(jsonrpc_err(
            id.unwrap_or(Value::Null),
            -32600,
            "missing JSON-RPC 'method' field",
        ));
    };

    // JSON-RPC notifications omit `id` and receive HTTP 202 with no response object.
    let req_id = id?;
    let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));

    match method {
        "initialize" => {
            let requested_ver = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_PROTOCOL_VERSION);
            let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested_ver) {
                requested_ver
            } else {
                DEFAULT_PROTOCOL_VERSION
            };
            Some(jsonrpc_ok(
                req_id,
                json!({
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": { "listChanged": false },
                        "resources": { "listChanged": false },
                        "prompts": { "listChanged": false },
                    },
                    "serverInfo": {
                        "name": quixigate_core::brand::BRAND_SLUG,
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "instructions": format!(
                        "{} hosted MCP server providing SSRF-hardened web search/fetch, live model catalog discovery, and gateway spend telemetry.",
                        quixigate_core::brand::BRAND_NAME
                    ),
                }),
            ))
        }
        "ping" => Some(jsonrpc_ok(req_id, json!({}))),
        "resources/list" => Some(jsonrpc_ok(req_id, json!({ "resources": [] }))),
        "resources/templates/list" => Some(jsonrpc_ok(req_id, json!({ "resourceTemplates": [] }))),
        "prompts/list" => Some(jsonrpc_ok(req_id, json!({ "prompts": [] }))),
        "tools/list" => Some(jsonrpc_ok(
            req_id,
            json!({ "tools": build_tools_catalog(state, principal) }),
        )),
        "tools/call" => Some(execute_tool_call(state, principal, headers, req_id, &params).await),
        other => Some(jsonrpc_err(
            req_id,
            -32601,
            format!("method '{other}' not found"),
        )),
    }
}

fn build_tools_catalog(state: &AppState, principal: &AuthenticatedPrincipal) -> Vec<Value> {
    let mut tools = vec![
        json!({
            "name": "quixigate_list_models",
            "description": "List routable LLM models, smart aliases, local runtimes, and capabilities permitted for this caller.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional substring filter on model or provider ID (e.g. 'claude', 'ollama', 'qwen')."
                    }
                }
            }
        }),
        json!({
            "name": "quixigate_web_search",
            "description": "Search the live web using QuixiGate's zero-config multi-provider cascade (Exa MCP, Parallel, Brave, Tavily, Serper, Bing, DuckDuckGo) with SSRF protection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (1-10, default 5)."
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "quixigate_web_fetch",
            "description": "Fetch and extract clean readable text from a public HTTP/HTTPS URL with private-IP/metadata SSRF blocking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Public https:// or http:// URL to read."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum text characters to return (default 12000, max 40000)."
                    }
                },
                "required": ["url"]
            }
        }),
    ];

    if can_read_spend(state, principal) {
        tools.push(json!({
            "name": "quixigate_spend_report",
            "description": "Aggregate exact NanoUsd spend, OAuth/local savings, requests, and tokens grouped by model, provider, tag, key, team, user, or day.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["model", "provider", "tag", "key", "team", "user", "day"],
                        "description": "Dimension to group spend rows by (default: 'model')."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional tag filter."
                    }
                }
            }
        }));
    }

    if is_full_admin(principal) {
        tools.push(json!({
            "name": "quixigate_gateway_health",
            "description": "Admin-only: inspect live gateway health, active credentials, cache hit stats, and 24h spend/savings summary.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }));
        tools.push(json!({
            "name": "quixigate_list_keys",
            "description": "Admin-only: list virtual keys (redacted prefixes, tags, allowed models, and spend vs budget).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }));
        tools.push(json!({
            "name": "quixigate_audit_logs",
            "description": "Admin-only: query immutable management audit logs (actor, action, table, object_id, diff).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum audit rows to return (default 25, max 200)."
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Optional table filter (e.g. 'keys', 'users', 'teams', 'routes', 'settings')."
                    },
                    "action": {
                        "type": "string",
                        "description": "Optional action filter (e.g. 'create', 'update', 'delete', 'block')."
                    }
                }
            }
        }));
    }

    tools
}

async fn execute_tool_call(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    headers: &HeaderMap,
    req_id: Value,
    params: &Value,
) -> Value {
    let Some(tool_name) = params.get("name").and_then(Value::as_str).map(str::trim) else {
        return jsonrpc_err(req_id, -32602, "missing 'params.name' in tools/call");
    };
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let started = Instant::now();
    let result: std::result::Result<Value, String> = match tool_name {
        "quixigate_list_models" => run_list_models_tool(state, principal, &args),
        "quixigate_web_search" => {
            let Some(query) = args
                .get("query")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return jsonrpc_err(
                    req_id,
                    -32602,
                    "quixigate_web_search requires non-empty 'query'",
                );
            };
            let max_results = args
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .clamp(1, 10) as usize;
            state.router.execute_web_search(query, max_results).await
        }
        "quixigate_web_fetch" => {
            let Some(url) = args
                .get("url")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return jsonrpc_err(
                    req_id,
                    -32602,
                    "quixigate_web_fetch requires non-empty 'url'",
                );
            };
            let max_chars = args
                .get("max_chars")
                .and_then(Value::as_u64)
                .unwrap_or(12_000)
                .clamp(500, 40_000) as usize;
            state.router.execute_web_fetch(url, max_chars).await
        }
        "quixigate_spend_report" => run_spend_report_tool(state, principal, &args),
        "quixigate_gateway_health" => {
            if !is_full_admin(principal) {
                Err("administrator privileges required for quixigate_gateway_health".to_owned())
            } else {
                run_gateway_health_tool(state)
            }
        }
        "quixigate_list_keys" => {
            if !is_full_admin(principal) {
                Err("administrator privileges required for quixigate_list_keys".to_owned())
            } else {
                run_list_keys_tool(state)
            }
        }
        "quixigate_audit_logs" => {
            if !is_full_admin(principal) {
                Err("administrator privileges required for quixigate_audit_logs".to_owned())
            } else {
                run_audit_logs_tool(state, &args)
            }
        }
        unknown => {
            return jsonrpc_err(req_id, -32602, format!("unknown MCP tool '{unknown}'"));
        }
    };

    let duration_ms = started.elapsed().as_millis() as u64;
    let settings = state.db.get_settings().unwrap_or_default();
    if !should_suppress_request_log(headers, &settings) {
        let mut tags = extract_request_tags(headers, None);
        if let Some(vk) = &principal.virtual_key {
            for t in &vk.tags {
                if !tags.contains(t) {
                    tags.push(t.clone());
                }
            }
        }
        let (status_code, err_msg) = match &result {
            Ok(_) => (200u16, None),
            Err(e) => (400u16, Some(e.clone())),
        };
        let log = RequestLog {
            id: format!("mcp_{}", Uuid::new_v4().simple()),
            timestamp_ms: now_ms(),
            ingress_protocol: IngressProtocol::Mcp,
            requested_model: format!("mcp/{tool_name}"),
            resolved_model: format!("mcp/{tool_name}"),
            provider_id: "mcp".to_owned(),
            credential_id: None,
            user_id: Some(principal.user.id.clone()),
            team_id: principal.effective_team_id(),
            virtual_key_id: principal.virtual_key.as_ref().map(|k| k.id.clone()),
            tags,
            status_code,
            stream: false,
            fallback_attempts: 0,
            cache_hit: false,
            ttft_ms: None,
            duration_ms,
            input_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
            output_tokens: 0,
            cost_nanos: NanoUsd::ZERO,
            api_equiv_nanos: NanoUsd::ZERO,
            otel_trace_id: None,
            error_message: err_msg,
            request_body_json: None,
            response_body_json: None,
        };
        let _ = state.db.insert_request_log(&log);
    }

    match result {
        Ok(val) => {
            let pretty = serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string());
            jsonrpc_ok(
                req_id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": pretty,
                    }],
                    "structuredContent": val,
                    "isError": false,
                }),
            )
        }
        Err(err_text) => jsonrpc_ok(
            req_id,
            json!({
                "content": [{
                    "type": "text",
                    "text": err_text,
                }],
                "isError": true,
            }),
        ),
    }
}

fn run_list_models_tool(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    args: &Value,
) -> std::result::Result<Value, String> {
    let substr = args
        .get("filter")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());

    let routes = state.db.list_model_routes().map_err(|e| e.to_string())?;
    let creds = state.db.list_credentials().map_err(|e| e.to_string())?;
    let configured_providers: std::collections::HashSet<String> = creds
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.provider_id.clone())
        .collect();
    let allowed = principal
        .virtual_key
        .as_ref()
        .map(|k| k.allowed_models.clone())
        .unwrap_or_default();

    let mut items = Vec::new();
    for r in routes {
        if let Some(route_tid) = &r.team_id
            && principal.user.role != UserRole::Admin
        {
            let in_team = match &principal.virtual_key {
                Some(vk) => vk.team_id.as_deref() == Some(route_tid.as_str()),
                None => state
                    .db
                    .is_team_member(route_tid, &principal.user.id)
                    .unwrap_or(false),
            };
            if !in_team {
                continue;
            }
        }
        if let Some(f) = &substr
            && !r.alias.to_ascii_lowercase().contains(f)
        {
            continue;
        }
        items.push(json!({
            "id": r.alias,
            "kind": "smart_route",
            "team_id": r.team_id,
            "pool_strategy": r.pool_strategy,
            "targets": r.targets.len(),
        }));
    }

    for m in state.router.list_available_models() {
        if !is_catalog_model_routable(&m, &configured_providers) {
            continue;
        }
        if !allowed.is_empty()
            && !allowed.iter().any(|pat| {
                pat == "*"
                    || pat == &m.canonical_id
                    || pat == &m.bare_id
                    || (pat.ends_with('*') && m.canonical_id.starts_with(pat.trim_end_matches('*')))
            })
        {
            continue;
        }
        if let Some(f) = &substr
            && !m.canonical_id.to_ascii_lowercase().contains(f)
            && !m.author.to_ascii_lowercase().contains(f)
        {
            continue;
        }
        let first_provider = m
            .providers
            .keys()
            .next()
            .cloned()
            .unwrap_or_else(|| m.author.clone());
        items.push(json!({
            "id": m.canonical_id,
            "kind": if m.is_local { "local" } else { "provider" },
            "provider_id": first_provider,
            "supports_reasoning": m.supports_reasoning,
            "supports_tools": m.supports_tools,
            "context_window": m.context_window,
        }));
    }

    Ok(json!({
        "count": items.len(),
        "models": items,
    }))
}

fn run_spend_report_tool(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    args: &Value,
) -> std::result::Result<Value, String> {
    if !can_read_spend(state, principal) {
        return Err("spend report permission denied for this credential".to_owned());
    }
    let group_by = args
        .get("group_by")
        .and_then(Value::as_str)
        .unwrap_or("model");
    let mut filter = LogFilter {
        tag: args.get("tag").and_then(Value::as_str).map(str::to_owned),
        ..Default::default()
    };
    if principal.user.role != UserRole::Admin {
        if let Some(vk) = &principal.virtual_key {
            filter.virtual_key_id = Some(vk.id.clone());
        } else {
            let admin_teams = state
                .db
                .list_admin_team_ids(&principal.user.id)
                .map_err(|e| e.to_string())?;
            if let Some(first_team) = admin_teams.first() {
                filter.team_id = Some(first_team.clone());
            } else {
                filter.user_id = Some(principal.user.id.clone());
            }
        }
    }
    let rows = state
        .db
        .get_spend_report(group_by, &filter)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "group_by": group_by,
        "rows": rows,
    }))
}

fn run_gateway_health_tool(state: &AppState) -> std::result::Result<Value, String> {
    let metrics = state
        .db
        .get_dashboard_metrics()
        .map_err(|e| e.to_string())?;
    let creds = state.db.list_credentials().map_err(|e| e.to_string())?;
    let (memory_entries, sqlite_entries) = state.router.cache().entry_counts();
    Ok(json!({
        "status": "healthy",
        "uptime_ms": (now_ms() - state.started_at_ms).max(0),
        "credentials_total": creds.len(),
        "credentials_enabled": creds.iter().filter(|c| c.enabled).count(),
        "cache": {
            "memory_entries": memory_entries,
            "sqlite_entries": sqlite_entries,
        },
        "metrics_24h": metrics,
    }))
}

fn run_list_keys_tool(state: &AppState) -> std::result::Result<Value, String> {
    let keys = state.db.list_virtual_keys().map_err(|e| e.to_string())?;
    let sanitized: Vec<Value> = keys
        .into_iter()
        .map(|k| {
            json!({
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "user_id": k.user_id,
                "team_id": k.team_id,
                "tags": k.tags,
                "revoked": k.revoked_at_ms.is_some(),
                "spend_usd": k.spend_nanos.to_decimal_string(),
                "budget_limit_usd": k.budget_limit_nanos.map(|b| b.to_decimal_string()),
            })
        })
        .collect();
    Ok(json!({
        "count": sanitized.len(),
        "keys": sanitized,
    }))
}

fn run_audit_logs_tool(state: &AppState, args: &Value) -> std::result::Result<Value, String> {
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(25)
        .clamp(1, 200) as usize;
    let table_name = args.get("table_name").and_then(Value::as_str);
    let action = args.get("action").and_then(Value::as_str);
    let logs = state
        .db
        .list_audit_logs(table_name, action, None, None, limit, 0)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "count": logs.len(),
        "audit_logs": logs,
    }))
}
