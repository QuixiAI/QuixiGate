//! Complete `/api/admin/*` control-plane REST & SSE endpoints powering the
//! embedded QuixiGate Web UI and CLI operator commands.

use std::convert::Infallible;
use std::time::Instant;

use aisdk::core::{Message, UserMessage};
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use quixigate_auth::StartOAuthRequest;
use quixigate_auth::providers::{
    OPENROUTER_OIDC_TOKEN_URL, OpenRouterAccountInfo, exchange_openrouter_oidc_token,
    inspect_openrouter_account, openrouter_base_url_for_region, validate_anthropic_setup_token,
};
use quixigate_auth::quota::poll_and_save_quota;
use quixigate_core::{
    AuditLogEntry, AuthKind, BudgetWindow, CredentialRecord, CredentialSecret, GatewaySettings,
    LocalEndpointConfig, ModelRoute, NanoUsd, OtlpProtocol, QuixigateError, PoolStrategy,
    RouteTarget, SpendReportRow, User, UserRole, VirtualKey, now_ms,
};
use quixigate_db::{AuthenticatedPrincipal, Database, LogFilter};
use quixigate_providers::{
    AISDK_BUILTIN_PROVIDERS, LocalEndpointSpec, LocalRuntimeScanner, normalize_local_runtime_id,
};
use quixigate_router::CanonicalRequest;
use quixigate_router::catalog::MODELS_DEV_URL;
use quixigate_router::tools::fetch_mcp_http_tools;
use quixigate_telemetry::TestSpanResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;

use crate::auth::{
    authenticate_admin, authenticate_inference, authenticate_team_admin_or_global,
    extract_api_token, extract_session_cookie, format_clear_session_cookie, format_session_cookie,
    is_direct_loopback,
};
use crate::state::{ApiError, ApiResult, AppState};

const SESSION_COOKIE_MAX_AGE_SECS: u64 = 30 * 24 * 3600;

const LOCAL_API_PROVIDERS: &[(&str, &str)] = &[
    ("llamacpp", "llama.cpp Server (local / --api-key)"),
    ("ollama", "Ollama (local / authenticated)"),
    ("mlx", "Apple MLX Server (local)"),
    ("vllm", "vLLM Server (local / --api-key)"),
];

/// Synchronizes persisted `custom_local_endpoints` and any stored local-runtime
/// API keys / custom `base_url` overrides from SQLite into `LocalRuntimeScanner`.
pub fn sync_local_scanner_from_db(db: &Database, scanner: &LocalRuntimeScanner) {
    if let Ok(settings) = db.get_settings() {
        scanner.apply_custom_endpoints(&settings.custom_local_endpoints);
    }
    for id in ["llamacpp", "ollama", "lmstudio", "mlx", "vllm"] {
        scanner.set_api_key(id, None);
    }
    if let Ok(creds) = db.list_credentials() {
        for c in creds.into_iter().filter(|c| c.enabled) {
            let norm = normalize_local_runtime_id(&c.provider_id);
            if matches!(
                norm.as_str(),
                "llamacpp" | "ollama" | "lmstudio" | "mlx" | "vllm"
            ) {
                if let Some(url) = c.base_url.as_deref().filter(|u| !u.trim().is_empty()) {
                    let display = match norm.as_str() {
                        "llamacpp" => "llama.cpp Server",
                        "ollama" => "Ollama",
                        "lmstudio" => "LM Studio",
                        "mlx" => "Apple MLX",
                        "vllm" => "vLLM",
                        _ => &c.label,
                    };
                    scanner.add_or_update_endpoint(LocalEndpointSpec {
                        runtime_id: norm.clone(),
                        display_name: display.to_owned(),
                        base_url: url.trim().to_owned(),
                    });
                }
                if let Ok((_, CredentialSecret::ApiKey { key })) =
                    db.get_credential_with_secret(&c.id)
                {
                    scanner.set_api_key(&norm, Some(key));
                }
            }
        }
    }
}

fn parse_optional_usd(raw: Option<&str>) -> quixigate_core::Result<Option<NanoUsd>> {
    match raw.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => NanoUsd::parse_usd(s).map(Some),
        None => Ok(None),
    }
}

/// Parses per-model max budget maps (`{"gpt-4o": 5.0}`, `{"gpt-4o": "5.00"}`, or LiteLLM nested
/// `{"gpt-4o": {"max_budget": 5.0, "budget_limit": 5.0}}`) into exact `BTreeMap<String, NanoUsd>`.
pub fn parse_model_budget_map(
    val: Option<&Value>,
) -> quixigate_core::Result<Option<std::collections::BTreeMap<String, NanoUsd>>> {
    let Some(v) = val else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(Some(std::collections::BTreeMap::new()));
    }
    let Some(obj) = v.as_object() else {
        return Err(QuixigateError::Validation(
            "model_max_budget must be a JSON object mapping model ID/glob to USD limit".to_owned(),
        ));
    };
    let mut out = std::collections::BTreeMap::new();
    for (model_pat, entry) in obj {
        let pat = model_pat.trim();
        if pat.is_empty() || entry.is_null() {
            continue;
        }
        let nanos = if let Some(inner_obj) = entry.as_object() {
            if let Some(bn) = inner_obj.get("budget_nanos").and_then(Value::as_i64) {
                NanoUsd(bn)
            } else if let Some(mb) = inner_obj
                .get("max_budget")
                .or_else(|| inner_obj.get("budget_limit"))
            {
                if let Some(s) = mb.as_str() {
                    NanoUsd::parse_usd(s)?
                } else {
                    NanoUsd::parse_usd(&mb.to_string())?
                }
            } else {
                continue;
            }
        } else if let Some(s) = entry.as_str() {
            NanoUsd::parse_usd(s)?
        } else if entry.is_number() {
            NanoUsd::parse_usd(&entry.to_string())?
        } else {
            return Err(QuixigateError::Validation(format!(
                "invalid model_max_budget entry for '{pat}': {entry}"
            )));
        };
        out.insert(pat.to_owned(), nanos);
    }
    Ok(Some(out))
}

/// Appends an immutable management audit record (automatically redacting sensitive fields in `quixigate-db`).
pub fn record_audit(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    action: &str,
    table_name: &str,
    object_id: &str,
    before_value: Option<Value>,
    updated_values: Option<Value>,
) {
    let role_str = match principal.user.role {
        UserRole::Admin => "admin",
        UserRole::Member => "member",
        UserRole::Viewer => "viewer",
    };
    let _ = state.db.insert_audit_log(
        &principal.user.id,
        role_str,
        action,
        table_name,
        object_id,
        before_value,
        updated_values,
    );
}

// ── 1. Auth & Operator Session Endpoints ─────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct AuthStatusResponse {
    pub require_auth: bool,
    pub ui_enabled: bool,
    pub master_key_configured: bool,
    pub bound_loopback: bool,
    pub direct_loopback: bool,
    pub has_password_user: bool,
    pub authenticated: bool,
    pub via_loopback_trust: bool,
    pub via_master_key: bool,
    pub user: Option<User>,
}

pub async fn get_auth_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<AuthStatusResponse>> {
    let settings = state.db.get_settings()?;
    let has_password_user = state.db.has_any_password_user()?;
    let direct_loopback = is_direct_loopback(&state, &headers);
    let ui_enabled = state.is_ui_enabled();
    let master_key_configured = state.has_master_key();
    match authenticate_inference(&state, &headers, None) {
        Ok(p) => Ok(Json(AuthStatusResponse {
            require_auth: settings.require_auth,
            ui_enabled,
            master_key_configured,
            bound_loopback: direct_loopback,
            direct_loopback,
            has_password_user,
            authenticated: true,
            via_loopback_trust: p.via_loopback_trust,
            via_master_key: p.via_master_key,
            user: Some(p.user),
        })),
        Err(_) => Ok(Json(AuthStatusResponse {
            require_auth: settings.require_auth,
            ui_enabled,
            master_key_configured,
            bound_loopback: direct_loopback,
            direct_loopback,
            has_password_user,
            authenticated: false,
            via_loopback_trust: false,
            via_master_key: false,
            user: None,
        })),
    }
}

#[derive(Debug, Deserialize)]
pub struct BootstrapAuthRequest {
    pub username: Option<String>,
    pub password: String,
}

pub async fn post_auth_bootstrap(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BootstrapAuthRequest>,
) -> ApiResult<Response> {
    if state.db.has_any_password_user()? {
        return Err(ApiError(QuixigateError::Forbidden(
            "an operator password is already configured; sign in instead".to_owned(),
        )));
    }
    let has_admin_auth = authenticate_admin(&state, &headers, true).is_ok();
    if !is_direct_loopback(&state, &headers) && !has_admin_auth {
        return Err(ApiError(QuixigateError::Forbidden(
            "initial password bootstrap from remote IP requires Master Key".to_owned(),
        )));
    }
    if req.password.trim().len() < 6 {
        return Err(ApiError(QuixigateError::Validation(
            "password must be at least 6 characters".to_owned(),
        )));
    }

    let user = if let Some(uname) = req
        .username
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty() && *u != "local")
    {
        state.db.create_user(
            uname,
            uname,
            UserRole::Admin,
            Some(req.password.trim()),
            None,
            BudgetWindow::Monthly,
        )?
    } else {
        state
            .db
            .update_user("local", None, None, Some(req.password.trim()), None, None)?
    };

    let token = state.db.create_session(&user.id)?;
    let cookie = format_session_cookie(&token, SESSION_COOKIE_MAX_AGE_SECS, false);
    let mut resp = Json(json!({ "ok": true, "token": token, "user": user })).into_response();
    if let Ok(hv) = HeaderValue::from_str(&cookie) {
        resp.headers_mut().insert(header::SET_COOKIE, hv);
    }
    Ok(resp)
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

pub async fn post_auth_login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> ApiResult<Response> {
    let (user, token) = if state.verify_master_key(&req.password) {
        let target = req.username.trim();
        let mut usr = if !target.is_empty() {
            state
                .db
                .get_user(target)
                .or_else(|_| state.db.get_user("local"))?
        } else {
            state.db.get_user("local")?
        };
        usr.role = UserRole::Admin;
        let tok = state.db.create_session(&usr.id)?;
        (usr, tok)
    } else {
        state
            .db
            .authenticate_password(&req.username, &req.password)?
    };
    let cookie = format_session_cookie(&token, SESSION_COOKIE_MAX_AGE_SECS, false);
    let mut resp = Json(json!({ "ok": true, "token": token, "user": user })).into_response();
    if let Ok(hv) = HeaderValue::from_str(&cookie) {
        resp.headers_mut().insert(header::SET_COOKIE, hv);
    }
    Ok(resp)
}

pub async fn post_auth_logout(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Response> {
    if let Some(tok) = extract_session_cookie(&headers) {
        let _ = state.db.revoke_session(&tok);
    }
    if let Some(tok) = extract_api_token(&headers, None)
        && quixigate_core::brand::is_session_token(&tok)
    {
        let _ = state.db.revoke_session(&tok);
    }
    let mut resp = Json(json!({ "ok": true })).into_response();
    if let Ok(hv) = HeaderValue::from_str(&format_clear_session_cookie()) {
        resp.headers_mut().insert(header::SET_COOKIE, hv);
    }
    Ok(resp)
}

#[derive(Debug, Deserialize)]
pub struct SetPasswordRequest {
    pub password: String,
}

pub async fn post_auth_password(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetPasswordRequest>,
) -> ApiResult<Json<User>> {
    let principal = authenticate_admin(&state, &headers, false)?;
    if req.password.trim().len() < 6 {
        return Err(ApiError(QuixigateError::Validation(
            "password must be at least 6 characters".to_owned(),
        )));
    }
    let updated = state.db.update_user(
        &principal.user.id,
        None,
        None,
        Some(req.password.trim()),
        None,
        None,
    )?;
    Ok(Json(updated))
}

// ── 2. Command Center Overview ───────────────────────────────────────────────

pub async fn get_overview(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let metrics = state.db.get_dashboard_metrics()?;
    let credentials = state.db.list_credentials()?;
    let local_runtimes = state.router.local_scanner().list_statuses();
    let routes = state.db.list_model_routes()?;
    let settings = state.db.get_settings()?;
    let recent_traces = state.db.list_request_logs(&LogFilter {
        limit: 12,
        ..Default::default()
    })?;
    let openrouter_connected = credentials
        .iter()
        .any(|c| c.enabled && c.provider_id == "openrouter");
    let openrouter_presets_count = if openrouter_connected {
        state.router.catalog().list_presets().len()
    } else {
        0
    };

    Ok(Json(json!({
        "uptime_ms": (now_ms() - state.started_at_ms).max(0),
        "bound_loopback": is_direct_loopback(&state, &headers),
        "ui_enabled": state.is_ui_enabled(),
        "master_key_configured": state.has_master_key(),
        "metrics": metrics,
        "credentials": credentials,
        "local_runtimes": local_runtimes,
        "routes": routes,
        "settings": settings,
        "recent_traces": recent_traces,
        "openrouter_connected": openrouter_connected,
        "openrouter_presets_count": openrouter_presets_count,
    })))
}

pub async fn get_doctor(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let now = now_ms();
    let settings = state.db.get_settings()?;
    let creds = state.db.list_credentials()?;
    let healthy_seats = creds.iter().filter(|c| c.is_healthy_for(None, now)).count();
    let dead_seats = creds
        .iter()
        .filter(|c| c.status == quixigate_core::CredentialStatus::Dead)
        .count();
    let routes = state.db.list_model_routes()?;
    let local_runtimes = state.router.local_scanner().list_statuses();
    let healthy_locals = local_runtimes.iter().filter(|r| r.healthy).count();
    let (cache_ram, cache_sqlite) = state.router.cache().entry_counts();
    let metrics = state.db.get_dashboard_metrics()?;

    Ok(Json(json!({
        "ok": dead_seats == 0,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_ms": (now - state.started_at_ms).max(0),
        "db_path": state.db.db_path().display().to_string(),
        "vault_source": state.db.vault().source().as_str(),
        "ui_enabled": state.is_ui_enabled(),
        "master_key_configured": state.has_master_key(),
        "require_auth": settings.require_auth,
        "active_requests": metrics.active_requests,
        "cache": {
            "enabled_default": settings.cache_enabled_default,
            "ttl_secs": settings.cache_ttl_secs,
            "ram_entries": cache_ram,
            "sqlite_entries": cache_sqlite,
        },
        "seats": {
            "total": creds.len(),
            "healthy": healthy_seats,
            "dead": dead_seats,
        },
        "routes_count": routes.len(),
        "local_runtimes": {
            "enabled": settings.local_discovery_enabled,
            "healthy": healthy_locals,
            "statuses": local_runtimes,
        },
        "mcp_servers": settings.mcp_server_urls,
        "otlp": {
            "enabled": settings.otlp_enabled,
            "endpoint": settings.otlp_endpoint,
            "protocol": settings.otlp_protocol,
            "project_name": settings.openinference_project_name,
        }
    })))
}

pub async fn get_spend_summary(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let metrics = state.db.get_dashboard_metrics()?;
    let users = state.db.list_users()?;
    let teams = state.db.list_teams()?;
    let keys = state.db.list_virtual_keys()?;
    Ok(Json(json!({
        "metrics": metrics,
        "users": users,
        "teams": teams,
        "keys": keys,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct ResetSpendRequest {
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub virtual_key_id: Option<String>,
    #[serde(default)]
    pub all: bool,
}

pub async fn post_spend_reset(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<ResetSpendRequest>>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let req = body.map(|Json(b)| b).unwrap_or(ResetSpendRequest {
        all: true,
        ..Default::default()
    });
    state.db.reset_spend(
        req.user_id.as_deref(),
        req.team_id.as_deref(),
        req.virtual_key_id.as_deref(),
        req.all,
    )?;
    Ok(Json(json!({ "ok": true })))
}

// ── 3. Providers, Credentials & 9 OAuth Wizards ──────────────────────────────

pub async fn get_providers_catalog(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let mut api_providers: Vec<Value> = LOCAL_API_PROVIDERS
        .iter()
        .chain(AISDK_BUILTIN_PROVIDERS.iter())
        .map(|(id, name)| json!({ "provider_id": id, "display_name": name }))
        .collect();
    // Deduplicate in case a provider_id appears in both lists
    let mut seen = std::collections::HashSet::new();
    api_providers.retain(|v| {
        v["provider_id"]
            .as_str()
            .is_some_and(|id| seen.insert(id.to_owned()))
    });
    Ok(Json(json!({
        "oauth_providers": state.oauth_wizard.catalog(),
        "api_providers": api_providers,
    })))
}

pub async fn get_credentials(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<CredentialRecord>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.list_credentials()?))
}

#[derive(Debug, Deserialize)]
pub struct UpsertCredentialRequest {
    pub id: Option<String>,
    pub provider_id: String,
    pub label: String,
    #[serde(default = "default_api_key_kind")]
    pub auth_kind: AuthKind,
    pub api_key: Option<String>,
    pub setup_token: Option<String>,
    pub base_url: Option<String>,
    pub priority: Option<i32>,
    pub enabled: Option<bool>,
    pub allow_api_overflow: Option<bool>,
}

const fn default_api_key_kind() -> AuthKind {
    AuthKind::ApiKey
}

pub async fn post_credential(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UpsertCredentialRequest>,
) -> ApiResult<Json<CredentialRecord>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let secret = match req.auth_kind {
        AuthKind::ApiKey => {
            let key = req
                .api_key
                .as_deref()
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .ok_or_else(|| QuixigateError::Validation("API key cannot be empty".to_owned()))?;
            CredentialSecret::ApiKey {
                key: key.to_owned(),
            }
        }
        AuthKind::SetupToken => {
            let tok = req.setup_token.as_deref().unwrap_or_default();
            validate_anthropic_setup_token(tok)?
        }
        AuthKind::LocalNoauth => CredentialSecret::LocalNoauth,
        AuthKind::Oauth => {
            return Err(ApiError(QuixigateError::Validation(
                "use /api/admin/oauth/start to enroll OAuth subscription seats".to_owned(),
            )));
        }
    };

    let norm_provider = normalize_local_runtime_id(&req.provider_id);
    let is_local = matches!(
        norm_provider.as_str(),
        "llamacpp" | "ollama" | "lmstudio" | "mlx" | "vllm" | "local"
    );
    let effective_provider = if is_local {
        norm_provider.as_str()
    } else {
        req.provider_id.trim()
    };
    let default_prio = if req.auth_kind.is_zero_marginal_cost() || is_local {
        50
    } else {
        200
    };
    let rec = state.db.upsert_credential(
        req.id.as_deref(),
        effective_provider,
        &req.label,
        req.auth_kind,
        req.base_url.as_deref(),
        req.priority.unwrap_or(default_prio),
        req.enabled.unwrap_or(true),
        req.allow_api_overflow.unwrap_or(true),
        &secret,
    )?;
    record_audit(
        &state,
        &principal,
        if req.id.is_some() { "update" } else { "create" },
        "credentials",
        &rec.id,
        None,
        serde_json::to_value(&rec).ok(),
    );
    if is_local {
        sync_local_scanner_from_db(&state.db, state.router.local_scanner());
        state.router.local_scanner().scan_all().await;
    } else if effective_provider == "openrouter" && rec.enabled {
        let _ = sync_openrouter_for_state(&state).await;
        if let Ok(refreshed) = state.db.get_credential(&rec.id) {
            return Ok(Json(refreshed));
        }
    }
    Ok(Json(rec))
}

pub async fn get_credential_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<CredentialRecord>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.get_credential(&id)?))
}

#[derive(Debug, Deserialize)]
pub struct PatchCredentialRequest {
    pub label: Option<String>,
    pub base_url: Option<Option<String>>,
    pub priority: Option<i32>,
    pub enabled: Option<bool>,
    pub allow_api_overflow: Option<bool>,
    pub api_key: Option<String>,
    pub setup_token: Option<String>,
}

pub async fn patch_credential(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PatchCredentialRequest>,
) -> ApiResult<Json<CredentialRecord>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    if let Some(raw_key) = req.api_key.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
        state.db.update_credential_secret(
            &id,
            &CredentialSecret::ApiKey {
                key: raw_key.to_owned(),
            },
            None,
        )?;
    } else if let Some(raw_tok) = req
        .setup_token
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        let secret = validate_anthropic_setup_token(raw_tok)?;
        state.db.update_credential_secret(&id, &secret, None)?;
    }
    let base_ref = req.base_url.as_ref().map(|opt| opt.as_deref());
    let updated = state.db.update_credential_meta(
        &id,
        req.label.as_deref(),
        base_ref,
        req.priority,
        req.enabled,
        req.allow_api_overflow,
    )?;
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    reconcile_openrouter_presets_on_cred_change(&state);
    if updated.provider_id == "openrouter" && updated.enabled {
        let _ = sync_openrouter_for_state(&state).await;
    }
    Ok(Json(updated))
}

pub async fn post_refresh_credential_token(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<CredentialRecord>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let (record, _secret) = state.router.refresher().ensure_fresh(&id).await?;
    Ok(Json(record))
}

pub async fn delete_credential(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let before = state
        .db
        .get_credential(&id)
        .ok()
        .and_then(|c| serde_json::to_value(c).ok());
    state.db.delete_credential(&id)?;
    record_audit(&state, &principal, "delete", "credentials", &id, before, None);
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    reconcile_openrouter_presets_on_cred_change(&state);
    Ok(Json(json!({ "deleted": true, "id": id })))
}

pub async fn post_reset_credential_health(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<CredentialRecord>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    Ok(Json(state.db.reset_credential_health(&id)?))
}

pub async fn post_refresh_credential_quota(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<CredentialRecord>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let _ = poll_and_save_quota(
        &state.db,
        state.router.refresher(),
        &id,
        &state.quota_endpoints,
    )
    .await?;
    Ok(Json(state.db.get_credential(&id)?))
}

#[derive(Debug, Default, Deserialize)]
pub struct TestCredentialRequest {
    pub model: Option<String>,
}

pub async fn post_test_credential(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Option<Json<TestCredentialRequest>>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let started = Instant::now();
    let (record, _secret) = state.router.refresher().ensure_fresh(&id).await?;

    if let Some(Json(TestCredentialRequest { model: Some(m) })) = body
        && !m.trim().is_empty()
    {
        let pinned = format!("{}@{}", m.trim(), record.id);
        let probe = state
            .router
            .execute_text(CanonicalRequest {
                requested_model: pinned,
                messages: vec![Message::User(UserMessage::new("Reply with OK."))],
                max_output_tokens: Some(16),
                cache_preference: Some(false),
                ..Default::default()
            })
            .await?;
        return Ok(Json(json!({
            "ok": true,
            "latency_ms": started.elapsed().as_millis() as u64,
            "resolved_model": probe.resolved_model,
            "preview": probe.text,
            "credential": state.db.get_credential(&id)?,
        })));
    }

    Ok(Json(json!({
        "ok": true,
        "latency_ms": started.elapsed().as_millis() as u64,
        "credential": state.db.get_credential(&id)?,
    })))
}

pub async fn post_oauth_start(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<StartOAuthRequest>,
) -> ApiResult<Json<quixigate_auth::OAuthWizardSession>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let session = state.oauth_wizard.start_flow(req).await?;
    Ok(Json(session))
}

pub async fn get_oauth_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
) -> ApiResult<Json<quixigate_auth::OAuthWizardSession>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let session = state.oauth_wizard.get_session(&session_id)?;
    if session.provider_id == "openrouter"
        && session.credential.is_some()
        && state.router.catalog().list_presets().is_empty()
    {
        let _ = sync_openrouter_for_state(&state).await;
    }
    Ok(Json(session))
}

#[derive(Debug, Deserialize)]
pub struct CompleteOAuthRequest {
    pub code: String,
}

pub async fn post_oauth_complete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(req): Json<CompleteOAuthRequest>,
) -> ApiResult<Json<quixigate_auth::OAuthWizardSession>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let session = state
        .oauth_wizard
        .complete_manual_code(&session_id, &req.code)
        .await?;
    if session.provider_id == "openrouter" && session.credential.is_some() {
        let _ = sync_openrouter_for_state(&state).await;
    }
    Ok(Json(session))
}

// ── 3b. First-Class OpenRouter Hub (`/api/admin/openrouter/*`) ───────────────

/// Clears synced OpenRouter `@preset/*` entries immediately whenever zero enabled
/// `"openrouter"` credentials remain, guaranteeing OpenRouter is never assumed when disconnected.
pub fn reconcile_openrouter_presets_on_cred_change(state: &AppState) {
    let has_openrouter = state
        .db
        .list_credentials()
        .map(|cs| cs.iter().any(|c| c.enabled && c.provider_id == "openrouter"))
        .unwrap_or(false);
    if !has_openrouter {
        state.router.catalog().clear_presets();
    }
}

/// Polls credits/quotas for all enabled `"openrouter"` seats and syncs live `/models`
/// and `@preset/*` `/presets` into `ModelCatalog`.
pub async fn sync_openrouter_for_state(state: &AppState) -> quixigate_core::Result<(usize, usize)> {
    let creds = state.db.list_credentials()?;
    let or_seats: Vec<CredentialRecord> = creds
        .into_iter()
        .filter(|c| c.enabled && c.provider_id == "openrouter")
        .collect();
    if or_seats.is_empty() {
        state.router.catalog().clear_presets();
        return Ok((0, 0));
    }

    for seat in &or_seats {
        let _ = poll_and_save_quota(
            &state.db,
            state.router.refresher(),
            &seat.id,
            &state.quota_endpoints,
        )
        .await;
    }

    let first = &or_seats[0];
    let (fresh_rec, secret) = state.router.refresher().ensure_fresh(&first.id).await?;
    let bearer = match secret {
        CredentialSecret::ApiKey { key } => Some(key),
        CredentialSecret::OpenRouterOidc { access, .. } => Some(access),
        _ => None,
    };
    let base_url = fresh_rec
        .base_url
        .as_deref()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or(openrouter_base_url_for_region(None));

    state
        .router
        .catalog()
        .sync_openrouter(state.router.refresher().http(), base_url, bearer.as_deref())
        .await
        .map_err(|e| QuixigateError::Internal(format!("OpenRouter catalog sync failed: {e}")))
}

fn mask_secret_preview(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() <= 10 {
        return "••••••••".to_owned();
    }
    format!("{}…{}", &trimmed[..8], &trimmed[trimmed.len() - 4..])
}

pub async fn get_openrouter_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let creds = state.db.list_credentials()?;
    let seats: Vec<CredentialRecord> = creds
        .into_iter()
        .filter(|c| c.provider_id == "openrouter")
        .collect();
    let connected = seats.iter().any(|s| s.enabled);
    let settings = state.db.get_settings()?;

    let env_key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());
    let oidc_env_detected = ["OPENROUTER_OIDC_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
        .into_iter()
        .find(|name| {
            std::env::var(name)
                .ok()
                .is_some_and(|v| !v.trim().is_empty())
        });

    let presets = if connected {
        state.router.catalog().list_presets()
    } else {
        Vec::new()
    };
    let catalog_model_count = state
        .router
        .list_available_models()
        .iter()
        .filter(|m| m.providers.contains_key("openrouter") && !m.is_preset)
        .count();

    Ok(Json(json!({
        "connected": connected,
        "seats": seats,
        "env_key_detected": env_key.is_some(),
        "env_key_masked": env_key.as_deref().map(mask_secret_preview),
        "oidc_env_detected": oidc_env_detected,
        "catalog_model_count": catalog_model_count,
        "preset_count": presets.len(),
        "presets": presets,
        "app_url": settings.openrouter_app_url,
        "app_title": settings.openrouter_app_title,
        "burst_enabled": settings.openrouter_burst_enabled,
        "burst_model": settings.openrouter_burst_model,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct ConnectOpenRouterRequest {
    /// `"pkce"` | `"api_key"` | `"env"` | `"oidc"`
    pub method: String,
    pub id: Option<String>,
    pub label: Option<String>,
    pub api_key: Option<String>,
    pub env_var: Option<String>,
    pub subject_token: Option<String>,
    pub subject_token_env: Option<String>,
    pub federation_policy_id: Option<String>,
    pub token_endpoint: Option<String>,
    pub base_url: Option<String>,
    pub region: Option<String>,
    pub priority: Option<i32>,
    pub allow_api_overflow: Option<bool>,
}

pub async fn post_openrouter_connect(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConnectOpenRouterRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let method = req.method.trim().to_ascii_lowercase();
    let effective_base_url = req
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| openrouter_base_url_for_region(req.region.as_deref()).to_owned());

    match method.as_str() {
        "pkce" | "browser" => {
            let session = state
                .oauth_wizard
                .start_flow(StartOAuthRequest {
                    provider_id: "openrouter".to_owned(),
                    label: req.label.or_else(|| Some("OpenRouter (Browser PKCE)".to_owned())),
                    priority: Some(req.priority.unwrap_or(200)),
                    region: req.region,
                    ..Default::default()
                })
                .await?;
            Ok(Json(json!({
                "mode": "pkce",
                "session": session,
            })))
        }
        "env" | "api_key" => {
            let raw_key = if method == "env" {
                let var_name = req
                    .env_var
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("OPENROUTER_API_KEY");
                std::env::var(var_name)
                    .ok()
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        QuixigateError::Validation(format!(
                            "environment variable '{var_name}' is not set or empty"
                        ))
                    })?
            } else {
                req.api_key
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        QuixigateError::Validation("OpenRouter API key cannot be empty".to_owned())
                    })?
                    .to_owned()
            };

            let account: OpenRouterAccountInfo = inspect_openrouter_account(
                state.router.refresher().http(),
                &effective_base_url,
                &raw_key,
            )
            .await?;

            let derived_label = req
                .label
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| account.key_label.clone())
                .unwrap_or_else(|| {
                    if method == "env" {
                        "OpenRouter (Env Key)".to_owned()
                    } else {
                        "OpenRouter (API Key)".to_owned()
                    }
                });

            let rec = state.db.upsert_credential(
                req.id.as_deref(),
                "openrouter",
                &derived_label,
                AuthKind::ApiKey,
                Some(&effective_base_url),
                req.priority.unwrap_or(200),
                true,
                req.allow_api_overflow.unwrap_or(true),
                &CredentialSecret::ApiKey { key: raw_key },
            )?;

            let (synced_models, synced_presets) =
                sync_openrouter_for_state(&state).await.unwrap_or((0, 0));
            let refreshed = state.db.get_credential(&rec.id).unwrap_or(rec);

            Ok(Json(json!({
                "mode": "connected",
                "method": method,
                "credential": refreshed,
                "account": account,
                "synced_models": synced_models,
                "synced_presets": synced_presets,
                "presets": state.router.catalog().list_presets(),
            })))
        }
        "oidc" => {
            let env_var = req
                .subject_token_env
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let subject_token = match req
                .subject_token
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(tok) => tok.to_owned(),
                None => {
                    let var_name = env_var.unwrap_or("OPENROUTER_OIDC_TOKEN");
                    std::env::var(var_name)
                        .ok()
                        .map(|s| s.trim().to_owned())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            QuixigateError::Validation(format!(
                                "provide 'subject_token' or set non-empty environment variable '{var_name}'"
                            ))
                        })?
                }
            };
            let token_endpoint = req
                .token_endpoint
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(OPENROUTER_OIDC_TOKEN_URL);

            let secret = exchange_openrouter_oidc_token(
                state.router.refresher().http(),
                token_endpoint,
                &subject_token,
                req.federation_policy_id.as_deref(),
                env_var,
            )
            .await?;

            let access_tok = match &secret {
                CredentialSecret::OpenRouterOidc { access, .. } => access.clone(),
                _ => String::new(),
            };
            let account = inspect_openrouter_account(
                state.router.refresher().http(),
                &effective_base_url,
                &access_tok,
            )
            .await
            .unwrap_or_default();

            let label = req
                .label
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| account.key_label.clone())
                .unwrap_or_else(|| "OpenRouter (OIDC Workload Identity)".to_owned());

            let rec = state.db.upsert_credential(
                req.id.as_deref(),
                "openrouter",
                &label,
                AuthKind::Oauth,
                Some(&effective_base_url),
                req.priority.unwrap_or(200),
                true,
                req.allow_api_overflow.unwrap_or(true),
                &secret,
            )?;

            let (synced_models, synced_presets) =
                sync_openrouter_for_state(&state).await.unwrap_or((0, 0));
            let refreshed = state.db.get_credential(&rec.id).unwrap_or(rec);

            Ok(Json(json!({
                "mode": "connected",
                "method": "oidc",
                "credential": refreshed,
                "account": account,
                "synced_models": synced_models,
                "synced_presets": synced_presets,
                "presets": state.router.catalog().list_presets(),
            })))
        }
        other => Err(ApiError(QuixigateError::Validation(format!(
            "unsupported OpenRouter connect method '{other}'; expected 'pkce', 'env', 'api_key', or 'oidc'"
        )))),
    }
}

pub async fn post_openrouter_sync(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let (synced_models, synced_presets) = sync_openrouter_for_state(&state).await?;
    let seats: Vec<CredentialRecord> = state
        .db
        .list_credentials()?
        .into_iter()
        .filter(|c| c.provider_id == "openrouter")
        .collect();
    Ok(Json(json!({
        "ok": true,
        "synced_models": synced_models,
        "synced_presets": synced_presets,
        "presets": if seats.iter().any(|s| s.enabled) {
            state.router.catalog().list_presets()
        } else {
            Vec::new()
        },
        "seats": seats,
    })))
}

// ── 4. Models, Smart Router & Local Runtimes ─────────────────────────────────

pub async fn get_admin_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let models = state.router.list_available_models();
    let routes = state.db.list_model_routes()?;
    let creds = state.db.list_credentials()?;
    let openrouter_connected = creds
        .iter()
        .any(|c| c.enabled && c.provider_id == "openrouter");
    let presets = if openrouter_connected {
        state.router.catalog().list_presets()
    } else {
        Vec::new()
    };
    Ok(Json(json!({
        "models": models,
        "routes": routes,
        "credentials": creds,
        "presets": presets,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct SyncModelsRequest {
    pub url: Option<String>,
}

pub async fn post_sync_models(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<SyncModelsRequest>>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let target_url = body
        .and_then(|Json(b)| b.url)
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| MODELS_DEV_URL.to_owned());
    let count = state
        .router
        .catalog()
        .sync_from_url(state.router.refresher().http(), &target_url)
        .await
        .map_err(|e| QuixigateError::Internal(format!("models.dev sync failed: {e}")))?;
    Ok(Json(json!({ "ok": true, "synced_models": count })))
}

pub async fn get_routes(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<ModelRoute>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.list_model_routes()?))
}

#[derive(Debug, Deserialize)]
pub struct UpsertRouteRequest {
    pub alias: String,
    pub targets: Vec<RouteTarget>,
    #[serde(default)]
    pub pool_strategy: PoolStrategy,
    pub allow_paid_overflow: Option<bool>,
    pub input_nanos_per_m: Option<i64>,
    pub output_nanos_per_m: Option<i64>,
    pub cache_read_nanos_per_m: Option<i64>,
    pub team_id: Option<String>,
}

pub async fn post_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UpsertRouteRequest>,
) -> ApiResult<Json<ModelRoute>> {
    let team_id_clean = req
        .team_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let principal = if let Some(tid) = &team_id_clean {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    if req.targets.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "route must contain at least one target model".to_owned(),
        )));
    }
    let existing = state.db.get_model_route(&req.alias)?;
    if let Some(ex) = &existing
        && principal.user.role != UserRole::Admin
        && ex.team_id != team_id_clean
    {
        return Err(ApiError(QuixigateError::Forbidden(
            "team admin cannot overwrite a global route or another team's route".to_owned(),
        )));
    }
    let now = now_ms();
    let route = state.db.upsert_model_route(&ModelRoute {
        alias: req.alias,
        targets: req.targets,
        pool_strategy: req.pool_strategy,
        allow_paid_overflow: req.allow_paid_overflow.unwrap_or(true),
        input_nanos_per_m: req.input_nanos_per_m,
        output_nanos_per_m: req.output_nanos_per_m,
        cache_read_nanos_per_m: req.cache_read_nanos_per_m,
        team_id: team_id_clean,
        created_by: existing
            .as_ref()
            .and_then(|r| r.created_by.clone())
            .or_else(|| Some(principal.user.id.clone())),
        updated_by: Some(principal.user.id.clone()),
        created_at_ms: existing.as_ref().and_then(|r| r.created_at_ms).or(Some(now)),
        updated_at_ms: now,
    })?;
    record_audit(
        &state,
        &principal,
        if existing.is_some() { "update" } else { "create" },
        "routes",
        &route.alias,
        existing.and_then(|e| serde_json::to_value(e).ok()),
        serde_json::to_value(&route).ok(),
    );
    Ok(Json(route))
}

pub async fn get_route_by_alias(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(alias): Path<String>,
) -> ApiResult<Json<ModelRoute>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let route = state
        .db
        .get_model_route(&alias)?
        .ok_or_else(|| QuixigateError::NotFound(format!("route '{alias}' not found")))?;
    Ok(Json(route))
}

#[derive(Debug, Deserialize)]
pub struct PatchRouteRequest {
    pub targets: Option<Vec<RouteTarget>>,
    pub pool_strategy: Option<PoolStrategy>,
    pub allow_paid_overflow: Option<bool>,
    pub input_nanos_per_m: Option<Option<i64>>,
    pub output_nanos_per_m: Option<Option<i64>>,
    pub cache_read_nanos_per_m: Option<Option<i64>>,
    pub team_id: Option<Option<String>>,
}

pub async fn put_or_patch_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(alias): Path<String>,
    Json(req): Json<PatchRouteRequest>,
) -> ApiResult<Json<ModelRoute>> {
    let existing = state.db.get_model_route(&alias)?;
    let next_team_id = match &req.team_id {
        Some(opt) => opt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        None => existing.as_ref().and_then(|r| r.team_id.clone()),
    };
    let principal = if let Some(tid) = &next_team_id {
        let p = authenticate_team_admin_or_global(&state, &headers, tid)?;
        if p.user.role != UserRole::Admin
            && let Some(ex) = &existing
            && ex.team_id.as_deref() != Some(tid.as_str())
        {
            return Err(ApiError(QuixigateError::Forbidden(
                "team admin cannot reassign a route owned by another scope".to_owned(),
            )));
        }
        p
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let targets = req
        .targets
        .or_else(|| existing.as_ref().map(|r| r.targets.clone()))
        .unwrap_or_default();
    if targets.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "route must contain at least one target model".to_owned(),
        )));
    }
    let now = now_ms();
    let route = state.db.upsert_model_route(&ModelRoute {
        alias,
        targets,
        pool_strategy: req
            .pool_strategy
            .or_else(|| existing.as_ref().map(|r| r.pool_strategy))
            .unwrap_or(PoolStrategy::FillFirst),
        allow_paid_overflow: req
            .allow_paid_overflow
            .or_else(|| existing.as_ref().map(|r| r.allow_paid_overflow))
            .unwrap_or(true),
        input_nanos_per_m: match req.input_nanos_per_m {
            Some(v) => v,
            None => existing.as_ref().and_then(|r| r.input_nanos_per_m),
        },
        output_nanos_per_m: match req.output_nanos_per_m {
            Some(v) => v,
            None => existing.as_ref().and_then(|r| r.output_nanos_per_m),
        },
        cache_read_nanos_per_m: match req.cache_read_nanos_per_m {
            Some(v) => v,
            None => existing.as_ref().and_then(|r| r.cache_read_nanos_per_m),
        },
        team_id: next_team_id,
        created_by: existing
            .as_ref()
            .and_then(|r| r.created_by.clone())
            .or_else(|| Some(principal.user.id.clone())),
        updated_by: Some(principal.user.id.clone()),
        created_at_ms: existing.as_ref().and_then(|r| r.created_at_ms).or(Some(now)),
        updated_at_ms: now,
    })?;
    record_audit(
        &state,
        &principal,
        if existing.is_some() { "update" } else { "create" },
        "routes",
        &route.alias,
        existing.and_then(|e| serde_json::to_value(e).ok()),
        serde_json::to_value(&route).ok(),
    );
    Ok(Json(route))
}

pub async fn delete_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(alias): Path<String>,
) -> ApiResult<Json<Value>> {
    let existing = state.db.get_model_route(&alias)?;
    let principal = if let Some(tid) = existing.as_ref().and_then(|r| r.team_id.as_deref()) {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    state.db.delete_model_route(&alias)?;
    record_audit(
        &state,
        &principal,
        "delete",
        "routes",
        &alias,
        existing.and_then(|e| serde_json::to_value(e).ok()),
        None,
    );
    Ok(Json(json!({ "deleted": true, "alias": alias })))
}

pub async fn get_local_runtime_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<quixigate_core::LocalRuntimeStatus>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let norm = normalize_local_runtime_id(&id);
    let status = state
        .router
        .local_scanner()
        .list_statuses()
        .into_iter()
        .find(|s| s.runtime_id == norm)
        .ok_or_else(|| QuixigateError::NotFound(format!("local runtime '{id}' not found")))?;
    Ok(Json(status))
}

pub async fn delete_local_runtime(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let norm = normalize_local_runtime_id(&id);
    let mut settings = state.db.get_settings()?;
    settings
        .custom_local_endpoints
        .retain(|e| normalize_local_runtime_id(&e.runtime_id) != norm);
    state.db.update_settings(&settings)?;
    state.router.local_scanner().remove_endpoint(&norm);
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    Ok(Json(json!({ "deleted": true, "runtime_id": norm })))
}

pub async fn get_local_runtimes(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<quixigate_core::LocalRuntimeStatus>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.router.local_scanner().list_statuses()))
}

pub async fn post_scan_local_runtimes(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<quixigate_core::LocalRuntimeStatus>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    Ok(Json(state.router.local_scanner().scan_all().await))
}

#[derive(Debug, Deserialize)]
pub struct RegisterLocalRuntimeRequest {
    pub runtime_id: String,
    pub display_name: String,
    pub base_url: String,
    pub api_key: Option<String>,
}

pub async fn post_local_runtime(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RegisterLocalRuntimeRequest>,
) -> ApiResult<Json<Vec<quixigate_core::LocalRuntimeStatus>>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let norm_id = normalize_local_runtime_id(&req.runtime_id);
    let display_name = req.display_name.trim().to_owned();
    let base_url = req.base_url.trim().to_owned();
    if norm_id.is_empty() || base_url.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "runtime_id and base_url are required".to_owned(),
        )));
    }

    let mut settings = state.db.get_settings()?;
    if let Some(existing) = settings
        .custom_local_endpoints
        .iter_mut()
        .find(|e| normalize_local_runtime_id(&e.runtime_id) == norm_id)
    {
        existing.runtime_id = norm_id.clone();
        existing.display_name = display_name.clone();
        existing.base_url = base_url.clone();
    } else {
        settings.custom_local_endpoints.push(LocalEndpointConfig {
            runtime_id: norm_id.clone(),
            display_name: display_name.clone(),
            base_url: base_url.clone(),
        });
    }
    state.db.update_settings(&settings)?;

    if let Some(raw_key) = req.api_key.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
        let existing_cred_id = state
            .db
            .list_credentials()?
            .into_iter()
            .find(|c| normalize_local_runtime_id(&c.provider_id) == norm_id)
            .map(|c| c.id);
        state.db.upsert_credential(
            existing_cred_id.as_deref(),
            &norm_id,
            &format!("{display_name} Key"),
            AuthKind::ApiKey,
            Some(&base_url),
            50,
            true,
            true,
            &CredentialSecret::ApiKey {
                key: raw_key.to_owned(),
            },
        )?;
    }

    state
        .router
        .local_scanner()
        .add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: norm_id,
            display_name,
            base_url,
        });
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    Ok(Json(state.router.local_scanner().scan_all().await))
}

#[derive(Debug, Deserialize)]
pub struct LocalModelControlRequest {
    pub model: String,
}

pub async fn post_load_local_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(runtime_id): Path<String>,
    Json(req): Json<LocalModelControlRequest>,
) -> ApiResult<Json<Vec<quixigate_core::LocalRuntimeStatus>>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let model = req.model.trim();
    if model.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "model name cannot be empty".to_owned(),
        )));
    }
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    state
        .router
        .local_scanner()
        .load_model(&runtime_id, model)
        .await?;
    Ok(Json(state.router.local_scanner().list_statuses()))
}

pub async fn post_unload_local_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(runtime_id): Path<String>,
    Json(req): Json<LocalModelControlRequest>,
) -> ApiResult<Json<Vec<quixigate_core::LocalRuntimeStatus>>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let model = req.model.trim();
    if model.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "model name cannot be empty".to_owned(),
        )));
    }
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    state
        .router
        .local_scanner()
        .unload_model(&runtime_id, model)
        .await?;
    Ok(Json(state.router.local_scanner().list_statuses()))
}

// ── 5. Users, Teams & `sk-qx-...` Virtual Keys ──────────────────────────────

pub async fn get_users(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<User>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.list_users()?))
}

pub async fn get_user_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<User>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.get_user(&id)?))
}

#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub display_name: Option<String>,
    pub role: Option<UserRole>,
    pub password: Option<String>,
    pub budget_usd: Option<String>,
    pub budget_nanos: Option<i64>,
    #[serde(default)]
    pub budget_window: BudgetWindow,
}

pub async fn post_user(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateUserRequest>,
) -> ApiResult<Json<User>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let budget = match req.budget_nanos {
        Some(n) => Some(NanoUsd(n)),
        None => parse_optional_usd(req.budget_usd.as_deref())?,
    };
    let user = state.db.create_user(
        &req.username,
        req.display_name.as_deref().unwrap_or(&req.username),
        req.role.unwrap_or(UserRole::Member),
        req.password.as_deref(),
        budget,
        req.budget_window,
    )?;
    record_audit(
        &state,
        &principal,
        "create",
        "users",
        &user.id,
        None,
        serde_json::to_value(&user).ok(),
    );
    Ok(Json(user))
}

#[derive(Debug, Deserialize)]
pub struct PatchUserRequest {
    pub display_name: Option<String>,
    pub role: Option<UserRole>,
    pub password: Option<String>,
    pub budget_usd: Option<Option<String>>,
    pub budget_nanos: Option<Option<i64>>,
    pub budget_window: Option<BudgetWindow>,
    pub blocked: Option<bool>,
}

pub async fn patch_user(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PatchUserRequest>,
) -> ApiResult<Json<User>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let before = state
        .db
        .get_user(&id)
        .ok()
        .and_then(|u| serde_json::to_value(u).ok());
    let budget = if let Some(bn) = req.budget_nanos {
        Some(bn.map(NanoUsd))
    } else if let Some(bu) = &req.budget_usd {
        Some(parse_optional_usd(bu.as_deref())?)
    } else {
        None
    };
    let mut updated = state.db.update_user(
        &id,
        req.display_name.as_deref(),
        req.role,
        req.password.as_deref(),
        budget,
        req.budget_window,
    )?;
    if let Some(blocked) = req.blocked {
        updated = state.db.set_user_blocked(&updated.id, blocked)?;
    }
    record_audit(
        &state,
        &principal,
        "update",
        "users",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

pub async fn post_reset_user_spend(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<User>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    state.db.reset_spend(Some(&id), None, None, false)?;
    let updated = state.db.get_user(&id)?;
    record_audit(
        &state,
        &principal,
        "reset_spend",
        "users",
        &updated.id,
        None,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

pub async fn delete_user(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let before = state
        .db
        .get_user(&id)
        .ok()
        .and_then(|u| serde_json::to_value(u).ok());
    state.db.delete_user(&id)?;
    record_audit(&state, &principal, "delete", "users", &id, before, None);
    Ok(Json(json!({ "deleted": true, "id": id })))
}

pub async fn get_teams(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<quixigate_core::Team>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.list_teams()?))
}

pub async fn get_team_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<quixigate_core::Team>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.get_team(&id)?))
}

#[derive(Debug, Deserialize)]
pub struct TeamMemberRoleInput {
    pub user_id: String,
    #[serde(default = "default_member_role")]
    pub role: UserRole,
}

const fn default_member_role() -> UserRole {
    UserRole::Member
}

#[derive(Debug, Deserialize)]
pub struct CreateTeamRequest {
    pub name: String,
    pub budget_usd: Option<String>,
    pub budget_nanos: Option<i64>,
    #[serde(default)]
    pub budget_window: BudgetWindow,
    #[serde(default)]
    pub member_ids: Vec<String>,
    #[serde(default)]
    pub members: Vec<TeamMemberRoleInput>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub model_max_budget: Option<Value>,
}

pub async fn post_team(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateTeamRequest>,
) -> ApiResult<Json<quixigate_core::Team>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let budget = match req.budget_nanos {
        Some(n) => Some(NanoUsd(n)),
        None => parse_optional_usd(req.budget_usd.as_deref())?,
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let mut team = state
        .db
        .create_team(&req.name, budget, req.budget_window, &req.member_ids)?;
    if !req.members.is_empty() {
        let with_roles: Vec<(String, UserRole)> = req
            .members
            .into_iter()
            .map(|m| {
                let uid = state
                    .db
                    .get_user(&m.user_id)
                    .map(|u| u.id)
                    .unwrap_or(m.user_id);
                (uid, m.role)
            })
            .collect();
        team = state.db.add_team_members_with_roles(&team.id, &with_roles)?;
    }
    if !req.tags.is_empty() || model_budgets.is_some() {
        team = state.db.set_team_policy(
            &team.id,
            (!req.tags.is_empty()).then_some(req.tags.as_slice()),
            model_budgets.as_ref(),
        )?;
    }
    record_audit(
        &state,
        &principal,
        "create",
        "teams",
        &team.id,
        None,
        serde_json::to_value(&team).ok(),
    );
    Ok(Json(team))
}

#[derive(Debug, Deserialize)]
pub struct PatchTeamRequest {
    pub name: Option<String>,
    pub budget_usd: Option<Option<String>>,
    pub budget_nanos: Option<Option<i64>>,
    pub budget_window: Option<BudgetWindow>,
    pub member_ids: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub model_max_budget: Option<Value>,
}

pub async fn patch_team(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PatchTeamRequest>,
) -> ApiResult<Json<quixigate_core::Team>> {
    let changing_budget =
        req.budget_nanos.is_some() || req.budget_usd.is_some() || req.budget_window.is_some();
    let principal = if changing_budget {
        authenticate_admin(&state, &headers, true)?
    } else {
        authenticate_team_admin_or_global(&state, &headers, &id)?
    };
    let before = state
        .db
        .get_team(&id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    let budget = if let Some(bn) = req.budget_nanos {
        Some(bn.map(NanoUsd))
    } else if let Some(bu) = &req.budget_usd {
        Some(parse_optional_usd(bu.as_deref())?)
    } else {
        None
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let mut updated = state.db.update_team(
        &id,
        req.name.as_deref(),
        budget,
        req.budget_window,
        req.member_ids.as_deref(),
    )?;
    if req.tags.is_some() || model_budgets.is_some() {
        updated = state.db.set_team_policy(
            &updated.id,
            req.tags.as_deref(),
            model_budgets.as_ref(),
        )?;
    }
    record_audit(
        &state,
        &principal,
        "update",
        "teams",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

#[derive(Debug, Deserialize)]
pub struct TeamMembersRequest {
    #[serde(default)]
    pub member_ids: Vec<String>,
    #[serde(default)]
    pub members: Vec<TeamMemberRoleInput>,
}

pub async fn post_add_team_members(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<TeamMembersRequest>,
) -> ApiResult<Json<quixigate_core::Team>> {
    let principal = authenticate_team_admin_or_global(&state, &headers, &id)?;
    let before = state
        .db
        .get_team(&id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    let mut updated = if !req.member_ids.is_empty() {
        state.db.add_team_members(&id, &req.member_ids)?
    } else {
        state.db.get_team(&id)?
    };
    if !req.members.is_empty() {
        let with_roles: Vec<(String, UserRole)> = req
            .members
            .into_iter()
            .map(|m| {
                let uid = state
                    .db
                    .get_user(&m.user_id)
                    .map(|u| u.id)
                    .unwrap_or(m.user_id);
                (uid, m.role)
            })
            .collect();
        updated = state.db.add_team_members_with_roles(&id, &with_roles)?;
    }
    record_audit(
        &state,
        &principal,
        "member_add",
        "teams",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

pub async fn delete_team_member(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, user_id)): Path<(String, String)>,
) -> ApiResult<Json<quixigate_core::Team>> {
    let principal = authenticate_team_admin_or_global(&state, &headers, &id)?;
    let before = state
        .db
        .get_team(&id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    let updated = state.db.remove_team_member(&id, &user_id)?;
    record_audit(
        &state,
        &principal,
        "member_delete",
        "teams",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

pub async fn post_reset_team_spend(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<quixigate_core::Team>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    state.db.reset_spend(None, Some(&id), None, false)?;
    let updated = state.db.get_team(&id)?;
    record_audit(
        &state,
        &principal,
        "reset_spend",
        "teams",
        &updated.id,
        None,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

pub async fn delete_team(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let before = state
        .db
        .get_team(&id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    state.db.delete_team(&id)?;
    record_audit(&state, &principal, "delete", "teams", &id, before, None);
    Ok(Json(json!({ "deleted": true, "id": id })))
}

pub async fn get_keys(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<VirtualKey>>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let all = state.db.list_virtual_keys()?;
    if matches!(principal.user.role, UserRole::Admin | UserRole::Viewer) {
        return Ok(Json(all));
    }
    let admin_teams = state.db.list_admin_team_ids(&principal.user.id)?;
    if admin_teams.is_empty() {
        return Err(ApiError(QuixigateError::Forbidden(
            "administrator or team admin privileges required".to_owned(),
        )));
    }
    Ok(Json(
        all.into_iter()
            .filter(|k| {
                k.team_id
                    .as_ref()
                    .is_some_and(|tid| admin_teams.contains(tid))
            })
            .collect(),
    ))
}

pub async fn get_key_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<VirtualKey>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    let vk = state.db.get_virtual_key(&resolved)?;
    let is_own = principal
        .virtual_key
        .as_ref()
        .is_some_and(|k| k.id == vk.id);
    if !is_own && !matches!(principal.user.role, UserRole::Admin | UserRole::Viewer) {
        let is_team_adm = vk
            .team_id
            .as_deref()
            .is_some_and(|tid| state.db.is_team_admin(tid, &principal.user.id).unwrap_or(false));
        if !is_team_adm {
            return Err(ApiError(QuixigateError::Forbidden(
                "administrator or team admin privileges required".to_owned(),
            )));
        }
    }
    Ok(Json(vk))
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyRequest {
    pub name: String,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub budget_usd: Option<String>,
    pub budget_nanos: Option<i64>,
    #[serde(default)]
    pub budget_window: BudgetWindow,
    pub rpm_limit: Option<u32>,
    pub tpm_limit: Option<u64>,
    pub max_concurrent: Option<u32>,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    pub log_payloads: Option<bool>,
    pub expires_at_ms: Option<i64>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub model_max_budget: Option<Value>,
    #[serde(default, alias = "allowed_ips")]
    pub allowed_cidrs: Vec<String>,
    pub allow_spend_read: Option<bool>,
}

pub async fn post_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateKeyRequest>,
) -> ApiResult<Json<Value>> {
    let team_id_clean = req
        .team_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let principal = if let Some(tid) = team_id_clean {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let budget = match req.budget_nanos {
        Some(n) => Some(NanoUsd(n)),
        None => parse_optional_usd(req.budget_usd.as_deref())?,
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let owner_id = req
        .user_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&principal.user.id);
    let (mut key, raw_key) = state.db.create_virtual_key(
        &req.name,
        owner_id,
        team_id_clean,
        budget,
        req.budget_window,
        req.rpm_limit,
        req.tpm_limit,
        req.max_concurrent,
        &req.allowed_models,
        req.log_payloads,
        req.expires_at_ms,
    )?;
    if !req.tags.is_empty()
        || model_budgets.is_some()
        || !req.allowed_cidrs.is_empty()
        || req.allow_spend_read.is_some()
    {
        key = state.db.set_virtual_key_policy(
            &key.id,
            (!req.tags.is_empty()).then_some(req.tags.as_slice()),
            model_budgets.as_ref(),
            (!req.allowed_cidrs.is_empty()).then_some(req.allowed_cidrs.as_slice()),
            req.allow_spend_read,
        )?;
    }
    record_audit(
        &state,
        &principal,
        "create",
        "keys",
        &key.id,
        None,
        serde_json::to_value(&key).ok(),
    );
    Ok(Json(json!({
        "key": key,
        "raw_key": raw_key,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct PatchKeyRequest {
    pub name: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<Option<String>>,
    pub budget_usd: Option<Option<String>>,
    pub budget_nanos: Option<Option<i64>>,
    pub budget_window: Option<BudgetWindow>,
    pub rpm_limit: Option<Option<u32>>,
    pub tpm_limit: Option<Option<u64>>,
    pub max_concurrent: Option<Option<u32>>,
    pub allowed_models: Option<Vec<String>>,
    pub log_payloads: Option<Option<bool>>,
    pub expires_at_ms: Option<Option<i64>>,
    pub tags: Option<Vec<String>>,
    pub model_max_budget: Option<Value>,
    #[serde(alias = "allowed_ips")]
    pub allowed_cidrs: Option<Vec<String>>,
    pub allow_spend_read: Option<bool>,
}

pub async fn patch_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PatchKeyRequest>,
) -> ApiResult<Json<VirtualKey>> {
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    let existing = state.db.get_virtual_key(&resolved)?;
    let principal = if let Some(tid) = existing.team_id.as_deref()
        && req
            .team_id
            .as_ref()
            .is_none_or(|opt| opt.as_deref() == Some(tid))
    {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let before = serde_json::to_value(&existing).ok();
    let budget = if let Some(bn) = req.budget_nanos {
        Some(bn.map(NanoUsd))
    } else if let Some(bu) = &req.budget_usd {
        Some(parse_optional_usd(bu.as_deref())?)
    } else {
        None
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let mut updated = state.db.update_virtual_key(
        &resolved,
        req.name.as_deref(),
        req.user_id.as_deref(),
        req.team_id.as_ref().map(|o| o.as_deref()),
        budget,
        req.budget_window,
        req.rpm_limit,
        req.tpm_limit,
        req.max_concurrent,
        req.allowed_models.as_deref(),
        req.log_payloads,
        req.expires_at_ms,
    )?;
    if req.tags.is_some()
        || model_budgets.is_some()
        || req.allowed_cidrs.is_some()
        || req.allow_spend_read.is_some()
    {
        updated = state.db.set_virtual_key_policy(
            &updated.id,
            req.tags.as_deref(),
            model_budgets.as_ref(),
            req.allowed_cidrs.as_deref(),
            req.allow_spend_read,
        )?;
    }
    record_audit(
        &state,
        &principal,
        "update",
        "keys",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(updated))
}

#[derive(Debug, Default, Deserialize)]
pub struct RotateKeyRequest {
    pub grace_seconds: Option<u64>,
}

pub async fn post_rotate_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Option<Json<RotateKeyRequest>>,
) -> ApiResult<Json<Value>> {
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    let existing = state.db.get_virtual_key(&resolved)?;
    let principal = if let Some(tid) = existing.team_id.as_deref() {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let grace = body.and_then(|Json(b)| b.grace_seconds).unwrap_or(300);
    let (key, raw_key) = state.db.rotate_virtual_key(&resolved, grace)?;
    record_audit(
        &state,
        &principal,
        "rotate",
        "keys",
        &key.id,
        serde_json::to_value(&existing).ok(),
        serde_json::to_value(&key).ok(),
    );
    Ok(Json(json!({
        "key": key,
        "raw_key": raw_key,
        "grace_seconds": grace,
    })))
}

pub async fn post_revoke_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<VirtualKey>> {
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    let existing = state.db.get_virtual_key(&resolved)?;
    let principal = if let Some(tid) = existing.team_id.as_deref() {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let vk = state.db.revoke_virtual_key(&resolved)?;
    record_audit(
        &state,
        &principal,
        "block",
        "keys",
        &vk.id,
        serde_json::to_value(&existing).ok(),
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(vk))
}

pub async fn post_unrevoke_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<VirtualKey>> {
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    let existing = state.db.get_virtual_key(&resolved)?;
    let principal = if let Some(tid) = existing.team_id.as_deref() {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let vk = state.db.unrevoke_virtual_key(&resolved)?;
    record_audit(
        &state,
        &principal,
        "unblock",
        "keys",
        &vk.id,
        serde_json::to_value(&existing).ok(),
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(vk))
}

pub async fn post_reset_key_spend(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<VirtualKey>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    state
        .db
        .reset_spend(None, None, Some(&resolved), false)?;
    let vk = state.db.get_virtual_key(&resolved)?;
    record_audit(
        &state,
        &principal,
        "reset_spend",
        "keys",
        &vk.id,
        None,
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(vk))
}

pub async fn delete_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let resolved = state.db.resolve_virtual_key_id(&id)?;
    let existing = state.db.get_virtual_key(&resolved)?;
    let principal = if let Some(tid) = existing.team_id.as_deref() {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    state.db.delete_virtual_key(&resolved)?;
    record_audit(
        &state,
        &principal,
        "delete",
        "keys",
        &resolved,
        serde_json::to_value(&existing).ok(),
        None,
    );
    Ok(Json(json!({ "deleted": true, "id": resolved })))
}

// ── 6. Traces, Settings, MCP, Cache, Arize Phoenix Preset & Config Sync ──────

#[derive(Debug, Default, Deserialize)]
pub struct TracesQuery {
    pub model: Option<String>,
    pub provider_id: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub virtual_key_id: Option<String>,
    pub tag: Option<String>,
    pub status_min: Option<u16>,
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub async fn get_traces(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<TracesQuery>,
) -> ApiResult<Json<Vec<quixigate_core::RequestLog>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let logs = state.db.list_request_logs(&LogFilter {
        model: q.model.filter(|s| !s.trim().is_empty()),
        provider_id: q.provider_id.filter(|s| !s.trim().is_empty()),
        user_id: q.user_id.filter(|s| !s.trim().is_empty()),
        team_id: q.team_id.filter(|s| !s.trim().is_empty()),
        virtual_key_id: q.virtual_key_id.filter(|s| !s.trim().is_empty()),
        tag: q.tag.filter(|s| !s.trim().is_empty()),
        status_min: q.status_min,
        since_ms: q.since_ms,
        until_ms: q.until_ms,
        limit: q.limit.unwrap_or(100),
        offset: q.offset.unwrap_or(0),
    })?;
    Ok(Json(logs))
}

#[derive(Debug, Default, Deserialize)]
pub struct DeleteTracesQuery {
    pub before_ms: Option<i64>,
}

pub async fn delete_traces(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<DeleteTracesQuery>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let deleted = state.db.clear_request_logs(q.before_ms)?;
    Ok(Json(json!({ "ok": true, "deleted_rows": deleted })))
}

pub async fn get_trace_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<quixigate_core::RequestLog>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.get_request_log(&id)?))
}

pub async fn get_traces_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Response> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let mut sub = state.telemetry.subscribe_traces();
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        loop {
            match sub.recv().await {
                Ok(log) => {
                    if let Ok(json_str) = serde_json::to_string(&log)
                        && tx
                            .send(Ok(Event::default().event("trace").data(json_str)))
                            .await
                            .is_err()
                    {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

pub async fn get_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<GatewaySettings>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.get_settings()?))
}

pub async fn put_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(settings): Json<GatewaySettings>,
) -> ApiResult<Json<GatewaySettings>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let prev = state.db.get_settings()?;
    if !prev.ui_enabled && settings.ui_enabled && !AppState::is_ui_compiled_in() {
        return Err(ApiError(QuixigateError::Validation(
            "Web UI assets were excluded at compile time (--no-default-features); rebuild with default 'ui' feature to enable /ui".to_owned(),
        )));
    }
    state.db.update_settings(&settings)?;
    if prev.ui_enabled != settings.ui_enabled {
        state.clear_ui_override();
    }
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    let saved = state.db.get_settings()?;
    record_audit(
        &state,
        &principal,
        "update",
        "settings",
        "gateway_settings",
        serde_json::to_value(&prev).ok(),
        serde_json::to_value(&saved).ok(),
    );
    Ok(Json(saved))
}

pub async fn patch_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(patch): Json<Value>,
) -> ApiResult<Json<GatewaySettings>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let Some(patch_obj) = patch.as_object() else {
        return Err(ApiError(QuixigateError::Validation(
            "settings patch must be a JSON object".to_owned(),
        )));
    };
    let touched_ui = patch_obj.contains_key("ui_enabled");
    let prev = state.db.get_settings()?;
    let mut current = serde_json::to_value(&prev)
        .map_err(|e| ApiError(QuixigateError::Internal(e.to_string())))?;
    if let Some(cur_obj) = current.as_object_mut() {
        for (k, v) in patch_obj {
            let canonical_key = match k.as_str() {
                "oidc_issuer_url" => "jwt_issuer_url",
                "jwt_public_key_url" => "jwt_jwks_url",
                other => other,
            };
            cur_obj.insert(canonical_key.to_owned(), v.clone());
        }
    }
    let merged: GatewaySettings = serde_json::from_value(current)
        .map_err(|e| ApiError(QuixigateError::Validation(format!("invalid settings patch: {e}"))))?;
    if touched_ui && merged.ui_enabled && !AppState::is_ui_compiled_in() {
        return Err(ApiError(QuixigateError::Validation(
            "Web UI assets were excluded at compile time (--no-default-features); rebuild with default 'ui' feature to enable /ui".to_owned(),
        )));
    }
    state.db.update_settings(&merged)?;
    if touched_ui {
        state.clear_ui_override();
    }
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    let saved = state.db.get_settings()?;
    record_audit(
        &state,
        &principal,
        "update",
        "settings",
        "gateway_settings",
        serde_json::to_value(&prev).ok(),
        Some(patch),
    );
    Ok(Json(saved))
}

#[derive(Debug, Default, Deserialize)]
pub struct SetUiStatusRequest {
    pub enabled: Option<bool>,
    pub ui_enabled: Option<bool>,
    /// Defaults to `true` (persists to SQLite `GatewaySettings.ui_enabled`).
    /// Pass `false` to toggle `/ui` ephemerally in memory until process restart.
    pub persist: Option<bool>,
}

pub async fn get_ui_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.ui_status_json()))
}

pub async fn set_ui_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetUiStatusRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let enabled = req.enabled.or(req.ui_enabled).ok_or_else(|| {
        ApiError(QuixigateError::Validation(
            "missing boolean field 'enabled' (or 'ui_enabled')".to_owned(),
        ))
    })?;
    let status = state.set_ui_enabled(enabled, req.persist.unwrap_or(true))?;
    Ok(Json(status))
}

pub async fn post_enable_ui(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let status = state.set_ui_enabled(true, true)?;
    Ok(Json(status))
}

pub async fn post_disable_ui(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let status = state.set_ui_enabled(false, true)?;
    Ok(Json(status))
}

#[derive(Debug, Default, Deserialize)]
pub struct SetBurstRequest {
    pub enabled: Option<bool>,
    pub model: Option<String>,
    pub burst_model: Option<String>,
    pub app_url: Option<String>,
    pub app_title: Option<String>,
}

fn burst_status_payload(state: &AppState) -> ApiResult<Value> {
    let settings = state.db.get_settings()?;
    let creds = state.db.list_credentials()?;
    let openrouter_seats = creds
        .iter()
        .filter(|c| c.enabled && c.provider_id == "openrouter")
        .count();
    Ok(json!({
        "enabled": settings.openrouter_burst_enabled,
        "burst_model": settings.openrouter_burst_model,
        "openrouter_connected": openrouter_seats > 0,
        "openrouter_seats": openrouter_seats,
        "app_url": settings.openrouter_app_url,
        "app_title": settings.openrouter_app_title,
    }))
}

pub async fn get_burst_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(burst_status_payload(&state)?))
}

pub async fn set_burst_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetBurstRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let mut settings = state.db.get_settings()?;
    if let Some(en) = req.enabled {
        settings.openrouter_burst_enabled = en;
    }
    if let Some(m) = req
        .model
        .or(req.burst_model)
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
    {
        settings.openrouter_burst_model = m;
    }
    if let Some(url) = req.app_url.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()) {
        settings.openrouter_app_url = url;
    }
    if let Some(title) = req.app_title.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()) {
        settings.openrouter_app_title = title;
    }
    state.db.update_settings(&settings)?;
    Ok(Json(burst_status_payload(&state)?))
}

pub async fn post_enable_burst(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<SetBurstRequest>>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let mut settings = state.db.get_settings()?;
    settings.openrouter_burst_enabled = true;
    if let Some(Json(req)) = body
        && let Some(m) = req
            .model
            .or(req.burst_model)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
    {
        settings.openrouter_burst_model = m;
    }
    state.db.update_settings(&settings)?;
    Ok(Json(burst_status_payload(&state)?))
}

pub async fn post_disable_burst(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let mut settings = state.db.get_settings()?;
    settings.openrouter_burst_enabled = false;
    state.db.update_settings(&settings)?;
    Ok(Json(burst_status_payload(&state)?))
}

#[derive(Debug, Deserialize)]
pub struct McpServerRequest {
    pub url: String,
}

pub async fn get_mcp_servers(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let settings = state.db.get_settings()?;
    Ok(Json(json!({ "mcp_server_urls": settings.mcp_server_urls })))
}

pub async fn post_mcp_server(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<McpServerRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let url = req.url.trim();
    if url.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "MCP server URL cannot be empty".to_owned(),
        )));
    }
    let mut settings = state.db.get_settings()?;
    if !settings.mcp_server_urls.iter().any(|u| u == url) {
        settings.mcp_server_urls.push(url.to_owned());
        state.db.update_settings(&settings)?;
    }
    Ok(Json(json!({
        "ok": true,
        "mcp_server_urls": settings.mcp_server_urls,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct DeleteMcpQuery {
    pub url: Option<String>,
}

pub async fn delete_mcp_server(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<DeleteMcpQuery>,
    body: Option<Json<McpServerRequest>>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let target = q
        .url
        .or_else(|| body.map(|Json(b)| b.url))
        .unwrap_or_default();
    let trimmed = target.trim();
    if trimmed.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "specify 'url' query parameter or JSON body".to_owned(),
        )));
    }
    let mut settings = state.db.get_settings()?;
    let before = settings.mcp_server_urls.len();
    settings.mcp_server_urls.retain(|u| u != trimmed);
    let removed = settings.mcp_server_urls.len() != before;
    if removed {
        state.db.update_settings(&settings)?;
    }
    Ok(Json(json!({
        "deleted": removed,
        "mcp_server_urls": settings.mcp_server_urls,
    })))
}

pub async fn post_probe_mcp_server(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<McpServerRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let url = req.url.trim();
    if url.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "MCP server URL cannot be empty".to_owned(),
        )));
    }
    let tools = fetch_mcp_http_tools(state.router.refresher().http(), url).await;
    let tool_summaries: Vec<Value> = tools
        .into_iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        })
        .collect();
    Ok(Json(json!({
        "url": url,
        "tool_count": tool_summaries.len(),
        "tools": tool_summaries,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct PhoenixPresetRequest {
    pub phoenix_ui_url: Option<String>,
    pub otlp_endpoint: Option<String>,
    pub project_name: Option<String>,
    pub api_key: Option<String>,
}

pub async fn post_phoenix_preset(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<PhoenixPresetRequest>>,
) -> ApiResult<Json<GatewaySettings>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let mut settings = state.db.get_settings()?;
    let req = body.map(|Json(b)| b).unwrap_or_default();

    let ui_base = req
        .phoenix_ui_url
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| "http://localhost:6006".to_owned());
    let endpoint = req
        .otlp_endpoint
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| format!("{}/v1/traces", ui_base.trim_end_matches('/')));

    settings.otlp_enabled = true;
    settings.otlp_protocol = OtlpProtocol::HttpProtobuf;
    settings.phoenix_ui_url = ui_base;
    settings.otlp_endpoint = endpoint;
    settings.openinference_project_name = req
        .project_name
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "quixigate".to_owned());
    if let Some(k) = req.api_key {
        settings.otlp_api_key = (!k.trim().is_empty()).then(|| k.trim().to_owned());
    }
    state.db.update_settings(&settings)?;
    Ok(Json(settings))
}

pub async fn post_test_span(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<TestSpanResult>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let settings = state.db.get_settings()?;
    let res = state.telemetry.send_test_span(&settings).await?;
    Ok(Json(res))
}

pub async fn get_cache_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let settings = state.db.get_settings()?;
    let (memory_entries, sqlite_entries) = state.router.cache().entry_counts();
    Ok(Json(json!({
        "status": "healthy",
        "enabled_by_default": settings.cache_enabled_default,
        "ttl_secs": settings.cache_ttl_secs,
        "memory_entries": memory_entries,
        "sqlite_entries": sqlite_entries,
    })))
}

pub async fn post_clear_cache(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let removed = state.router.cache().clear_all()?;
    Ok(Json(json!({ "ok": true, "cleared_entries": removed })))
}

#[derive(Debug, Default, Deserialize)]
pub struct ExportConfigQuery {
    pub format: Option<String>,
    pub include_secrets: Option<bool>,
}

pub async fn get_export_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<ExportConfigQuery>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let fmt = q.format.as_deref().unwrap_or("toml");
    let content = state
        .db
        .export_config(fmt, q.include_secrets.unwrap_or(false))?;
    Ok(Json(json!({ "format": fmt, "content": content })))
}

#[derive(Debug, Deserialize)]
pub struct ImportConfigRequest {
    pub content: String,
    pub format: Option<String>,
}

pub async fn post_import_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ImportConfigRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let applied = state.db.import_config(&req.content, req.format.as_deref())?;
    state.clear_ui_override();
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    Ok(Json(json!({ "ok": true, "applied_items": applied })))
}

#[derive(Debug, Default, Deserialize)]
pub struct SpendExportQuery {
    pub format: Option<String>,
    pub limit: Option<usize>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub virtual_key_id: Option<String>,
    pub tag: Option<String>,
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,
}

fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_owned()
    }
}

pub async fn get_spend_export(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SpendExportQuery>,
) -> ApiResult<Response> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let logs = state.db.list_request_logs(&LogFilter {
        model: None,
        provider_id: None,
        user_id: q.user_id.filter(|s| !s.trim().is_empty()),
        team_id: q.team_id.filter(|s| !s.trim().is_empty()),
        virtual_key_id: q.virtual_key_id.filter(|s| !s.trim().is_empty()),
        tag: q.tag.filter(|s| !s.trim().is_empty()),
        status_min: None,
        since_ms: q.since_ms,
        until_ms: q.until_ms,
        limit: q.limit.unwrap_or(1_000).min(50_000),
        offset: 0,
    })?;

    let fmt = q.format.as_deref().unwrap_or("csv").to_ascii_lowercase();
    if fmt == "json" {
        return Ok(Json(json!({
            "count": logs.len(),
            "logs": logs,
        }))
        .into_response());
    }

    let mut csv = String::from(
        "id,timestamp_ms,ingress_protocol,requested_model,resolved_model,provider_id,credential_id,user_id,team_id,virtual_key_id,status_code,stream,cache_hit,ttft_ms,duration_ms,input_tokens,cached_input_tokens,reasoning_tokens,output_tokens,cost_nanos,cost_usd,api_equiv_nanos,api_equiv_usd\n",
    );
    for l in &logs {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_escape(&l.id),
            l.timestamp_ms,
            csv_escape(l.ingress_protocol.as_str()),
            csv_escape(&l.requested_model),
            csv_escape(&l.resolved_model),
            csv_escape(&l.provider_id),
            csv_escape(l.credential_id.as_deref().unwrap_or("")),
            csv_escape(l.user_id.as_deref().unwrap_or("")),
            csv_escape(l.team_id.as_deref().unwrap_or("")),
            csv_escape(l.virtual_key_id.as_deref().unwrap_or("")),
            l.status_code,
            l.stream,
            l.cache_hit,
            l.ttft_ms.map(|v| v.to_string()).unwrap_or_default(),
            l.duration_ms,
            l.input_tokens,
            l.cached_input_tokens,
            l.reasoning_tokens,
            l.output_tokens,
            l.cost_nanos.nanos(),
            l.cost_nanos.to_decimal_string(),
            l.api_equiv_nanos.nanos(),
            l.api_equiv_nanos.to_decimal_string(),
        ));
    }

    Ok((
        StatusCode::OK,
        [
            ("content-type", "text/csv; charset=utf-8"),
            (
                "content-disposition",
                "attachment; filename=\"quixigate-spend-export.csv\"",
            ),
        ],
        csv,
    )
        .into_response())
}

#[derive(Debug, Default, Deserialize)]
pub struct SpendReportQuery {
    pub group_by: Option<String>,
    pub model: Option<String>,
    pub provider_id: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    #[serde(alias = "api_key")]
    pub virtual_key_id: Option<String>,
    pub tag: Option<String>,
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,
}

/// `GET /api/admin/spend/report` & `GET /spend/report` — Multi-dimensional `NanoUsd` spend & savings
/// breakdown (`?group_by=model|provider|tag|key|team|user|day`). Supports self-service access for
/// Virtual Keys with `allow_spend_read = true` (scoped to their own key) and Team Admins (scoped to their team).
pub async fn get_spend_report(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SpendReportQuery>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let group_by = q
        .group_by
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("model");

    let mut filter = LogFilter {
        model: q.model.filter(|s| !s.trim().is_empty()),
        provider_id: q.provider_id.filter(|s| !s.trim().is_empty()),
        user_id: q.user_id.filter(|s| !s.trim().is_empty()),
        team_id: q.team_id.filter(|s| !s.trim().is_empty()),
        virtual_key_id: q.virtual_key_id.filter(|s| !s.trim().is_empty()),
        tag: q.tag.filter(|s| !s.trim().is_empty()),
        since_ms: q.since_ms,
        until_ms: q.until_ms,
        ..Default::default()
    };

    let is_global_viewer = matches!(principal.user.role, UserRole::Admin | UserRole::Viewer)
        && principal.virtual_key.is_none();

    if !is_global_viewer {
        if let Some(vk) = &principal.virtual_key {
            if principal.user.role == UserRole::Admin && vk.allow_spend_read {
                // Admin key with explicit spend read permission still scopes to requested filter or own key
            } else if vk.allow_spend_read {
                filter.virtual_key_id = Some(vk.id.clone());
                filter.user_id = None;
                filter.team_id = None;
            } else if principal.user.role != UserRole::Admin {
                return Err(ApiError(QuixigateError::Forbidden(
                    "virtual key does not have allow_spend_read (permissions.get_spend_routes) enabled"
                        .to_owned(),
                )));
            }
        } else {
            let admin_teams = state.db.list_admin_team_ids(&principal.user.id)?;
            if admin_teams.is_empty() {
                return Err(ApiError(QuixigateError::Forbidden(
                    "administrator, team admin, or virtual key allow_spend_read permission required"
                        .to_owned(),
                )));
            }
            if let Some(requested_tid) = &filter.team_id {
                if !admin_teams.contains(requested_tid) {
                    return Err(ApiError(QuixigateError::Forbidden(format!(
                        "not a team admin for team '{requested_tid}'"
                    ))));
                }
            } else {
                filter.team_id = Some(admin_teams[0].clone());
            }
        }
    }

    let rows: Vec<SpendReportRow> = state.db.get_spend_report(group_by, &filter)?;
    let total_spend_nanos: i64 = rows.iter().map(|r| r.spend_nanos.nanos()).sum();
    let total_savings_nanos: i64 = rows.iter().map(|r| r.savings_nanos.nanos()).sum();
    Ok(Json(json!({
        "group_by": group_by,
        "count": rows.len(),
        "total_spend_nanos": total_spend_nanos,
        "total_spend_usd": NanoUsd(total_spend_nanos).to_decimal_string(),
        "total_savings_nanos": total_savings_nanos,
        "total_savings_usd": NanoUsd(total_savings_nanos).to_decimal_string(),
        "data": rows,
        "rows": rows,
        "results": rows,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct AuditLogsQuery {
    pub table_name: Option<String>,
    pub action: Option<String>,
    pub actor_id: Option<String>,
    pub object_id: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// `GET /api/admin/audit-logs` & `GET /audit/logs` — Immutable management audit trail.
pub async fn get_audit_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<AuditLogsQuery>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let limit = q.limit.unwrap_or(100).clamp(1, 1_000);
    let offset = q.offset.unwrap_or(0);
    let logs: Vec<AuditLogEntry> = state.db.list_audit_logs(
        q.table_name.as_deref(),
        q.action.as_deref(),
        q.actor_id.as_deref(),
        q.object_id.as_deref(),
        limit,
        offset,
    )?;
    Ok(Json(json!({
        "count": logs.len(),
        "audit_logs": logs,
    })))
}

/// `GET /api/admin/audit-logs/{id}` & `GET /audit/{id}` — Fetch single audit log entry.
pub async fn get_audit_log_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<AuditLogEntry>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(state.db.get_audit_log(&id)?))
}
