//! LiteLLM-compatible Management API endpoints (`/key/*`, `/user/*`, `/team/*`,
//! `/model/*`, `/v1/model/info`, `/model_group/info`, `/credentials*`, `/spend/*`,
//! `/global/*`, `/cache/*`, `/config/*`, `/health/liveliness`).
//!
//! Every operation is backed directly by QuixiGate's SQLite (`WAL`), AES-256-GCM vault,
//! and exact `NanoUsd(i64)` accounting so headless scripts, Terraform providers, and
//! existing LiteLLM tooling can manage QuixiGate with zero Web UI dependency.

use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::routing::{delete, get, patch, post};
use axum::Json;
use quixigate_core::{
    AuthKind, BudgetWindow, CredentialSecret, ModelRoute, NanoUsd, QuixigateError, PoolStrategy,
    RouteTarget, Team, User, UserRole, VirtualKey, now_ms,
};
use quixigate_db::LogFilter;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::admin::{parse_model_budget_map, record_audit, sync_local_scanner_from_db};
use crate::auth::{
    authenticate_admin, authenticate_inference, authenticate_team_admin_or_global,
    extract_api_token,
};
use crate::state::{ApiError, ApiResult, AppState};

pub fn router() -> Router<AppState> {
    Router::new()
        // Virtual Keys (`/key/*`)
        .route("/key/generate", post(post_key_generate))
        .route("/key/update", post(post_key_update))
        .route("/key/delete", post(post_key_delete))
        .route("/key/info", get(get_key_info))
        .route("/key/list", get(get_key_list))
        .route("/key/regenerate", post(post_key_regenerate_body))
        .route("/key/{key_id}/regenerate", post(post_key_regenerate_path))
        .route("/key/block", post(post_key_block))
        .route("/key/unblock", post(post_key_unblock))
        .route("/key/health", post(post_key_health))
        // Users (`/user/*`)
        .route("/user/new", post(post_user_new))
        .route("/user/update", post(post_user_update))
        .route("/user/delete", post(post_user_delete))
        .route("/user/block", post(post_user_block))
        .route("/user/unblock", post(post_user_unblock))
        .route("/user/info", get(get_user_info))
        .route("/user/list", get(get_user_list))
        // Teams (`/team/*`)
        .route("/team/new", post(post_team_new))
        .route("/team/update", post(post_team_update))
        .route("/team/delete", post(post_team_delete))
        .route("/team/info", get(get_team_info))
        .route("/team/list", get(get_team_list))
        .route("/team/member_add", post(post_team_member_add))
        .route("/team/member_delete", post(post_team_member_delete))
        // Models & Credentials (`/model/*`, `/v1/model/info`, `/model_group/info`, `/credentials*`)
        .route("/model/new", post(post_model_new))
        .route("/model/update", post(post_model_update))
        .route("/model/{model_id}/update", patch(patch_model_by_id))
        .route("/model/delete", post(post_model_delete))
        .route("/model/info", get(get_model_info))
        .route("/v1/model/info", get(get_model_info))
        .route("/model_group/info", get(get_model_group_info))
        .route(
            "/credentials",
            get(get_litellm_credentials).post(post_litellm_credential),
        )
        .route("/credentials/{id}", delete(delete_litellm_credential))
        // Spend, Audit, Cache, Config, UI, Health
        .route("/spend/logs", get(get_spend_logs))
        .route("/spend/report", get(crate::admin::get_spend_report))
        .route("/spend/tags", get(get_spend_tags))
        .route("/spend/export", get(crate::admin::get_spend_export))
        .route("/spend/calculate", post(post_spend_calculate))
        .route("/global/spend", get(get_global_spend))
        .route("/global/spend/report", get(get_global_spend_report))
        .route("/global/spend/reset", post(post_global_spend_reset))
        .route("/audit/logs", get(crate::admin::get_audit_logs))
        .route("/audit/{id}", get(crate::admin::get_audit_log_by_id))
        .route("/cache/ping", get(get_cache_ping))
        .route("/cache/flushall", post(post_cache_flushall))
        .route("/cache/delete", post(post_cache_flushall))
        .route("/config/yaml", get(get_config_yaml).post(post_config_yaml))
        .route(
            "/config/ui",
            get(crate::admin::get_ui_status)
                .post(crate::admin::set_ui_status)
                .put(crate::admin::set_ui_status)
                .patch(crate::admin::set_ui_status),
        )
        .route("/health/liveliness", get(get_health_liveliness))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn nanos_to_json_number(n: NanoUsd) -> Value {
    let dec = n.to_decimal_string();
    serde_json::from_str::<Value>(&dec).unwrap_or(Value::String(dec))
}

fn parse_budget_value(val: Option<&Value>) -> quixigate_core::Result<Option<NanoUsd>> {
    let Some(v) = val else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(None);
    }
    if let Some(s) = v.as_str() {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        return NanoUsd::parse_usd(trimmed).map(Some);
    }
    if v.is_number() {
        return NanoUsd::parse_usd(&v.to_string()).map(Some);
    }
    Err(QuixigateError::Validation(format!(
        "invalid budget amount: {v}"
    )))
}

fn parse_budget_window_str(raw: Option<&str>) -> BudgetWindow {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return BudgetWindow::Total;
    };
    match s.to_ascii_lowercase().as_str() {
        "daily" | "1d" | "24h" | "day" => BudgetWindow::Daily,
        "weekly" | "7d" | "1w" | "week" => BudgetWindow::Weekly,
        "monthly" | "30d" | "1mo" | "month" => BudgetWindow::Monthly,
        _ => BudgetWindow::Total,
    }
}

fn parse_duration_to_expires_ms(raw: Option<&str>) -> Option<i64> {
    let s = raw?.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    let secs: i64 = if let Some(d) = s.strip_suffix('d') {
        d.trim().parse::<i64>().ok()?.saturating_mul(86_400)
    } else if let Some(h) = s.strip_suffix('h') {
        h.trim().parse::<i64>().ok()?.saturating_mul(3_600)
    } else if let Some(m) = s.strip_suffix('m') {
        m.trim().parse::<i64>().ok()?.saturating_mul(60)
    } else if let Some(sec) = s.strip_suffix('s') {
        sec.trim().parse::<i64>().ok()?
    } else {
        s.parse::<i64>().ok()?
    };
    Some(now_ms().saturating_add(secs.saturating_mul(1_000)))
}

fn parse_litellm_role(raw: Option<&str>) -> Option<UserRole> {
    match raw?.trim().to_ascii_lowercase().as_str() {
        "proxy_admin" | "admin" | "owner" => Some(UserRole::Admin),
        "proxy_admin_viewer" | "internal_user_viewer" | "viewer" | "read_only" => {
            Some(UserRole::Viewer)
        }
        "internal_user" | "member" | "user" | "customer" => Some(UserRole::Member),
        _ => None,
    }
}

fn format_model_budgets_json(
    map: &std::collections::BTreeMap<String, NanoUsd>,
) -> serde_json::Map<String, Value> {
    map.iter()
        .map(|(k, v)| (k.clone(), nanos_to_json_number(*v)))
        .collect()
}

fn format_virtual_key_litellm(vk: &VirtualKey, raw_key: Option<&str>) -> Value {
    let mut obj = json!({
        "token": vk.id,
        "token_id": vk.id,
        "key_name": vk.key_prefix,
        "key_alias": vk.name,
        "user_id": vk.user_id,
        "team_id": vk.team_id,
        "spend": nanos_to_json_number(vk.spend_nanos),
        "spend_nanos": vk.spend_nanos.nanos(),
        "spend_usd": vk.spend_nanos.to_decimal_string(),
        "max_budget": vk.budget_limit_nanos.map(nanos_to_json_number),
        "max_budget_nanos": vk.budget_limit_nanos.map(NanoUsd::nanos),
        "model_max_budget": format_model_budgets_json(&vk.model_budgets_nanos),
        "budget_duration": vk.budget_window.as_str(),
        "budget_reset_at_ms": vk.window_reset_at_ms,
        "rpm_limit": vk.rpm_limit,
        "tpm_limit": vk.tpm_limit,
        "max_parallel_requests": vk.max_concurrent,
        "models": vk.allowed_models,
        "tags": vk.tags,
        "metadata": { "tags": vk.tags },
        "allowed_ips": vk.allowed_cidrs,
        "allowed_cidrs": vk.allowed_cidrs,
        "allow_spend_read": vk.allow_spend_read,
        "permissions": {
            "get_spend_routes": vk.allow_spend_read,
        },
        "blocked": vk.revoked_at_ms.is_some(),
        "revoked_at_ms": vk.revoked_at_ms,
        "expires_at_ms": vk.expires_at_ms,
        "created_at_ms": vk.created_at_ms,
    });
    if let Some(raw) = raw_key
        && let Some(map) = obj.as_object_mut()
    {
        map.insert("key".to_owned(), Value::String(raw.to_owned()));
    }
    obj
}

fn format_user_litellm(u: &User) -> Value {
    json!({
        "user_id": u.id,
        "username": u.username,
        "user_alias": u.display_name,
        "user_role": match u.role {
            UserRole::Admin => "proxy_admin",
            UserRole::Member => "internal_user",
            UserRole::Viewer => "internal_user_viewer",
        },
        "role": u.role,
        "blocked": u.blocked,
        "spend": nanos_to_json_number(u.spend_nanos),
        "spend_nanos": u.spend_nanos.nanos(),
        "spend_usd": u.spend_nanos.to_decimal_string(),
        "max_budget": u.budget_limit_nanos.map(nanos_to_json_number),
        "max_budget_nanos": u.budget_limit_nanos.map(NanoUsd::nanos),
        "budget_duration": u.budget_window.as_str(),
        "created_at_ms": u.created_at_ms,
    })
}

fn format_team_litellm(t: &Team) -> Value {
    json!({
        "team_id": t.id,
        "team_alias": t.name,
        "spend": nanos_to_json_number(t.spend_nanos),
        "spend_nanos": t.spend_nanos.nanos(),
        "spend_usd": t.spend_nanos.to_decimal_string(),
        "max_budget": t.budget_limit_nanos.map(nanos_to_json_number),
        "max_budget_nanos": t.budget_limit_nanos.map(NanoUsd::nanos),
        "model_max_budget": format_model_budgets_json(&t.model_budgets_nanos),
        "budget_duration": t.budget_window.as_str(),
        "members": t.member_ids,
        "members_with_roles": t.members,
        "tags": t.tags,
        "created_at_ms": t.created_at_ms,
    })
}

// ── 1. Virtual Key Management (`/key/*`) ─────────────────────────────────────

fn extract_tags_from_litellm(explicit: &[String], metadata: Option<&Value>) -> Vec<String> {
    let mut out: Vec<String> = explicit
        .iter()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    if let Some(arr) = metadata
        .and_then(|m| m.get("tags"))
        .and_then(Value::as_array)
    {
        for v in arr {
            if let Some(s) = v.as_str().map(str::trim).filter(|s| !s.is_empty())
                && !out.iter().any(|existing| existing == s)
            {
                out.push(s.to_owned());
            }
        }
    }
    out
}

fn extract_allow_spend_read(explicit: Option<bool>, permissions: Option<&Value>) -> Option<bool> {
    if let Some(b) = explicit {
        return Some(b);
    }
    permissions.and_then(|p| {
        if let Some(b) = p.get("get_spend_routes").and_then(Value::as_bool) {
            return Some(b);
        }
        p.get("get_spend_routes")
            .and_then(Value::as_array)
            .map(|arr| !arr.is_empty())
    })
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmKeyGenerateRequest {
    pub key: Option<String>,
    pub key_alias: Option<String>,
    pub name: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub max_budget: Option<Value>,
    pub budget_nanos: Option<i64>,
    pub model_max_budget: Option<Value>,
    pub budget_duration: Option<String>,
    pub rpm_limit: Option<u32>,
    pub tpm_limit: Option<u64>,
    pub max_parallel_requests: Option<u32>,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub metadata: Option<Value>,
    #[serde(default, alias = "allowed_cidrs")]
    pub allowed_ips: Vec<String>,
    pub permissions: Option<Value>,
    pub allow_spend_read: Option<bool>,
    pub duration: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub log_payloads: Option<bool>,
}

pub async fn post_key_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<LiteLlmKeyGenerateRequest>>,
) -> ApiResult<Json<Value>> {
    let req = body.map(|Json(b)| b).unwrap_or_default();
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
    let alias = req
        .key_alias
        .or(req.name)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("key-{}", now_ms() % 1_000_000));
    let budget = match req.budget_nanos {
        Some(n) => Some(NanoUsd(n)),
        None => parse_budget_value(req.max_budget.as_ref())?,
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let merged_tags = extract_tags_from_litellm(&req.tags, req.metadata.as_ref());
    let allow_spend_read = extract_allow_spend_read(req.allow_spend_read, req.permissions.as_ref());
    let window = parse_budget_window_str(req.budget_duration.as_deref());
    let expires = req
        .expires_at_ms
        .or_else(|| parse_duration_to_expires_ms(req.duration.as_deref()));
    let owner_id = req
        .user_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&principal.user.id);

    let (mut vk, raw_key) = state.db.create_virtual_key_with_token(
        &alias,
        owner_id,
        team_id_clean,
        budget,
        window,
        req.rpm_limit,
        req.tpm_limit,
        req.max_parallel_requests,
        &req.models,
        req.log_payloads,
        expires,
        req.key.as_deref(),
    )?;
    if !merged_tags.is_empty()
        || model_budgets.is_some()
        || !req.allowed_ips.is_empty()
        || allow_spend_read.is_some()
    {
        vk = state.db.set_virtual_key_policy(
            &vk.id,
            (!merged_tags.is_empty()).then_some(merged_tags.as_slice()),
            model_budgets.as_ref(),
            (!req.allowed_ips.is_empty()).then_some(req.allowed_ips.as_slice()),
            allow_spend_read,
        )?;
    }
    record_audit(
        &state,
        &principal,
        "create",
        "keys",
        &vk.id,
        None,
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(format_virtual_key_litellm(&vk, Some(&raw_key))))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmKeyUpdateRequest {
    pub key: String,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<Option<String>>,
    pub max_budget: Option<Value>,
    pub budget_nanos: Option<Option<i64>>,
    pub model_max_budget: Option<Value>,
    pub budget_duration: Option<String>,
    pub rpm_limit: Option<Option<u32>>,
    pub tpm_limit: Option<Option<u64>>,
    pub max_parallel_requests: Option<Option<u32>>,
    pub models: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub metadata: Option<Value>,
    #[serde(alias = "allowed_cidrs")]
    pub allowed_ips: Option<Vec<String>>,
    pub permissions: Option<Value>,
    pub allow_spend_read: Option<bool>,
    pub duration: Option<String>,
    pub blocked: Option<bool>,
    pub log_payloads: Option<Option<bool>>,
}

pub async fn post_key_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmKeyUpdateRequest>,
) -> ApiResult<Json<Value>> {
    let resolved = state.db.resolve_virtual_key_id(&req.key)?;
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
    } else if let Some(mb) = &req.max_budget {
        if mb.is_null() {
            Some(None)
        } else {
            Some(parse_budget_value(Some(mb))?)
        }
    } else {
        None
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let next_tags: Option<Vec<String>> = if let Some(explicit) = &req.tags {
        Some(extract_tags_from_litellm(explicit, req.metadata.as_ref()))
    } else if req
        .metadata
        .as_ref()
        .and_then(|m| m.get("tags"))
        .is_some()
    {
        Some(extract_tags_from_litellm(&[], req.metadata.as_ref()))
    } else {
        None
    };
    let allow_spend_read = extract_allow_spend_read(req.allow_spend_read, req.permissions.as_ref());
    let window = req
        .budget_duration
        .as_deref()
        .map(|s| parse_budget_window_str(Some(s)));
    let expires = req
        .duration
        .as_deref()
        .map(|d| parse_duration_to_expires_ms(Some(d)));

    let mut vk = state.db.update_virtual_key(
        &resolved,
        req.key_alias.as_deref(),
        req.user_id.as_deref(),
        req.team_id.as_ref().map(|o| o.as_deref()),
        budget,
        window,
        req.rpm_limit,
        req.tpm_limit,
        req.max_parallel_requests,
        req.models.as_deref(),
        req.log_payloads,
        expires,
    )?;
    if next_tags.is_some()
        || model_budgets.is_some()
        || req.allowed_ips.is_some()
        || allow_spend_read.is_some()
    {
        vk = state.db.set_virtual_key_policy(
            &vk.id,
            next_tags.as_deref(),
            model_budgets.as_ref(),
            req.allowed_ips.as_deref(),
            allow_spend_read,
        )?;
    }
    if let Some(blocked) = req.blocked {
        vk = if blocked {
            state.db.revoke_virtual_key(&vk.id)?
        } else {
            state.db.unrevoke_virtual_key(&vk.id)?
        };
    }
    record_audit(
        &state,
        &principal,
        "update",
        "keys",
        &vk.id,
        before,
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(format_virtual_key_litellm(&vk, None)))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmKeyDeleteRequest {
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub key_aliases: Vec<String>,
}

pub async fn post_key_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmKeyDeleteRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let mut deleted = Vec::new();
    for target in req.keys.iter().chain(req.key_aliases.iter()) {
        if let Ok(id) = state.db.resolve_virtual_key_id(target) {
            let before = state
                .db
                .get_virtual_key(&id)
                .ok()
                .and_then(|k| serde_json::to_value(k).ok());
            state.db.delete_virtual_key(&id)?;
            record_audit(&state, &principal, "delete", "keys", &id, before, None);
            deleted.push(id);
        }
    }
    Ok(Json(json!({ "deleted_keys": deleted })))
}

#[derive(Debug, Default, Deserialize)]
pub struct KeyInfoQuery {
    pub key: Option<String>,
}

pub async fn get_key_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<KeyInfoQuery>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    if let Some(target_key) = q.key.as_deref().filter(|s| !s.trim().is_empty()) {
        // Querying arbitrary keys requires admin role unless inspecting one's own key.
        let resolved = state.db.resolve_virtual_key_id(target_key)?;
        let is_own = principal
            .virtual_key
            .as_ref()
            .is_some_and(|vk| vk.id == resolved);
        if !is_own && principal.user.role != UserRole::Admin {
            return Err(ApiError(QuixigateError::Forbidden(
                "administrator privileges required to inspect another virtual key".to_owned(),
            )));
        }
        let vk = state.db.get_virtual_key(&resolved)?;
        return Ok(Json(json!({
            "key": vk.key_prefix,
            "info": format_virtual_key_litellm(&vk, None),
        })));
    }

    if let Some(vk) = &principal.virtual_key {
        return Ok(Json(json!({
            "key": vk.key_prefix,
            "info": format_virtual_key_litellm(vk, None),
        })));
    }

    // Authenticated via Master Key or loopback admin without `?key=`: return caller principal info.
    Ok(Json(json!({
        "key": extract_api_token(&headers, None).map(|_| "master_key").unwrap_or("loopback"),
        "info": {
            "user_id": principal.user.id,
            "role": principal.user.role,
            "via_master_key": principal.via_master_key,
            "via_loopback_trust": principal.via_loopback_trust,
            "spend": nanos_to_json_number(principal.user.spend_nanos),
            "spend_nanos": principal.user.spend_nanos.nanos(),
            "max_budget": principal.user.budget_limit_nanos.map(nanos_to_json_number),
        }
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct KeyListQuery {
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub key_alias: Option<String>,
    pub return_full_object: Option<bool>,
}

pub async fn get_key_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<KeyListQuery>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let admin_teams = if matches!(principal.user.role, UserRole::Admin | UserRole::Viewer) {
        None
    } else {
        let teams = state.db.list_admin_team_ids(&principal.user.id)?;
        if teams.is_empty() {
            return Err(ApiError(QuixigateError::Forbidden(
                "administrator or team admin privileges required".to_owned(),
            )));
        }
        Some(teams)
    };
    let full = q.return_full_object.unwrap_or(true);
    let keys: Vec<VirtualKey> = state
        .db
        .list_virtual_keys()?
        .into_iter()
        .filter(|k| {
            let team_scope_ok = match &admin_teams {
                None => true,
                Some(teams) => k
                    .team_id
                    .as_ref()
                    .is_some_and(|tid| teams.contains(tid)),
            };
            team_scope_ok
                && q.user_id
                    .as_deref()
                    .is_none_or(|u| k.user_id == u.trim())
                && q.team_id
                    .as_deref()
                    .is_none_or(|t| k.team_id.as_deref() == Some(t.trim()))
                && q.key_alias
                    .as_deref()
                    .is_none_or(|a| k.name.eq_ignore_ascii_case(a.trim()))
        })
        .collect();
    let total = keys.len();
    let items: Vec<Value> = if full {
        keys.iter()
            .map(|k| format_virtual_key_litellm(k, None))
            .collect()
    } else {
        keys.iter().map(|k| Value::String(k.id.clone())).collect()
    };
    Ok(Json(json!({
        "keys": items,
        "total_count": total,
        "current_page": 1,
        "total_pages": 1,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct KeyTargetRequest {
    pub key: Option<String>,
    pub grace_seconds: Option<u64>,
}

pub async fn post_key_regenerate_body(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<KeyTargetRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let target = req.key.as_deref().ok_or_else(|| {
        ApiError(QuixigateError::Validation(
            "field 'key' is required".to_owned(),
        ))
    })?;
    let grace = req.grace_seconds.unwrap_or(300);
    let (vk, raw_key) = state.db.rotate_virtual_key(target, grace)?;
    Ok(Json(format_virtual_key_litellm(&vk, Some(&raw_key))))
}

pub async fn post_key_regenerate_path(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
    body: Option<Json<KeyTargetRequest>>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let grace = body.and_then(|Json(b)| b.grace_seconds).unwrap_or(300);
    let (vk, raw_key) = state.db.rotate_virtual_key(&key_id, grace)?;
    Ok(Json(format_virtual_key_litellm(&vk, Some(&raw_key))))
}

pub async fn post_key_block(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<KeyTargetRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let target = req.key.as_deref().ok_or_else(|| {
        ApiError(QuixigateError::Validation(
            "field 'key' is required".to_owned(),
        ))
    })?;
    let vk = state.db.revoke_virtual_key(target)?;
    record_audit(
        &state,
        &principal,
        "block",
        "keys",
        &vk.id,
        None,
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(format_virtual_key_litellm(&vk, None)))
}

pub async fn post_key_unblock(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<KeyTargetRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let target = req.key.as_deref().ok_or_else(|| {
        ApiError(QuixigateError::Validation(
            "field 'key' is required".to_owned(),
        ))
    })?;
    let vk = state.db.unrevoke_virtual_key(target)?;
    record_audit(
        &state,
        &principal,
        "unblock",
        "keys",
        &vk.id,
        None,
        serde_json::to_value(&vk).ok(),
    );
    Ok(Json(format_virtual_key_litellm(&vk, None)))
}

pub async fn post_key_health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    Ok(Json(json!({
        "key": "healthy",
        "user_id": principal.user.id,
        "key_id": principal.virtual_key.as_ref().map(|k| k.id.clone()),
        "via_master_key": principal.via_master_key,
        "via_loopback_trust": principal.via_loopback_trust,
    })))
}

// ── 2. User Management (`/user/*`) ───────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmUserNewRequest {
    pub user_id: Option<String>,
    pub username: Option<String>,
    pub user_email: Option<String>,
    pub user_alias: Option<String>,
    pub user_role: Option<String>,
    pub password: Option<String>,
    pub max_budget: Option<Value>,
    pub budget_nanos: Option<i64>,
    pub budget_duration: Option<String>,
    pub auto_create_key: Option<bool>,
    pub key_alias: Option<String>,
    pub team_id: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
}

pub async fn post_user_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmUserNewRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let username = req
        .username
        .or(req.user_id)
        .or(req.user_email)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("user-{}", now_ms() % 1_000_000));
    let display = req.user_alias.clone().unwrap_or_else(|| username.clone());
    let role = parse_litellm_role(req.user_role.as_deref()).unwrap_or(UserRole::Member);
    let budget = match req.budget_nanos {
        Some(n) => Some(NanoUsd(n)),
        None => parse_budget_value(req.max_budget.as_ref())?,
    };
    let window = parse_budget_window_str(req.budget_duration.as_deref());
    let user = state.db.create_user(
        &username,
        &display,
        role,
        req.password.as_deref(),
        budget,
        window,
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

    let mut out = format_user_litellm(&user);
    if req.auto_create_key.unwrap_or(true) {
        let key_alias = req
            .key_alias
            .unwrap_or_else(|| format!("{}-default", user.username));
        let (vk, raw_key) = state.db.create_virtual_key(
            &key_alias,
            &user.id,
            req.team_id.as_deref().filter(|s| !s.trim().is_empty()),
            budget,
            window,
            None,
            None,
            None,
            &req.models,
            None,
            None,
        )?;
        record_audit(
            &state,
            &principal,
            "create",
            "keys",
            &vk.id,
            None,
            serde_json::to_value(&vk).ok(),
        );
        if let Some(map) = out.as_object_mut() {
            map.insert("key".to_owned(), Value::String(raw_key));
            map.insert("token".to_owned(), Value::String(vk.id));
        }
    }
    Ok(Json(out))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmUserUpdateRequest {
    pub user_id: String,
    pub user_alias: Option<String>,
    pub user_role: Option<String>,
    pub password: Option<String>,
    pub max_budget: Option<Value>,
    pub budget_nanos: Option<Option<i64>>,
    pub budget_duration: Option<String>,
    pub blocked: Option<bool>,
}

pub async fn post_user_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmUserUpdateRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let before = state
        .db
        .get_user(&req.user_id)
        .ok()
        .and_then(|u| serde_json::to_value(u).ok());
    let budget = if let Some(bn) = req.budget_nanos {
        Some(bn.map(NanoUsd))
    } else if let Some(mb) = &req.max_budget {
        if mb.is_null() {
            Some(None)
        } else {
            Some(parse_budget_value(Some(mb))?)
        }
    } else {
        None
    };
    let role = parse_litellm_role(req.user_role.as_deref());
    let window = req
        .budget_duration
        .as_deref()
        .map(|s| parse_budget_window_str(Some(s)));
    let mut updated = state.db.update_user(
        &req.user_id,
        req.user_alias.as_deref(),
        role,
        req.password.as_deref(),
        budget,
        window,
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
    Ok(Json(format_user_litellm(&updated)))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmUserDeleteRequest {
    #[serde(default)]
    pub user_ids: Vec<String>,
}

pub async fn post_user_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmUserDeleteRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let mut deleted = Vec::new();
    for id in &req.user_ids {
        let before = state
            .db
            .get_user(id)
            .ok()
            .and_then(|u| serde_json::to_value(u).ok());
        state.db.delete_user(id)?;
        record_audit(&state, &principal, "delete", "users", id, before, None);
        deleted.push(id.clone());
    }
    Ok(Json(json!({ "deleted_users": deleted })))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmUserBlockRequest {
    pub user_id: Option<String>,
    #[serde(default)]
    pub user_ids: Vec<String>,
}

pub async fn post_user_block(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmUserBlockRequest>,
) -> ApiResult<Json<Value>> {
    set_users_blocked_impl(state, headers, req, true).await
}

pub async fn post_user_unblock(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmUserBlockRequest>,
) -> ApiResult<Json<Value>> {
    set_users_blocked_impl(state, headers, req, false).await
}

async fn set_users_blocked_impl(
    state: AppState,
    headers: HeaderMap,
    req: LiteLlmUserBlockRequest,
    blocked: bool,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let mut targets = req.user_ids;
    if let Some(single) = req.user_id.filter(|s| !s.trim().is_empty())
        && !targets.contains(&single)
    {
        targets.push(single);
    }
    if targets.is_empty() {
        return Err(ApiError(QuixigateError::Validation(
            "provide 'user_id' or 'user_ids'".to_owned(),
        )));
    }
    let mut updated_users = Vec::new();
    for target in &targets {
        let before = state
            .db
            .get_user(target)
            .ok()
            .and_then(|u| serde_json::to_value(u).ok());
        let user = state.db.set_user_blocked(target, blocked)?;
        record_audit(
            &state,
            &principal,
            if blocked { "block" } else { "unblock" },
            "users",
            &user.id,
            before,
            serde_json::to_value(&user).ok(),
        );
        updated_users.push(format_user_litellm(&user));
    }
    Ok(Json(json!({
        "blocked": blocked,
        "blocked_users": updated_users,
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct UserInfoQuery {
    pub user_id: Option<String>,
}

pub async fn get_user_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<UserInfoQuery>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, false)?;
    let target = q
        .user_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&principal.user.id);
    let user = state.db.get_user(target)?;
    let keys: Vec<Value> = state
        .db
        .list_virtual_keys()?
        .into_iter()
        .filter(|k| k.user_id == user.id)
        .map(|k| format_virtual_key_litellm(&k, None))
        .collect();
    let teams: Vec<Value> = state
        .db
        .list_teams()?
        .into_iter()
        .filter(|t| t.member_ids.contains(&user.id))
        .map(|t| format_team_litellm(&t))
        .collect();
    Ok(Json(json!({
        "user_id": user.id,
        "user_info": format_user_litellm(&user),
        "keys": keys,
        "teams": teams,
    })))
}

pub async fn get_user_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let users: Vec<Value> = state
        .db
        .list_users()?
        .iter()
        .map(format_user_litellm)
        .collect();
    let count = users.len();
    Ok(Json(json!({
        "users": users,
        "total": count,
        "page": 1,
        "total_pages": 1,
    })))
}

// ── 3. Team Management (`/team/*`) ───────────────────────────────────────────

fn parse_team_member_role(raw: Option<&str>) -> UserRole {
    match raw.unwrap_or("user").trim().to_ascii_lowercase().as_str() {
        "admin" | "team_admin" | "owner" => UserRole::Admin,
        "viewer" | "read_only" => UserRole::Viewer,
        _ => UserRole::Member,
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmTeamNewRequest {
    pub team_alias: Option<String>,
    pub name: Option<String>,
    pub max_budget: Option<Value>,
    pub budget_nanos: Option<i64>,
    pub model_max_budget: Option<Value>,
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub metadata: Option<Value>,
    #[serde(default)]
    pub member_ids: Vec<String>,
    #[serde(default)]
    pub members_with_roles: Vec<Value>,
}

pub async fn post_team_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmTeamNewRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    let name = req
        .team_alias
        .or(req.name)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            ApiError(QuixigateError::Validation(
                "team_alias (or name) is required".to_owned(),
            ))
        })?;
    let budget = match req.budget_nanos {
        Some(n) => Some(NanoUsd(n)),
        None => parse_budget_value(req.max_budget.as_ref())?,
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let merged_tags = extract_tags_from_litellm(&req.tags, req.metadata.as_ref());
    let window = parse_budget_window_str(req.budget_duration.as_deref());
    let mut role_pairs: Vec<(String, UserRole)> = Vec::new();
    for entry in &req.members_with_roles {
        if let Some(uid) = entry.get("user_id").and_then(Value::as_str)
            && !uid.trim().is_empty()
        {
            let resolved = state
                .db
                .get_user(uid)
                .map(|u| u.id)
                .unwrap_or_else(|_| uid.trim().to_owned());
            let role = parse_team_member_role(entry.get("role").and_then(Value::as_str));
            role_pairs.push((resolved, role));
        }
    }
    let mut team = state
        .db
        .create_team(&name, budget, window, &req.member_ids)?;
    if !role_pairs.is_empty() {
        team = state.db.add_team_members_with_roles(&team.id, &role_pairs)?;
    }
    if !merged_tags.is_empty() || model_budgets.is_some() {
        team = state.db.set_team_policy(
            &team.id,
            (!merged_tags.is_empty()).then_some(merged_tags.as_slice()),
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
    Ok(Json(format_team_litellm(&team)))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmTeamUpdateRequest {
    pub team_id: String,
    pub team_alias: Option<String>,
    pub max_budget: Option<Value>,
    pub budget_nanos: Option<Option<i64>>,
    pub model_max_budget: Option<Value>,
    pub budget_duration: Option<String>,
    pub tags: Option<Vec<String>>,
    pub metadata: Option<Value>,
    pub member_ids: Option<Vec<String>>,
}

pub async fn post_team_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmTeamUpdateRequest>,
) -> ApiResult<Json<Value>> {
    let changing_budget =
        req.max_budget.is_some() || req.budget_nanos.is_some() || req.budget_duration.is_some();
    let principal = if changing_budget {
        authenticate_admin(&state, &headers, true)?
    } else {
        authenticate_team_admin_or_global(&state, &headers, &req.team_id)?
    };
    let before = state
        .db
        .get_team(&req.team_id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    let budget = if let Some(bn) = req.budget_nanos {
        Some(bn.map(NanoUsd))
    } else if let Some(mb) = &req.max_budget {
        if mb.is_null() {
            Some(None)
        } else {
            Some(parse_budget_value(Some(mb))?)
        }
    } else {
        None
    };
    let model_budgets = parse_model_budget_map(req.model_max_budget.as_ref())?;
    let next_tags = if let Some(explicit) = &req.tags {
        Some(extract_tags_from_litellm(explicit, req.metadata.as_ref()))
    } else if req
        .metadata
        .as_ref()
        .and_then(|m| m.get("tags"))
        .is_some()
    {
        Some(extract_tags_from_litellm(&[], req.metadata.as_ref()))
    } else {
        None
    };
    let window = req
        .budget_duration
        .as_deref()
        .map(|s| parse_budget_window_str(Some(s)));
    let mut team = state.db.update_team(
        &req.team_id,
        req.team_alias.as_deref(),
        budget,
        window,
        req.member_ids.as_deref(),
    )?;
    if next_tags.is_some() || model_budgets.is_some() {
        team = state.db.set_team_policy(
            &team.id,
            next_tags.as_deref(),
            model_budgets.as_ref(),
        )?;
    }
    record_audit(
        &state,
        &principal,
        "update",
        "teams",
        &team.id,
        before,
        serde_json::to_value(&team).ok(),
    );
    Ok(Json(format_team_litellm(&team)))
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmTeamDeleteRequest {
    #[serde(default)]
    pub team_ids: Vec<String>,
}

pub async fn post_team_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmTeamDeleteRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_admin(&state, &headers, true)?;
    for id in &req.team_ids {
        let before = state
            .db
            .get_team(id)
            .ok()
            .and_then(|t| serde_json::to_value(t).ok());
        state.db.delete_team(id)?;
        record_audit(&state, &principal, "delete", "teams", id, before, None);
    }
    Ok(Json(json!({ "deleted_teams": req.team_ids })))
}

#[derive(Debug, Default, Deserialize)]
pub struct TeamInfoQuery {
    pub team_id: String,
}

pub async fn get_team_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<TeamInfoQuery>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    if !matches!(principal.user.role, UserRole::Admin | UserRole::Viewer)
        && !state.db.is_team_member(&q.team_id, &principal.user.id)?
    {
        return Err(ApiError(QuixigateError::Forbidden(
            "team membership or administrator privileges required".to_owned(),
        )));
    }
    let team = state.db.get_team(&q.team_id)?;
    let keys: Vec<Value> = state
        .db
        .list_virtual_keys()?
        .into_iter()
        .filter(|k| k.team_id.as_deref() == Some(team.id.as_str()))
        .map(|k| format_virtual_key_litellm(&k, None))
        .collect();
    Ok(Json(json!({
        "team_id": team.id,
        "team_info": format_team_litellm(&team),
        "keys": keys,
    })))
}

pub async fn get_team_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Vec<Value>>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    Ok(Json(
        state
            .db
            .list_teams()?
            .iter()
            .map(format_team_litellm)
            .collect(),
    ))
}

#[derive(Debug, Default, Deserialize)]
pub struct TeamMemberAddRequest {
    pub team_id: String,
    pub member: Option<Value>,
    #[serde(default)]
    pub member_ids: Vec<String>,
}

pub async fn post_team_member_add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TeamMemberAddRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_team_admin_or_global(&state, &headers, &req.team_id)?;
    let before = state
        .db
        .get_team(&req.team_id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    let mut role_pairs: Vec<(String, UserRole)> = req
        .member_ids
        .into_iter()
        .map(|u| {
            let uid = state.db.get_user(&u).map(|usr| usr.id).unwrap_or(u);
            (uid, UserRole::Member)
        })
        .collect();
    if let Some(m) = &req.member {
        let items: Vec<&Value> = match m {
            Value::Array(arr) => arr.iter().collect(),
            other => vec![other],
        };
        for item in items {
            if let Some(s) = item.as_str() {
                let uid = state
                    .db
                    .get_user(s)
                    .map(|usr| usr.id)
                    .unwrap_or_else(|_| s.to_owned());
                role_pairs.push((uid, UserRole::Member));
            } else if let Some(uid_raw) = item.get("user_id").and_then(Value::as_str) {
                let uid = state
                    .db
                    .get_user(uid_raw)
                    .map(|usr| usr.id)
                    .unwrap_or_else(|_| uid_raw.to_owned());
                let role = parse_team_member_role(item.get("role").and_then(Value::as_str));
                role_pairs.push((uid, role));
            }
        }
    }
    let updated = state
        .db
        .add_team_members_with_roles(&req.team_id, &role_pairs)?;
    record_audit(
        &state,
        &principal,
        "member_add",
        "teams",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(format_team_litellm(&updated)))
}

#[derive(Debug, Deserialize)]
pub struct TeamMemberDeleteRequest {
    pub team_id: String,
    pub user_id: String,
}

pub async fn post_team_member_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TeamMemberDeleteRequest>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_team_admin_or_global(&state, &headers, &req.team_id)?;
    let before = state
        .db
        .get_team(&req.team_id)
        .ok()
        .and_then(|t| serde_json::to_value(t).ok());
    let resolved_uid = state
        .db
        .get_user(&req.user_id)
        .map(|u| u.id)
        .unwrap_or(req.user_id);
    let updated = state.db.remove_team_member(&req.team_id, &resolved_uid)?;
    record_audit(
        &state,
        &principal,
        "member_delete",
        "teams",
        &updated.id,
        before,
        serde_json::to_value(&updated).ok(),
    );
    Ok(Json(format_team_litellm(&updated)))
}

// ── 4. Model & Credential Management (`/model/*`, `/credentials*`) ───────────

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmParams {
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct LiteLlmModelInfo {
    pub id: Option<String>,
    pub team_id: Option<String>,
    pub input_cost_per_token: Option<Value>,
    pub output_cost_per_token: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct LiteLlmModelNewRequest {
    pub model_name: String,
    pub litellm_params: LiteLlmParams,
    pub model_info: Option<LiteLlmModelInfo>,
    pub team_id: Option<String>,
}

fn parse_per_token_cost_to_per_million_nanos(v: Option<&Value>) -> Option<i64> {
    let val = v?;
    let per_tok = NanoUsd::parse_usd(&val.to_string()).ok()?;
    Some(per_tok.nanos().saturating_mul(1_000_000))
}

fn resolve_provider_and_model(params: &LiteLlmParams) -> (String, String) {
    if let Some(explicit) = params
        .custom_llm_provider
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        return (explicit.trim().to_owned(), params.model.trim().to_owned());
    }
    if let Some((prov, rest)) = params.model.trim().split_once('/') {
        return (prov.to_owned(), rest.to_owned());
    }
    ("openai".to_owned(), params.model.trim().to_owned())
}

pub async fn post_model_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmModelNewRequest>,
) -> ApiResult<Json<Value>> {
    let team_id_clean = req
        .team_id
        .as_deref()
        .or_else(|| req.model_info.as_ref().and_then(|m| m.team_id.as_deref()))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let has_raw_key = req
        .litellm_params
        .api_key
        .as_deref()
        .is_some_and(|k| !k.trim().is_empty());
    let principal = if let Some(tid) = &team_id_clean
        && !has_raw_key
    {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    let alias = req
        .model_info
        .as_ref()
        .and_then(|i| i.id.as_deref())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&req.model_name)
        .trim()
        .to_owned();
    let (provider_id, upstream_model) = resolve_provider_and_model(&req.litellm_params);

    // If an explicit `api_key` or `api_base` is supplied in `litellm_params`, persist it safely in the AES-256-GCM vault!
    let mut created_credential_id: Option<String> = None;
    if let Some(raw_key) = req
        .litellm_params
        .api_key
        .as_deref()
        .filter(|k| !k.trim().is_empty())
    {
        let rec = state.db.upsert_credential(
            None,
            &provider_id,
            &format!("{alias} ({provider_id})"),
            AuthKind::ApiKey,
            req.litellm_params
                .api_base
                .as_deref()
                .filter(|u| !u.trim().is_empty()),
            0,
            true,
            true,
            &CredentialSecret::ApiKey {
                key: raw_key.trim().to_owned(),
            },
        )?;
        sync_local_scanner_from_db(&state.db, state.router.local_scanner());
        created_credential_id = Some(rec.id);
    }

    let existing = state.db.get_model_route(&alias)?;
    if let Some(ex) = &existing
        && principal.user.role != UserRole::Admin
        && ex.team_id != team_id_clean
    {
        return Err(ApiError(QuixigateError::Forbidden(
            "team admin cannot overwrite a global route or another team's route".to_owned(),
        )));
    }
    let mut targets = existing.as_ref().map(|r| r.targets.clone()).unwrap_or_default();
    let new_target = RouteTarget {
        model: upstream_model.clone(),
        provider_id: Some(provider_id.clone()),
        credential_id: created_credential_id.clone(),
    };
    if !targets
        .iter()
        .any(|t| t.provider_id == new_target.provider_id && t.model == new_target.model)
    {
        targets.push(new_target);
    }

    let input_nanos = req
        .model_info
        .as_ref()
        .and_then(|m| parse_per_token_cost_to_per_million_nanos(m.input_cost_per_token.as_ref()))
        .or_else(|| existing.as_ref().and_then(|e| e.input_nanos_per_m));
    let output_nanos = req
        .model_info
        .as_ref()
        .and_then(|m| parse_per_token_cost_to_per_million_nanos(m.output_cost_per_token.as_ref()))
        .or_else(|| existing.as_ref().and_then(|e| e.output_nanos_per_m));

    let now = now_ms();
    let effective_team = team_id_clean.or_else(|| existing.as_ref().and_then(|e| e.team_id.clone()));
    let saved = state.db.upsert_model_route(&ModelRoute {
        alias: alias.clone(),
        targets,
        pool_strategy: existing
            .as_ref()
            .map(|e| e.pool_strategy)
            .unwrap_or(PoolStrategy::FillFirst),
        allow_paid_overflow: true,
        input_nanos_per_m: input_nanos,
        output_nanos_per_m: output_nanos,
        cache_read_nanos_per_m: existing.as_ref().and_then(|e| e.cache_read_nanos_per_m),
        team_id: effective_team,
        created_by: existing
            .as_ref()
            .and_then(|e| e.created_by.clone())
            .or_else(|| Some(principal.user.id.clone())),
        updated_by: Some(principal.user.id.clone()),
        created_at_ms: existing.as_ref().and_then(|e| e.created_at_ms).or(Some(now)),
        updated_at_ms: now,
    })?;
    record_audit(
        &state,
        &principal,
        if existing.is_some() { "update" } else { "create" },
        "routes",
        &saved.alias,
        existing.and_then(|e| serde_json::to_value(e).ok()),
        serde_json::to_value(&saved).ok(),
    );

    Ok(Json(json!({
        "model_id": saved.alias,
        "model_name": saved.alias,
        "credential_id": created_credential_id,
        "route": saved,
    })))
}

pub async fn post_model_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmModelNewRequest>,
) -> ApiResult<Json<Value>> {
    post_model_new(State(state), headers, Json(req)).await
}

pub async fn patch_model_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model_id): Path<String>,
    Json(mut req): Json<LiteLlmModelNewRequest>,
) -> ApiResult<Json<Value>> {
    if req.model_name.trim().is_empty() {
        req.model_name = model_id;
    }
    post_model_new(State(state), headers, Json(req)).await
}

#[derive(Debug, Deserialize)]
pub struct LiteLlmModelDeleteRequest {
    pub id: String,
}

pub async fn post_model_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmModelDeleteRequest>,
) -> ApiResult<Json<Value>> {
    let target = req.id.trim();
    let existing = state.db.get_model_route(target)?;
    let principal = if let Some(tid) = existing.as_ref().and_then(|r| r.team_id.as_deref()) {
        authenticate_team_admin_or_global(&state, &headers, tid)?
    } else {
        authenticate_admin(&state, &headers, true)?
    };
    state.db.delete_model_route(target)?;
    record_audit(
        &state,
        &principal,
        "delete",
        "routes",
        target,
        existing.and_then(|e| serde_json::to_value(e).ok()),
        None,
    );
    Ok(Json(json!({
        "message": format!("Model '{target}' deleted"),
        "deleted": true,
    })))
}

pub async fn get_model_info(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    let routes = state.db.list_model_routes()?;
    let mut data = Vec::new();

    for route in &routes {
        if let Some(route_tid) = &route.team_id
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
        for t in &route.targets {
            let model_str = match &t.provider_id {
                Some(p) => format!("{p}/{}", t.model),
                None => t.model.clone(),
            };
            data.push(json!({
                "model_name": route.alias,
                "litellm_params": {
                    "model": model_str,
                    "custom_llm_provider": t.provider_id,
                    "credential_id": t.credential_id,
                },
                "model_info": {
                    "id": route.alias,
                    "team_id": route.team_id,
                    "created_by": route.created_by,
                    "updated_by": route.updated_by,
                    "created_at_ms": route.created_at_ms,
                    "updated_at_ms": route.updated_at_ms,
                    "pool_strategy": route.pool_strategy,
                    "allow_paid_overflow": route.allow_paid_overflow,
                    "input_nanos_per_m": route.input_nanos_per_m,
                    "output_nanos_per_m": route.output_nanos_per_m,
                }
            }));
        }
    }

    let configured_providers: std::collections::HashSet<String> = state
        .db
        .list_credentials()?
        .into_iter()
        .filter(|c| c.enabled)
        .map(|c| c.provider_id)
        .collect();
    for m in state
        .router
        .list_available_models()
        .into_iter()
        .filter(|m| crate::codecs::common::is_catalog_model_routable(m, &configured_providers))
    {
        let first_provider = m
            .providers
            .keys()
            .next()
            .cloned()
            .unwrap_or_else(|| m.author.clone());
        data.push(json!({
            "model_name": m.canonical_id,
            "litellm_params": {
                "model": m.canonical_id,
                "custom_llm_provider": first_provider,
            },
            "model_info": {
                "id": m.canonical_id,
                "bare_id": m.bare_id,
                "max_input_tokens": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "input_nanos_per_m": m.input_nanos_per_m.nanos(),
                "output_nanos_per_m": m.output_nanos_per_m.nanos(),
                "supports_function_calling": m.supports_tools,
                "supports_reasoning": m.supports_reasoning,
                "supports_vision": m.supports_vision,
                "supports_response_schema": m.supports_structured_output,
                "is_local": m.is_local,
                "is_preset": m.is_preset,
            }
        }));
    }

    Ok(Json(json!({ "data": data })))
}

pub async fn get_model_group_info(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    let configured_providers: std::collections::HashSet<String> = state
        .db
        .list_credentials()?
        .into_iter()
        .filter(|c| c.enabled)
        .map(|c| c.provider_id)
        .collect();
    let models = state.router.list_available_models();
    let groups: Vec<Value> = models
        .into_iter()
        .filter(|m| crate::codecs::common::is_catalog_model_routable(m, &configured_providers))
        .map(|m| {
            json!({
                "model_group": m.canonical_id,
                "providers": m.providers.keys().cloned().collect::<Vec<_>>(),
                "max_input_tokens": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "input_nanos_per_m": m.input_nanos_per_m.nanos(),
                "output_nanos_per_m": m.output_nanos_per_m.nanos(),
                "supports_function_calling": m.supports_tools,
                "supports_reasoning": m.supports_reasoning,
                "supports_vision": m.supports_vision,
                "is_local": m.is_local,
                "is_preset": m.is_preset,
            })
        })
        .collect();
    Ok(Json(json!({ "data": groups })))
}

#[derive(Debug, Deserialize)]
pub struct LiteLlmCredentialCreateRequest {
    pub credential_name: String,
    pub credential_values: Value,
    pub credential_info: Option<Value>,
}

pub async fn get_litellm_credentials(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let creds: Vec<Value> = state
        .db
        .list_credentials()?
        .into_iter()
        .map(|c| {
            json!({
                "credential_id": c.id,
                "credential_name": c.label,
                "provider_id": c.provider_id,
                "auth_kind": c.auth_kind,
                "base_url": c.base_url,
                "status": c.status,
                "enabled": c.enabled,
                "priority": c.priority,
            })
        })
        .collect();
    Ok(Json(json!({ "credentials": creds })))
}

pub async fn post_litellm_credential(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LiteLlmCredentialCreateRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let provider_id = req
        .credential_info
        .as_ref()
        .and_then(|v| v.get("custom_llm_provider").or_else(|| v.get("provider_id")))
        .and_then(Value::as_str)
        .or_else(|| req.credential_values.get("provider_id").and_then(Value::as_str))
        .unwrap_or("openai");
    let api_key = req
        .credential_values
        .get("api_key")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            ApiError(QuixigateError::Validation(
                "credential_values.api_key is required".to_owned(),
            ))
        })?;
    let base_url = req
        .credential_values
        .get("api_base")
        .or_else(|| req.credential_values.get("base_url"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty());

    let rec = state.db.upsert_credential(
        None,
        provider_id,
        req.credential_name.trim(),
        AuthKind::ApiKey,
        base_url,
        0,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: api_key.trim().to_owned(),
        },
    )?;
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    if rec.provider_id == "openrouter" && rec.enabled {
        let _ = crate::admin::sync_openrouter_for_state(&state).await;
    }
    Ok(Json(json!({
        "credential_id": rec.id,
        "credential_name": rec.label,
        "provider_id": rec.provider_id,
    })))
}

pub async fn delete_litellm_credential(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    state.db.delete_credential(&id)?;
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    crate::admin::reconcile_openrouter_presets_on_cred_change(&state);
    Ok(Json(json!({ "deleted": true, "credential_id": id })))
}

// ── 5. Spend, Cache, YAML Config & Health ────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
pub struct SpendLogsQuery {
    pub api_key: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub model: Option<String>,
    pub tag: Option<String>,
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub async fn get_spend_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SpendLogsQuery>,
) -> ApiResult<Json<Vec<Value>>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let mut filter = LogFilter {
        model: q.model.filter(|s| !s.trim().is_empty()),
        provider_id: None,
        user_id: q.user_id.filter(|s| !s.trim().is_empty()),
        team_id: q.team_id.filter(|s| !s.trim().is_empty()),
        virtual_key_id: q.api_key.filter(|s| !s.trim().is_empty()),
        tag: q.tag.filter(|s| !s.trim().is_empty()),
        status_min: None,
        since_ms: q.since_ms,
        until_ms: q.until_ms,
        limit: q.limit.unwrap_or(100),
        offset: q.offset.unwrap_or(0),
    };

    let is_global_viewer = matches!(principal.user.role, UserRole::Admin | UserRole::Viewer)
        && principal.virtual_key.is_none();
    if !is_global_viewer {
        if let Some(vk) = &principal.virtual_key {
            if vk.allow_spend_read {
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

    let logs = state.db.list_request_logs(&filter)?;

    let formatted: Vec<Value> = logs
        .into_iter()
        .map(|l| {
            json!({
                "request_id": l.id,
                "call_type": l.ingress_protocol.as_str(),
                "api_key": l.virtual_key_id,
                "user": l.user_id,
                "team_id": l.team_id,
                "tags": l.tags,
                "request_tags": l.tags,
                "model": l.resolved_model,
                "requested_model": l.requested_model,
                "custom_llm_provider": l.provider_id,
                "spend": nanos_to_json_number(l.cost_nanos),
                "spend_nanos": l.cost_nanos.nanos(),
                "spend_usd": l.cost_nanos.to_decimal_string(),
                "api_equiv_nanos": l.api_equiv_nanos.nanos(),
                "total_tokens": l.input_tokens + l.output_tokens,
                "prompt_tokens": l.input_tokens,
                "completion_tokens": l.output_tokens,
                "cached_tokens": l.cached_input_tokens,
                "reasoning_tokens": l.reasoning_tokens,
                "cache_hit": l.cache_hit,
                "status_code": l.status_code,
                "duration_ms": l.duration_ms,
                "ttft_ms": l.ttft_ms,
                "timestamp_ms": l.timestamp_ms,
            })
        })
        .collect();
    Ok(Json(formatted))
}

pub async fn get_spend_tags(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(mut q): Query<crate::admin::SpendReportQuery>,
) -> ApiResult<Json<Value>> {
    q.group_by = Some("tag".to_owned());
    crate::admin::get_spend_report(State(state), headers, Query(q)).await
}

#[derive(Debug, Default, Deserialize)]
pub struct SpendCalculateRequest {
    pub model: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub messages: Option<Vec<Value>>,
    pub completion_response: Option<Value>,
}

pub async fn post_spend_calculate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SpendCalculateRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let model = req
        .model
        .or_else(|| {
            req.completion_response
                .as_ref()
                .and_then(|c| c.get("model"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "openai/gpt-4o".to_owned());

    let usage_obj = req
        .completion_response
        .as_ref()
        .and_then(|c| c.get("usage"));

    let input_tokens = req
        .prompt_tokens
        .or_else(|| {
            usage_obj
                .and_then(|u| u.get("prompt_tokens").or_else(|| u.get("input_tokens")))
                .and_then(Value::as_u64)
        })
        .unwrap_or_else(|| {
            req.messages
                .as_ref()
                .map(|msgs| {
                    let chars: usize = msgs.iter().map(|m| m.to_string().len()).sum();
                    ((chars / 4) as u64).max(1)
                })
                .unwrap_or(0)
        });
    let output_tokens = req
        .completion_tokens
        .or_else(|| {
            usage_obj
                .and_then(|u| u.get("completion_tokens").or_else(|| u.get("output_tokens")))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    let cached_tokens = req.cached_tokens.unwrap_or(0);

    let provider_id = req.custom_llm_provider.as_deref().unwrap_or("openai");
    let route = state.db.get_model_route(&model).ok().flatten();
    let (cost, api_equiv) = state.router.catalog().compute_cost(
        provider_id,
        AuthKind::ApiKey,
        &model,
        input_tokens,
        cached_tokens,
        output_tokens,
        route.as_ref(),
    );
    Ok(Json(json!({
        "cost": nanos_to_json_number(cost),
        "cost_nanos": cost.nanos(),
        "cost_usd": cost.to_decimal_string(),
        "api_equiv_nanos": api_equiv.nanos(),
        "model": model,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
    })))
}

pub async fn get_global_spend(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let metrics = state.db.get_dashboard_metrics()?;
    Ok(Json(json!({
        "spend": nanos_to_json_number(metrics.spend_30d_nanos),
        "spend_nanos": metrics.spend_30d_nanos.nanos(),
        "spend_usd": metrics.spend_30d_nanos.to_decimal_string(),
        "spend_24h_nanos": metrics.spend_24h_nanos.nanos(),
        "oauth_and_local_savings_30d_nanos": metrics.savings_30d_nanos.nanos(),
        "oauth_and_local_savings_30d_usd": metrics.savings_30d_nanos.to_decimal_string(),
    })))
}

pub async fn get_global_spend_report(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    crate::admin::get_spend_summary(State(state), headers).await
}

pub async fn post_global_spend_reset(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    state.db.reset_spend(None, None, None, true)?;
    Ok(Json(json!({ "ok": true, "message": "Global spend counters reset to 0" })))
}

pub async fn get_cache_ping(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, false)?;
    let settings = state.db.get_settings()?;
    let (memory_entries, sqlite_entries) = state.router.cache().entry_counts();
    Ok(Json(json!({
        "status": "healthy",
        "cache_type": "blake3_moka_sqlite",
        "ping_response": true,
        "enabled_by_default": settings.cache_enabled_default,
        "ttl_secs": settings.cache_ttl_secs,
        "memory_entries": memory_entries,
        "sqlite_entries": sqlite_entries,
    })))
}

pub async fn post_cache_flushall(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let cleared = state.router.cache().clear_all()?;
    Ok(Json(json!({
        "status": "success",
        "cleared_entries": cleared,
    })))
}

pub async fn get_config_yaml(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let yaml_str = state.db.export_config("yaml", false)?;
    let parsed: Value = serde_yaml_ng::from_str(&yaml_str).unwrap_or(Value::Null);
    Ok(Json(json!({
        "yaml": yaml_str,
        "config": parsed,
    })))
}

#[derive(Debug, Deserialize)]
pub struct ConfigYamlImportRequest {
    pub yaml: Option<String>,
    #[serde(flatten)]
    pub raw_object: serde_json::Map<String, Value>,
}

pub async fn post_config_yaml(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ConfigYamlImportRequest>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_admin(&state, &headers, true)?;
    let content = match req.yaml {
        Some(s) => s,
        None => serde_yaml_ng::to_string(&req.raw_object)
            .map_err(|e| ApiError(QuixigateError::Validation(format!("invalid YAML payload: {e}"))))?,
    };
    let applied = state.db.import_config(&content, Some("yaml"))?;
    state.clear_ui_override();
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    Ok(Json(json!({ "ok": true, "applied_items": applied })))
}

pub async fn get_health_liveliness() -> Json<Value> {
    Json(json!({ "status": "I'm alive!", "ok": true }))
}
