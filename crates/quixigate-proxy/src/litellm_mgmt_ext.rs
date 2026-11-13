//! Extended LiteLLM Control-Plane Compatibility Endpoints:
//! - Virtual Keys (`/key/service-account/generate`, `/key/bulk_update`, `/team/key/bulk_update`,
//!   `/v2/key/info`, `/key/{key_id}/reset_spend`, `/key/aliases`)
//! - Users & Sessions (`/v2/user/info`, `/users/me`, `/user/available_roles`, `/user/available_users`,
//!   `/user/bulk_update`, `/users/bulk`, `/users/bulk_delete`, `/user/password/change`,
//!   `/session/logout`, `/user/daily/activity`)
//! - End-Users / Customers (`/customer/*` and `/end_user/*` — full CRUD, block/unblock, daily activity)
//! - Teams (`PATCH /team/{team_id}`, `/v2/team/list`, `/team/available`, `/team/{team_id}/members/me`,
//!   `/team/member_update`, `/team/bulk_member_add`, `/teams/{team_id}/members/bulk_*`,
//!   `/team/block`, `/team/unblock`, `/team/{team_id}/disable_logging`, `/team/model/{add,delete}`,
//!   `/team/{team_id}/member/{user_id}/reset_*`, `/team/daily/activity`, `/team/spend/by_user`)
//! - Standalone Budgets (`/budget/*`, `/budgets`, `/management/v1/budgets`)
//! - Managed Tags (`/tag/*`)
//! - Model Access Groups (`/access_group/*`, `/v1/access_group*`)
//! - Fallbacks (`/fallback`, `/fallback/{*model}`)
//! - Models, Router, Credentials (`/v2/model/info`, `/model/{block,unblock,settings,deprecations,cost_map/source}`,
//!   `/reload/model_cost_map`, `/router/{settings,fields}`, `/adaptive_router/state`,
//!   `PATCH /credentials/{id}`, `/credentials/by_name/{*name}`, `/credentials/by_model/{*model_id}`)
//! - MCP Registry & REST Bridge (`/v1/mcp/server*`, `/v1/mcp/tools`, `/mcp-rest/*`)
//! - Spend, Activity, Audit, Health, Config & Public Catalog (`/audit`, `/spend/*`, `/global/*`,
//!   `/gateway/daily/activity`, `/provider/budgets`, `/test`, `/routes`, `/settings`, `/health/*`,
//!   `/config/{update,list}`, `/public/*`, `/model_hub*`)

use std::collections::BTreeMap;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post, put};
use quixigate_core::{
    AccessGroupRecord, AuthKind, BudgetRecord, CredentialSecret, EndUser, ModelRoute,
    NANOS_PER_USD, NanoUsd, PoolStrategy, QuixigateError, RouteTarget, TagRecord, UserRole, now_ms,
};
use quixigate_db::LogFilter;
use quixigate_providers::AISDK_BUILTIN_PROVIDERS;
use quixigate_router::catalog::MODELS_DEV_URL;
use quixigate_router::tools::fetch_mcp_http_tools;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::admin::{parse_model_budget_map, record_audit, sync_local_scanner_from_db};
use crate::auth::{
    authenticate_admin, authenticate_inference, extract_api_token, extract_session_cookie,
    format_clear_session_cookie,
};
use crate::litellm_mgmt::{
    format_model_budgets_json, format_team_litellm, format_user_litellm,
    format_virtual_key_litellm, nanos_to_json_number, parse_budget_value, parse_budget_window_str,
    parse_litellm_role,
};
use crate::state::{ApiResult, AppState};

fn auth_admin(
    state: &AppState,
    headers: &HeaderMap,
) -> quixigate_core::Result<quixigate_db::AuthenticatedPrincipal> {
    authenticate_admin(state, headers, true)
}

fn auth_team_or_admin(
    state: &AppState,
    headers: &HeaderMap,
) -> quixigate_core::Result<quixigate_db::AuthenticatedPrincipal> {
    let principal = authenticate_inference(state, headers, None)?;
    if principal.user.role == UserRole::Admin {
        return Ok(principal);
    }
    if !state.db.list_admin_team_ids(&principal.user.id)?.is_empty() {
        return Ok(principal);
    }
    Err(QuixigateError::Forbidden(
        "global administrator or team administrator privileges required".to_owned(),
    ))
}

pub async fn options_ok() -> StatusCode {
    StatusCode::NO_CONTENT
}

pub fn router() -> Router<AppState> {
    Router::new()
        // ── Virtual Keys ─────────────────────────────────────────────────────
        .route(
            "/key/service-account/generate",
            post(post_key_service_account_generate),
        )
        .route("/key/bulk_update", post(post_key_bulk_update))
        .route("/team/key/bulk_update", post(post_key_bulk_update))
        .route("/v2/key/info", post(post_v2_key_info))
        .route("/key/{key_id}/reset_spend", post(post_key_reset_spend))
        .route("/key/aliases", get(get_key_aliases))
        // ── Users & Session ──────────────────────────────────────────────────
        .route("/v2/user/info", get(get_v2_user_info))
        .route("/users/me", get(get_users_me))
        .route("/management/v1/users/me", get(get_users_me))
        .route("/user/available_roles", get(get_user_available_roles))
        .route("/user/available_users", get(get_user_available_users))
        .route("/user/bulk_update", post(post_user_bulk_update))
        .route("/users/bulk", post(post_users_bulk_create))
        .route("/management/v1/users/bulk", post(post_users_bulk_create))
        .route("/users/bulk_delete", post(post_users_bulk_delete))
        .route(
            "/management/v1/users/bulk_delete",
            post(post_users_bulk_delete),
        )
        .route("/user/password/change", post(post_user_password_change))
        .route("/session/logout", post(post_session_logout))
        .route("/user/daily/activity", get(get_daily_activity))
        // ── Customers / End-Users (16 routes) ────────────────────────────────
        .route("/customer/new", post(post_end_user_new))
        .route("/end_user/new", post(post_end_user_new))
        .route("/customer/info", get(get_end_user_info))
        .route("/end_user/info", get(get_end_user_info))
        .route("/customer/update", post(post_end_user_update))
        .route("/end_user/update", post(post_end_user_update))
        .route("/customer/delete", post(post_end_user_delete))
        .route("/end_user/delete", post(post_end_user_delete))
        .route("/customer/list", get(get_end_user_list))
        .route("/end_user/list", get(get_end_user_list))
        .route("/customer/block", post(post_end_user_block))
        .route("/end_user/block", post(post_end_user_block))
        .route("/customer/unblock", post(post_end_user_unblock))
        .route("/end_user/unblock", post(post_end_user_unblock))
        .route("/customer/daily/activity", get(get_daily_activity))
        .route("/end_user/daily/activity", get(get_daily_activity))
        // ── Teams ────────────────────────────────────────────────────────────
        .route("/team/{team_id}", patch(patch_team_by_id))
        .route("/v2/team/list", get(get_v2_team_list))
        .route("/team/available", get(get_team_available))
        .route("/team/{team_id}/members/me", get(get_team_member_me))
        .route("/team/member_update", post(post_team_member_update))
        .route("/team/bulk_member_add", post(post_team_bulk_member_add))
        .route(
            "/teams/{team_id}/members/bulk_delete",
            post(post_team_members_bulk_delete),
        )
        .route(
            "/management/v1/teams/{team_id}/members/bulk_delete",
            post(post_team_members_bulk_delete),
        )
        .route(
            "/teams/{team_id}/members/bulk_update",
            post(post_team_members_bulk_update),
        )
        .route(
            "/management/v1/teams/{team_id}/members/bulk_update",
            post(post_team_members_bulk_update),
        )
        .route("/team/block", post(post_team_block))
        .route("/team/unblock", post(post_team_unblock))
        .route(
            "/team/{team_id}/disable_logging",
            post(post_team_disable_logging),
        )
        .route(
            "/team/{team_id}/member/{user_id}/reset_spend",
            post(post_team_member_reset_spend),
        )
        .route(
            "/team/{team_id}/member/{user_id}/reset_budget",
            post(post_team_member_reset_spend),
        )
        .route("/team/model/add", post(post_team_model_add))
        .route("/team/model/delete", post(post_team_model_delete))
        .route("/team/daily/activity", get(get_daily_activity))
        .route("/team/spend/by_user", get(get_team_spend_by_user))
        // ── Budgets ──────────────────────────────────────────────────────────
        .route("/budget/new", post(post_budget_new))
        .route("/budget/update", post(post_budget_update))
        .route(
            "/budget/info",
            get(get_budget_info_query).post(post_budget_info),
        )
        .route("/budget/settings", get(get_budget_settings))
        .route("/budget/list", get(get_budget_list))
        .route("/budget/delete", post(post_budget_delete))
        .route("/budgets", get(get_budget_list))
        .route("/management/v1/budgets", get(get_budget_list))
        // ── Tags ─────────────────────────────────────────────────────────────
        .route("/tag/new", post(post_tag_new))
        .route("/tag/update", post(post_tag_update))
        .route("/tag/info", get(get_tag_info_query).post(post_tag_info))
        .route("/tag/list", get(get_tag_list))
        .route("/tag/delete", post(post_tag_delete))
        .route("/tag/daily/activity", get(get_daily_activity))
        // ── Access Groups ────────────────────────────────────────────────────
        .route("/access_group/new", post(post_access_group_new))
        .route("/access_group/list", get(get_access_group_list))
        .route(
            "/access_group/{access_group}/info",
            get(get_access_group_info),
        )
        .route(
            "/access_group/{access_group}/update",
            put(put_access_group_update),
        )
        .route(
            "/access_group/{access_group}/delete",
            delete(delete_access_group_by_name),
        )
        .route(
            "/access_group/{access_group}/budget",
            get(get_access_group_budget)
                .put(put_access_group_budget)
                .delete(delete_access_group_budget),
        )
        .route(
            "/v1/access_group",
            get(get_access_group_list).post(post_access_group_new),
        )
        .route(
            "/v1/access_group/{access_group}",
            get(get_access_group_info)
                .put(put_access_group_update)
                .patch(put_access_group_update)
                .delete(delete_access_group_by_name),
        )
        // ── Fallbacks ────────────────────────────────────────────────────────
        .route("/fallback", post(post_fallback))
        .route(
            "/fallback/{*model}",
            get(get_fallback_by_model).delete(delete_fallback_by_model),
        )
        // ── Models, Router & Credentials ─────────────────────────────────────
        .route("/v2/model/info", get(get_v2_model_info))
        .route("/model/block", post(post_model_block))
        .route("/model/unblock", post(post_model_unblock))
        .route("/model/settings", get(get_model_settings))
        .route("/model/cost_map/source", get(get_model_cost_map_source))
        .route("/model/deprecations", get(get_model_deprecations))
        .route("/v1/model/deprecations", get(get_model_deprecations))
        .route("/reload/model_cost_map", post(post_reload_model_cost_map))
        .route("/router/settings", get(get_router_settings))
        .route("/router/fields", get(get_router_fields))
        .route("/adaptive_router/state", get(get_adaptive_router_state))
        .route("/credentials/by_name/{*name}", get(get_credential_by_name))
        .route(
            "/credentials/by_model/{*model_id}",
            get(get_credentials_by_model),
        )
        // ── MCP Registry & REST Bridge ───────────────────────────────────────
        .route(
            "/v1/mcp/server",
            get(get_mcp_servers)
                .post(post_or_put_mcp_server)
                .put(post_or_put_mcp_server),
        )
        .route("/v1/mcp/server/health", get(get_mcp_servers_health))
        .route(
            "/v1/mcp/server/{server_id}",
            get(get_mcp_server_by_id)
                .put(put_mcp_server_by_id)
                .patch(put_mcp_server_by_id)
                .delete(delete_mcp_server_by_id),
        )
        .route("/v1/mcp/tools", get(get_mcp_tools))
        .route("/mcp-rest/tools/list", get(get_mcp_tools))
        .route("/mcp-rest/tools/call", post(post_mcp_rest_tools_call))
        .route(
            "/mcp-rest/test/connection",
            post(post_mcp_rest_test_connection),
        )
        .route(
            "/mcp-rest/test/tools/list",
            post(post_mcp_rest_test_tools_list),
        )
        // ── Spend, Activity, Audit, Health, Config, Public & Introspection ──
        .route("/audit", get(crate::admin::get_audit_logs))
        .route("/spend/keys", get(get_spend_keys))
        .route("/spend/users", get(get_spend_users))
        .route("/spend/logs/v2", get(get_spend_logs_v2))
        .route("/key/spend/report", get(get_key_spend_report))
        .route("/user/spend/report", get(get_user_spend_report))
        .route("/team/spend/report", get(get_team_spend_report))
        .route("/spend_logs/end_users", get(get_spend_logs_end_users))
        .route(
            "/management/v1/spend_logs/end_users",
            get(get_spend_logs_end_users),
        )
        .route("/spend_logs/users", get(get_spend_users))
        .route("/management/v1/spend_logs/users", get(get_spend_users))
        .route("/global/activity", get(get_daily_activity))
        .route("/gateway/daily/activity", get(get_daily_activity))
        .route("/global/spend/keys", get(get_spend_keys))
        .route("/global/spend/teams", get(get_global_spend_teams))
        .route("/global/spend/models", get(get_global_spend_models))
        .route("/global/spend/provider", get(get_global_spend_provider))
        .route("/global/spend/logs", get(get_spend_logs_v2))
        .route("/global/spend/tags", get(get_global_spend_tags))
        .route("/global/spend/all_tag_names", get(get_global_all_tag_names))
        .route("/global/spend/refresh", post(post_global_spend_refresh))
        .route("/global/spend/end_users", post(post_global_spend_end_users))
        .route("/global/all_end_users", get(get_end_user_list))
        .route("/provider/budgets", get(get_provider_budgets))
        .route("/test", get(get_test_ping))
        .route("/routes", get(get_routes_introspection))
        .route("/settings", get(get_gateway_settings_litellm))
        .route("/health/services", get(get_health_services))
        .route(
            "/health/readiness/details",
            get(get_health_readiness_details),
        )
        .route("/health/backlog", get(get_health_backlog))
        .route("/health/drain", get(get_health_drain))
        .route("/health/test_connection", post(post_health_test_connection))
        .route("/config/update", post(post_config_update))
        .route("/config/list", get(get_config_list))
        .route("/public/model_hub", get(get_public_model_hub))
        .route("/public/v1/model_hub", get(get_public_model_hub))
        .route("/model_hub", get(get_public_model_hub))
        .route(
            "/public/v1/model_hub/{facet}",
            get(get_public_model_hub_facet),
        )
        .route("/model_hub/{facet}", get(get_public_model_hub_facet))
        .route("/public/model_hub/info", get(get_public_model_hub_info))
        .route("/public/providers", get(get_public_providers))
        .route("/public/providers/fields", get(get_public_provider_fields))
        .route("/public/endpoints", get(get_public_endpoints))
        .route(
            "/public/litellm_model_cost_map",
            get(get_public_model_cost_map),
        )
}

// ── 1. Virtual Key Extensions ────────────────────────────────────────────────

async fn post_key_service_account_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let sa_id = body
        .get("service_account_id")
        .or_else(|| body.get("key_alias"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("sa-{}", &Uuid::new_v4().simple().to_string()[..8]));
    if let Some(obj) = body.as_object_mut() {
        obj.entry("key_alias".to_owned())
            .or_insert_with(|| Value::String(sa_id.clone()));
    }
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let max_budget = parse_budget_value(body.get("max_budget"))?;
    let window = parse_budget_window_str(body.get("budget_duration").and_then(Value::as_str));
    let rpm_limit = body
        .get("rpm_limit")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    let tpm_limit = body.get("tpm_limit").and_then(Value::as_u64);
    let max_conc = body
        .get("max_parallel_requests")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    let models: Vec<String> = body
        .get("models")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let tags: Vec<String> = body
        .get("tags")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let (mut vk, raw) = state.db.create_virtual_key(
        &sa_id,
        &principal.user.id,
        team_id,
        max_budget,
        window,
        rpm_limit,
        tpm_limit,
        max_conc,
        &models,
        None,
        None,
    )?;
    let mut eff_tags = tags;
    eff_tags.push(format!("service_account:{sa_id}"));
    vk = state
        .db
        .set_virtual_key_policy(&vk.id, Some(&eff_tags), None, None, None)?;
    record_audit(
        &state,
        &principal,
        "create_service_account_key",
        "virtual_keys",
        &vk.id,
        None,
        Some(json!({ "service_account_id": sa_id, "team_id": vk.team_id })),
    );
    let mut out = format_virtual_key_litellm(&vk, Some(&raw));
    if let Some(obj) = out.as_object_mut() {
        obj.insert("service_account_id".to_owned(), Value::String(sa_id));
    }
    Ok(Json(out))
}

async fn post_key_bulk_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let items: Vec<Value> = if let Some(arr) = body.get("keys").and_then(Value::as_array) {
        arr.clone()
    } else if let Some(arr) = body.as_array() {
        arr.clone()
    } else {
        return Err(QuixigateError::Validation("expected 'keys' array".to_owned()).into());
    };

    let mut updated = Vec::new();
    for item in items {
        let Some(key_ref) = item
            .get("key")
            .or_else(|| item.get("token"))
            .or_else(|| item.get("key_id"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        let budget = match item.get("max_budget") {
            Some(v) => Some(parse_budget_value(Some(v))?),
            None => None,
        };
        let window = item
            .get("budget_duration")
            .and_then(Value::as_str)
            .map(|s| parse_budget_window_str(Some(s)));
        let alias = item.get("key_alias").and_then(Value::as_str);
        let team_id = item.get("team_id").map(|v| v.as_str());
        let models = item.get("models").and_then(Value::as_array).map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        });
        let mut vk = state.db.update_virtual_key(
            key_ref,
            alias,
            None,
            team_id,
            budget,
            window,
            None,
            None,
            None,
            models.as_deref(),
            None,
            None,
        )?;
        if let Some(blocked) = item.get("blocked").and_then(Value::as_bool) {
            vk = if blocked {
                state.db.revoke_virtual_key(&vk.id)?
            } else {
                state.db.unrevoke_virtual_key(&vk.id)?
            };
        }
        updated.push(format_virtual_key_litellm(&vk, None));
    }
    record_audit(
        &state,
        &principal,
        "bulk_update",
        "virtual_keys",
        "bulk",
        None,
        Some(json!({ "count": updated.len() })),
    );
    Ok(Json(json!({
        "results": updated,
        "total_updated": updated.len(),
    })))
}

async fn post_v2_key_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested = body
        .get("key")
        .or_else(|| body.get("token"))
        .and_then(Value::as_str)
        .or_else(|| extract_api_token(&headers, None).as_deref().map(|_| ""))
        .unwrap_or("");
    let target = if requested.trim().is_empty() {
        extract_api_token(&headers, None).ok_or_else(|| {
            QuixigateError::Validation("missing 'key' in body or Authorization header".to_owned())
        })?
    } else {
        requested.trim().to_owned()
    };
    let resolved_id = state.db.resolve_virtual_key_id(&target)?;
    let vk = state.db.get_virtual_key(&resolved_id)?;
    if principal.user.role != UserRole::Admin
        && vk.user_id != principal.user.id
        && principal.virtual_key.as_ref().map(|k| k.id.as_str()) != Some(vk.id.as_str())
    {
        return Err(QuixigateError::Forbidden(
            "cannot inspect another user's virtual key".to_owned(),
        )
        .into());
    }
    Ok(Json(json!({
        "key": target,
        "info": format_virtual_key_litellm(&vk, None),
    })))
}

async fn post_key_reset_spend(
    State(state): State<AppState>,
    Path(key_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let resolved = state.db.resolve_virtual_key_id(&key_id)?;
    state.db.reset_spend(None, None, Some(&resolved), false)?;
    let vk = state.db.get_virtual_key(&resolved)?;
    record_audit(
        &state,
        &principal,
        "reset_spend",
        "virtual_keys",
        &vk.id,
        None,
        None,
    );
    Ok(Json(format_virtual_key_litellm(&vk, None)))
}

async fn get_key_aliases(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let keys = state.db.list_virtual_keys()?;
    let filtered: Vec<Value> = keys
        .into_iter()
        .filter(|k| principal.user.role == UserRole::Admin || k.user_id == principal.user.id)
        .map(|k| {
            json!({
                "token": k.id,
                "key_alias": k.name,
                "key_name": k.key_prefix,
                "team_id": k.team_id,
            })
        })
        .collect();
    Ok(Json(json!({ "aliases": filtered })))
}

// ── 2. Users & Session Extensions ────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
struct UserInfoQuery {
    user_id: Option<String>,
}

async fn get_v2_user_info(
    State(state): State<AppState>,
    Query(q): Query<UserInfoQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let uid = q.user_id.as_deref().unwrap_or(&principal.user.id);
    if principal.user.role != UserRole::Admin
        && uid != principal.user.id
        && uid != principal.user.username
    {
        return Err(
            QuixigateError::Forbidden("cannot inspect another user's profile".to_owned()).into(),
        );
    }
    let user = state.db.get_user(uid)?;
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

async fn get_users_me(State(state): State<AppState>, headers: HeaderMap) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let teams: Vec<Value> = state
        .db
        .list_teams()?
        .into_iter()
        .filter(|t| t.member_ids.contains(&principal.user.id))
        .map(|t| format_team_litellm(&t))
        .collect();
    let mut profile = format_user_litellm(&principal.user);
    if let Some(obj) = profile.as_object_mut() {
        obj.insert("teams".to_owned(), Value::Array(teams));
    }
    Ok(Json(profile))
}

async fn get_user_available_roles() -> Json<Value> {
    Json(json!({
        "roles": {
            "proxy_admin": { "ui_label": "Proxy Admin", "quixigate_role": "admin" },
            "proxy_admin_viewer": { "ui_label": "Proxy Admin Viewer", "quixigate_role": "viewer" },
            "internal_user": { "ui_label": "Internal User", "quixigate_role": "member" },
            "internal_user_viewer": { "ui_label": "Internal User Viewer", "quixigate_role": "viewer" }
        }
    }))
}

async fn get_user_available_users(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _admin = auth_admin(&state, &headers)?;
    let users: Vec<Value> = state
        .db
        .list_users()?
        .iter()
        .map(format_user_litellm)
        .collect();
    Ok(Json(json!({ "users": users, "total": users.len() })))
}

async fn post_user_bulk_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let users_arr = body
        .get("users")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| body.as_array().cloned())
        .unwrap_or_default();
    let all_users = body
        .get("all_users")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut updated = Vec::new();
    if all_users {
        let role = parse_litellm_role(body.get("user_role").and_then(Value::as_str));
        let budget = match body.get("max_budget") {
            Some(v) => Some(parse_budget_value(Some(v))?),
            None => None,
        };
        let window = body
            .get("budget_duration")
            .and_then(Value::as_str)
            .map(|s| parse_budget_window_str(Some(s)));
        for u in state.db.list_users()? {
            let next = state
                .db
                .update_user(&u.id, None, role, None, budget, window)?;
            updated.push(format_user_litellm(&next));
        }
    } else {
        for item in users_arr {
            let Some(uid) = item
                .get("user_id")
                .or_else(|| item.get("username"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let role = parse_litellm_role(item.get("user_role").and_then(Value::as_str));
            let alias = item.get("user_alias").and_then(Value::as_str);
            let budget = match item.get("max_budget") {
                Some(v) => Some(parse_budget_value(Some(v))?),
                None => None,
            };
            let window = item
                .get("budget_duration")
                .and_then(Value::as_str)
                .map(|s| parse_budget_window_str(Some(s)));
            let next = state
                .db
                .update_user(uid, alias, role, None, budget, window)?;
            updated.push(format_user_litellm(&next));
        }
    }
    record_audit(
        &state,
        &principal,
        "bulk_update",
        "users",
        "bulk",
        None,
        Some(json!({ "count": updated.len() })),
    );
    Ok(Json(json!({
        "results": updated,
        "total_updated": updated.len(),
    })))
}

async fn post_users_bulk_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let items = body
        .get("users")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| body.as_array().cloned())
        .ok_or_else(|| QuixigateError::Validation("expected 'users' array".to_owned()))?;
    let mut created = Vec::new();
    for item in items {
        let username = item
            .get("user_email")
            .or_else(|| item.get("username"))
            .or_else(|| item.get("user_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("user_{}", &Uuid::new_v4().simple().to_string()[..10]));
        let display = item
            .get("user_alias")
            .and_then(Value::as_str)
            .unwrap_or(&username);
        let role = parse_litellm_role(item.get("user_role").and_then(Value::as_str))
            .unwrap_or(UserRole::Member);
        let budget = parse_budget_value(item.get("max_budget"))?;
        let window = parse_budget_window_str(item.get("budget_duration").and_then(Value::as_str));
        let user = state
            .db
            .create_user(&username, display, role, None, budget, window)?;
        created.push(format_user_litellm(&user));
    }
    record_audit(
        &state,
        &principal,
        "bulk_create",
        "users",
        "bulk",
        None,
        Some(json!({ "count": created.len() })),
    );
    Ok(Json(
        json!({ "results": created, "total_created": created.len() }),
    ))
}

async fn post_users_bulk_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let ids: Vec<String> = body
        .get("user_ids")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let mut deleted = Vec::new();
    for uid in ids {
        if state.db.delete_user(&uid).is_ok() {
            deleted.push(uid);
        }
    }
    record_audit(
        &state,
        &principal,
        "bulk_delete",
        "users",
        "bulk",
        None,
        Some(json!({ "deleted": deleted })),
    );
    Ok(Json(
        json!({ "deleted_users": deleted, "count": deleted.len() }),
    ))
}

async fn post_user_password_change(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let new_password = body
        .get("new_password")
        .or_else(|| body.get("password"))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'new_password' is required".to_owned()))?;
    if new_password.len() < 8 {
        return Err(QuixigateError::Validation(
            "new_password must be at least 8 characters".to_owned(),
        )
        .into());
    }
    let target_uid = body
        .get("user_id")
        .and_then(Value::as_str)
        .unwrap_or(&principal.user.id);
    if target_uid != principal.user.id && principal.user.role != UserRole::Admin {
        return Err(
            QuixigateError::Forbidden("cannot change another user's password".to_owned()).into(),
        );
    }
    state
        .db
        .update_user(target_uid, None, None, Some(new_password), None, None)?;
    Ok(Json(json!({ "status": "success", "user_id": target_uid })))
}

async fn post_session_logout(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(tok) = extract_session_cookie(&headers)
        .or_else(|| extract_api_token(&headers, None))
        .filter(|t| t.starts_with(quixigate_core::brand::SESSION_TOKEN_PREFIX))
    {
        let _ = state.db.revoke_session(&tok);
    }
    let mut resp = Json(json!({ "status": "logged_out" })).into_response();
    if let Ok(cookie) = HeaderValue::from_str(&format_clear_session_cookie()) {
        resp.headers_mut().insert(header::SET_COOKIE, cookie);
    }
    resp
}

// ── 3. Customers / End-Users (`/customer/*`, `/end_user/*`) ──────────────────

fn format_end_user_litellm(eu: &EndUser, budget: Option<&BudgetRecord>) -> Value {
    let budget_json = budget.map(format_budget_litellm);
    json!({
        "user_id": eu.user_id,
        "alias": eu.alias,
        "blocked": eu.blocked,
        "spend": nanos_to_json_number(eu.spend_nanos),
        "spend_nanos": eu.spend_nanos.nanos(),
        "spend_usd": eu.spend_nanos.to_decimal_string(),
        "budget_id": eu.budget_id,
        "max_budget": eu.budget_limit_nanos.map(nanos_to_json_number),
        "budget_duration": eu.budget_window.as_str(),
        "budget_reset_at_ms": eu.window_reset_at_ms,
        "allowed_models": eu.allowed_models,
        "litellm_budget_table": budget_json,
        "metadata": eu.metadata,
        "created_at_ms": eu.created_at_ms,
        "updated_at_ms": eu.updated_at_ms,
    })
}

async fn post_end_user_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let user_id = body
        .get("user_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("'user_id' is required".to_owned()))?;
    let alias = body.get("alias").and_then(Value::as_str);
    let blocked = body.get("blocked").and_then(Value::as_bool);
    let budget_id = body.get("budget_id").map(Value::as_str);
    let max_budget = match body.get("max_budget") {
        Some(v) => Some(parse_budget_value(Some(v))?),
        None => None,
    };
    let window = body
        .get("budget_duration")
        .and_then(Value::as_str)
        .map(|s| parse_budget_window_str(Some(s)));
    let models = body
        .get("allowed_models")
        .or_else(|| body.get("models"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        });
    let eu = state.db.upsert_end_user(
        user_id,
        alias,
        blocked,
        budget_id,
        max_budget,
        window,
        models.as_deref(),
        body.get("metadata"),
    )?;
    let linked = eu
        .budget_id
        .as_deref()
        .and_then(|bid| state.db.get_budget(bid).ok());
    record_audit(
        &state,
        &principal,
        "create",
        "end_users",
        &eu.user_id,
        None,
        Some(format_end_user_litellm(&eu, linked.as_ref())),
    );
    Ok(Json(format_end_user_litellm(&eu, linked.as_ref())))
}

async fn post_end_user_update(
    state: State<AppState>,
    headers: HeaderMap,
    body: Json<Value>,
) -> ApiResult<Json<Value>> {
    post_end_user_new(state, headers, body).await
}

#[derive(Debug, Deserialize, Default)]
struct EndUserInfoQuery {
    end_user_id: Option<String>,
    user_id: Option<String>,
}

async fn get_end_user_info(
    State(state): State<AppState>,
    Query(q): Query<EndUserInfoQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_team_or_admin(&state, &headers)?;
    let id = q
        .end_user_id
        .as_deref()
        .or(q.user_id.as_deref())
        .ok_or_else(|| {
            QuixigateError::Validation("'end_user_id' query param is required".to_owned())
        })?;
    let eu = state.db.get_end_user(id)?;
    let linked = eu
        .budget_id
        .as_deref()
        .and_then(|bid| state.db.get_budget(bid).ok());
    Ok(Json(format_end_user_litellm(&eu, linked.as_ref())))
}

async fn get_end_user_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_team_or_admin(&state, &headers)?;
    let rows: Vec<Value> = state
        .db
        .list_end_users()?
        .into_iter()
        .map(|eu| {
            let linked = eu
                .budget_id
                .as_deref()
                .and_then(|bid| state.db.get_budget(bid).ok());
            format_end_user_litellm(&eu, linked.as_ref())
        })
        .collect();
    Ok(Json(Value::Array(rows)))
}

async fn post_end_user_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let mut ids: Vec<String> = body
        .get("user_ids")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if let Some(single) = body
        .get("user_id")
        .or_else(|| body.get("end_user_id"))
        .and_then(Value::as_str)
    {
        ids.push(single.trim().to_owned());
    }
    let deleted = state.db.delete_end_users(&ids)?;
    record_audit(
        &state,
        &principal,
        "delete",
        "end_users",
        "bulk",
        None,
        Some(json!({ "user_ids": ids, "deleted": deleted })),
    );
    Ok(Json(json!({ "deleted": deleted, "user_ids": ids })))
}

async fn set_end_user_block_flag(
    state: AppState,
    headers: HeaderMap,
    body: Value,
    blocked: bool,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let mut ids: Vec<String> = body
        .get("user_ids")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if let Some(single) = body
        .get("user_id")
        .or_else(|| body.get("end_user_id"))
        .and_then(Value::as_str)
    {
        ids.push(single.trim().to_owned());
    }
    if ids.is_empty() {
        return Err(
            QuixigateError::Validation("'user_ids' or 'user_id' required".to_owned()).into(),
        );
    }
    let mut out = Vec::new();
    for id in &ids {
        let eu = state.db.set_end_user_blocked(id, blocked)?;
        out.push(format_end_user_litellm(&eu, None));
    }
    record_audit(
        &state,
        &principal,
        if blocked { "block" } else { "unblock" },
        "end_users",
        &ids[0],
        None,
        Some(json!({ "blocked": blocked, "user_ids": ids })),
    );
    Ok(Json(json!({ "results": out, "blocked": blocked })))
}

async fn post_end_user_block(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    set_end_user_block_flag(state, headers, body, true).await
}

async fn post_end_user_unblock(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    set_end_user_block_flag(state, headers, body, false).await
}

// ── 4. Team Extensions ───────────────────────────────────────────────────────

async fn patch_team_by_id(
    State(state): State<AppState>,
    Path(team_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let existing = state.db.get_team(&team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&existing.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let alias = body
        .get("team_alias")
        .or_else(|| body.get("name"))
        .and_then(Value::as_str);
    let budget = match body.get("max_budget") {
        Some(v) => Some(parse_budget_value(Some(v))?),
        None => None,
    };
    let window = body
        .get("budget_duration")
        .and_then(Value::as_str)
        .map(|s| parse_budget_window_str(Some(s)));
    let mut team = state
        .db
        .update_team(&existing.id, alias, budget, window, None)?;
    if let Some(blocked) = body.get("blocked").and_then(Value::as_bool) {
        team = state.db.set_team_blocked(&team.id, blocked)?;
    }
    if let Some(dl) = body.get("disable_logging").and_then(Value::as_bool) {
        team = state.db.set_team_disable_logging(&team.id, dl)?;
    }
    if let Some(models_arr) = body.get("models").and_then(Value::as_array) {
        let models: Vec<String> = models_arr
            .iter()
            .filter_map(Value::as_str)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        team = state.db.set_team_allowed_models(&team.id, &models)?;
    }
    let mmb = parse_model_budget_map(body.get("model_max_budget"))?;
    let tags = body.get("tags").and_then(Value::as_array).map(|arr| {
        arr.iter()
            .filter_map(Value::as_str)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
    });
    if mmb.is_some() || tags.is_some() {
        team = state
            .db
            .set_team_policy(&team.id, tags.as_deref(), mmb.as_ref())?;
    }
    record_audit(
        &state,
        &principal,
        "patch",
        "teams",
        &team.id,
        None,
        Some(format_team_litellm(&team)),
    );
    Ok(Json(format_team_litellm(&team)))
}

#[derive(Debug, Deserialize, Default)]
struct TeamListQuery {
    page: Option<usize>,
    page_size: Option<usize>,
    user_id: Option<String>,
}

async fn get_v2_team_list(
    State(state): State<AppState>,
    Query(q): Query<TeamListQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let page = q.page.unwrap_or(1).max(1);
    let page_size = q.page_size.unwrap_or(100).clamp(1, 500);
    let mut teams = state.db.list_teams()?;
    if principal.user.role != UserRole::Admin {
        teams.retain(|t| t.member_ids.contains(&principal.user.id));
    }
    if let Some(uid) = q.user_id.as_deref() {
        teams.retain(|t| t.member_ids.iter().any(|m| m == uid));
    }
    let total = teams.len();
    let total_pages = total.div_ceil(page_size).max(1);
    let slice: Vec<Value> = teams
        .into_iter()
        .skip((page - 1) * page_size)
        .take(page_size)
        .map(|t| format_team_litellm(&t))
        .collect();
    Ok(Json(json!({
        "teams": slice,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    })))
}

async fn get_team_available(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let teams: Vec<Value> = state
        .db
        .list_teams()?
        .into_iter()
        .filter(|t| !t.blocked)
        .map(|t| format_team_litellm(&t))
        .collect();
    Ok(Json(Value::Array(teams)))
}

async fn get_team_member_me(
    State(state): State<AppState>,
    Path(team_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let team = state.db.get_team(&team_id)?;
    let member = team
        .members
        .iter()
        .find(|m| m.user_id == principal.user.id)
        .ok_or_else(|| {
            QuixigateError::NotFound(format!(
                "user '{}' is not a member of team '{}'",
                principal.user.id, team.id
            ))
        })?;
    Ok(Json(json!({
        "team_id": team.id,
        "user_id": member.user_id,
        "role": member.role,
        "spend": nanos_to_json_number(principal.user.spend_nanos),
        "max_budget": principal.user.budget_limit_nanos.map(nanos_to_json_number),
    })))
}

fn parse_member_pairs(val: Option<&Value>) -> Vec<(String, UserRole)> {
    let mut pairs = Vec::new();
    let Some(v) = val else {
        return pairs;
    };
    if let Some(arr) = v.as_array() {
        for item in arr {
            if let Some(uid) = item.as_str() {
                pairs.push((uid.trim().to_owned(), UserRole::Member));
            } else if let Some(uid) = item.get("user_id").and_then(Value::as_str) {
                let role = parse_litellm_role(item.get("role").and_then(Value::as_str))
                    .unwrap_or(UserRole::Member);
                pairs.push((uid.trim().to_owned(), role));
            }
        }
    } else if let Some(uid) = v.get("user_id").and_then(Value::as_str) {
        let role =
            parse_litellm_role(v.get("role").and_then(Value::as_str)).unwrap_or(UserRole::Member);
        pairs.push((uid.trim().to_owned(), role));
    }
    pairs
}

async fn post_team_member_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'team_id' is required".to_owned()))?;
    let team = state.db.get_team(team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&team.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let user_id = body
        .get("user_id")
        .or_else(|| body.get("member").and_then(|m| m.get("user_id")))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'user_id' is required".to_owned()))?;
    let role = parse_litellm_role(
        body.get("role")
            .or_else(|| body.get("member").and_then(|m| m.get("role")))
            .and_then(Value::as_str),
    )
    .unwrap_or(UserRole::Member);
    let updated = state
        .db
        .add_team_members_with_roles(&team.id, &[(user_id.to_owned(), role)])?;
    if let Some(mb) = body
        .get("max_budget_in_team")
        .or_else(|| body.get("max_budget"))
    {
        let limit = parse_budget_value(Some(mb))?;
        let _ = state
            .db
            .update_user(user_id, None, None, None, Some(limit), None);
    }
    record_audit(
        &state,
        &principal,
        "member_update",
        "teams",
        &updated.id,
        None,
        Some(json!({ "user_id": user_id, "role": role })),
    );
    Ok(Json(format_team_litellm(&updated)))
}

async fn post_team_bulk_member_add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'team_id' is required".to_owned()))?;
    let team = state.db.get_team(team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&team.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let pairs = if body.get("all_users").and_then(Value::as_bool) == Some(true) {
        state
            .db
            .list_users()?
            .into_iter()
            .map(|u| (u.id, UserRole::Member))
            .collect()
    } else {
        parse_member_pairs(body.get("members").or_else(|| body.get("member")))
    };
    let updated = state.db.add_team_members_with_roles(&team.id, &pairs)?;
    record_audit(
        &state,
        &principal,
        "bulk_member_add",
        "teams",
        &updated.id,
        None,
        Some(json!({ "added": pairs.len() })),
    );
    Ok(Json(format_team_litellm(&updated)))
}

async fn post_team_members_bulk_delete(
    State(state): State<AppState>,
    Path(team_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let mut team = state.db.get_team(&team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&team.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let user_ids: Vec<String> = body
        .get("user_ids")
        .or_else(|| body.get("members"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.as_str()
                        .or_else(|| v.get("user_id").and_then(Value::as_str))
                })
                .map(|s| s.trim().to_owned())
                .collect()
        })
        .unwrap_or_default();
    for uid in &user_ids {
        team = state.db.remove_team_member(&team.id, uid)?;
    }
    record_audit(
        &state,
        &principal,
        "bulk_member_delete",
        "teams",
        &team.id,
        None,
        Some(json!({ "removed": user_ids })),
    );
    Ok(Json(format_team_litellm(&team)))
}

async fn post_team_members_bulk_update(
    State(state): State<AppState>,
    Path(team_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let team = state.db.get_team(&team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&team.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let pairs = parse_member_pairs(body.get("members").or(Some(&body)));
    let updated = state.db.add_team_members_with_roles(&team.id, &pairs)?;
    Ok(Json(format_team_litellm(&updated)))
}

async fn post_team_block(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'team_id' is required".to_owned()))?;
    let team = state.db.set_team_blocked(team_id, true)?;
    record_audit(&state, &principal, "block", "teams", &team.id, None, None);
    Ok(Json(format_team_litellm(&team)))
}

async fn post_team_unblock(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'team_id' is required".to_owned()))?;
    let team = state.db.set_team_blocked(team_id, false)?;
    record_audit(&state, &principal, "unblock", "teams", &team.id, None, None);
    Ok(Json(format_team_litellm(&team)))
}

async fn post_team_disable_logging(
    State(state): State<AppState>,
    Path(team_id): Path<String>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let existing = state.db.get_team(&team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&existing.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let disable = body
        .as_ref()
        .and_then(|Json(v)| v.get("disable_logging").and_then(Value::as_bool))
        .unwrap_or(true);
    let team = state.db.set_team_disable_logging(&existing.id, disable)?;
    record_audit(
        &state,
        &principal,
        "disable_logging",
        "teams",
        &team.id,
        None,
        Some(json!({ "disable_logging": disable })),
    );
    Ok(Json(format_team_litellm(&team)))
}

async fn post_team_member_reset_spend(
    State(state): State<AppState>,
    Path((team_id, user_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let team = state.db.get_team(&team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&team.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    state.db.reset_spend(Some(&user_id), None, None, false)?;
    Ok(Json(json!({
        "status": "success",
        "team_id": team.id,
        "user_id": user_id,
    })))
}

async fn post_team_model_add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'team_id' is required".to_owned()))?;
    let existing = state.db.get_team(team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&existing.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let mut next_models = existing.allowed_models.clone();
    if let Some(arr) = body.get("models").and_then(Value::as_array) {
        for v in arr {
            if let Some(m) = v.as_str().map(str::trim).filter(|s| !s.is_empty())
                && !next_models.iter().any(|e| e == m)
            {
                next_models.push(m.to_owned());
            }
        }
    }
    let updated = state
        .db
        .set_team_allowed_models(&existing.id, &next_models)?;
    Ok(Json(format_team_litellm(&updated)))
}

async fn post_team_model_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_team_or_admin(&state, &headers)?;
    let team_id = body
        .get("team_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'team_id' is required".to_owned()))?;
    let existing = state.db.get_team(team_id)?;
    if principal.user.role != UserRole::Admin
        && !state.db.is_team_admin(&existing.id, &principal.user.id)?
    {
        return Err(
            QuixigateError::Forbidden("not an administrator of this team".to_owned()).into(),
        );
    }
    let remove: Vec<String> = body
        .get("models")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .collect()
        })
        .unwrap_or_default();
    let next_models: Vec<String> = existing
        .allowed_models
        .into_iter()
        .filter(|m| !remove.contains(m))
        .collect();
    let updated = state
        .db
        .set_team_allowed_models(&existing.id, &next_models)?;
    Ok(Json(format_team_litellm(&updated)))
}

#[derive(Debug, Deserialize, Default)]
struct TeamSpendQuery {
    team_id: Option<String>,
}

async fn get_team_spend_by_user(
    State(state): State<AppState>,
    Query(q): Query<TeamSpendQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_team_or_admin(&state, &headers)?;
    let filter = LogFilter {
        team_id: q.team_id,
        limit: 500,
        ..Default::default()
    };
    let rows = state.db.get_spend_report("user", &filter)?;
    let formatted: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "user_id": r.group_key,
                "spend": nanos_to_json_number(r.spend_nanos),
                "total_tokens": r.total_tokens,
                "requests": r.requests,
            })
        })
        .collect();
    Ok(Json(json!({ "results": formatted })))
}

// ── 5. Standalone Budgets (`/budget/*`, `/budgets`) ──────────────────────────

fn format_budget_litellm(b: &BudgetRecord) -> Value {
    json!({
        "budget_id": b.budget_id,
        "max_budget": b.max_budget_nanos.map(nanos_to_json_number),
        "soft_budget": b.soft_budget_nanos.map(nanos_to_json_number),
        "max_parallel_requests": b.max_parallel_requests,
        "tpm_limit": b.tpm_limit,
        "rpm_limit": b.rpm_limit,
        "budget_duration": b.budget_duration,
        "model_max_budget": format_model_budgets_json(&b.model_max_budget_nanos),
        "budget_reset_at_ms": b.budget_reset_at_ms,
        "created_at_ms": b.created_at_ms,
        "updated_at_ms": b.updated_at_ms,
    })
}

fn parse_budget_record_body(
    body: &Value,
    default_id: Option<String>,
) -> quixigate_core::Result<BudgetRecord> {
    let budget_id = body
        .get("budget_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or(default_id)
        .unwrap_or_else(|| format!("bdg_{}", Uuid::new_v4().simple()));
    let max_budget_nanos = parse_budget_value(body.get("max_budget"))?;
    let soft_budget_nanos = parse_budget_value(body.get("soft_budget"))?;
    let max_parallel_requests = body
        .get("max_parallel_requests")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    let tpm_limit = body.get("tpm_limit").and_then(Value::as_u64);
    let rpm_limit = body
        .get("rpm_limit")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    let budget_duration = body
        .get("budget_duration")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let reset_at = budget_duration
        .as_deref()
        .and_then(|d| parse_budget_window_str(Some(d)).next_reset_ms(now_ms()));
    let model_max_budget_nanos =
        parse_model_budget_map(body.get("model_max_budget"))?.unwrap_or_default();
    let now = now_ms();
    Ok(BudgetRecord {
        budget_id,
        max_budget_nanos,
        soft_budget_nanos,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        budget_duration,
        model_max_budget_nanos,
        budget_reset_at_ms: reset_at,
        created_at_ms: now,
        updated_at_ms: now,
    })
}

async fn post_budget_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let rec = parse_budget_record_body(&body, None)?;
    let saved = state.db.upsert_budget(&rec)?;
    record_audit(
        &state,
        &principal,
        "create",
        "budgets",
        &saved.budget_id,
        None,
        Some(format_budget_litellm(&saved)),
    );
    Ok(Json(format_budget_litellm(&saved)))
}

async fn post_budget_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let id = body
        .get("budget_id")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'budget_id' is required".to_owned()))?
        .to_owned();
    let rec = parse_budget_record_body(&body, Some(id))?;
    let saved = state.db.upsert_budget(&rec)?;
    record_audit(
        &state,
        &principal,
        "update",
        "budgets",
        &saved.budget_id,
        None,
        Some(format_budget_litellm(&saved)),
    );
    Ok(Json(format_budget_litellm(&saved)))
}

#[derive(Debug, Deserialize, Default)]
struct BudgetInfoQuery {
    budget_id: Option<String>,
}

async fn get_budget_info_query(
    State(state): State<AppState>,
    Query(q): Query<BudgetInfoQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let Some(id) = q.budget_id.as_deref() else {
        let all: Vec<Value> = state
            .db
            .list_budgets()?
            .iter()
            .map(format_budget_litellm)
            .collect();
        return Ok(Json(Value::Array(all)));
    };
    let rec = state.db.get_budget(id)?;
    Ok(Json(format_budget_litellm(&rec)))
}

async fn post_budget_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let ids: Vec<String> = body
        .get("budgets")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty()
        && let Some(single) = body.get("budget_id").and_then(Value::as_str)
    {
        let rec = state.db.get_budget(single)?;
        return Ok(Json(Value::Array(vec![format_budget_litellm(&rec)])));
    }
    let out: Vec<Value> = ids
        .iter()
        .filter_map(|id| state.db.get_budget(id).ok())
        .map(|b| format_budget_litellm(&b))
        .collect();
    Ok(Json(Value::Array(out)))
}

async fn get_budget_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    Ok(Json(json!([
        { "field_name": "max_budget", "field_type": "Float", "field_description": "Max budget in USD" },
        { "field_name": "soft_budget", "field_type": "Float", "field_description": "Soft alert threshold in USD" },
        { "field_name": "max_parallel_requests", "field_type": "Integer", "field_description": "Max concurrent in-flight requests" },
        { "field_name": "tpm_limit", "field_type": "Integer", "field_description": "Tokens per minute limit" },
        { "field_name": "rpm_limit", "field_type": "Integer", "field_description": "Requests per minute limit" },
        { "field_name": "budget_duration", "field_type": "String", "field_description": "Reset window (daily, weekly, monthly, total)" },
        { "field_name": "model_max_budget", "field_type": "Dict", "field_description": "Per-model USD budget map" }
    ])))
}

async fn get_budget_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let rows: Vec<Value> = state
        .db
        .list_budgets()?
        .iter()
        .map(format_budget_litellm)
        .collect();
    Ok(Json(Value::Array(rows)))
}

async fn post_budget_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let id = body
        .get("id")
        .or_else(|| body.get("budget_id"))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'id' or 'budget_id' is required".to_owned()))?;
    state.db.delete_budget(id)?;
    record_audit(&state, &principal, "delete", "budgets", id, None, None);
    Ok(Json(json!({ "deleted": true, "budget_id": id })))
}

// ── 6. Managed Tags (`/tag/*`) ───────────────────────────────────────────────

fn format_tag_litellm(t: &TagRecord) -> Value {
    json!({
        "tag_name": t.tag_name,
        "name": t.tag_name,
        "description": t.description,
        "models": t.models,
        "budget_id": t.budget_id,
        "max_budget": t.max_budget_nanos.map(nanos_to_json_number),
        "spend": nanos_to_json_number(t.spend_nanos),
        "spend_nanos": t.spend_nanos.nanos(),
        "metadata": t.metadata,
        "created_at_ms": t.created_at_ms,
        "updated_at_ms": t.updated_at_ms,
    })
}

async fn post_tag_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let tag_name = body
        .get("name")
        .or_else(|| body.get("tag_name"))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'name' or 'tag_name' is required".to_owned()))?
        .trim()
        .to_owned();
    let max_budget_nanos = parse_budget_value(body.get("max_budget"))?;
    let models: Vec<String> = body
        .get("models")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let now = now_ms();
    let rec = TagRecord {
        tag_name,
        description: body
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        models,
        budget_id: body
            .get("budget_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        max_budget_nanos,
        spend_nanos: NanoUsd::ZERO,
        metadata: body.get("metadata").cloned().unwrap_or_else(|| json!({})),
        created_at_ms: now,
        updated_at_ms: now,
    };
    let saved = state.db.upsert_tag(&rec)?;
    record_audit(
        &state,
        &principal,
        "upsert",
        "tags",
        &saved.tag_name,
        None,
        Some(format_tag_litellm(&saved)),
    );
    Ok(Json(format_tag_litellm(&saved)))
}

async fn post_tag_update(
    state: State<AppState>,
    headers: HeaderMap,
    body: Json<Value>,
) -> ApiResult<Json<Value>> {
    post_tag_new(state, headers, body).await
}

#[derive(Debug, Deserialize, Default)]
struct TagInfoQuery {
    name: Option<String>,
    tag_name: Option<String>,
}

async fn get_tag_info_query(
    State(state): State<AppState>,
    Query(q): Query<TagInfoQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let name =
        q.name.as_deref().or(q.tag_name.as_deref()).ok_or_else(|| {
            QuixigateError::Validation("'name' query param is required".to_owned())
        })?;
    let tag = state.db.get_tag(name)?;
    Ok(Json(format_tag_litellm(&tag)))
}

async fn post_tag_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let names: Vec<String> = body
        .get("names")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if names.is_empty()
        && let Some(single) = body
            .get("name")
            .or_else(|| body.get("tag_name"))
            .and_then(Value::as_str)
    {
        let tag = state.db.get_tag(single)?;
        return Ok(Json(json!({ single: format_tag_litellm(&tag) })));
    }
    let mut map = serde_json::Map::new();
    for n in names {
        if let Ok(t) = state.db.get_tag(&n) {
            map.insert(n, format_tag_litellm(&t));
        }
    }
    Ok(Json(Value::Object(map)))
}

async fn get_tag_list(State(state): State<AppState>, headers: HeaderMap) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let mut map = serde_json::Map::new();
    for t in state.db.list_tags()? {
        map.insert(t.tag_name.clone(), format_tag_litellm(&t));
    }
    Ok(Json(Value::Object(map)))
}

async fn post_tag_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let name = body
        .get("name")
        .or_else(|| body.get("tag_name"))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'name' is required".to_owned()))?;
    state.db.delete_tag(name)?;
    record_audit(&state, &principal, "delete", "tags", name, None, None);
    Ok(Json(json!({ "deleted": true, "name": name })))
}

// ── 7. Access Groups (`/access_group/*`, `/v1/access_group*`) ────────────────

fn format_access_group_litellm(ag: &AccessGroupRecord) -> Value {
    json!({
        "access_group": ag.access_group,
        "access_group_id": ag.access_group,
        "access_group_name": ag.access_group,
        "description": ag.description,
        "model_names": ag.model_names,
        "assigned_model_ids": ag.model_names,
        "mcp_server_ids": ag.mcp_server_ids,
        "assigned_mcp_server_ids": ag.mcp_server_ids,
        "max_budget": ag.max_budget_nanos.map(nanos_to_json_number),
        "tpm_limit": ag.tpm_limit,
        "rpm_limit": ag.rpm_limit,
        "spend": nanos_to_json_number(ag.spend_nanos),
        "created_at_ms": ag.created_at_ms,
        "updated_at_ms": ag.updated_at_ms,
    })
}

fn parse_access_group_body(
    body: &Value,
    fallback_name: Option<&str>,
) -> quixigate_core::Result<AccessGroupRecord> {
    let name = body
        .get("access_group")
        .or_else(|| body.get("access_group_name"))
        .or_else(|| body.get("access_group_id"))
        .and_then(Value::as_str)
        .or(fallback_name)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("'access_group' name is required".to_owned()))?
        .to_owned();
    let model_names: Vec<String> = body
        .get("model_names")
        .or_else(|| body.get("assigned_model_ids"))
        .or_else(|| body.get("models"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let mcp_server_ids: Vec<String> = body
        .get("mcp_server_ids")
        .or_else(|| body.get("assigned_mcp_server_ids"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let max_budget_nanos = parse_budget_value(body.get("max_budget"))?;
    let tpm_limit = body.get("tpm_limit").and_then(Value::as_u64);
    let rpm_limit = body
        .get("rpm_limit")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    let now = now_ms();
    Ok(AccessGroupRecord {
        access_group: name,
        description: body
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        model_names,
        mcp_server_ids,
        max_budget_nanos,
        tpm_limit,
        rpm_limit,
        spend_nanos: NanoUsd::ZERO,
        created_at_ms: now,
        updated_at_ms: now,
    })
}

async fn post_access_group_new(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let rec = parse_access_group_body(&body, None)?;
    let saved = state.db.upsert_access_group(&rec)?;
    record_audit(
        &state,
        &principal,
        "upsert",
        "access_groups",
        &saved.access_group,
        None,
        Some(format_access_group_litellm(&saved)),
    );
    Ok(Json(format_access_group_litellm(&saved)))
}

async fn get_access_group_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let list: Vec<Value> = state
        .db
        .list_access_groups()?
        .iter()
        .map(format_access_group_litellm)
        .collect();
    Ok(Json(json!({ "access_groups": list, "data": list })))
}

async fn get_access_group_info(
    State(state): State<AppState>,
    Path(access_group): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let ag = state.db.get_access_group(&access_group)?;
    Ok(Json(format_access_group_litellm(&ag)))
}

async fn put_access_group_update(
    State(state): State<AppState>,
    Path(access_group): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let existing = state.db.get_access_group(&access_group).ok();
    let mut rec = parse_access_group_body(&body, Some(&access_group))?;
    if body.get("model_names").is_none()
        && body.get("assigned_model_ids").is_none()
        && body.get("models").is_none()
        && let Some(ex) = &existing
    {
        rec.model_names = ex.model_names.clone();
    }
    let saved = state.db.upsert_access_group(&rec)?;
    record_audit(
        &state,
        &principal,
        "update",
        "access_groups",
        &saved.access_group,
        None,
        Some(format_access_group_litellm(&saved)),
    );
    Ok(Json(format_access_group_litellm(&saved)))
}

async fn delete_access_group_by_name(
    State(state): State<AppState>,
    Path(access_group): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    state.db.delete_access_group(&access_group)?;
    record_audit(
        &state,
        &principal,
        "delete",
        "access_groups",
        &access_group,
        None,
        None,
    );
    Ok(Json(
        json!({ "deleted": true, "access_group": access_group }),
    ))
}

async fn get_access_group_budget(
    State(state): State<AppState>,
    Path(access_group): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let ag = state.db.get_access_group(&access_group)?;
    Ok(Json(json!({
        "access_group": ag.access_group,
        "max_budget": ag.max_budget_nanos.map(nanos_to_json_number),
        "tpm_limit": ag.tpm_limit,
        "rpm_limit": ag.rpm_limit,
        "spend": nanos_to_json_number(ag.spend_nanos),
    })))
}

async fn put_access_group_budget(
    State(state): State<AppState>,
    Path(access_group): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let mut ag = state.db.get_access_group(&access_group)?;
    if body.get("max_budget").is_some() {
        ag.max_budget_nanos = parse_budget_value(body.get("max_budget"))?;
    }
    if let Some(tpm) = body.get("tpm_limit").and_then(Value::as_u64) {
        ag.tpm_limit = Some(tpm);
    }
    if let Some(rpm) = body.get("rpm_limit").and_then(Value::as_u64) {
        ag.rpm_limit = Some(rpm as u32);
    }
    let saved = state.db.upsert_access_group(&ag)?;
    record_audit(
        &state,
        &principal,
        "update_budget",
        "access_groups",
        &saved.access_group,
        None,
        Some(format_access_group_litellm(&saved)),
    );
    Ok(Json(format_access_group_litellm(&saved)))
}

async fn delete_access_group_budget(
    State(state): State<AppState>,
    Path(access_group): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let mut ag = state.db.get_access_group(&access_group)?;
    ag.max_budget_nanos = None;
    ag.tpm_limit = None;
    ag.rpm_limit = None;
    let saved = state.db.upsert_access_group(&ag)?;
    record_audit(
        &state,
        &principal,
        "delete_budget",
        "access_groups",
        &saved.access_group,
        None,
        None,
    );
    Ok(Json(format_access_group_litellm(&saved)))
}

// ── 8. Fallbacks (`/fallback`, `/fallback/{*model}`) ─────────────────────────

async fn post_fallback(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("'model' is required".to_owned()))?;
    let fallbacks: Vec<String> = body
        .get("fallback_models")
        .or_else(|| body.get("fallbacks"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .ok_or_else(|| {
            QuixigateError::Validation("'fallback_models' array is required".to_owned())
        })?;

    let existing = state.db.get_model_route(model)?;
    let primary_target = existing
        .as_ref()
        .and_then(|r| r.targets.first().cloned())
        .unwrap_or_else(|| RouteTarget {
            model: model.to_owned(),
            provider_id: None,
            credential_id: None,
        });
    let mut targets = vec![primary_target];
    for fb in &fallbacks {
        if !targets.iter().any(|t| t.model == *fb) {
            targets.push(RouteTarget {
                model: fb.clone(),
                provider_id: None,
                credential_id: None,
            });
        }
    }
    let now = now_ms();
    let route = ModelRoute {
        alias: model.to_owned(),
        targets,
        pool_strategy: existing
            .as_ref()
            .map(|r| r.pool_strategy)
            .unwrap_or(PoolStrategy::FillFirst),
        allow_paid_overflow: existing.as_ref().is_none_or(|r| r.allow_paid_overflow),
        blocked: existing.as_ref().is_some_and(|r| r.blocked),
        input_nanos_per_m: existing.as_ref().and_then(|r| r.input_nanos_per_m),
        output_nanos_per_m: existing.as_ref().and_then(|r| r.output_nanos_per_m),
        cache_read_nanos_per_m: existing.as_ref().and_then(|r| r.cache_read_nanos_per_m),
        team_id: existing.as_ref().and_then(|r| r.team_id.clone()),
        created_by: existing
            .as_ref()
            .and_then(|r| r.created_by.clone())
            .or_else(|| Some(principal.user.id.clone())),
        updated_by: Some(principal.user.id.clone()),
        created_at_ms: existing
            .as_ref()
            .and_then(|r| r.created_at_ms)
            .or(Some(now)),
        updated_at_ms: now,
    };
    let saved = state.db.upsert_model_route(&route)?;
    record_audit(
        &state,
        &principal,
        "set_fallbacks",
        "model_routes",
        &saved.alias,
        None,
        Some(json!({ "fallback_models": fallbacks })),
    );
    Ok(Json(json!({
        "model": saved.alias,
        "fallback_models": fallbacks,
        "targets": saved.targets,
    })))
}

async fn get_fallback_by_model(
    State(state): State<AppState>,
    Path(raw_model): Path<String>,
    headers: HeaderMap,
    uri: Uri,
) -> ApiResult<Response> {
    let model = raw_model.trim_start_matches('/').trim();
    if model == "login" {
        return Ok(crate::litellm_unsupported::fallback_handler(Method::GET, uri).await);
    }
    let _principal = auth_admin(&state, &headers)?;
    let Some(route) = state.db.get_model_route(model)? else {
        return Ok(Json(json!({
            "model": model,
            "fallback_models": [],
        }))
        .into_response());
    };
    let fallbacks: Vec<String> = route
        .targets
        .iter()
        .skip(1)
        .map(|t| t.model.clone())
        .collect();
    Ok(Json(json!({
        "model": route.alias,
        "fallback_models": fallbacks,
        "targets": route.targets,
    }))
    .into_response())
}

async fn delete_fallback_by_model(
    State(state): State<AppState>,
    Path(raw_model): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let model = raw_model.trim_start_matches('/').trim();
    if let Some(mut route) = state.db.get_model_route(model)? {
        if route.targets.len() <= 1 && route.targets.first().is_some_and(|t| t.model == model) {
            state.db.delete_model_route(model)?;
        } else {
            route.targets.truncate(1);
            state.db.upsert_model_route(&route)?;
        }
    }
    record_audit(
        &state,
        &principal,
        "delete_fallbacks",
        "model_routes",
        model,
        None,
        None,
    );
    Ok(Json(json!({ "deleted": true, "model": model })))
}

// ── 9. Models, Router & Credentials Extensions ───────────────────────────────

async fn get_v2_model_info(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let routes = state.db.list_model_routes()?;
    let catalog = state.router.list_available_models();
    let mut data = Vec::new();
    for r in routes {
        data.push(json!({
            "model_name": r.alias,
            "blocked": r.blocked,
            "litellm_params": {
                "model": r.targets.first().map(|t| t.model.as_str()).unwrap_or(&r.alias),
                "targets": r.targets,
            },
            "model_info": {
                "id": r.alias,
                "db_model": true,
                "team_id": r.team_id,
            }
        }));
    }
    for m in catalog {
        data.push(json!({
            "model_name": m.canonical_id,
            "blocked": false,
            "litellm_params": { "model": m.canonical_id },
            "model_info": {
                "id": m.canonical_id,
                "max_tokens": m.max_output_tokens,
                "max_input_tokens": m.context_window,
                "input_cost_per_token": (m.input_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64),
                "output_cost_per_token": (m.output_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64),
            }
        }));
    }
    Ok(Json(json!({ "data": data, "total_count": data.len() })))
}

async fn post_model_block(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let model = body
        .get("model_id")
        .or_else(|| body.get("model"))
        .or_else(|| body.get("model_name"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            QuixigateError::Validation("'model_id' or 'model' is required".to_owned())
        })?;
    let r = state.db.set_model_route_blocked(model, true)?;
    record_audit(
        &state,
        &principal,
        "block",
        "model_routes",
        &r.alias,
        None,
        None,
    );
    Ok(Json(json!({ "model_id": r.alias, "blocked": r.blocked })))
}

async fn post_model_unblock(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let model = body
        .get("model_id")
        .or_else(|| body.get("model"))
        .or_else(|| body.get("model_name"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            QuixigateError::Validation("'model_id' or 'model' is required".to_owned())
        })?;
    let r = state.db.set_model_route_blocked(model, false)?;
    record_audit(
        &state,
        &principal,
        "unblock",
        "model_routes",
        &r.alias,
        None,
        None,
    );
    Ok(Json(json!({ "model_id": r.alias, "blocked": r.blocked })))
}

async fn get_model_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    Ok(Json(json!([
        { "field_name": "model_name", "field_type": "String" },
        { "field_name": "litellm_params", "field_type": "Dict" },
        { "field_name": "model_info", "field_type": "Dict" }
    ])))
}

async fn get_model_cost_map_source() -> Json<Value> {
    Json(json!({
        "source": MODELS_DEV_URL,
        "embedded_fallback": true,
        "accounting_unit": "NanoUsd(i64)",
    }))
}

async fn get_model_deprecations() -> Json<Value> {
    Json(json!({ "deprecated_models": [], "count": 0 }))
}

async fn post_reload_model_cost_map(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let count = state
        .router
        .catalog()
        .sync_from_url(state.router.inference_http(), MODELS_DEV_URL)
        .await
        .unwrap_or_else(|_| state.router.list_available_models().len());
    Ok(Json(json!({
        "status": "reloaded",
        "models_loaded": count,
        "source": MODELS_DEV_URL,
    })))
}

async fn get_router_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let settings = state.db.get_settings()?;
    Ok(Json(json!({
        "routing_strategy": "fill_first",
        "supported_strategies": ["fill_first", "round_robin", "least_busy", "latency_based"],
        "enable_burst_overflow": settings.openrouter_burst_enabled,
        "response_cache_enabled": settings.cache_enabled_default,
        "response_cache_ttl_secs": settings.cache_ttl_secs,
        "num_retries": 3,
    })))
}

async fn get_router_fields() -> Json<Value> {
    Json(json!({
        "fields": [
            { "name": "routing_strategy", "options": ["fill_first", "round_robin", "least_busy", "latency_based"] },
            { "name": "enable_burst_overflow", "type": "boolean" },
            { "name": "response_cache_enabled", "type": "boolean" }
        ]
    }))
}

async fn get_adaptive_router_state(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let creds = state.db.list_credentials()?;
    Ok(Json(json!({
        "status": "active",
        "credentials_tracked": creds.len(),
        "healthy_credentials": creds.iter().filter(|c| c.enabled && c.status.as_str() == "ok").count(),
    })))
}

pub async fn patch_litellm_credential(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let cred_vals = body.get("credential_values").unwrap_or(&body);
    let info = body.get("credential_info").unwrap_or(&body);
    let label = body
        .get("credential_name")
        .or_else(|| body.get("label"))
        .and_then(Value::as_str);
    let base_url = cred_vals
        .get("api_base")
        .or_else(|| cred_vals.get("base_url"))
        .map(Value::as_str);
    let priority = info
        .get("priority")
        .and_then(Value::as_i64)
        .map(|v| v as i32);
    let enabled = info.get("enabled").and_then(Value::as_bool);
    let overflow = info.get("allow_api_overflow").and_then(Value::as_bool);

    let updated = state
        .db
        .update_credential_meta(&id, label, base_url, priority, enabled, overflow)?;
    if let Some(raw_key) = cred_vals
        .get("api_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let secret = if updated.auth_kind == AuthKind::SetupToken {
            CredentialSecret::AnthropicSetupToken {
                token: raw_key.to_owned(),
            }
        } else {
            CredentialSecret::ApiKey {
                key: raw_key.to_owned(),
            }
        };
        state
            .db
            .update_credential_secret(&updated.id, &secret, updated.base_url.as_deref())?;
    }
    sync_local_scanner_from_db(&state.db, state.router.local_scanner());
    record_audit(
        &state,
        &principal,
        "patch",
        "credentials",
        &updated.id,
        None,
        Some(json!({ "label": updated.label, "enabled": updated.enabled })),
    );
    Ok(Json(json!({
        "credential_id": updated.id,
        "credential_name": updated.label,
        "credential_info": {
            "provider_id": updated.provider_id,
            "auth_kind": updated.auth_kind,
            "base_url": updated.base_url,
            "priority": updated.priority,
            "enabled": updated.enabled,
            "status": updated.status,
        }
    })))
}

async fn get_credential_by_name(
    State(state): State<AppState>,
    Path(raw_name): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let name = raw_name.trim_start_matches('/').trim();
    let cred = state
        .db
        .list_credentials()?
        .into_iter()
        .find(|c| c.label == name || c.id == name)
        .ok_or_else(|| QuixigateError::NotFound(format!("credential '{name}' not found")))?;
    Ok(Json(json!({
        "credential_id": cred.id,
        "credential_name": cred.label,
        "credential_info": {
            "provider_id": cred.provider_id,
            "auth_kind": cred.auth_kind,
            "base_url": cred.base_url,
            "priority": cred.priority,
            "enabled": cred.enabled,
            "status": cred.status,
        }
    })))
}

async fn get_credentials_by_model(
    State(state): State<AppState>,
    Path(raw_model): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let model = raw_model.trim_start_matches('/').trim();
    let provider_hint = model.split_once('/').map(|(p, _)| p).unwrap_or("");
    let creds: Vec<Value> = state
        .db
        .list_credentials()?
        .into_iter()
        .filter(|c| {
            provider_hint.is_empty()
                || c.provider_id == provider_hint
                || c.provider_id == "openrouter"
        })
        .map(|c| {
            json!({
                "credential_id": c.id,
                "credential_name": c.label,
                "provider_id": c.provider_id,
                "status": c.status,
            })
        })
        .collect();
    Ok(Json(json!({ "credentials": creds })))
}

// ── 10. MCP Registry (`/v1/mcp/server*`, `/v1/mcp/tools`, `/mcp-rest/*`) ─────

fn mcp_server_id_for_url(idx: usize, url: &str) -> String {
    let sanitized: String = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(32)
        .collect();
    format!("mcp_{idx}_{sanitized}")
}

fn format_mcp_servers(urls: &[String]) -> Vec<Value> {
    urls.iter()
        .enumerate()
        .map(|(idx, url)| {
            let sid = mcp_server_id_for_url(idx, url);
            json!({
                "server_id": sid,
                "server_name": sid,
                "url": url,
                "transport": "http",
                "status": "active",
            })
        })
        .collect()
}

async fn get_mcp_servers(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.get_settings()?;
    Ok(Json(Value::Array(format_mcp_servers(
        &settings.mcp_server_urls,
    ))))
}

async fn post_or_put_mcp_server(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let url = body
        .get("url")
        .or_else(|| body.get("server_url"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("'url' is required".to_owned()))?;
    let mut settings = state.db.get_settings()?;
    if !settings.mcp_server_urls.iter().any(|u| u == url) {
        settings.mcp_server_urls.push(url.to_owned());
        state.db.update_settings(&settings)?;
    }
    let idx = settings
        .mcp_server_urls
        .iter()
        .position(|u| u == url)
        .unwrap_or(0);
    let sid = mcp_server_id_for_url(idx, url);
    record_audit(
        &state,
        &principal,
        "register_mcp_server",
        "settings",
        &sid,
        None,
        Some(json!({ "url": url })),
    );
    Ok(Json(json!({
        "server_id": sid,
        "server_name": body.get("server_name").and_then(Value::as_str).unwrap_or(&sid),
        "url": url,
        "transport": "http",
    })))
}

async fn get_mcp_servers_health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.get_settings()?;
    let mut statuses = Vec::new();
    for (idx, url) in settings.mcp_server_urls.iter().enumerate() {
        let tools = fetch_mcp_http_tools(state.router.inference_http(), url).await;
        statuses.push(json!({
            "server_id": mcp_server_id_for_url(idx, url),
            "url": url,
            "healthy": true,
            "tools_discovered": tools.len(),
        }));
    }
    Ok(Json(json!({
        "builtin_mcp_enabled": settings.mcp_server_enabled,
        "servers": statuses,
    })))
}

async fn get_mcp_server_by_id(
    State(state): State<AppState>,
    Path(server_id): Path<String>,
    headers: HeaderMap,
    uri: Uri,
) -> ApiResult<Response> {
    if server_id == "submissions" {
        return Ok(crate::litellm_unsupported::fallback_handler(Method::GET, uri).await);
    }
    let _principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.get_settings()?;
    let found = format_mcp_servers(&settings.mcp_server_urls)
        .into_iter()
        .find(|v| {
            v["server_id"].as_str() == Some(server_id.as_str())
                || v["url"].as_str() == Some(server_id.as_str())
        })
        .ok_or_else(|| QuixigateError::NotFound(format!("MCP server '{server_id}' not found")))?;
    Ok(Json(found).into_response())
}

async fn put_mcp_server_by_id(
    State(state): State<AppState>,
    Path(server_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let new_url = body
        .get("url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("'url' is required".to_owned()))?;
    let mut settings = state.db.get_settings()?;
    let mut replaced = false;
    for (idx, existing_url) in settings.mcp_server_urls.iter_mut().enumerate() {
        if mcp_server_id_for_url(idx, existing_url) == server_id || *existing_url == server_id {
            *existing_url = new_url.to_owned();
            replaced = true;
            break;
        }
    }
    if !replaced {
        settings.mcp_server_urls.push(new_url.to_owned());
    }
    state.db.update_settings(&settings)?;
    record_audit(
        &state,
        &principal,
        "update_mcp_server",
        "settings",
        &server_id,
        None,
        Some(json!({ "url": new_url })),
    );
    Ok(Json(
        json!({ "server_id": server_id, "url": new_url, "transport": "http" }),
    ))
}

async fn delete_mcp_server_by_id(
    State(state): State<AppState>,
    Path(server_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let mut settings = state.db.get_settings()?;
    let before = settings.mcp_server_urls.len();
    settings.mcp_server_urls = settings
        .mcp_server_urls
        .into_iter()
        .enumerate()
        .filter(|(idx, url)| mcp_server_id_for_url(*idx, url) != server_id && *url != server_id)
        .map(|(_, url)| url)
        .collect();
    let removed = before.saturating_sub(settings.mcp_server_urls.len());
    state.db.update_settings(&settings)?;
    record_audit(
        &state,
        &principal,
        "delete_mcp_server",
        "settings",
        &server_id,
        None,
        Some(json!({ "removed": removed })),
    );
    Ok(Json(
        json!({ "deleted": removed > 0, "server_id": server_id }),
    ))
}

async fn get_mcp_tools(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.get_settings()?;
    let mut tools = vec![
        json!({ "name": "quixigate_web_search", "source": "builtin", "description": "Live web search across 6-engine cascade" }),
        json!({ "name": "quixigate_web_fetch", "source": "builtin", "description": "SSRF-guarded URL readable extraction" }),
        json!({ "name": "quixigate_list_models", "source": "builtin", "description": "List available routed models" }),
    ];
    for url in &settings.mcp_server_urls {
        for t in fetch_mcp_http_tools(state.router.inference_http(), url).await {
            tools.push(json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "source": url,
            }));
        }
    }
    Ok(Json(json!({ "tools": tools })))
}

async fn post_mcp_rest_tools_call(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let name = body
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'name' is required".to_owned()))?;
    let arguments = body.get("arguments").cloned().unwrap_or_else(|| json!({}));
    if name == "quixigate_web_search" {
        let q = arguments.get("query").and_then(Value::as_str).unwrap_or("");
        let res = state
            .router
            .execute_web_search_with_tool(q, 5, None)
            .await
            .map_err(|e| QuixigateError::Internal(e.to_string()))?;
        return Ok(Json(
            json!({ "content": [{ "type": "text", "text": res.to_string() }] }),
        ));
    }
    let settings = state.db.get_settings()?;
    let target_url = body
        .get("server_url")
        .and_then(Value::as_str)
        .or_else(|| settings.mcp_server_urls.first().map(String::as_str))
        .ok_or_else(|| {
            QuixigateError::Validation("no MCP server URL configured for tool call".to_owned())
        })?;
    let rpc = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": name, "arguments": arguments }
    });
    let resp = state
        .router
        .inference_http()
        .post(target_url)
        .timeout(Duration::from_secs(15))
        .json(&rpc)
        .send()
        .await
        .map_err(|e| QuixigateError::Upstream {
            provider: "mcp".to_owned(),
            status: 502,
            message: format!("MCP call failed: {e}"),
        })?;
    let val: Value = resp.json().await.unwrap_or_default();
    Ok(Json(val.get("result").cloned().unwrap_or(val)))
}

async fn post_mcp_rest_test_connection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let url = body
        .get("url")
        .or_else(|| body.get("server_url"))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'url' is required".to_owned()))?;
    let tools = fetch_mcp_http_tools(state.router.inference_http(), url).await;
    Ok(Json(json!({
        "status": "ok",
        "url": url,
        "tools_count": tools.len(),
    })))
}

async fn post_mcp_rest_test_tools_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let url = body
        .get("url")
        .or_else(|| body.get("server_url"))
        .and_then(Value::as_str)
        .ok_or_else(|| QuixigateError::Validation("'url' is required".to_owned()))?;
    let tools: Vec<Value> = fetch_mcp_http_tools(state.router.inference_http(), url)
        .await
        .into_iter()
        .map(|t| json!({ "name": t.name, "description": t.description, "inputSchema": t.input_schema }))
        .collect();
    Ok(Json(json!({ "tools": tools })))
}

// ── 11. Spend, Activity, Health, Config, Public & Introspection ──────────────

#[derive(Debug, Deserialize, Default)]
struct SpendFilterQuery {
    team_id: Option<String>,
    user_id: Option<String>,
    api_key: Option<String>,
    page: Option<usize>,
    page_size: Option<usize>,
}

fn ensure_spend_access(state: &AppState, headers: &HeaderMap) -> quixigate_core::Result<()> {
    let principal = authenticate_inference(state, headers, None)?;
    if let Some(vk) = &principal.virtual_key
        && !vk.allow_spend_read
    {
        return Err(QuixigateError::Forbidden(
            "virtual key does not have get_spend_routes permission".to_owned(),
        ));
    }
    Ok(())
}

async fn get_daily_activity(
    State(state): State<AppState>,
    Query(q): Query<SpendFilterQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let filter = LogFilter {
        team_id: q.team_id,
        user_id: q.user_id,
        virtual_key_id: q.api_key,
        limit: 500,
        ..Default::default()
    };
    let rows = state.db.get_spend_report("day", &filter)?;
    let mut total_spend = NanoUsd::ZERO;
    let mut total_tokens = 0u64;
    let mut total_reqs = 0u64;
    let results: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            total_spend = NanoUsd(total_spend.nanos().saturating_add(r.spend_nanos.nanos()));
            total_tokens += r.total_tokens;
            total_reqs += r.requests;
            json!({
                "date": r.group_key,
                "metrics": {
                    "spend": nanos_to_json_number(r.spend_nanos),
                    "prompt_tokens": r.input_tokens,
                    "completion_tokens": r.output_tokens,
                    "cache_read_input_tokens": r.cached_tokens,
                    "total_tokens": r.total_tokens,
                    "api_requests": r.requests,
                }
            })
        })
        .collect();
    Ok(Json(json!({
        "results": results,
        "metadata": {
            "total_spend": nanos_to_json_number(total_spend),
            "total_tokens": total_tokens,
            "total_api_requests": total_reqs,
        }
    })))
}

async fn get_spend_keys(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let keys: Vec<Value> = state
        .db
        .list_virtual_keys()?
        .iter()
        .map(|k| format_virtual_key_litellm(k, None))
        .collect();
    Ok(Json(Value::Array(keys)))
}

async fn get_spend_users(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let users: Vec<Value> = state
        .db
        .list_users()?
        .iter()
        .map(format_user_litellm)
        .collect();
    Ok(Json(Value::Array(users)))
}

async fn get_spend_logs_v2(
    State(state): State<AppState>,
    Query(q): Query<SpendFilterQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let page = q.page.unwrap_or(1).max(1);
    let page_size = q.page_size.unwrap_or(50).clamp(1, 500);
    let logs = state.db.list_request_logs(&LogFilter {
        team_id: q.team_id,
        user_id: q.user_id,
        virtual_key_id: q.api_key,
        limit: page_size,
        offset: (page - 1) * page_size,
        ..Default::default()
    })?;
    let data: Vec<Value> = logs
        .into_iter()
        .map(|l| {
            json!({
                "request_id": l.id,
                "model": l.resolved_model,
                "requested_model": l.requested_model,
                "custom_llm_provider": l.provider_id,
                "spend": nanos_to_json_number(l.cost_nanos),
                "total_tokens": l.input_tokens + l.output_tokens,
                "prompt_tokens": l.input_tokens,
                "completion_tokens": l.output_tokens,
                "user": l.user_id,
                "team_id": l.team_id,
                "api_key": l.virtual_key_id,
                "request_tags": l.tags,
                "status_code": l.status_code,
                "startTime": l.timestamp_ms,
            })
        })
        .collect();
    Ok(Json(json!({
        "data": data,
        "page": page,
        "page_size": page_size,
    })))
}

async fn build_grouped_spend_report(
    state: &AppState,
    headers: &HeaderMap,
    group_by: &str,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(state, headers)?;
    let rows = state.db.get_spend_report(
        group_by,
        &LogFilter {
            limit: 500,
            ..Default::default()
        },
    )?;
    let out: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "group_key": r.group_key,
                "spend": nanos_to_json_number(r.spend_nanos),
                "spend_nanos": r.spend_nanos.nanos(),
                "total_tokens": r.total_tokens,
                "requests": r.requests,
            })
        })
        .collect();
    Ok(Json(Value::Array(out)))
}

async fn get_key_spend_report(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "key").await
}

async fn get_user_spend_report(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "user").await
}

async fn get_team_spend_report(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "team").await
}

async fn get_global_spend_teams(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "team").await
}

async fn get_global_spend_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "model").await
}

async fn get_global_spend_provider(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "provider").await
}

async fn get_global_spend_tags(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    build_grouped_spend_report(&state, &headers, "tag").await
}

async fn get_global_all_tag_names(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let mut tags: Vec<String> = state
        .db
        .list_tags()?
        .into_iter()
        .map(|t| t.tag_name)
        .collect();
    for r in state
        .db
        .get_spend_report("tag", &LogFilter::default())
        .unwrap_or_default()
    {
        if r.group_key != "(untagged)" && !tags.contains(&r.group_key) {
            tags.push(r.group_key);
        }
    }
    tags.sort();
    Ok(Json(json!({ "tag_names": tags })))
}

async fn get_spend_logs_end_users(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let rows: Vec<Value> = state
        .db
        .list_end_users()?
        .into_iter()
        .map(|eu| format_end_user_litellm(&eu, None))
        .collect();
    Ok(Json(Value::Array(rows)))
}

async fn post_global_spend_end_users(
    state: State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    get_spend_logs_end_users(state, headers).await
}

async fn post_global_spend_refresh() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "mode": "real_time_sqlite_wal",
        "message": "QuixiGate commits NanoUsd spend atomically on every request; no materialized view refresh is required."
    }))
}

async fn get_provider_budgets(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    ensure_spend_access(&state, &headers)?;
    let rows = state
        .db
        .get_spend_report("provider", &LogFilter::default())?;
    let mut providers = BTreeMap::new();
    for r in rows {
        providers.insert(
            r.group_key,
            json!({
                "spend": nanos_to_json_number(r.spend_nanos),
                "budget_limit": null,
            }),
        );
    }
    Ok(Json(json!({ "providers": providers })))
}

async fn get_test_ping() -> Json<Value> {
    Json(json!({ "status": "connected", "service": "quixigate" }))
}

async fn get_routes_introspection(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let spec = crate::openapi::handle_openapi_json(State(state), headers).await;
    let paths: Vec<String> = spec
        .0
        .get("paths")
        .and_then(Value::as_object)
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    Ok(Json(json!({ "routes": paths, "count": paths.len() })))
}

async fn get_gateway_settings_litellm(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let s = state.db.get_settings()?;
    Ok(Json(json!({
        "values": {
            "require_auth": s.require_auth,
            "ui_enabled": state.is_ui_enabled(),
            "enable_burst_overflow": s.openrouter_burst_enabled,
            "response_cache_enabled": s.cache_enabled_default,
            "response_cache_ttl_secs": s.cache_ttl_secs,
            "secret_scanner_mode": s.secret_scanner_mode,
            "mcp_server_enabled": s.mcp_server_enabled,
            "otlp_endpoint": s.otlp_endpoint,
        }
    })))
}

async fn get_health_services(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let creds = state.db.list_credentials()?;
    Ok(Json(json!({
        "status": "healthy",
        "sqlite": "ok",
        "vault": "ok",
        "configured_credentials": creds.len(),
    })))
}

async fn get_health_readiness_details(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let creds = state.db.list_credentials()?;
    let models = state.router.list_available_models();
    Ok(Json(json!({
        "status": "connected",
        "db": "connected",
        "cache": "sqlite_wal",
        "credentials_count": creds.len(),
        "models_available": models.len(),
        "version": env!("CARGO_PKG_VERSION"),
    })))
}

async fn get_health_backlog(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let metrics = state.db.get_dashboard_metrics()?;
    Ok(Json(json!({
        "active_requests": metrics.active_requests,
        "queued_logs": 0,
    })))
}

async fn get_health_drain() -> Json<Value> {
    Json(json!({ "draining": false, "status": "serving" }))
}

async fn post_health_test_connection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let litellm_params = body.get("litellm_params").unwrap_or(&body);
    let model = litellm_params
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("openai/gpt-4o-mini");
    let has_api_key = litellm_params
        .get("api_key")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty());
    let resolved = state.router.resolve_http_candidates(model).await;
    Ok(Json(json!({
        "status": if has_api_key || resolved.is_ok() { "success" } else { "unconfigured" },
        "model": model,
    })))
}

async fn post_config_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = auth_admin(&state, &headers)?;
    let mut settings = state.db.get_settings()?;
    let gs = body.get("general_settings").unwrap_or(&body);
    if let Some(b) = gs.get("require_auth").and_then(Value::as_bool) {
        settings.require_auth = b;
    }
    if let Some(b) = gs
        .get("enable_burst_overflow")
        .or_else(|| gs.get("openrouter_burst_enabled"))
        .and_then(Value::as_bool)
    {
        settings.openrouter_burst_enabled = b;
    }
    if let Some(b) = gs
        .get("response_cache_enabled")
        .or_else(|| gs.get("cache_enabled_default"))
        .or_else(|| gs.get("cache"))
        .and_then(Value::as_bool)
    {
        settings.cache_enabled_default = b;
    }
    state.db.update_settings(&settings)?;
    record_audit(
        &state,
        &principal,
        "config_update",
        "settings",
        "global",
        None,
        Some(body),
    );
    Ok(Json(json!({ "status": "updated" })))
}

async fn get_config_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = auth_admin(&state, &headers)?;
    let settings = state.db.get_settings()?;
    let routes = state.db.list_model_routes()?;
    Ok(Json(json!({
        "general_settings": settings,
        "model_list": routes,
    })))
}

async fn get_public_model_hub(State(state): State<AppState>) -> Json<Value> {
    let models: Vec<Value> = state
        .router
        .list_available_models()
        .into_iter()
        .map(|m| {
            json!({
                "model_group": m.canonical_id,
                "providers": [m.author],
                "max_input_tokens": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "supports_vision": m.supports_vision,
                "supports_function_calling": m.supports_tools,
                "supports_reasoning": m.supports_reasoning,
            })
        })
        .collect();
    Json(Value::Array(models))
}

async fn get_public_model_hub_facet(
    State(state): State<AppState>,
    Path(facet): Path<String>,
) -> Json<Value> {
    let models = state.router.list_available_models();
    match facet.as_str() {
        "providers" => {
            let mut provs: Vec<String> = models.into_iter().map(|m| m.author).collect();
            provs.sort();
            provs.dedup();
            Json(json!({ "providers": provs }))
        }
        _ => Json(json!({ "facet": facet, "total": models.len() })),
    }
}

async fn get_public_model_hub_info(State(state): State<AppState>) -> Json<Value> {
    let settings = state.db.get_settings().unwrap_or_default();
    Json(json!({
        "docs_title": settings.docs_title.unwrap_or_else(|| "QuixiGate Model Hub".to_owned()),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn get_public_providers() -> Json<Value> {
    let list: Vec<Value> = AISDK_BUILTIN_PROVIDERS
        .iter()
        .map(|(id, display_name)| {
            let env_key = format!("{}_API_KEY", id.to_ascii_uppercase().replace('-', "_"));
            json!({
                "provider": id,
                "display_name": display_name,
                "env_key": env_key,
            })
        })
        .collect();
    Json(Value::Array(list))
}

async fn get_public_provider_fields() -> Json<Value> {
    let list: Vec<Value> = AISDK_BUILTIN_PROVIDERS
        .iter()
        .map(|(id, display_name)| {
            let env_key = format!("{}_API_KEY", id.to_ascii_uppercase().replace('-', "_"));
            json!({
                "provider": id,
                "provider_display_name": display_name,
                "credential_fields": [
                    { "key": "api_key", "label": format!("{display_name} API Key ({env_key})"), "field_type": "password", "required": true },
                    { "key": "api_base", "label": "Custom Base URL", "field_type": "text", "required": false }
                ]
            })
        })
        .collect();
    Json(Value::Array(list))
}

async fn get_public_endpoints() -> Json<Value> {
    Json(json!({
        "endpoints": [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1beta/models/{model}:generateContent",
            "/v1/embeddings",
            "/v1/images/generations",
            "/v1/audio/speech",
            "/v1/audio/transcriptions",
            "/v1/moderations",
            "/v1/rerank",
            "/v1/search",
            "/mcp"
        ]
    }))
}

async fn get_public_model_cost_map(State(state): State<AppState>) -> Json<Value> {
    let mut map = serde_json::Map::new();
    for m in state.router.list_available_models() {
        map.insert(
            m.canonical_id.clone(),
            json!({
                "max_input_tokens": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "input_cost_per_token": (m.input_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64),
                "output_cost_per_token": (m.output_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64),
                "litellm_provider": m.author,
            }),
        );
    }
    Json(Value::Object(map))
}
