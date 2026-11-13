//! `POST /v1/embeddings`, `GET /v1/models`, OpenRouter-compatible `/api/v1/key` &
//! `/api/v1/credits`, `/health/*`, and Prometheus `/metrics`.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STD;
use quixigate_core::{IngressProtocol, NANOS_PER_USD, QuixigateError, UserRole, now_ms};
use quixigate_telemetry::OpenInferencePayload;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::{
    attach_gateway_headers, authorize_inference_request, finalize_request_accounting,
    is_catalog_model_routable,
};
use crate::state::{ApiResult, AppState};

pub async fn handle_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("missing 'model' field".to_owned()))?
        .to_owned();

    let inputs: Vec<String> = match &body["input"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => {
            return Err(QuixigateError::Validation(
                "'input' must be a string or array of strings".to_owned(),
            )
            .into());
        }
    };
    if inputs.is_empty() {
        return Err(
            QuixigateError::Validation("embeddings 'input' cannot be empty".to_owned()).into(),
        );
    }

    let dimensions = body["dimensions"].as_u64().map(|v| v as usize);
    let base64_format = body["encoding_format"].as_str() == Some("base64");

    let est_tokens: u64 = inputs.iter().map(|s| ((s.len() / 4) as u64).max(1)).sum();
    let (_, est_cost) = state
        .router
        .estimate_preflight_budget(&requested_model, est_tokens, 0)?;
    let (_conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        &requested_model,
        est_tokens,
        est_cost,
    )?;

    let request_id = format!("emb_{}", Uuid::new_v4().simple());
    state.telemetry.inc_active();
    let res = state
        .router
        .execute_embeddings(&requested_model, &inputs, dimensions)
        .await;
    state.telemetry.dec_active();

    let routed = res?;
    let data_items: Vec<Value> = routed
        .output
        .vectors
        .iter()
        .enumerate()
        .map(|(idx, vec_f32)| {
            let emb_val = if base64_format {
                let mut raw = Vec::with_capacity(vec_f32.len() * 4);
                for &val in vec_f32 {
                    raw.extend_from_slice(&val.to_le_bytes());
                }
                Value::String(BASE64_STD.encode(raw))
            } else {
                json!(vec_f32)
            };
            json!({
                "object": "embedding",
                "index": idx,
                "embedding": emb_val
            })
        })
        .collect();

    let resp_json = json!({
        "object": "list",
        "data": data_items,
        "model": routed.resolved_model,
        "usage": {
            "prompt_tokens": routed.output.prompt_tokens,
            "total_tokens": routed.output.total_tokens
        }
    });

    let oi = OpenInferencePayload {
        invocation_parameters_json: dimensions.map(|d| json!({ "dimensions": d }).to_string()),
        input_messages: inputs
            .into_iter()
            .map(|t| ("input".to_owned(), t))
            .collect(),
        output_text: None,
        output_tool_calls: Vec::new(),
        tool_schemas_json: Vec::new(),
        session_id: None,
        tags: vec!["quixigate".to_owned(), "embeddings".to_owned()],
    };

    finalize_request_accounting(
        &state,
        &principal,
        budget_guard,
        request_id.clone(),
        IngressProtocol::Embeddings,
        requested_model,
        routed.resolved_model.clone(),
        routed.provider_id.clone(),
        routed.credential_id,
        200,
        false,
        routed.fallback_attempts,
        false,
        None,
        routed.duration_ms,
        routed.output.prompt_tokens,
        0,
        0,
        0,
        routed.cost_nanos,
        routed.api_equiv_nanos,
        None,
        Some(body.to_string()),
        None,
        oi,
        None,
        Vec::new(),
    );

    let mut resp = Json(resp_json).into_response();
    attach_gateway_headers(
        resp.headers_mut(),
        &request_id,
        &routed.provider_id,
        &routed.resolved_model,
        false,
        routed.cost_nanos,
        routed.api_equiv_nanos,
    );
    Ok(resp)
}

#[derive(Debug, Default, Deserialize)]
pub struct ListModelsQuery {
    /// Pass `?all=true` to return the entire `models.dev` reference catalog even for
    /// unconfigured providers; defaults to `false` so clients only see routable models.
    #[serde(default)]
    pub all: bool,
}

/// `GET /v1/models` — OpenAI & OpenRouter-enriched live model catalog returning custom aliases,
/// active OAuth/API seats, and auto-discovered local runtimes (pass `?all=true` for full catalog).
pub async fn handle_list_models(
    State(state): State<AppState>,
    Query(query): Query<ListModelsQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let routes = state.db.list_model_routes()?;
    let creds = state.db.list_credentials()?;
    let configured_providers: std::collections::HashSet<String> = creds
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.provider_id.clone())
        .collect();

    let catalog_models = state.router.list_available_models();
    let now_sec = now_ms() / 1000;
    let allowed = principal
        .virtual_key
        .as_ref()
        .map(|k| k.allowed_models.clone())
        .unwrap_or_default();

    let mut data = Vec::with_capacity(routes.len() + catalog_models.len());

    for r in &routes {
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
        if !allowed.is_empty()
            && !allowed.iter().any(|pat| {
                pat == "*"
                    || pat == &r.alias
                    || (pat.ends_with('*') && r.alias.starts_with(pat.trim_end_matches('*')))
            })
        {
            continue;
        }
        data.push(json!({
            "id": r.alias,
            "object": "model",
            "created": r.updated_at_ms / 1000,
            "owned_by": "quixigate-alias",
            "name": format!("{} (Smart Route)", r.alias),
            "context_length": 200_000,
            "quixigate": {
                "is_alias": true,
                "pool_strategy": r.pool_strategy,
                "allow_paid_overflow": r.allow_paid_overflow,
                "team_id": r.team_id,
                "created_by": r.created_by,
                "updated_by": r.updated_by,
                "targets": r.targets,
            }
        }));
    }

    for m in catalog_models {
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
        let has_ready_seat = is_catalog_model_routable(&m, &configured_providers);
        if !query.all && !has_ready_seat {
            continue;
        }
        let per_tok_prompt =
            (m.input_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64);
        let per_tok_comp =
            (m.output_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64);

        data.push(json!({
            "id": m.canonical_id,
            "bare_id": m.bare_id,
            "object": "model",
            "created": now_sec,
            "owned_by": m.author,
            "name": m.display_name,
            "description": m.description,
            "context_length": m.context_window,
            "pricing": {
                "prompt": format!("{per_tok_prompt:.10}"),
                "completion": format!("{per_tok_comp:.10}"),
                "input_nanos_per_m": m.input_nanos_per_m.nanos(),
                "output_nanos_per_m": m.output_nanos_per_m.nanos(),
            },
            "quixigate": {
                "ready": has_ready_seat,
                "is_local": m.is_local,
                "is_preset": m.is_preset,
                "providers": m.providers.keys().collect::<Vec<_>>(),
                "supports_tools": m.supports_tools,
                "supports_reasoning": m.supports_reasoning,
                "supports_vision": m.supports_vision,
                "supports_structured_output": m.supports_structured_output,
            }
        }));
    }

    Ok(Json(json!({
        "object": "list",
        "data": data
    })))
}

/// `GET /api/v1/presets` — OpenRouter-compatible preset catalog returning synced OpenRouter
/// `@preset/<slug>` presets (strictly when OpenRouter is connected) plus QuixiGate Smart Route aliases.
pub async fn handle_openrouter_presets(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let creds = state.db.list_credentials()?;
    let openrouter_connected = creds
        .iter()
        .any(|c| c.enabled && c.provider_id == "openrouter");
    let routes = state.db.list_model_routes()?;

    let mut data = Vec::new();
    if openrouter_connected {
        for p in state.router.catalog().list_presets() {
            data.push(json!({
                "id": p.id,
                "slug": p.slug,
                "model": format!("@preset/{}", p.slug),
                "name": p.name,
                "description": p.description,
                "underlying_model": p.underlying_model,
                "updated_at": p.updated_at,
                "source": "openrouter",
            }));
        }
    }
    for r in routes {
        let first_target = r.targets.first().map(|t| t.model.clone());
        data.push(json!({
            "id": format!("route_{}", r.alias),
            "slug": r.alias,
            "model": r.alias,
            "name": format!("{} (Smart Route)", r.alias),
            "description": "QuixiGate Smart Router alias",
            "underlying_model": first_target,
            "updated_at": r.updated_at_ms.to_string(),
            "source": "quixigate-route",
        }));
    }
    Ok(Json(json!({ "data": data })))
}

/// `GET /api/v1/key` & `GET /api/v1/auth/key` — OpenRouter-compatible key introspection.
pub async fn handle_openrouter_key(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    if let Some(vk) = principal.virtual_key {
        let usage_usd = (vk.spend_nanos.nanos() as f64) / (NANOS_PER_USD as f64);
        let limit_usd = vk
            .budget_limit_nanos
            .map(|l| (l.nanos() as f64) / (NANOS_PER_USD as f64));
        let remaining_usd = limit_usd.map(|l| (l - usage_usd).max(0.0));
        Ok(Json(json!({
            "data": {
                "label": vk.name,
                "key_prefix": vk.key_prefix,
                "usage": usage_usd,
                "usage_nanos": vk.spend_nanos.nanos(),
                "limit": limit_usd,
                "limit_nanos": vk.budget_limit_nanos.map(|v| v.nanos()),
                "limit_remaining": remaining_usd,
                "budget_window": vk.budget_window,
                "is_free_tier": false,
                "rate_limit": {
                    "requests": vk.rpm_limit.unwrap_or(0),
                    "interval": "1m"
                }
            }
        })))
    } else {
        let u = principal.user;
        let usage_usd = (u.spend_nanos.nanos() as f64) / (NANOS_PER_USD as f64);
        let limit_usd = u
            .budget_limit_nanos
            .map(|l| (l.nanos() as f64) / (NANOS_PER_USD as f64));
        Ok(Json(json!({
            "data": {
                "label": u.display_name,
                "usage": usage_usd,
                "usage_nanos": u.spend_nanos.nanos(),
                "limit": limit_usd,
                "limit_remaining": limit_usd.map(|l| (l - usage_usd).max(0.0)),
                "is_free_tier": false
            }
        })))
    }
}

/// `GET /api/v1/credits` — OpenRouter-compatible credit & budget summary.
pub async fn handle_openrouter_credits(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let (spend, limit) = if let Some(vk) = &principal.virtual_key {
        (vk.spend_nanos, vk.budget_limit_nanos)
    } else {
        (
            principal.user.spend_nanos,
            principal.user.budget_limit_nanos,
        )
    };
    let total_usage = (spend.nanos() as f64) / (NANOS_PER_USD as f64);
    let total_credits = limit
        .map(|l| (l.nanos() as f64) / (NANOS_PER_USD as f64))
        .unwrap_or(total_usage + 1000.0);
    Ok(Json(json!({
        "data": {
            "total_credits": total_credits,
            "total_usage": total_usage,
            "usage_nanos": spend.nanos(),
            "limit_nanos": limit.map(|l| l.nanos()),
        }
    })))
}

pub async fn handle_liveness() -> Json<Value> {
    Json(json!({ "status": "ok", "service": "quixigate" }))
}

pub async fn handle_readiness(State(state): State<AppState>) -> Response {
    match state.db.get_settings() {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "uptime_ms": (now_ms() - state.started_at_ms).max(0),
                "catalog_models": state.router.list_available_models().len(),
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "unhealthy", "error": err.to_string() })),
        )
            .into_response(),
    }
}

pub async fn handle_prometheus_metrics(State(state): State<AppState>) -> Response {
    let body = state.telemetry.render_prometheus_metrics();
    let mut resp = (StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    resp
}
