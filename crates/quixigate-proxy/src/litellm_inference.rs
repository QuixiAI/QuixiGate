//! LiteLLM Data-Plane & Developer Utility Compatibility Endpoints:
//! - Unprefixed & SDK aliases (`/models`, `/v1/models/{*id}`, `/models/{*tail}`, `/chat/completions`,
//!   `/embeddings`, `/responses` HTTP+WS, `/openai/v1/responses`, `/messages`, `/messages/count_tokens`,
//!   Cursor `/cursor/*`, `/mcp/enabled`)
//! - Azure OpenAI & Legacy Engine aliases (`/openai/deployments/{model}/*`, `/engines/{model}/*`)
//! - Legacy OpenAI Text Completions (`POST /v1/completions`, `POST /completions` — JSON + SSE + `mock_response`)
//! - Responses API helpers (`POST /{v1/,openai/v1/,}responses/{input_tokens,compact}`)
//! - Multimodal Images (`POST /{v1/,}images/{generations,edits,variations}`)
//! - Multimodal Audio (`POST /{v1/,}audio/{speech,transcriptions,translations}`)
//! - Moderations (`POST /{v1/,}moderations` — upstream + local guardrail fallback)
//! - Rerank (`POST /{v1/,v2/,}rerank` — upstream Cohere/Jina/Voyage/OpenAI-compat + local lexical scorer)
//! - Web Search (`POST /{v1/,}search`, `POST /{v1/,}search/{tool}`, `GET /{v1/,}search/tools`)
//! - Developer Utilities (`/utils/{token_counter,supported_openai_params,model_info,transform_request}`,
//!   `POST /cost/estimate`, `GET /{v2/,}guardrails/list`, `POST /{guardrails/,}apply_guardrail`)

use std::convert::Infallible;
use std::time::Instant;

use aisdk::core::{Message, UserMessage};
use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use quixigate_core::{
    AuthKind, IngressProtocol, ModelSpec, NANOS_PER_USD, NanoUsd, QuixigateError,
    SecretScannerMode, UserRole, now_ms,
};
use quixigate_router::guardrails::{detect_first_secret, redact_secrets_in_str};
use quixigate_router::{CanonicalRequest, RoutedStreamEvent};
use quixigate_telemetry::OpenInferencePayload;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::codecs::{anthropic, gemini, misc, openai_chat, openai_responses};
use crate::state::{ApiError, ApiResult, AppState};

pub fn router() -> Router<AppState> {
    Router::new()
        // Unprefixed & model-retrieve aliases
        .route("/models", axum::routing::get(misc::handle_list_models))
        .route(
            "/v1/models/{*model_id}",
            axum::routing::get(handle_retrieve_model),
        )
        .route(
            "/models/{*tail}",
            axum::routing::get(handle_retrieve_model).post(handle_unprefixed_models_post),
        )
        .route(
            "/chat/completions",
            axum::routing::post(openai_chat::handle_chat_completions),
        )
        .route("/embeddings", axum::routing::post(misc::handle_embeddings))
        .route(
            "/responses",
            axum::routing::post(openai_responses::handle_responses_post)
                .get(openai_responses::handle_responses_ws),
        )
        .route(
            "/openai/v1/responses",
            axum::routing::post(openai_responses::handle_responses_post)
                .get(openai_responses::handle_responses_ws),
        )
        .route(
            "/v1/responses/input_tokens",
            axum::routing::post(handle_responses_input_tokens),
        )
        .route(
            "/responses/input_tokens",
            axum::routing::post(handle_responses_input_tokens),
        )
        .route(
            "/openai/v1/responses/input_tokens",
            axum::routing::post(handle_responses_input_tokens),
        )
        .route(
            "/v1/responses/compact",
            axum::routing::post(handle_responses_compact),
        )
        .route(
            "/responses/compact",
            axum::routing::post(handle_responses_compact),
        )
        .route(
            "/openai/v1/responses/compact",
            axum::routing::post(handle_responses_compact),
        )
        .route("/messages", axum::routing::post(anthropic::handle_messages))
        .route(
            "/messages/count_tokens",
            axum::routing::post(anthropic::handle_count_tokens),
        )
        // Cursor IDE aliases
        .route(
            "/cursor/models",
            axum::routing::get(misc::handle_list_models),
        )
        .route(
            "/cursor/v1/models",
            axum::routing::get(misc::handle_list_models),
        )
        .route(
            "/cursor/chat/completions",
            axum::routing::post(openai_chat::handle_chat_completions),
        )
        // MCP enabled probe
        .route("/mcp/enabled", axum::routing::get(handle_mcp_enabled))
        .route("/enabled", axum::routing::get(handle_mcp_enabled))
        // Legacy OpenAI Text Completions
        .route(
            "/v1/completions",
            axum::routing::post(handle_legacy_completions),
        )
        .route(
            "/completions",
            axum::routing::post(handle_legacy_completions),
        )
        // Azure OpenAI & Legacy Engines Deployment Aliases
        .route(
            "/openai/deployments/{model}/chat/completions",
            axum::routing::post(handle_deployment_chat_completions),
        )
        .route(
            "/engines/{model}/chat/completions",
            axum::routing::post(handle_deployment_chat_completions),
        )
        .route(
            "/openai/deployments/{model}/completions",
            axum::routing::post(handle_deployment_completions),
        )
        .route(
            "/engines/{model}/completions",
            axum::routing::post(handle_deployment_completions),
        )
        .route(
            "/openai/deployments/{model}/embeddings",
            axum::routing::post(handle_deployment_embeddings),
        )
        .route(
            "/engines/{model}/embeddings",
            axum::routing::post(handle_deployment_embeddings),
        )
        .route(
            "/openai/deployments/{model}/images/generations",
            axum::routing::post(handle_deployment_images_generations),
        )
        .route(
            "/openai/deployments/{model}/images/edits",
            axum::routing::post(handle_deployment_images_edits),
        )
        // Multimodal Images
        .route(
            "/v1/images/generations",
            axum::routing::post(handle_images_generations),
        )
        .route(
            "/images/generations",
            axum::routing::post(handle_images_generations),
        )
        .route("/v1/images/edits", axum::routing::post(handle_images_edits))
        .route("/images/edits", axum::routing::post(handle_images_edits))
        .route(
            "/v1/images/variations",
            axum::routing::post(handle_images_variations),
        )
        .route(
            "/images/variations",
            axum::routing::post(handle_images_variations),
        )
        // Multimodal Audio
        .route("/v1/audio/speech", axum::routing::post(handle_audio_speech))
        .route("/audio/speech", axum::routing::post(handle_audio_speech))
        .route(
            "/v1/audio/transcriptions",
            axum::routing::post(handle_audio_transcriptions),
        )
        .route(
            "/audio/transcriptions",
            axum::routing::post(handle_audio_transcriptions),
        )
        .route(
            "/v1/audio/translations",
            axum::routing::post(handle_audio_translations),
        )
        .route(
            "/audio/translations",
            axum::routing::post(handle_audio_translations),
        )
        // Moderations & Rerank
        .route("/v1/moderations", axum::routing::post(handle_moderations))
        .route("/moderations", axum::routing::post(handle_moderations))
        .route("/v1/rerank", axum::routing::post(handle_rerank))
        .route("/v2/rerank", axum::routing::post(handle_rerank))
        .route("/rerank", axum::routing::post(handle_rerank))
        // Web Search (6-Engine Waterfall)
        .route("/v1/search", axum::routing::post(handle_search))
        .route("/search", axum::routing::post(handle_search))
        .route("/v1/search/tools", axum::routing::get(handle_search_tools))
        .route("/search/tools", axum::routing::get(handle_search_tools))
        .route(
            "/v1/search/{search_tool_name}",
            axum::routing::post(handle_search_by_tool),
        )
        .route(
            "/search/{search_tool_name}",
            axum::routing::post(handle_search_by_tool),
        )
        // Developer Utilities & Guardrails
        .route(
            "/utils/token_counter",
            axum::routing::post(handle_utils_token_counter),
        )
        .route(
            "/utils/supported_openai_params",
            axum::routing::get(handle_utils_supported_openai_params),
        )
        .route(
            "/utils/model_info",
            axum::routing::get(handle_utils_model_info),
        )
        .route(
            "/utils/transform_request",
            axum::routing::post(handle_utils_transform_request),
        )
        .route("/cost/estimate", axum::routing::post(handle_cost_estimate))
        .route(
            "/guardrails/list",
            axum::routing::get(handle_guardrails_list),
        )
        .route(
            "/v2/guardrails/list",
            axum::routing::get(handle_guardrails_list),
        )
        .route(
            "/apply_guardrail",
            axum::routing::post(handle_apply_guardrail),
        )
        .route(
            "/guardrails/apply_guardrail",
            axum::routing::post(handle_apply_guardrail),
        )
}

async fn handle_mcp_enabled(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let settings = state.db.get_settings()?;
    Ok(Json(json!({ "enabled": settings.mcp_server_enabled })))
}

async fn handle_unprefixed_models_post(
    State(state): State<AppState>,
    Path(tail): Path<String>,
    Query(query): Query<gemini::GeminiQuery>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> ApiResult<Response> {
    let trimmed = tail.trim_start_matches('/');
    if let Some((_, action)) = trimmed.rsplit_once(':')
        && matches!(
            action,
            "generateContent" | "streamGenerateContent" | "countTokens"
        )
    {
        let parsed: Value = serde_json::from_slice(&body)
            .map_err(|e| QuixigateError::Validation(format!("invalid JSON body: {e}")))?;
        return gemini::handle_gemini_model_action(
            State(state),
            Path(tail),
            Query(query),
            headers,
            Json(parsed),
        )
        .await;
    }
    Ok(crate::litellm_unsupported::fallback_handler(Method::POST, uri).await)
}

/// `GET /v1/models/{*model_id}` & `GET /models/{*tail}` — OpenAI `models.retrieve(model_id)` compatibility.
pub async fn handle_retrieve_model(
    State(state): State<AppState>,
    Path(raw_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let model_id = raw_id.trim_start_matches('/').trim();
    if model_id.is_empty() {
        return Err(QuixigateError::Validation("model_id cannot be empty".to_owned()).into());
    }

    if let Some(r) = state.db.get_model_route(model_id)? {
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
                return Err(
                    QuixigateError::NotFound(format!("model '{model_id}' not found")).into(),
                );
            }
        }
        return Ok(Json(json!({
            "id": r.alias,
            "object": "model",
            "created": r.updated_at_ms / 1000,
            "owned_by": "quixigate-alias",
            "quixigate": {
                "is_alias": true,
                "blocked": r.blocked,
                "pool_strategy": r.pool_strategy,
                "targets": r.targets,
            }
        })));
    }

    let now_sec = now_ms() / 1000;
    let catalog_models = state.router.list_available_models();
    if let Some(m) = catalog_models
        .into_iter()
        .find(|m| m.canonical_id == model_id || m.bare_id == model_id)
    {
        let per_tok_prompt =
            (m.input_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64);
        let per_tok_comp =
            (m.output_nanos_per_m.nanos() as f64) / 1_000_000.0 / (NANOS_PER_USD as f64);
        return Ok(Json(json!({
            "id": model_id,
            "canonical_id": m.canonical_id,
            "object": "model",
            "created": now_sec,
            "owned_by": m.author,
            "context_length": m.context_window,
            "pricing": {
                "prompt": format!("{per_tok_prompt:.10}"),
                "completion": format!("{per_tok_comp:.10}"),
                "input_nanos_per_m": m.input_nanos_per_m.nanos(),
                "output_nanos_per_m": m.output_nanos_per_m.nanos(),
            }
        })));
    }

    Err(QuixigateError::NotFound(format!("model '{model_id}' not found")).into())
}

fn inject_deployment_model(mut body: Value, deployment_model: &str) -> Value {
    if let Some(obj) = body.as_object_mut() {
        let missing = obj
            .get("model")
            .and_then(Value::as_str)
            .is_none_or(|s| s.trim().is_empty());
        if missing {
            obj.insert(
                "model".to_owned(),
                Value::String(deployment_model.to_owned()),
            );
        }
    }
    body
}

async fn handle_deployment_chat_completions(
    State(state): State<AppState>,
    Path(model): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let patched = inject_deployment_model(body, &model);
    openai_chat::handle_chat_completions(State(state), headers, Json(patched)).await
}

async fn handle_deployment_completions(
    State(state): State<AppState>,
    Path(model): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let patched = inject_deployment_model(body, &model);
    handle_legacy_completions(State(state), headers, Json(patched)).await
}

async fn handle_deployment_embeddings(
    State(state): State<AppState>,
    Path(model): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let patched = inject_deployment_model(body, &model);
    misc::handle_embeddings(State(state), headers, Json(patched)).await
}

async fn handle_deployment_images_generations(
    State(state): State<AppState>,
    Path(model): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let patched = inject_deployment_model(body, &model);
    handle_images_generations(State(state), headers, Json(patched)).await
}

async fn handle_deployment_images_edits(
    State(state): State<AppState>,
    Path(model): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Response> {
    forward_multipart_endpoint(
        state,
        headers,
        body,
        "images/edits",
        IngressProtocol::Images,
        Some(model),
    )
    .await
}

/// `POST /v1/completions` & `POST /completions` — Legacy OpenAI Text Completions JSON + SSE codec.
pub async fn handle_legacy_completions(
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

    let prompts: Vec<String> = match &body["prompt"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) if arr.iter().all(Value::is_string) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        Value::Null => vec![String::new()],
        other => vec![other.to_string()],
    };
    let primary_prompt = prompts.first().cloned().unwrap_or_default();
    let echo = body["echo"].as_bool().unwrap_or(false);
    let suffix = body["suffix"].as_str().map(str::to_owned);
    let max_out = body["max_tokens"].as_u64().map(|v| v as u32);
    let stop_sequences = match &body["stop"] {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(arr) => Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
        ),
        _ => None,
    };

    let req = CanonicalRequest {
        requested_model: requested_model.clone(),
        system: None,
        messages: vec![Message::User(UserMessage::new(primary_prompt.clone()))],
        client_tools: Vec::new(),
        schema: None,
        temperature: scale_temperature(body["temperature"].as_f64()),
        top_p: scale_top_p(body["top_p"].as_f64()),
        top_k: body["top_k"].as_u64().map(|v| v as u32),
        max_output_tokens: max_out,
        stop_sequences,
        presence_penalty: body["presence_penalty"].as_f64().map(|v| v as f32),
        frequency_penalty: body["frequency_penalty"].as_f64().map(|v| v as f32),
        seed: body["seed"].as_u64().map(|v| v as u32),
        reasoning_effort: None,
        extra_headers: extract_passthrough_headers(&headers),
        extra_body: Map::new(),
        openrouter_body: extract_openrouter_body(&body),
        cache_preference: parse_cache_preference(&headers),
    };

    let est_in = estimate_request_tokens(&req);
    let (est_total, est_cost) = state.router.estimate_preflight_budget(
        &requested_model,
        est_in,
        u64::from(max_out.unwrap_or(256)),
    )?;
    let (conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        &requested_model,
        est_total,
        est_cost,
    )?;

    let stream_requested = body["stream"].as_bool().unwrap_or(false);
    let request_id = format!("cmpl_{}", Uuid::new_v4().simple());
    let created_sec = now_ms() / 1000;

    let format_final_text = |generated: &str| -> String {
        let mut out = String::new();
        if echo {
            out.push_str(&primary_prompt);
        }
        out.push_str(generated);
        if let Some(sfx) = &suffix {
            out.push_str(sfx);
        }
        out
    };

    if let Some(mock_text) = body["mock_response"].as_str() {
        let mut guard_req = req.clone();
        quixigate_router::guardrails::apply_request_guardrails(
            &mut guard_req,
            &state.db.get_settings()?,
        )?;
        let full_text = format_final_text(mock_text);
        let out_tokens = ((mock_text.len() / 4) as u64).max(1);
        let _ = budget_guard.commit(NanoUsd::ZERO);
        drop(conc_guard);

        if stream_requested {
            let chunk = json!({
                "id": request_id,
                "object": "text_completion",
                "created": created_sec,
                "model": requested_model,
                "choices": [{
                    "text": full_text,
                    "index": 0,
                    "logprobs": Value::Null,
                    "finish_reason": "stop"
                }]
            });
            let events = vec![
                Ok::<Event, Infallible>(Event::default().data(chunk.to_string())),
                Ok::<Event, Infallible>(Event::default().data("[DONE]")),
            ];
            return Ok(Sse::new(futures::stream::iter(events))
                .keep_alive(KeepAlive::default())
                .into_response());
        }

        let choices: Vec<Value> = prompts
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let mut t = String::new();
                if echo {
                    t.push_str(p);
                }
                t.push_str(mock_text);
                if let Some(sfx) = &suffix {
                    t.push_str(sfx);
                }
                json!({
                    "text": t,
                    "index": idx,
                    "logprobs": Value::Null,
                    "finish_reason": "stop"
                })
            })
            .collect();
        let mut resp = Json(json!({
            "id": request_id,
            "object": "text_completion",
            "created": created_sec,
            "model": requested_model,
            "choices": choices,
            "usage": {
                "prompt_tokens": est_in,
                "completion_tokens": out_tokens,
                "total_tokens": est_in + out_tokens
            }
        }))
        .into_response();
        attach_gateway_headers(
            resp.headers_mut(),
            &request_id,
            "mock",
            &requested_model,
            false,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        return Ok(resp);
    }

    let oi = build_openinference_inputs(&req, None);
    let raw_req = Some(body.to_string());
    let started = Instant::now();
    state.telemetry.inc_active();

    if !stream_requested {
        let res = state.router.execute_text(req).await;
        state.telemetry.dec_active();
        drop(conc_guard);

        match res {
            Ok(gen_out) => {
                let text = format_final_text(&gen_out.text);
                let resp_json = json!({
                    "id": request_id,
                    "object": "text_completion",
                    "created": created_sec,
                    "model": gen_out.resolved_model,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "logprobs": Value::Null,
                        "finish_reason": gen_out.finish_reason.as_deref().unwrap_or("stop")
                    }],
                    "usage": {
                        "prompt_tokens": gen_out.input_tokens,
                        "completion_tokens": gen_out.output_tokens,
                        "total_tokens": gen_out.input_tokens + gen_out.output_tokens
                    }
                });
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id.clone(),
                    IngressProtocol::TextCompletions,
                    requested_model,
                    gen_out.resolved_model.clone(),
                    gen_out.provider_id.clone(),
                    gen_out.credential_id,
                    200,
                    false,
                    gen_out.fallback_attempts,
                    gen_out.cache_hit,
                    None,
                    gen_out.duration_ms,
                    gen_out.input_tokens,
                    gen_out.cached_input_tokens,
                    gen_out.reasoning_tokens,
                    gen_out.output_tokens,
                    gen_out.cost_nanos,
                    gen_out.api_equiv_nanos,
                    None,
                    raw_req,
                    Some(resp_json.to_string()),
                    oi,
                    Some(text),
                    Vec::new(),
                );
                let mut resp = Json(resp_json).into_response();
                attach_gateway_headers(
                    resp.headers_mut(),
                    &request_id,
                    &gen_out.provider_id,
                    &gen_out.resolved_model,
                    gen_out.cache_hit,
                    gen_out.cost_nanos,
                    gen_out.api_equiv_nanos,
                );
                Ok(resp)
            }
            Err(err) => {
                let prov = error_provider_id(&err);
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::TextCompletions,
                    requested_model.clone(),
                    requested_model,
                    prov,
                    None,
                    err.status_code(),
                    false,
                    0,
                    false,
                    None,
                    started.elapsed().as_millis() as u64,
                    0,
                    0,
                    0,
                    0,
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
                    Some(err.to_string()),
                    raw_req,
                    None,
                    oi,
                    None,
                    Vec::new(),
                );
                Err(ApiError(err))
            }
        }
    } else {
        let routed = match state.router.execute_stream(req).await {
            Ok(s) => s,
            Err(err) => {
                state.telemetry.dec_active();
                drop(conc_guard);
                let prov = error_provider_id(&err);
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::TextCompletions,
                    requested_model.clone(),
                    requested_model,
                    prov,
                    None,
                    err.status_code(),
                    true,
                    0,
                    false,
                    None,
                    started.elapsed().as_millis() as u64,
                    0,
                    0,
                    0,
                    0,
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
                    Some(err.to_string()),
                    raw_req,
                    None,
                    oi,
                    None,
                    Vec::new(),
                );
                return Err(ApiError(err));
            }
        };

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        let state_bg = state.clone();
        let req_id_bg = request_id.clone();
        let resolved_hdr = routed.resolved_model.clone();
        let provider_hdr = routed.provider_id.clone();
        let cache_hit_hdr = routed.cache_hit;

        tokio::spawn(async move {
            let _hold_conc = conc_guard;
            let mut stream = routed.events;
            let mut summary_opt = None;
            let mut stream_err = None;

            if echo && !primary_prompt.is_empty() {
                let echo_chunk = json!({
                    "id": req_id_bg,
                    "object": "text_completion",
                    "created": created_sec,
                    "model": routed.resolved_model,
                    "choices": [{
                        "text": primary_prompt,
                        "index": 0,
                        "logprobs": Value::Null,
                        "finish_reason": Value::Null
                    }]
                });
                let _ = tx
                    .send(Ok(Event::default().data(echo_chunk.to_string())))
                    .await;
            }

            while let Some(ev) = stream.next().await {
                match ev {
                    RoutedStreamEvent::TextDelta(delta) => {
                        let chunk = json!({
                            "id": req_id_bg,
                            "object": "text_completion",
                            "created": created_sec,
                            "model": routed.resolved_model,
                            "choices": [{
                                "text": delta,
                                "index": 0,
                                "logprobs": Value::Null,
                                "finish_reason": Value::Null
                            }]
                        });
                        if tx
                            .send(Ok(Event::default().data(chunk.to_string())))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    RoutedStreamEvent::Finish(sum) => {
                        if let Some(sfx) = &suffix {
                            let sfx_chunk = json!({
                                "id": req_id_bg,
                                "object": "text_completion",
                                "created": created_sec,
                                "model": routed.resolved_model,
                                "choices": [{
                                    "text": sfx,
                                    "index": 0,
                                    "logprobs": Value::Null,
                                    "finish_reason": Value::Null
                                }]
                            });
                            let _ = tx
                                .send(Ok(Event::default().data(sfx_chunk.to_string())))
                                .await;
                        }
                        let final_chunk = json!({
                            "id": req_id_bg,
                            "object": "text_completion",
                            "created": created_sec,
                            "model": sum.resolved_model,
                            "choices": [{
                                "text": "",
                                "index": 0,
                                "logprobs": Value::Null,
                                "finish_reason": sum.finish_reason.as_deref().unwrap_or("stop")
                            }]
                        });
                        let _ = tx
                            .send(Ok(Event::default().data(final_chunk.to_string())))
                            .await;
                        summary_opt = Some(sum);
                    }
                    RoutedStreamEvent::Error(e) => {
                        stream_err = Some(e);
                    }
                    _ => {}
                }
            }

            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            state_bg.telemetry.dec_active();

            if let Some(sum) = summary_opt {
                finalize_request_accounting(
                    &state_bg,
                    &principal,
                    budget_guard,
                    req_id_bg,
                    IngressProtocol::TextCompletions,
                    requested_model,
                    sum.resolved_model,
                    sum.provider_id,
                    sum.credential_id,
                    200,
                    true,
                    sum.fallback_attempts,
                    routed.cache_hit,
                    sum.ttft_ms,
                    sum.duration_ms,
                    sum.input_tokens,
                    sum.cached_input_tokens,
                    sum.reasoning_tokens,
                    sum.output_tokens,
                    sum.cost_nanos,
                    sum.api_equiv_nanos,
                    stream_err,
                    raw_req,
                    None,
                    oi,
                    Some(sum.text),
                    Vec::new(),
                );
            }
        });

        let mut resp = Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response();
        attach_gateway_headers(
            resp.headers_mut(),
            &request_id,
            &provider_hdr,
            &resolved_hdr,
            cache_hit_hdr,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        Ok(resp)
    }
}

/// `POST /{v1/,openai/v1/,}responses/input_tokens` — Responses API input token estimator.
pub async fn handle_responses_input_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let mut chars = body["instructions"].as_str().map_or(0, str::len);
    match &body["input"] {
        Value::String(s) => chars += s.len(),
        Value::Array(items) => {
            for item in items {
                chars += extract_content_text(item.get("content").unwrap_or(item)).len() + 12;
            }
        }
        other if !other.is_null() => chars += other.to_string().len(),
        _ => {}
    }
    if let Some(tools) = body["tools"].as_array() {
        for t in tools {
            chars += t.to_string().len();
        }
    }
    let input_tokens = ((chars / 4) as u64).max(1) + 4;
    Ok(Json(json!({
        "object": "response.input_tokens",
        "input_tokens": input_tokens
    })))
}

/// `POST /{v1/,openai/v1/,}responses/compact` — Stateless Responses API context compaction helper.
pub async fn handle_responses_compact(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("gpt-4o-mini")
        .to_owned();

    let raw_input = match &body["input"] {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .map(|item| {
                let role = item["role"].as_str().unwrap_or("user");
                let text = extract_content_text(item.get("content").unwrap_or(item));
                format!("{role}: {text}")
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    };

    let compact_prompt = format!(
        "Summarize the following conversation context concisely, preserving all key facts, decisions, code identifiers, and open tasks:\n\n{raw_input}"
    );
    let req = CanonicalRequest {
        requested_model: requested_model.clone(),
        system: body["instructions"].as_str().map(str::to_owned),
        messages: vec![Message::User(UserMessage::new(compact_prompt))],
        max_output_tokens: Some(1024),
        ..Default::default()
    };

    let est_in = estimate_request_tokens(&req);
    let (est_total, est_cost) =
        state
            .router
            .estimate_preflight_budget(&requested_model, est_in, 1024)?;
    let (conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        &requested_model,
        est_total,
        est_cost,
    )?;

    let resp_id = format!("resp_cmp_{}", Uuid::new_v4().simple());
    let created_sec = now_ms() / 1000;

    if let Some(mock_text) = body["mock_response"].as_str() {
        let _ = budget_guard.commit(NanoUsd::ZERO);
        drop(conc_guard);
        let out_toks = ((mock_text.len() / 4) as u64).max(1);
        return Ok(Json(json!({
            "id": resp_id,
            "object": "response.compaction",
            "created_at": created_sec,
            "model": requested_model,
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": mock_text }]
            }],
            "usage": {
                "input_tokens": est_in,
                "output_tokens": out_toks,
                "total_tokens": est_in + out_toks
            }
        })));
    }

    let oi = build_openinference_inputs(&req, None);
    state.telemetry.inc_active();
    let res = state.router.execute_text(req).await;
    state.telemetry.dec_active();
    drop(conc_guard);

    let gen_out = res?;
    finalize_request_accounting(
        &state,
        &principal,
        budget_guard,
        resp_id.clone(),
        IngressProtocol::Responses,
        requested_model,
        gen_out.resolved_model.clone(),
        gen_out.provider_id,
        gen_out.credential_id,
        200,
        false,
        gen_out.fallback_attempts,
        gen_out.cache_hit,
        None,
        gen_out.duration_ms,
        gen_out.input_tokens,
        gen_out.cached_input_tokens,
        gen_out.reasoning_tokens,
        gen_out.output_tokens,
        gen_out.cost_nanos,
        gen_out.api_equiv_nanos,
        None,
        Some(body.to_string()),
        None,
        oi,
        Some(gen_out.text.clone()),
        Vec::new(),
    );

    Ok(Json(json!({
        "id": resp_id,
        "object": "response.compaction",
        "created_at": created_sec,
        "model": gen_out.resolved_model,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": gen_out.text }]
        }],
        "usage": {
            "input_tokens": gen_out.input_tokens,
            "output_tokens": gen_out.output_tokens,
            "total_tokens": gen_out.input_tokens + gen_out.output_tokens
        }
    })))
}

// ── Multimodal Images & Audio ────────────────────────────────────────────────

/// `POST /v1/images/generations` & `POST /images/generations`
pub async fn handle_images_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("gpt-image-1")
        .to_owned();
    let count = body["n"].as_u64().unwrap_or(1).max(1);
    let est_tokens = count * 1_000;
    let (_, est_cost) =
        state
            .router
            .estimate_preflight_budget(&requested_model, est_tokens, est_tokens)?;
    let (_conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        &requested_model,
        est_tokens,
        est_cost,
    )?;

    let request_id = format!("img_{}", Uuid::new_v4().simple());
    let created_sec = now_ms() / 1000;

    if let Some(mock_url) = body["mock_response"].as_str() {
        let _ = budget_guard.commit(NanoUsd::ZERO);
        let mut resp = Json(json!({
            "created": created_sec,
            "data": [{ "url": mock_url, "revised_prompt": body["prompt"] }]
        }))
        .into_response();
        attach_gateway_headers(
            resp.headers_mut(),
            &request_id,
            "mock",
            &requested_model,
            false,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        return Ok(resp);
    }

    let candidates = state
        .router
        .resolve_http_candidates(&requested_model)
        .await?;
    if candidates.is_empty() {
        return Err(QuixigateError::NoAvailableRoute(requested_model).into());
    }

    let started = Instant::now();
    let mut last_err: Option<QuixigateError> = None;

    for (idx, cand) in candidates.iter().enumerate() {
        if let Some(obj) = body.as_object_mut() {
            obj.insert(
                "model".to_owned(),
                Value::String(cand.upstream_model.clone()),
            );
        }
        let url = format!("{}/images/generations", cand.base_url);
        let mut req_builder = state
            .router
            .inference_http()
            .post(&url)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(k) = cand.api_key.as_deref().filter(|s| !s.is_empty()) {
            req_builder = req_builder.bearer_auth(k);
        }
        for (hk, hv) in &cand.extra_headers {
            req_builder = req_builder.header(hk, hv);
        }

        match req_builder.json(&body).send().await {
            Ok(up_resp) => {
                let status = up_resp.status();
                let resp_bytes = up_resp.bytes().await.unwrap_or_default();
                if !status.is_success() {
                    let msg = serde_json::from_slice::<Value>(&resp_bytes)
                        .ok()
                        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
                        .unwrap_or_else(|| String::from_utf8_lossy(&resp_bytes).into_owned());
                    last_err = Some(QuixigateError::Upstream {
                        status: status.as_u16(),
                        provider: cand.provider_id.clone(),
                        message: msg,
                    });
                    continue;
                }
                if let Some(cid) = &cand.credential_id {
                    let _ = state.db.mark_credential_used(cid);
                }
                let parsed: Value =
                    serde_json::from_slice(&resp_bytes).unwrap_or_else(|_| json!({}));
                let in_toks = parsed["usage"]["input_tokens"]
                    .as_u64()
                    .unwrap_or(est_tokens);
                let out_toks = parsed["usage"]["output_tokens"]
                    .as_u64()
                    .unwrap_or(est_tokens);
                let (cost_nanos, api_equiv_nanos) = state.router.catalog().compute_cost(
                    &cand.provider_id,
                    cand.auth_kind,
                    &cand.upstream_model,
                    in_toks,
                    0,
                    out_toks,
                    cand.route.as_ref(),
                );
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id.clone(),
                    IngressProtocol::Images,
                    requested_model,
                    cand.upstream_model.clone(),
                    cand.provider_id.clone(),
                    cand.credential_id.clone(),
                    status.as_u16(),
                    false,
                    idx as u32,
                    false,
                    None,
                    started.elapsed().as_millis() as u64,
                    in_toks,
                    0,
                    0,
                    out_toks,
                    cost_nanos,
                    api_equiv_nanos,
                    None,
                    None,
                    None,
                    OpenInferencePayload::default(),
                    None,
                    Vec::new(),
                );
                let mut resp = (
                    status,
                    [(header::CONTENT_TYPE, "application/json")],
                    resp_bytes,
                )
                    .into_response();
                attach_gateway_headers(
                    resp.headers_mut(),
                    &request_id,
                    &cand.provider_id,
                    &cand.upstream_model,
                    false,
                    cost_nanos,
                    api_equiv_nanos,
                );
                return Ok(resp);
            }
            Err(e) => {
                let (st, why) = quixigate_providers::describe_reqwest_error(&e);
                last_err = Some(QuixigateError::Upstream {
                    status: st.map(|s| s.as_u16()).unwrap_or(502),
                    provider: cand.provider_id.clone(),
                    message: why,
                });
            }
        }
    }

    Err(last_err
        .unwrap_or(QuixigateError::NoAvailableRoute(requested_model))
        .into())
}

async fn handle_images_edits(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Response> {
    forward_multipart_endpoint(
        state,
        headers,
        body,
        "images/edits",
        IngressProtocol::Images,
        None,
    )
    .await
}

async fn handle_images_variations(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Response> {
    forward_multipart_endpoint(
        state,
        headers,
        body,
        "images/variations",
        IngressProtocol::Images,
        None,
    )
    .await
}

/// `POST /v1/audio/speech` & `POST /audio/speech` — Text-to-Speech binary audio streaming/pass-through.
pub async fn handle_audio_speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("missing 'model' field".to_owned()))?
        .to_owned();
    let input_text = body["input"]
        .as_str()
        .ok_or_else(|| {
            QuixigateError::Validation("missing 'input' text for audio/speech".to_owned())
        })?
        .to_owned();

    let est_tokens = ((input_text.len() / 4) as u64).max(1);
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

    let request_id = format!("tts_{}", Uuid::new_v4().simple());

    if body["mock_response"].is_string() {
        let _ = budget_guard.commit(NanoUsd::ZERO);
        let fake_mp3 = Bytes::from_static(b"ID3\x04\x00\x00\x00\x00\x00\x00QUIXIGATE_MOCK_AUDIO");
        let mut resp = (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "audio/mpeg")],
            fake_mp3,
        )
            .into_response();
        attach_gateway_headers(
            resp.headers_mut(),
            &request_id,
            "mock",
            &requested_model,
            false,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        return Ok(resp);
    }

    let candidates = state
        .router
        .resolve_http_candidates(&requested_model)
        .await?;
    if candidates.is_empty() {
        return Err(QuixigateError::NoAvailableRoute(requested_model).into());
    }

    let started = Instant::now();
    let mut last_err: Option<QuixigateError> = None;

    for (idx, cand) in candidates.iter().enumerate() {
        if let Some(obj) = body.as_object_mut() {
            obj.insert(
                "model".to_owned(),
                Value::String(cand.upstream_model.clone()),
            );
        }
        let url = format!("{}/audio/speech", cand.base_url);
        let mut req_builder = state
            .router
            .inference_http()
            .post(&url)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(k) = cand.api_key.as_deref().filter(|s| !s.is_empty()) {
            req_builder = req_builder.bearer_auth(k);
        }
        for (hk, hv) in &cand.extra_headers {
            req_builder = req_builder.header(hk, hv);
        }

        match req_builder.json(&body).send().await {
            Ok(up_resp) => {
                let status = up_resp.status();
                let ct = up_resp
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("audio/mpeg")
                    .to_owned();
                let resp_bytes = up_resp.bytes().await.unwrap_or_default();
                if !status.is_success() {
                    let msg = String::from_utf8_lossy(&resp_bytes).into_owned();
                    last_err = Some(QuixigateError::Upstream {
                        status: status.as_u16(),
                        provider: cand.provider_id.clone(),
                        message: msg,
                    });
                    continue;
                }
                if let Some(cid) = &cand.credential_id {
                    let _ = state.db.mark_credential_used(cid);
                }
                let (cost_nanos, api_equiv_nanos) = state.router.catalog().compute_cost(
                    &cand.provider_id,
                    cand.auth_kind,
                    &cand.upstream_model,
                    est_tokens,
                    0,
                    0,
                    cand.route.as_ref(),
                );
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id.clone(),
                    IngressProtocol::Audio,
                    requested_model,
                    cand.upstream_model.clone(),
                    cand.provider_id.clone(),
                    cand.credential_id.clone(),
                    status.as_u16(),
                    false,
                    idx as u32,
                    false,
                    None,
                    started.elapsed().as_millis() as u64,
                    est_tokens,
                    0,
                    0,
                    0,
                    cost_nanos,
                    api_equiv_nanos,
                    None,
                    None,
                    None,
                    OpenInferencePayload::default(),
                    None,
                    Vec::new(),
                );
                let mut resp = (status, resp_bytes).into_response();
                if let Ok(hv) = HeaderValue::from_str(&ct) {
                    resp.headers_mut().insert(header::CONTENT_TYPE, hv);
                }
                attach_gateway_headers(
                    resp.headers_mut(),
                    &request_id,
                    &cand.provider_id,
                    &cand.upstream_model,
                    false,
                    cost_nanos,
                    api_equiv_nanos,
                );
                return Ok(resp);
            }
            Err(e) => {
                let (st, why) = quixigate_providers::describe_reqwest_error(&e);
                last_err = Some(QuixigateError::Upstream {
                    status: st.map(|s| s.as_u16()).unwrap_or(502),
                    provider: cand.provider_id.clone(),
                    message: why,
                });
            }
        }
    }

    Err(last_err
        .unwrap_or(QuixigateError::NoAvailableRoute(requested_model))
        .into())
}

async fn handle_audio_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Response> {
    forward_multipart_endpoint(
        state,
        headers,
        body,
        "audio/transcriptions",
        IngressProtocol::Audio,
        None,
    )
    .await
}

async fn handle_audio_translations(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Response> {
    forward_multipart_endpoint(
        state,
        headers,
        body,
        "audio/translations",
        IngressProtocol::Audio,
        None,
    )
    .await
}

/// Extracts `model` from JSON or `multipart/form-data` payload bytes without re-encoding binary parts.
fn extract_model_from_raw_body(body: &[u8], default_model: &str) -> String {
    if let Ok(val) = serde_json::from_slice::<Value>(body)
        && let Some(m) = val["model"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    {
        return m.to_owned();
    }
    let lossy = String::from_utf8_lossy(&body[..body.len().min(16_384)]);
    for marker in ["name=\"model\"", "name='model'"] {
        if let Some(pos) = lossy.find(marker) {
            let after = &lossy[pos + marker.len()..];
            if let Some(sep) = after
                .find("\r\n\r\n")
                .map(|i| (i, 4))
                .or_else(|| after.find("\n\n").map(|i| (i, 2)))
            {
                let val_start = &after[sep.0 + sep.1..];
                if let Some(line) = val_start.lines().next() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() && !trimmed.starts_with("--") {
                        return trimmed.to_owned();
                    }
                }
            }
        }
    }
    default_model.to_owned()
}

async fn forward_multipart_endpoint(
    state: AppState,
    headers: HeaderMap,
    body: Bytes,
    upstream_subpath: &'static str,
    protocol: IngressProtocol,
    model_override: Option<String>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let default_model = if upstream_subpath.starts_with("audio/") {
        "whisper-1"
    } else {
        "gpt-image-1"
    };
    let requested_model = model_override
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| extract_model_from_raw_body(&body, default_model));

    let est_tokens = ((body.len() / 256) as u64).max(32);
    let (_, est_cost) = state
        .router
        .estimate_preflight_budget(&requested_model, est_tokens, 0)?;
    let (_conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        None,
        &requested_model,
        est_tokens,
        est_cost,
    )?;

    let request_id = format!("mm_{}", Uuid::new_v4().simple());
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data")
        .to_owned();

    let candidates = state
        .router
        .resolve_http_candidates(&requested_model)
        .await?;
    if candidates.is_empty() {
        return Err(QuixigateError::NoAvailableRoute(requested_model).into());
    }

    let started = Instant::now();
    let mut last_err: Option<QuixigateError> = None;

    for (idx, cand) in candidates.iter().enumerate() {
        let url = format!("{}/{upstream_subpath}", cand.base_url);
        let mut req_builder = state
            .router
            .inference_http()
            .post(&url)
            .header(header::CONTENT_TYPE, &content_type)
            .body(body.clone());
        if let Some(k) = cand.api_key.as_deref().filter(|s| !s.is_empty()) {
            req_builder = req_builder.bearer_auth(k);
        }
        for (hk, hv) in &cand.extra_headers {
            req_builder = req_builder.header(hk, hv);
        }

        match req_builder.send().await {
            Ok(up_resp) => {
                let status = up_resp.status();
                let resp_ct = up_resp
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/json")
                    .to_owned();
                let resp_bytes = up_resp.bytes().await.unwrap_or_default();
                if !status.is_success() {
                    let msg = String::from_utf8_lossy(&resp_bytes).into_owned();
                    last_err = Some(QuixigateError::Upstream {
                        status: status.as_u16(),
                        provider: cand.provider_id.clone(),
                        message: msg,
                    });
                    continue;
                }
                if let Some(cid) = &cand.credential_id {
                    let _ = state.db.mark_credential_used(cid);
                }
                let (cost_nanos, api_equiv_nanos) = state.router.catalog().compute_cost(
                    &cand.provider_id,
                    cand.auth_kind,
                    &cand.upstream_model,
                    est_tokens,
                    0,
                    0,
                    cand.route.as_ref(),
                );
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id.clone(),
                    protocol,
                    requested_model,
                    cand.upstream_model.clone(),
                    cand.provider_id.clone(),
                    cand.credential_id.clone(),
                    status.as_u16(),
                    false,
                    idx as u32,
                    false,
                    None,
                    started.elapsed().as_millis() as u64,
                    est_tokens,
                    0,
                    0,
                    0,
                    cost_nanos,
                    api_equiv_nanos,
                    None,
                    None,
                    None,
                    OpenInferencePayload::default(),
                    None,
                    Vec::new(),
                );
                let mut resp = (status, resp_bytes).into_response();
                if let Ok(hv) = HeaderValue::from_str(&resp_ct) {
                    resp.headers_mut().insert(header::CONTENT_TYPE, hv);
                }
                attach_gateway_headers(
                    resp.headers_mut(),
                    &request_id,
                    &cand.provider_id,
                    &cand.upstream_model,
                    false,
                    cost_nanos,
                    api_equiv_nanos,
                );
                return Ok(resp);
            }
            Err(e) => {
                let (st, why) = quixigate_providers::describe_reqwest_error(&e);
                last_err = Some(QuixigateError::Upstream {
                    status: st.map(|s| s.as_u16()).unwrap_or(502),
                    provider: cand.provider_id.clone(),
                    message: why,
                });
            }
        }
    }

    Err(last_err
        .unwrap_or(QuixigateError::NoAvailableRoute(requested_model))
        .into())
}

// ── Moderations & Rerank ─────────────────────────────────────────────────────

/// `POST /v1/moderations` & `POST /moderations` — Dispatches to upstream OpenAI-compatible moderation
/// endpoint when configured, falling back automatically to QuixiGate's local guardrail classifier.
pub async fn handle_moderations(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("omni-moderation-latest")
        .to_owned();

    let inputs: Vec<String> = match &body["input"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| extract_content_text(v))
            })
            .collect(),
        _ => {
            return Err(QuixigateError::Validation(
                "'input' must be a string or array for /v1/moderations".to_owned(),
            )
            .into());
        }
    };
    if inputs.is_empty() {
        return Err(
            QuixigateError::Validation("moderation 'input' cannot be empty".to_owned()).into(),
        );
    }

    let est_tokens: u64 = inputs.iter().map(|s| ((s.len() / 4) as u64).max(1)).sum();
    let (_conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        &requested_model,
        est_tokens,
        NanoUsd::ZERO,
    )?;

    let request_id = format!("modr_{}", Uuid::new_v4().simple());
    let started = Instant::now();

    // Try upstream HTTP candidate if configured (`openai` or explicit route)
    if let Ok(candidates) = state.router.resolve_http_candidates(&requested_model).await {
        for (idx, cand) in candidates.iter().enumerate() {
            if let Some(obj) = body.as_object_mut() {
                obj.insert(
                    "model".to_owned(),
                    Value::String(cand.upstream_model.clone()),
                );
            }
            let url = format!("{}/moderations", cand.base_url);
            let mut req_builder = state
                .router
                .inference_http()
                .post(&url)
                .header(header::CONTENT_TYPE, "application/json");
            if let Some(k) = cand.api_key.as_deref().filter(|s| !s.is_empty()) {
                req_builder = req_builder.bearer_auth(k);
            }
            if let Ok(up_resp) = req_builder.json(&body).send().await
                && up_resp.status().is_success()
                && let Ok(val) = up_resp.json::<Value>().await
            {
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id.clone(),
                    IngressProtocol::Moderations,
                    requested_model,
                    cand.upstream_model.clone(),
                    cand.provider_id.clone(),
                    cand.credential_id.clone(),
                    200,
                    false,
                    idx as u32,
                    false,
                    None,
                    started.elapsed().as_millis() as u64,
                    est_tokens,
                    0,
                    0,
                    0,
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
                    None,
                    None,
                    None,
                    OpenInferencePayload::default(),
                    None,
                    Vec::new(),
                );
                let mut resp = Json(val).into_response();
                attach_gateway_headers(
                    resp.headers_mut(),
                    &request_id,
                    &cand.provider_id,
                    &cand.upstream_model,
                    false,
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
                );
                return Ok(resp);
            }
        }
    }

    // Offline / local guardrail moderation fallback
    let settings = state.db.get_settings()?;
    let banned_lower: Vec<String> = settings
        .banned_keywords
        .iter()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let results: Vec<Value> = inputs
        .iter()
        .map(|txt| {
            let secret = detect_first_secret(txt);
            let lower = txt.to_lowercase();
            let kw_hit = banned_lower.iter().any(|kw| lower.contains(kw));
            let injection_hit = lower.contains("ignore previous instructions")
                || lower.contains("<|im_start|>")
                || lower.contains("system override");
            let flagged = secret.is_some() || kw_hit || injection_hit;
            json!({
                "flagged": flagged,
                "categories": {
                    "sexual": false,
                    "hate": false,
                    "harassment": false,
                    "self-harm": false,
                    "violence": false,
                    "illicit": kw_hit,
                    "quixigate/secret_leak": secret.is_some(),
                    "quixigate/banned_keyword": kw_hit,
                    "quixigate/prompt_injection": injection_hit,
                },
                "category_scores": {
                    "sexual": 0.0,
                    "hate": 0.0,
                    "harassment": 0.0,
                    "self-harm": 0.0,
                    "violence": 0.0,
                    "illicit": if kw_hit { 1.0 } else { 0.0 },
                    "quixigate/secret_leak": if secret.is_some() { 1.0 } else { 0.0 },
                    "quixigate/banned_keyword": if kw_hit { 1.0 } else { 0.0 },
                    "quixigate/prompt_injection": if injection_hit { 0.95 } else { 0.0 },
                }
            })
        })
        .collect();

    finalize_request_accounting(
        &state,
        &principal,
        budget_guard,
        request_id.clone(),
        IngressProtocol::Moderations,
        requested_model.clone(),
        "quixigate-local-moderation".to_owned(),
        "quixigate".to_owned(),
        None,
        200,
        false,
        0,
        false,
        None,
        started.elapsed().as_millis() as u64,
        est_tokens,
        0,
        0,
        0,
        NanoUsd::ZERO,
        NanoUsd::ZERO,
        None,
        None,
        None,
        OpenInferencePayload::default(),
        None,
        Vec::new(),
    );

    let mut resp = Json(json!({
        "id": request_id,
        "model": requested_model,
        "results": results,
    }))
    .into_response();
    attach_gateway_headers(
        resp.headers_mut(),
        &request_id,
        "quixigate",
        "quixigate-local-moderation",
        false,
        NanoUsd::ZERO,
        NanoUsd::ZERO,
    );
    Ok(resp)
}

/// `POST /v1/rerank`, `POST /v2/rerank`, `POST /rerank` — Cohere / Jina / Voyage / LiteLLM Rerank API.
pub async fn handle_rerank(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("quixigate-rerank")
        .to_owned();
    let query = body["query"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("missing 'query' string for /rerank".to_owned()))?
        .to_owned();
    let raw_docs = body["documents"].as_array().ok_or_else(|| {
        QuixigateError::Validation("'documents' must be an array for /rerank".to_owned())
    })?;
    if raw_docs.is_empty() {
        return Err(QuixigateError::Validation("'documents' cannot be empty".to_owned()).into());
    }

    let doc_texts: Vec<String> = raw_docs
        .iter()
        .map(|d| match d {
            Value::String(s) => s.clone(),
            Value::Object(map) => map
                .get("text")
                .or_else(|| map.get("content"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| d.to_string()),
            other => other.to_string(),
        })
        .collect();

    let top_n = body["top_n"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(doc_texts.len())
        .min(doc_texts.len());
    let return_documents = body["return_documents"].as_bool().unwrap_or(true);

    let est_tokens: u64 =
        ((query.len() + doc_texts.iter().map(String::len).sum::<usize>()) / 4).max(1) as u64;
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

    let request_id = format!("rerank_{}", Uuid::new_v4().simple());
    let started = Instant::now();

    let use_local_lexical = body["mock_response"].is_boolean()
        || body["mock_response"].is_string()
        || matches!(
            requested_model.as_str(),
            "local" | "quixigate-rerank" | "lexical"
        );

    if !use_local_lexical {
        let candidates = state
            .router
            .resolve_http_candidates(&requested_model)
            .await?;
        let mut last_err: Option<QuixigateError> = None;
        for (idx, cand) in candidates.iter().enumerate() {
            if let Some(obj) = body.as_object_mut() {
                obj.insert(
                    "model".to_owned(),
                    Value::String(cand.upstream_model.clone()),
                );
            }
            let url = format!("{}/rerank", cand.base_url);
            let mut req_builder = state
                .router
                .inference_http()
                .post(&url)
                .header(header::CONTENT_TYPE, "application/json");
            if let Some(k) = cand.api_key.as_deref().filter(|s| !s.is_empty()) {
                req_builder = req_builder.bearer_auth(k);
            }
            for (hk, hv) in &cand.extra_headers {
                req_builder = req_builder.header(hk, hv);
            }

            match req_builder.json(&body).send().await {
                Ok(up_resp) => {
                    let status = up_resp.status();
                    let resp_val: Value = up_resp.json().await.unwrap_or_else(|_| json!({}));
                    if !status.is_success() {
                        last_err = Some(QuixigateError::Upstream {
                            status: status.as_u16(),
                            provider: cand.provider_id.clone(),
                            message: resp_val["error"]["message"]
                                .as_str()
                                .or_else(|| resp_val["message"].as_str())
                                .unwrap_or("rerank upstream error")
                                .to_owned(),
                        });
                        continue;
                    }
                    if let Some(cid) = &cand.credential_id {
                        let _ = state.db.mark_credential_used(cid);
                    }
                    let (cost_nanos, api_equiv_nanos) = state.router.catalog().compute_cost(
                        &cand.provider_id,
                        cand.auth_kind,
                        &cand.upstream_model,
                        est_tokens,
                        0,
                        0,
                        cand.route.as_ref(),
                    );
                    finalize_request_accounting(
                        &state,
                        &principal,
                        budget_guard,
                        request_id.clone(),
                        IngressProtocol::Rerank,
                        requested_model,
                        cand.upstream_model.clone(),
                        cand.provider_id.clone(),
                        cand.credential_id.clone(),
                        200,
                        false,
                        idx as u32,
                        false,
                        None,
                        started.elapsed().as_millis() as u64,
                        est_tokens,
                        0,
                        0,
                        0,
                        cost_nanos,
                        api_equiv_nanos,
                        None,
                        None,
                        None,
                        OpenInferencePayload::default(),
                        None,
                        Vec::new(),
                    );
                    let mut resp = Json(resp_val).into_response();
                    attach_gateway_headers(
                        resp.headers_mut(),
                        &request_id,
                        &cand.provider_id,
                        &cand.upstream_model,
                        false,
                        cost_nanos,
                        api_equiv_nanos,
                    );
                    return Ok(resp);
                }
                Err(e) => {
                    let (st, why) = quixigate_providers::describe_reqwest_error(&e);
                    last_err = Some(QuixigateError::Upstream {
                        status: st.map(|s| s.as_u16()).unwrap_or(502),
                        provider: cand.provider_id.clone(),
                        message: why,
                    });
                }
            }
        }
        if let Some(err) = last_err {
            return Err(err.into());
        }
        if candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(requested_model).into());
        }
    }

    // Built-in deterministic lexical overlap scorer (`model = "quixigate-rerank"` / `mock_response`)
    let q_terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect();
    let mut scored: Vec<(usize, f64)> = doc_texts
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let d_lower = doc.to_lowercase();
            let hits = q_terms
                .iter()
                .filter(|term| d_lower.contains(term.as_str()))
                .count();
            let score = if q_terms.is_empty() {
                0.0
            } else {
                (hits as f64) / (q_terms.len() as f64)
            };
            (idx, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_n);

    let results: Vec<Value> = scored
        .into_iter()
        .map(|(index, relevance_score)| {
            if return_documents {
                json!({
                    "index": index,
                    "relevance_score": relevance_score,
                    "document": { "text": doc_texts[index] }
                })
            } else {
                json!({
                    "index": index,
                    "relevance_score": relevance_score
                })
            }
        })
        .collect();

    let _ = budget_guard.commit(NanoUsd::ZERO);
    let mut resp = Json(json!({
        "id": request_id,
        "model": requested_model,
        "results": results,
        "meta": { "billed_units": { "search_units": 1 } }
    }))
    .into_response();
    attach_gateway_headers(
        resp.headers_mut(),
        &request_id,
        "quixigate",
        &requested_model,
        false,
        NanoUsd::ZERO,
        NanoUsd::ZERO,
    );
    Ok(resp)
}

// ── Web Search (`/v1/search*`) ───────────────────────────────────────────────

async fn run_search_request(
    state: AppState,
    headers: HeaderMap,
    body: Value,
    tool_pin: Option<String>,
) -> ApiResult<Json<Value>> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let query = body["query"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            QuixigateError::Validation("missing 'query' string for /v1/search".to_owned())
        })?
        .to_owned();
    let max_results = body["max_results"]
        .as_u64()
        .or_else(|| body["top_k"].as_u64())
        .map(|v| v as usize)
        .unwrap_or(5)
        .clamp(1, 10);
    let pin = tool_pin.or_else(|| {
        body["search_tool_name"]
            .as_str()
            .or_else(|| body["provider"].as_str())
            .map(str::to_owned)
    });

    let (_conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        "quixigate-web-search",
        32,
        NanoUsd::ZERO,
    )?;

    if body["mock_response"].is_boolean() || body["mock_response"].is_string() {
        let _ = budget_guard.commit(NanoUsd::ZERO);
        return Ok(Json(json!({
            "object": "search",
            "query": query,
            "provider": pin.as_deref().unwrap_or("mock"),
            "results": [{
                "title": format!("Mock search result for {query}"),
                "url": "https://quixi.ai/docs",
                "snippet": body["mock_response"].as_str().unwrap_or("Mock web search snippet."),
            }]
        })));
    }

    let started = Instant::now();
    let search_out = state
        .router
        .execute_web_search_with_tool(&query, max_results, pin.as_deref())
        .await
        .map_err(|e| QuixigateError::Upstream {
            status: 502,
            provider: pin.clone().unwrap_or_else(|| "web_search".to_owned()),
            message: e,
        })?;

    let provider_id = search_out["provider"]["id"]
        .as_str()
        .unwrap_or("web_search")
        .to_owned();
    finalize_request_accounting(
        &state,
        &principal,
        budget_guard,
        format!("srch_{}", Uuid::new_v4().simple()),
        IngressProtocol::Search,
        "quixigate-web-search".to_owned(),
        provider_id.clone(),
        provider_id.clone(),
        None,
        200,
        false,
        0,
        search_out["cached"].as_bool().unwrap_or(false),
        None,
        started.elapsed().as_millis() as u64,
        32,
        0,
        0,
        0,
        NanoUsd::ZERO,
        NanoUsd::ZERO,
        None,
        None,
        None,
        OpenInferencePayload::default(),
        None,
        Vec::new(),
    );

    Ok(Json(json!({
        "object": "search",
        "query": query,
        "provider": provider_id,
        "cached": search_out["cached"],
        "results": search_out["results"],
    })))
}

pub async fn handle_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    run_search_request(state, headers, body, None).await
}

pub async fn handle_search_by_tool(
    State(state): State<AppState>,
    Path(search_tool_name): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    run_search_request(state, headers, body, Some(search_tool_name)).await
}

pub async fn handle_search_tools(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let creds = state.db.list_credentials()?;
    let has_prov = |p: &str| {
        creds
            .iter()
            .any(|c| c.enabled && c.provider_id.eq_ignore_ascii_case(p))
    };
    let tools = vec![
        json!({ "search_tool_name": "tavily", "provider": "tavily", "configured": has_prov("tavily") || std::env::var("TAVILY_API_KEY").is_ok() }),
        json!({ "search_tool_name": "exa", "provider": "exa", "configured": has_prov("exa") || std::env::var("EXA_API_KEY").is_ok() }),
        json!({ "search_tool_name": "brave", "provider": "brave", "configured": has_prov("brave") || std::env::var("BRAVE_API_KEY").is_ok() }),
        json!({ "search_tool_name": "perplexity", "provider": "perplexity", "configured": has_prov("perplexity") || std::env::var("PERPLEXITY_API_KEY").is_ok() }),
        json!({ "search_tool_name": "searxng", "provider": "searxng", "configured": std::env::var("SEARXNG_URL").is_ok() }),
        json!({ "search_tool_name": "exa_mcp", "provider": "exa_mcp", "configured": true, "keyless": true }),
        json!({ "search_tool_name": "parallel_mcp", "provider": "parallel_mcp", "configured": true, "keyless": true }),
        json!({ "search_tool_name": "duckduckgo", "provider": "duckduckgo", "configured": true, "keyless": true }),
    ];
    Ok(Json(json!({ "search_tools": tools, "data": tools })))
}

// ── Developer Utilities & Guardrails ─────────────────────────────────────────

/// `POST /utils/token_counter`
pub async fn handle_utils_token_counter(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let model = body["model"].as_str().unwrap_or("gpt-4o").to_owned();

    let total_tokens = if let Some(msgs) = body["messages"].as_array() {
        let (system, messages) = parse_openai_messages(msgs);
        let req = CanonicalRequest {
            requested_model: model.clone(),
            system,
            messages,
            client_tools: parse_openai_tools(&body),
            ..Default::default()
        };
        estimate_request_tokens(&req)
    } else {
        let text = body["prompt"]
            .as_str()
            .or_else(|| body["text"].as_str())
            .unwrap_or_default();
        ((text.len() / 4) as u64).max(usize::from(!text.is_empty()) as u64)
    };

    Ok(Json(json!({
        "total_tokens": total_tokens,
        "request_model": model,
        "model_used": model,
        "tokenizer_type": "quixigate_estimator"
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelUtilQuery {
    pub model: Option<String>,
    pub litellm_model_name: Option<String>,
}

/// `GET /utils/supported_openai_params`
pub async fn handle_utils_supported_openai_params(
    State(state): State<AppState>,
    Query(q): Query<ModelUtilQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let model = q
        .model
        .or(q.litellm_model_name)
        .unwrap_or_else(|| "gpt-4o".to_owned());
    let mut params = vec![
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "user",
        "metadata",
    ];
    let catalog = state.router.list_available_models();
    if catalog
        .iter()
        .any(|m| (m.canonical_id == model || m.bare_id == model) && m.supports_reasoning)
    {
        params.push("reasoning_effort");
    }
    Ok(Json(json!({
        "model": model,
        "supported_openai_params": params
    })))
}

/// `GET /utils/model_info`
pub async fn handle_utils_model_info(
    State(state): State<AppState>,
    Query(q): Query<ModelUtilQuery>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let model = q
        .model
        .or(q.litellm_model_name)
        .ok_or_else(|| QuixigateError::Validation("missing 'model' query parameter".to_owned()))?;
    let spec = ModelSpec::parse(&model);
    let route = state.db.get_model_route(&spec.base_model)?;
    let catalog = state.router.list_available_models();
    let cat_match = catalog
        .into_iter()
        .find(|m| m.canonical_id == spec.base_model || m.bare_id == spec.base_model);

    let (in_nanos_m, out_nanos_m, ctx, provider, tools, vision, reasoning) =
        if let Some(m) = cat_match {
            (
                route
                    .as_ref()
                    .and_then(|r| r.input_nanos_per_m)
                    .unwrap_or(m.input_nanos_per_m.nanos()),
                route
                    .as_ref()
                    .and_then(|r| r.output_nanos_per_m)
                    .unwrap_or(m.output_nanos_per_m.nanos()),
                m.context_window,
                m.author,
                m.supports_tools,
                m.supports_vision,
                m.supports_reasoning,
            )
        } else if let Some(r) = &route {
            (
                r.input_nanos_per_m.unwrap_or(0),
                r.output_nanos_per_m.unwrap_or(0),
                200_000,
                "quixigate-alias".to_owned(),
                true,
                true,
                true,
            )
        } else {
            return Err(
                QuixigateError::NotFound(format!("model '{model}' not found in catalog")).into(),
            );
        };

    let in_per_tok = (in_nanos_m as f64) / 1_000_000.0 / (NANOS_PER_USD as f64);
    let out_per_tok = (out_nanos_m as f64) / 1_000_000.0 / (NANOS_PER_USD as f64);

    Ok(Json(json!({
        "model_info": {
            "key": model,
            "max_tokens": ctx,
            "max_input_tokens": ctx,
            "max_output_tokens": (ctx / 4).max(4_096),
            "input_cost_per_token": in_per_tok,
            "output_cost_per_token": out_per_tok,
            "input_nanos_per_m": in_nanos_m,
            "output_nanos_per_m": out_nanos_m,
            "litellm_provider": provider,
            "mode": "chat",
            "supports_function_calling": tools,
            "supports_vision": vision,
            "supports_reasoning": reasoning,
        }
    })))
}

/// `POST /utils/transform_request` — Dry-run inspection of QuixiGate's canonical request normalization & routing.
pub async fn handle_utils_transform_request(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let call_type = body["call_type"]
        .as_str()
        .unwrap_or("completion")
        .to_owned();
    let req_body = body.get("request_body").unwrap_or(&body);
    let model = req_body["model"].as_str().unwrap_or("gpt-4o").to_owned();
    let spec = ModelSpec::parse(&model);
    let route = state.db.get_model_route(&spec.base_model)?;
    let creds = state.db.list_credentials()?;
    let settings = state.db.get_settings()?;
    let candidates = state.router.pool().resolve_candidates(
        &spec,
        route.as_ref(),
        &creds,
        state.router.catalog(),
        state.router.local_scanner(),
        &settings,
    );

    let raw_msgs = req_body["messages"].as_array().cloned().unwrap_or_default();
    let (system, parsed_msgs) = parse_openai_messages(&raw_msgs);
    let tools = parse_openai_tools(req_body);

    let candidate_views: Vec<Value> = candidates
        .iter()
        .map(|c| {
            json!({
                "provider_id": c.provider_id,
                "upstream_model": c.upstream_model,
                "credential_id": c.credential.as_ref().map(|cr| &cr.id),
                "base_url_override": c.base_url_override,
            })
        })
        .collect();

    Ok(Json(json!({
        "call_type": call_type,
        "requested_model": model,
        "online_grounding": spec.online,
        "normalized_system": system,
        "normalized_message_count": parsed_msgs.len(),
        "normalized_tools": tools,
        "resolved_candidates": candidate_views,
        "raw_request_body": req_body,
    })))
}

/// `POST /cost/estimate` — Exact `NanoUsd` per-request, daily, and monthly cost estimator.
pub async fn handle_cost_estimate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let model = body["model"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("missing 'model' field".to_owned()))?;
    let input_tokens = body["input_tokens"]
        .as_u64()
        .or_else(|| body["prompt_tokens"].as_u64())
        .unwrap_or(0);
    let output_tokens = body["output_tokens"]
        .as_u64()
        .or_else(|| body["completion_tokens"].as_u64())
        .unwrap_or(0);
    let cached_tokens = body["cached_tokens"].as_u64().unwrap_or(0);
    let requests_per_day = body["num_requests_per_day"]
        .as_f64()
        .unwrap_or(1.0)
        .max(0.0);

    let spec = ModelSpec::parse(model);
    let route = state.db.get_model_route(&spec.base_model)?;
    let provider = spec
        .pinned_target
        .as_deref()
        .or_else(|| spec.base_model.split_once('/').map(|(p, _)| p))
        .unwrap_or("openai");
    let (_, input_only_nanos) = state.router.catalog().compute_cost(
        provider,
        AuthKind::ApiKey,
        &spec.base_model,
        input_tokens,
        cached_tokens,
        0,
        route.as_ref(),
    );
    let (_, output_only_nanos) = state.router.catalog().compute_cost(
        provider,
        AuthKind::ApiKey,
        &spec.base_model,
        0,
        0,
        output_tokens,
        route.as_ref(),
    );
    let total_nanos = NanoUsd(
        input_only_nanos
            .nanos()
            .saturating_add(output_only_nanos.nanos()),
    );
    let per_req_usd = (total_nanos.nanos() as f64) / (NANOS_PER_USD as f64);
    let daily_cost = per_req_usd * requests_per_day;
    let monthly_cost = daily_cost * 30.0;

    Ok(Json(json!({
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_per_request": (input_only_nanos.nanos() as f64) / (NANOS_PER_USD as f64),
        "output_cost_per_request": (output_only_nanos.nanos() as f64) / (NANOS_PER_USD as f64),
        "cost_per_request": per_req_usd,
        "total_cost": per_req_usd,
        "cost_per_request_decimal": total_nanos.to_decimal_string(),
        "cost_nanos": total_nanos.nanos(),
        "total_cost_nanos": total_nanos.nanos(),
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
    })))
}

/// `GET /guardrails/list` & `GET /v2/guardrails/list`
pub async fn handle_guardrails_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.get_settings()?;
    let guardrails = vec![
        json!({
            "guardrail_id": "quixigate-secret-scanner",
            "guardrail_name": "quixigate-secret-scanner",
            "litellm_params": {
                "guardrail": "quixigate_secret_scanner",
                "mode": settings.secret_scanner_mode,
                "default_on": settings.secret_scanner_mode != SecretScannerMode::Off,
            },
            "guardrail_info": {
                "description": "In-process regex scanner for leaked API keys & PEM private keys (off | redact | block)."
            }
        }),
        json!({
            "guardrail_id": "quixigate-banned-keywords",
            "guardrail_name": "quixigate-banned-keywords",
            "litellm_params": {
                "guardrail": "quixigate_banned_keywords",
                "mode": "pre_call",
                "default_on": !settings.banned_keywords.is_empty(),
                "banned_keywords_count": settings.banned_keywords.len(),
            },
            "guardrail_info": {
                "description": "Case-insensitive pre-flight keyword & phrase blocker."
            }
        }),
    ];
    Ok(Json(json!({ "guardrails": guardrails })))
}

/// `POST /apply_guardrail` & `POST /guardrails/apply_guardrail`
pub async fn handle_apply_guardrail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.get_settings()?;
    let guardrail_name = body["guardrail_name"]
        .as_str()
        .unwrap_or("quixigate-secret-scanner");
    let raw_text = body["text"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            body["texts"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
        })
        .unwrap_or_default();

    let detected_secret = detect_first_secret(&raw_text);
    let redacted_text = redact_secrets_in_str(&raw_text);
    let lower = raw_text.to_lowercase();
    let matched_keywords: Vec<String> = settings
        .banned_keywords
        .iter()
        .filter(|kw| {
            let trimmed = kw.trim().to_lowercase();
            !trimmed.is_empty() && lower.contains(&trimmed)
        })
        .cloned()
        .collect();

    let blocked = !matched_keywords.is_empty()
        || (detected_secret.is_some()
            && (settings.secret_scanner_mode == SecretScannerMode::Block
                || guardrail_name.contains("block")));

    let triggered = blocked || redacted_text != raw_text;
    Ok(Json(json!({
        "guardrail_name": guardrail_name,
        "response_text": redacted_text,
        "triggered": triggered,
        "blocked": blocked,
        "action": if blocked { "blocked" } else if redacted_text != raw_text { "redacted" } else { "passed" },
        "detected_secret_type": detected_secret,
        "matched_banned_keywords": matched_keywords,
    })))
}
