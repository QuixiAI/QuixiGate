//! `POST /v1/chat/completions` — Two-Lane OpenAI Chat Completions Codec:
//! 1. **Zero-DOM Cut-Through Lane**: Borrowed `ChatEnvelopePeek` + SIMD raw-`Bytes` model/usage
//!    splicer + `CutThroughSseStream` with Actix-style greedy `poll_next` write coalescing (`32 KB`
//!    watermark, `0 ms` timer delay, zero `tokio::spawn`, zero `mpsc` channel, zero per-token JSON DOM)
//!    + RAII `Drop` spend/telemetry accounting.
//! 2. **Polyglot Transcoding Lane**: Full `CanonicalRequest` normalization for non-OpenAI upstreams
//!    (Anthropic, ChatGPT Codex, Gemini Code Assist), `:online` / MCP server-side tool loops,
//!    exact BLAKE3 response caching, secret redaction/blocking guardrails, and payload/OTLP capture.

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::BytesMut;
use futures::Stream;
use quixigate_core::{IngressProtocol, NanoUsd, QuixigateError, SecretScannerMode, now_ms};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::{
    CanonicalRequest, CutThroughJsonOutcome, CutThroughStreamOutcome, CutThroughWireFormat,
    RoutedStream, RoutedStreamEvent, StreamCompletionSummary,
};
use quixigate_telemetry::{OpenInferencePayload, TelemetryHub};
use serde::Deserialize;
use serde::de::IgnoredAny;
use serde_json::{Map, Value, json};

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

// ── Zero-DOM Request Envelope Peek ───────────────────────────────────────────

#[derive(Deserialize, Default)]
struct StreamOptionsPeek {
    #[serde(default)]
    include_usage: Option<bool>,
}

#[derive(Deserialize)]
struct ChatEnvelopePeek<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    messages: Option<MessagesArrayMarker>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(borrow, default)]
    mock_response: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    user: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    end_user_id: Option<Cow<'a, str>>,
    #[serde(default)]
    stream_options: Option<StreamOptionsPeek>,
    #[serde(borrow, default)]
    tags: Option<TagsPeek<'a>>,
    #[serde(borrow, default)]
    metadata: Option<MetadataPeek<'a>>,
    #[serde(default)]
    provider: Option<IgnoredAny>,
    #[serde(default)]
    transforms: Option<IgnoredAny>,
    #[serde(default)]
    models: Option<IgnoredAny>,
    #[serde(default)]
    plugins: Option<IgnoredAny>,
    #[serde(default)]
    route: Option<IgnoredAny>,
}

impl<'a> ChatEnvelopePeek<'a> {
    #[inline]
    fn has_openrouter_only_fields(&self) -> bool {
        self.provider.is_some()
            || self.transforms.is_some()
            || self.models.is_some()
            || self.plugins.is_some()
            || self.route.is_some()
    }
}

// ── Route Entrypoint ─────────────────────────────────────────────────────────

pub async fn handle_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    raw_body: Bytes,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.settings_arc();
    let capture_payloads = principal
        .virtual_key
        .as_ref()
        .and_then(|k| k.log_payloads)
        .unwrap_or(settings.capture_payloads_default);
    let cache_pref = parse_cache_preference(&headers);

    // ── Fast Lane Gate: Zero-DOM Peek + Cut-Through Bit-Shuffler ─────────────
    let can_attempt_peek = settings.enforced_params.is_empty()
        && settings.secret_scanner_mode == SecretScannerMode::Off
        && settings.banned_keywords.is_empty()
        && !capture_payloads
        && !settings.otlp_enabled
        && settings.mcp_server_urls.is_empty()
        && !state.router.has_mcp_urls();

    let mut pre_reserved: Option<(ConcurrencyGuard, BudgetGuard, String)> = None;

    if can_attempt_peek && let Ok(peek) = serde_json::from_slice::<ChatEnvelopePeek<'_>>(&raw_body)
    {
        let requested_model_str = peek
            .model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| QuixigateError::Validation("missing 'model' field".to_owned()))?;
        if peek.messages.is_none() {
            return Err(ApiError(QuixigateError::Validation(
                "'messages' must be an array".to_owned(),
            )));
        }

        let cache_wanted = cache_pref.unwrap_or(settings.cache_enabled_default);
        if peek.mock_response.is_none() && !cache_wanted {
            let est_in = ((raw_body.len() / 4) as u64).max(1) + 8;
            let max_out = peek
                .max_completion_tokens
                .or(peek.max_tokens)
                .unwrap_or(1024);
            let plan =
                state
                    .router
                    .prepare_cut_through_plan(requested_model_str, est_in, max_out)?;

            if !plan.spec().online {
                let req_tags =
                    collect_peek_tags(&headers, peek.metadata.as_ref(), peek.tags.as_ref());
                let end_user_id =
                    peek_end_user(&headers, peek.user.as_deref(), peek.end_user_id.as_deref());
                let (conc_guard, mut budget_guard) = state.db.authorize_and_reserve_full(
                    &principal,
                    requested_model_str,
                    plan.est_total_tokens,
                    plan.est_cost,
                    &req_tags,
                    end_user_id,
                )?;
                if should_suppress_request_log(&headers, &settings) {
                    budget_guard.set_suppress_log(true);
                }

                let stream_requested = peek.stream.unwrap_or(false);
                let has_include_usage = peek
                    .stream_options
                    .as_ref()
                    .and_then(|so| so.include_usage)
                    .unwrap_or(false);
                let has_or_only = peek.has_openrouter_only_fields();
                let passthrough_headers = extract_passthrough_headers_vec(&headers);
                let request_id = TelemetryHub::fast_prefixed_id("chatcmpl_");

                if stream_requested {
                    state.telemetry.inc_active();
                    let started = Instant::now();
                    match state
                        .router
                        .execute_cut_through_stream(
                            plan,
                            &raw_body,
                            has_include_usage,
                            has_or_only,
                            &passthrough_headers,
                        )
                        .await
                    {
                        Ok(CutThroughStreamOutcome::Stream(active)) => {
                            let provider_hdr = active.provider_id.clone();
                            let model_hdr = active.resolved_model.clone();
                            let req_id_hdr = request_id.clone();
                            let sse_stream = CutThroughSseStream::new(
                                active,
                                state,
                                principal,
                                conc_guard,
                                budget_guard,
                                CutThroughWireFormat::OpenAiChat,
                                IngressProtocol::ChatCompletions,
                                request_id,
                                est_in,
                            );
                            let mut resp = Response::new(Body::from_stream(sse_stream));
                            *resp.status_mut() = StatusCode::OK;
                            let hdrs = resp.headers_mut();
                            hdrs.insert(
                                header::CONTENT_TYPE,
                                HeaderValue::from_static("text/event-stream"),
                            );
                            hdrs.insert(
                                header::CACHE_CONTROL,
                                HeaderValue::from_static("no-cache"),
                            );
                            attach_gateway_headers(
                                hdrs,
                                &req_id_hdr,
                                &provider_hdr,
                                &model_hdr,
                                false,
                                NanoUsd::ZERO,
                                NanoUsd::ZERO,
                            );
                            return Ok(resp);
                        }
                        Ok(CutThroughStreamOutcome::FallbackTranscode) => {
                            state.telemetry.dec_active();
                            pre_reserved =
                                Some((conc_guard, budget_guard, requested_model_str.to_owned()));
                        }
                        Err(err) => {
                            state.telemetry.dec_active();
                            drop(conc_guard);
                            finalize_cut_through_accounting(
                                &state,
                                &principal,
                                budget_guard,
                                IngressProtocol::ChatCompletions,
                                request_id,
                                requested_model_str.to_owned(),
                                requested_model_str.to_owned(),
                                error_provider_id(&err),
                                None,
                                err.http_status(),
                                true,
                                0,
                                None,
                                started.elapsed().as_millis() as u64,
                                0,
                                0,
                                0,
                                0,
                                NanoUsd::ZERO,
                                NanoUsd::ZERO,
                                Some(err.to_string()),
                            );
                            return Err(ApiError(err));
                        }
                    }
                } else {
                    state.telemetry.inc_active();
                    let started = Instant::now();
                    match state
                        .router
                        .execute_cut_through_json(
                            plan,
                            &raw_body,
                            est_in,
                            has_or_only,
                            &passthrough_headers,
                        )
                        .await
                    {
                        Ok(CutThroughJsonOutcome::Ready(res)) => {
                            state.telemetry.dec_active();
                            drop(conc_guard);
                            let mut resp = Response::new(Body::from(res.body));
                            *resp.status_mut() = StatusCode::OK;
                            resp.headers_mut().insert(
                                header::CONTENT_TYPE,
                                HeaderValue::from_static("application/json"),
                            );
                            attach_gateway_headers(
                                resp.headers_mut(),
                                &request_id,
                                &res.provider_id,
                                &res.resolved_model,
                                false,
                                res.cost_nanos,
                                res.api_equiv_nanos,
                            );
                            finalize_cut_through_accounting(
                                &state,
                                &principal,
                                budget_guard,
                                IngressProtocol::ChatCompletions,
                                request_id,
                                requested_model_str.to_owned(),
                                res.resolved_model,
                                res.provider_id,
                                res.credential_id,
                                200,
                                false,
                                res.fallback_attempts,
                                None,
                                res.duration_ms,
                                res.input_tokens,
                                res.cached_input_tokens,
                                res.reasoning_tokens,
                                res.output_tokens,
                                res.cost_nanos,
                                res.api_equiv_nanos,
                                None,
                            );
                            return Ok(resp);
                        }
                        Ok(CutThroughJsonOutcome::FallbackTranscode) => {
                            state.telemetry.dec_active();
                            pre_reserved =
                                Some((conc_guard, budget_guard, requested_model_str.to_owned()));
                        }
                        Err(err) => {
                            state.telemetry.dec_active();
                            drop(conc_guard);
                            finalize_cut_through_accounting(
                                &state,
                                &principal,
                                budget_guard,
                                IngressProtocol::ChatCompletions,
                                request_id,
                                requested_model_str.to_owned(),
                                requested_model_str.to_owned(),
                                error_provider_id(&err),
                                None,
                                err.http_status(),
                                false,
                                0,
                                None,
                                started.elapsed().as_millis() as u64,
                                0,
                                0,
                                0,
                                0,
                                NanoUsd::ZERO,
                                NanoUsd::ZERO,
                                Some(err.to_string()),
                            );
                            return Err(ApiError(err));
                        }
                    }
                }
            }
        }
    }

    // ── Transcoding / Feature-Rich Lane (`serde_json::Value` + `aisdk`) ──────
    let body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| QuixigateError::Validation(format!("invalid JSON body: {e}")))?;

    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("missing 'model' field".to_owned()))?
        .to_owned();

    let raw_messages = body["messages"]
        .as_array()
        .ok_or_else(|| QuixigateError::Validation("'messages' must be an array".to_owned()))?;
    let (system, messages) = parse_openai_messages(raw_messages);
    let client_tools = parse_openai_tools(&body);

    let schema = match body["response_format"]["type"].as_str() {
        Some("json_schema") => body["response_format"]["json_schema"]
            .get("schema")
            .cloned()
            .or_else(|| body["response_format"].get("schema").cloned()),
        Some("json_object") => Some(json!({ "type": "object" })),
        _ => None,
    };

    let max_out = body["max_completion_tokens"]
        .as_u64()
        .or_else(|| body["max_tokens"].as_u64())
        .map(|v| v as u32);
    let stop_sequences = match &body["stop"] {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(arr) => Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
        ),
        _ => None,
    };

    let mut extra_body = Map::new();
    if let Some(tc) = body.get("tool_choice") {
        extra_body.insert("tool_choice".into(), tc.clone());
    }

    let req = CanonicalRequest {
        requested_model: requested_model.clone(),
        system,
        messages,
        client_tools,
        schema,
        temperature: scale_temperature(body["temperature"].as_f64()),
        top_p: scale_top_p(body["top_p"].as_f64()),
        top_k: body["top_k"].as_u64().map(|v| v as u32),
        max_output_tokens: max_out,
        stop_sequences,
        presence_penalty: body["presence_penalty"].as_f64().map(|v| v as f32),
        frequency_penalty: body["frequency_penalty"].as_f64().map(|v| v as f32),
        seed: body["seed"].as_u64().map(|v| v as u32),
        reasoning_effort: parse_reasoning_effort(
            body["reasoning_effort"]
                .as_str()
                .or_else(|| body["reasoning"]["effort"].as_str()),
        ),
        extra_headers: extract_passthrough_headers(&headers),
        extra_body,
        openrouter_body: extract_openrouter_body(&body),
        cache_preference: cache_pref,
    };

    let est_in = estimate_request_tokens(&req);
    let (conc_guard, budget_guard) = match pre_reserved {
        Some((cg, bg, _)) => (cg, bg),
        None => {
            let (est_total, est_cost) = state.router.estimate_preflight_budget(
                &requested_model,
                est_in,
                u64::from(max_out.unwrap_or(1024)),
            )?;
            authorize_inference_request(
                &state,
                &principal,
                &headers,
                Some(&body),
                &requested_model,
                est_total,
                est_cost,
            )?
        }
    };

    let stream_requested = body["stream"].as_bool().unwrap_or(false);
    let request_id = TelemetryHub::fast_prefixed_id("chatcmpl_");
    let created_sec = now_ms() / 1000;

    // Zero-cost instant `mock_response` short-circuit (`good.md` §21) for fast integration testing
    if let Some(mock_text) = body["mock_response"].as_str() {
        let mut guard_req = req.clone();
        quixigate_router::guardrails::apply_request_guardrails(&mut guard_req, &settings)?;
        let out_tokens = ((mock_text.len() / 4) as u64).max(1);
        let zero = NanoUsd(0);
        let _ = budget_guard.commit(zero);
        drop(conc_guard);
        if stream_requested {
            let chunk1 = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_sec,
                "model": requested_model,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant", "content": mock_text },
                    "finish_reason": Value::Null,
                }]
            });
            let chunk2 = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_sec,
                "model": requested_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": est_in,
                    "completion_tokens": out_tokens,
                    "total_tokens": est_in + out_tokens,
                }
            });
            let sse_body = format!("data: {chunk1}\n\ndata: {chunk2}\n\ndata: [DONE]\n\n");
            let mut resp = (
                StatusCode::OK,
                [("content-type", "text/event-stream")],
                sse_body,
            )
                .into_response();
            attach_gateway_headers(
                resp.headers_mut(),
                &request_id,
                "mock",
                &requested_model,
                false,
                zero,
                zero,
            );
            return Ok(resp);
        }
        let resp_json = json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created_sec,
            "model": requested_model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": mock_text },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": est_in,
                "completion_tokens": out_tokens,
                "total_tokens": est_in + out_tokens,
            }
        });
        let mut resp = Json(resp_json).into_response();
        attach_gateway_headers(
            resp.headers_mut(),
            &request_id,
            "mock",
            &requested_model,
            false,
            zero,
            zero,
        );
        return Ok(resp);
    }

    let need_deep_telemetry = capture_payloads || settings.otlp_enabled;
    let oi = if need_deep_telemetry {
        let session_id = headers
            .get("x-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        build_openinference_inputs(&req, session_id)
    } else {
        OpenInferencePayload::default()
    };
    let raw_req_str = capture_payloads.then(|| {
        std::str::from_utf8(&raw_body)
            .map(str::to_owned)
            .unwrap_or_else(|_| body.to_string())
    });
    let started = Instant::now();

    state.telemetry.inc_active();

    if !stream_requested {
        let result = state.router.execute_text(req).await;
        state.telemetry.dec_active();
        drop(conc_guard);

        match result {
            Ok(gen_out) => {
                let finish_reason = if !gen_out.tool_calls.is_empty() {
                    "tool_calls"
                } else {
                    match gen_out.finish_reason.as_deref() {
                        Some("length") => "length",
                        Some("content_filter") => "content_filter",
                        _ => "stop",
                    }
                };
                let tool_calls_json: Vec<Value> = gen_out
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.tool.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool.name,
                                "arguments": tc.input.to_string(),
                            }
                        })
                    })
                    .collect();

                let mut msg_obj = Map::new();
                msg_obj.insert("role".into(), json!("assistant"));
                if gen_out.text.is_empty() && !tool_calls_json.is_empty() {
                    msg_obj.insert("content".into(), Value::Null);
                } else {
                    msg_obj.insert("content".into(), json!(gen_out.text));
                }
                if let Some(r) = &gen_out.reasoning
                    && !r.is_empty()
                {
                    msg_obj.insert("reasoning_content".into(), json!(r));
                }
                if !tool_calls_json.is_empty() {
                    msg_obj.insert("tool_calls".into(), Value::Array(tool_calls_json));
                }

                let resp_json = json!({
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created_sec,
                    "model": gen_out.resolved_model,
                    "choices": [{
                        "index": 0,
                        "message": Value::Object(msg_obj),
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": gen_out.input_tokens,
                        "completion_tokens": gen_out.output_tokens,
                        "total_tokens": gen_out.input_tokens + gen_out.output_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": gen_out.cached_input_tokens,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": gen_out.reasoning_tokens,
                        }
                    }
                });

                let oi_tools = if need_deep_telemetry {
                    gen_out
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            (
                                tc.tool.id.clone(),
                                tc.tool.name.clone(),
                                tc.input.to_string(),
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id.clone(),
                    IngressProtocol::ChatCompletions,
                    requested_model,
                    gen_out.resolved_model.clone(),
                    gen_out.provider_id.clone(),
                    gen_out.credential_id.clone(),
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
                    raw_req_str,
                    capture_payloads.then(|| resp_json.to_string()),
                    oi,
                    need_deep_telemetry.then_some(gen_out.text),
                    oi_tools,
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
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::ChatCompletions,
                    requested_model.clone(),
                    requested_model,
                    error_provider_id(&err),
                    None,
                    err.http_status(),
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
                    raw_req_str,
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
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::ChatCompletions,
                    requested_model.clone(),
                    requested_model,
                    error_provider_id(&err),
                    None,
                    err.http_status(),
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
                    raw_req_str,
                    None,
                    oi,
                    None,
                    Vec::new(),
                );
                return Err(ApiError(err));
            }
        };

        let provider_hdr = routed.provider_id.clone();
        let model_hdr = routed.resolved_model.clone();
        let cache_hit = routed.cache_hit;
        let req_id_for_headers = request_id.clone();

        let sse_stream = ChatTranscodeSseStream::new(
            routed,
            state,
            principal,
            conc_guard,
            budget_guard,
            request_id,
            created_sec,
            requested_model,
            started,
            capture_payloads,
            need_deep_telemetry,
            raw_req_str,
            oi,
        );

        let mut resp = Response::new(Body::from_stream(sse_stream));
        *resp.status_mut() = StatusCode::OK;
        let hdrs = resp.headers_mut();
        hdrs.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        hdrs.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        attach_gateway_headers(
            hdrs,
            &req_id_for_headers,
            &provider_hdr,
            &model_hdr,
            cache_hit,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        Ok(resp)
    }
}

// ── Zero-Spawn `BytesMut` Transcoding SSE Stream (`/v1/chat/completions`) ────

const TRANSCODE_BATCH_CAP: usize = 32 * 1024;

struct ChatTranscodeSseStream {
    events: Pin<Box<dyn Stream<Item = RoutedStreamEvent> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
    request_id: String,
    created_sec: i64,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    fallback_attempts: u32,
    cache_hit: bool,
    started: Instant,
    capture_payloads: bool,
    need_deep_telemetry: bool,
    raw_req_str: Option<String>,
    oi: Option<OpenInferencePayload>,
    chunk_prefix: Bytes,
    tool_index_by_id: HashMap<String, usize>,
    next_tool_idx: usize,
    completed_summary: Option<StreamCompletionSummary>,
    stream_err: Option<String>,
    emitted_init: bool,
    eof: bool,
    buf: BytesMut,
}

impl ChatTranscodeSseStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        routed: RoutedStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        request_id: String,
        created_sec: i64,
        requested_model: String,
        started: Instant,
        capture_payloads: bool,
        need_deep_telemetry: bool,
        raw_req_str: Option<String>,
        oi: OpenInferencePayload,
    ) -> Self {
        let mut pfx = BytesMut::with_capacity(160);
        pfx.extend_from_slice(b"data: {\"id\":");
        put_json_escaped_str(&mut pfx, &request_id);
        pfx.extend_from_slice(b",\"object\":\"chat.completion.chunk\",\"created\":");
        put_i64_dec(&mut pfx, created_sec);
        pfx.extend_from_slice(b",\"model\":");
        put_json_escaped_str(&mut pfx, &routed.resolved_model);
        pfx.extend_from_slice(b",\"choices\":[{\"index\":0,\"delta\":{");

        Self {
            events: routed.events,
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
            request_id,
            created_sec,
            requested_model,
            resolved_model: routed.resolved_model,
            provider_id: routed.provider_id,
            credential_id: routed.credential_id,
            fallback_attempts: routed.fallback_attempts,
            cache_hit: routed.cache_hit,
            started,
            capture_payloads,
            need_deep_telemetry,
            raw_req_str,
            oi: Some(oi),
            chunk_prefix: pfx.freeze(),
            tool_index_by_id: HashMap::new(),
            next_tool_idx: 0,
            completed_summary: None,
            stream_err: None,
            emitted_init: false,
            eof: false,
            buf: BytesMut::with_capacity(4096),
        }
    }

    fn write_event(&mut self, ev: RoutedStreamEvent) -> bool {
        match ev {
            RoutedStreamEvent::TextDelta(delta) => {
                if !delta.is_empty() {
                    self.buf.extend_from_slice(&self.chunk_prefix);
                    self.buf.extend_from_slice(b"\"content\":");
                    put_json_escaped_str(&mut self.buf, &delta);
                    self.buf
                        .extend_from_slice(b"},\"finish_reason\":null}]}\n\n");
                }
                false
            }
            RoutedStreamEvent::ReasoningDelta(delta) => {
                if !delta.is_empty() {
                    self.buf.extend_from_slice(&self.chunk_prefix);
                    self.buf.extend_from_slice(b"\"reasoning_content\":");
                    put_json_escaped_str(&mut self.buf, &delta);
                    self.buf
                        .extend_from_slice(b"},\"finish_reason\":null}]}\n\n");
                }
                false
            }
            RoutedStreamEvent::ToolCallStart { id, name } => {
                let idx = *self.tool_index_by_id.entry(id.clone()).or_insert_with(|| {
                    let i = self.next_tool_idx;
                    self.next_tool_idx += 1;
                    i
                });
                self.buf.extend_from_slice(&self.chunk_prefix);
                self.buf.extend_from_slice(b"\"tool_calls\":[{\"index\":");
                put_usize_dec(&mut self.buf, idx);
                self.buf.extend_from_slice(b",\"id\":");
                put_json_escaped_str(&mut self.buf, &id);
                self.buf
                    .extend_from_slice(b",\"type\":\"function\",\"function\":{\"name\":");
                put_json_escaped_str(&mut self.buf, &name);
                self.buf
                    .extend_from_slice(b",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
                false
            }
            RoutedStreamEvent::ToolCallDelta { id, delta } => {
                let idx = *self.tool_index_by_id.entry(id).or_insert_with(|| {
                    let i = self.next_tool_idx;
                    self.next_tool_idx += 1;
                    i
                });
                self.buf.extend_from_slice(&self.chunk_prefix);
                self.buf.extend_from_slice(b"\"tool_calls\":[{\"index\":");
                put_usize_dec(&mut self.buf, idx);
                self.buf.extend_from_slice(b",\"function\":{\"arguments\":");
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf
                    .extend_from_slice(b"}}]},\"finish_reason\":null}]}\n\n");
                false
            }
            RoutedStreamEvent::ToolCallReady(tc) => {
                if !self.tool_index_by_id.contains_key(&tc.tool.id) {
                    let idx = self.next_tool_idx;
                    self.next_tool_idx += 1;
                    self.tool_index_by_id.insert(tc.tool.id.clone(), idx);
                    let args_str = tc.input.to_string();
                    self.buf.extend_from_slice(&self.chunk_prefix);
                    self.buf.extend_from_slice(b"\"tool_calls\":[{\"index\":");
                    put_usize_dec(&mut self.buf, idx);
                    self.buf.extend_from_slice(b",\"id\":");
                    put_json_escaped_str(&mut self.buf, &tc.tool.id);
                    self.buf
                        .extend_from_slice(b",\"type\":\"function\",\"function\":{\"name\":");
                    put_json_escaped_str(&mut self.buf, &tc.tool.name);
                    self.buf.extend_from_slice(b",\"arguments\":");
                    put_json_escaped_str(&mut self.buf, &args_str);
                    self.buf
                        .extend_from_slice(b"}}]},\"finish_reason\":null}]}\n\n");
                }
                false
            }
            RoutedStreamEvent::Finish(summary) => {
                let finish_reason = if !summary.tool_calls.is_empty() {
                    "tool_calls"
                } else {
                    match summary.finish_reason.as_deref() {
                        Some("length") => "length",
                        Some("content_filter") => "content_filter",
                        _ => "stop",
                    }
                };
                self.buf.extend_from_slice(b"data: {\"id\":");
                put_json_escaped_str(&mut self.buf, &self.request_id);
                self.buf
                    .extend_from_slice(b",\"object\":\"chat.completion.chunk\",\"created\":");
                put_i64_dec(&mut self.buf, self.created_sec);
                self.buf.extend_from_slice(b",\"model\":");
                put_json_escaped_str(&mut self.buf, &summary.resolved_model);
                self.buf.extend_from_slice(
                    b",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":",
                );
                put_json_escaped_str(&mut self.buf, finish_reason);
                self.buf
                    .extend_from_slice(b"}],\"usage\":{\"prompt_tokens\":");
                put_u64_dec(&mut self.buf, summary.input_tokens);
                self.buf.extend_from_slice(b",\"completion_tokens\":");
                put_u64_dec(&mut self.buf, summary.output_tokens);
                self.buf.extend_from_slice(b",\"total_tokens\":");
                put_u64_dec(&mut self.buf, summary.input_tokens + summary.output_tokens);
                self.buf
                    .extend_from_slice(b",\"prompt_tokens_details\":{\"cached_tokens\":");
                put_u64_dec(&mut self.buf, summary.cached_input_tokens);
                self.buf
                    .extend_from_slice(b"},\"completion_tokens_details\":{\"reasoning_tokens\":");
                put_u64_dec(&mut self.buf, summary.reasoning_tokens);
                self.buf.extend_from_slice(b"}}}\n\n");
                self.completed_summary = Some(summary);
                false
            }
            RoutedStreamEvent::Error(err_msg) => {
                self.buf
                    .extend_from_slice(b"data: {\"error\":{\"message\":");
                put_json_escaped_str(&mut self.buf, &err_msg);
                self.buf
                    .extend_from_slice(b",\"type\":\"upstream_error\"}}\n\n");
                self.stream_err = Some(err_msg);
                true
            }
            _ => false,
        }
    }
}

impl Stream for ChatTranscodeSseStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.eof {
            return Poll::Ready(None);
        }

        if !this.emitted_init {
            this.emitted_init = true;
            this.buf.extend_from_slice(&this.chunk_prefix);
            this.buf
                .extend_from_slice(b"\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n");
        }

        loop {
            match this.events.as_mut().poll_next(cx) {
                Poll::Ready(Some(ev)) => {
                    let stop_now = this.write_event(ev);
                    if stop_now {
                        this.buf.extend_from_slice(b"data: [DONE]\n\n");
                        this.eof = true;
                        return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                    }
                    if this.buf.len() >= TRANSCODE_BATCH_CAP {
                        return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                    }
                }
                Poll::Ready(None) => {
                    this.buf.extend_from_slice(b"data: [DONE]\n\n");
                    this.eof = true;
                    return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                }
                Poll::Pending => {
                    if !this.buf.is_empty() {
                        return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

impl Drop for ChatTranscodeSseStream {
    fn drop(&mut self) {
        self.state.telemetry.dec_active();
        drop(self.conc_guard.take());
        let (Some(budget_guard), Some(oi)) = (self.budget_guard.take(), self.oi.take()) else {
            return;
        };

        if let Some(sum) = self.completed_summary.take() {
            let oi_tools = if self.need_deep_telemetry {
                sum.tool_calls
                    .iter()
                    .map(|tc| {
                        (
                            tc.tool.id.clone(),
                            tc.tool.name.clone(),
                            tc.input.to_string(),
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let synth_resp = self.capture_payloads.then(|| {
                json!({
                    "content": sum.text,
                    "reasoning": sum.reasoning,
                })
                .to_string()
            });
            finalize_request_accounting(
                &self.state,
                &self.principal,
                budget_guard,
                std::mem::take(&mut self.request_id),
                IngressProtocol::ChatCompletions,
                std::mem::take(&mut self.requested_model),
                sum.resolved_model,
                sum.provider_id,
                sum.credential_id,
                200,
                true,
                sum.fallback_attempts,
                self.cache_hit,
                sum.ttft_ms,
                sum.duration_ms,
                sum.input_tokens,
                sum.cached_input_tokens,
                sum.reasoning_tokens,
                sum.output_tokens,
                sum.cost_nanos,
                sum.api_equiv_nanos,
                None,
                self.raw_req_str.take(),
                synth_resp,
                oi,
                self.need_deep_telemetry.then_some(sum.text),
                oi_tools,
            );
        } else {
            finalize_request_accounting(
                &self.state,
                &self.principal,
                budget_guard,
                std::mem::take(&mut self.request_id),
                IngressProtocol::ChatCompletions,
                std::mem::take(&mut self.requested_model),
                std::mem::take(&mut self.resolved_model),
                std::mem::take(&mut self.provider_id),
                self.credential_id.take(),
                if self.stream_err.is_some() { 502 } else { 200 },
                true,
                self.fallback_attempts,
                self.cache_hit,
                None,
                self.started.elapsed().as_millis() as u64,
                0,
                0,
                0,
                0,
                NanoUsd::ZERO,
                NanoUsd::ZERO,
                self.stream_err.take(),
                self.raw_req_str.take(),
                None,
                oi,
                None,
                Vec::new(),
            );
        }
    }
}
