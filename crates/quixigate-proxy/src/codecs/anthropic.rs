//! `POST /v1/messages` & `POST /v1/messages/count_tokens` —
//! Two-Lane Anthropic Messages JSON & SSE Codec (native Claude Code compatibility):
//! 1. **Zero-DOM Cut-Through Lane**: Borrowed `AnthropicEnvelopePeek` + raw-`Bytes` Anthropic splicer
//!    and `CutThroughSseStream` (`0` `tokio::spawn`, `0` `mpsc`, `0` per-token JSON DOM) when routing
//!    `/v1/messages` directly to Anthropic (`anthropic` / `anthropic-token`) or MiniMax (`minimax-portal`).
//! 2. **Zero-Spawn Polyglot Transcoding Lane**: Full `CanonicalRequest` normalization and direct
//!    `BytesMut` `AnthropicTranscodeSseStream` state machine (`0` `tokio::spawn`, `0` `mpsc`, `0` `json!`)
//!    for OpenRouter, OpenAI, Codex, Gemini, and local runtimes.

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use aisdk::core::language_model::{LanguageModelResponseContentType, ReasoningEffort};
use aisdk::core::tools::ToolDetails;
use aisdk::core::{AssistantMessage, Message, ToolCallInfo, ToolResultInfo, UserMessage};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::BytesMut;
use futures::Stream;
use quixigate_core::{IngressProtocol, NanoUsd, QuixigateError, SecretScannerMode};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::{
    CanonicalRequest, ClientToolSpec, CutThroughJsonOutcome, CutThroughStreamOutcome,
    CutThroughWireFormat, RoutedStream, RoutedStreamEvent, StreamCompletionSummary,
};
use quixigate_telemetry::{OpenInferencePayload, TelemetryHub};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

#[derive(Deserialize)]
struct AnthropicEnvelopePeek<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    messages: Option<MessagesArrayMarker>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(borrow, default)]
    tags: Option<TagsPeek<'a>>,
    #[serde(borrow, default)]
    metadata: Option<MetadataPeek<'a>>,
}

fn parse_anthropic_canonical(
    body: &Value,
    headers: &HeaderMap,
) -> quixigate_core::Result<CanonicalRequest> {
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            QuixigateError::Validation("missing 'model' in Anthropic request".to_owned())
        })?
        .to_owned();

    let system = match &body["system"] {
        Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        Value::Array(blocks) => {
            let joined: Vec<String> = blocks
                .iter()
                .filter_map(|b| b["text"].as_str().map(str::to_owned))
                .collect();
            (!joined.is_empty()).then(|| joined.join("\n\n"))
        }
        _ => None,
    };

    let raw_messages = body["messages"]
        .as_array()
        .ok_or_else(|| QuixigateError::Validation("'messages' must be an array".to_owned()))?;

    let mut messages = Vec::new();
    let mut tool_names_by_id: HashMap<String, String> = HashMap::new();

    for msg in raw_messages {
        let role = msg["role"].as_str().unwrap_or("user");
        match (&msg["content"], role) {
            (Value::String(s), "assistant") => {
                messages.push(Message::Assistant(AssistantMessage::new(
                    LanguageModelResponseContentType::Text(s.clone()),
                    None,
                )));
            }
            (Value::String(s), _) => {
                messages.push(Message::User(UserMessage::new(s.clone())));
            }
            (Value::Array(blocks), "assistant") => {
                let mut text_buf = Vec::new();
                for b in blocks {
                    match b["type"].as_str() {
                        Some("text") => {
                            if let Some(t) = b["text"].as_str() {
                                text_buf.push(t.to_owned());
                            }
                        }
                        Some("tool_use") => {
                            if !text_buf.is_empty() {
                                messages.push(Message::Assistant(AssistantMessage::new(
                                    LanguageModelResponseContentType::Text(text_buf.join("\n")),
                                    None,
                                )));
                                text_buf.clear();
                            }
                            let id = b["id"].as_str().unwrap_or("toolu_1").to_owned();
                            let name = b["name"].as_str().unwrap_or("tool").to_owned();
                            let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                            tool_names_by_id.insert(id.clone(), name.clone());
                            messages.push(Message::Assistant(AssistantMessage::new(
                                LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                                    tool: ToolDetails { id, name },
                                    input,
                                    extensions: Default::default(),
                                }),
                                None,
                            )));
                        }
                        _ => {}
                    }
                }
                if !text_buf.is_empty() {
                    messages.push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::Text(text_buf.join("\n")),
                        None,
                    )));
                }
            }
            (Value::Array(blocks), _) => {
                let mut user_text = Vec::new();
                for b in blocks {
                    match b["type"].as_str() {
                        Some("tool_result") => {
                            if !user_text.is_empty() {
                                messages
                                    .push(Message::User(UserMessage::new(user_text.join("\n"))));
                                user_text.clear();
                            }
                            let id = b["tool_use_id"].as_str().unwrap_or("toolu_1").to_owned();
                            let name = tool_names_by_id
                                .get(&id)
                                .cloned()
                                .unwrap_or_else(|| "tool".to_owned());
                            let content_str = extract_content_text(&b["content"]);
                            messages.push(Message::Tool(ToolResultInfo {
                                tool: ToolDetails { id, name },
                                output: Ok(Value::String(content_str)),
                            }));
                        }
                        Some("image") => {
                            let media = b["source"]["media_type"].as_str().unwrap_or("image/*");
                            user_text.push(format!("[image: {media}]"));
                        }
                        _ => {
                            if let Some(t) = b["text"].as_str() {
                                user_text.push(t.to_owned());
                            }
                        }
                    }
                }
                if !user_text.is_empty() {
                    messages.push(Message::User(UserMessage::new(user_text.join("\n"))));
                }
            }
            _ => {}
        }
    }

    let mut client_tools = Vec::new();
    if let Some(tools) = body["tools"].as_array() {
        for t in tools {
            if let Some(name) = t["name"].as_str() {
                client_tools.push(ClientToolSpec {
                    name: name.to_owned(),
                    description: t["description"].as_str().unwrap_or_default().to_owned(),
                    parameters: t
                        .get("input_schema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                });
            }
        }
    }

    let reasoning_effort = if body["thinking"]["type"].as_str() == Some("enabled") {
        let budget = body["thinking"]["budget_tokens"].as_u64().unwrap_or(4096);
        Some(if budget >= 16_000 {
            ReasoningEffort::High
        } else if budget >= 4_000 {
            ReasoningEffort::Medium
        } else {
            ReasoningEffort::Low
        })
    } else {
        None
    };

    let stop_sequences = body["stop_sequences"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect()
    });

    Ok(CanonicalRequest {
        requested_model,
        system,
        messages,
        client_tools,
        schema: None,
        temperature: scale_temperature(body["temperature"].as_f64()),
        top_p: scale_top_p(body["top_p"].as_f64()),
        top_k: body["top_k"].as_u64().map(|v| v as u32),
        max_output_tokens: body["max_tokens"].as_u64().map(|v| v as u32),
        stop_sequences,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        reasoning_effort,
        extra_headers: extract_passthrough_headers(headers),
        extra_body: Map::new(),
        openrouter_body: extract_openrouter_body(body),
        cache_preference: parse_cache_preference(headers),
    })
}

pub async fn handle_count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let _principal = authenticate_inference(&state, &headers, None)?;
    let req = parse_anthropic_canonical(&body, &headers)?;
    let input_tokens = estimate_request_tokens(&req);
    Ok(Json(json!({ "input_tokens": input_tokens })))
}

pub async fn handle_messages(
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
    let need_deep_telemetry = capture_payloads || settings.otlp_enabled;

    let mut pre_reserved: Option<(ConcurrencyGuard, BudgetGuard)> = None;

    // ── Fast Lane Gate: Zero-DOM `/v1/messages` -> Anthropic Cut-Through ─────
    {
        let cache_pref = parse_cache_preference(&headers);
        let can_attempt_peek = settings.enforced_params.is_empty()
            && settings.secret_scanner_mode == SecretScannerMode::Off
            && settings.banned_keywords.is_empty()
            && !capture_payloads
            && !settings.otlp_enabled
            && settings.mcp_server_urls.is_empty()
            && !state.router.has_mcp_urls();

        if can_attempt_peek
            && let Ok(peek) = serde_json::from_slice::<AnthropicEnvelopePeek<'_>>(&raw_body)
        {
            let requested_model_str = peek
                .model
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    QuixigateError::Validation("missing 'model' in Anthropic request".to_owned())
                })?;
            if peek.messages.is_none() {
                return Err(ApiError(QuixigateError::Validation(
                    "'messages' must be an array".to_owned(),
                )));
            }

            let cache_wanted = cache_pref.unwrap_or(settings.cache_enabled_default);
            if !cache_wanted {
                let est_in = ((raw_body.len() / 4) as u64).max(1) + 8;
                let max_out = peek.max_tokens.unwrap_or(4096);
                let plan =
                    state
                        .router
                        .prepare_cut_through_plan(requested_model_str, est_in, max_out)?;

                if !plan.spec().online
                    && plan.candidates.first().is_some_and(|c| {
                        matches!(
                            c.provider_id.as_str(),
                            "anthropic" | "anthropic-token" | "minimax-portal"
                        )
                    })
                {
                    let req_tags =
                        collect_peek_tags(&headers, peek.metadata.as_ref(), peek.tags.as_ref());
                    let end_user_id = peek_end_user(
                        &headers,
                        peek.metadata.as_ref().and_then(|m| m.user_id.as_deref()),
                        None,
                    );
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
                    let passthrough_headers = extract_passthrough_headers_vec(&headers);
                    let msg_id = TelemetryHub::fast_prefixed_id("msg_");

                    if stream_requested {
                        state.telemetry.inc_active();
                        let started = Instant::now();
                        match state
                            .router
                            .execute_anthropic_cut_through_stream(
                                plan,
                                &raw_body,
                                &passthrough_headers,
                            )
                            .await
                        {
                            Ok(CutThroughStreamOutcome::Stream(active)) => {
                                let provider_hdr = active.provider_id.clone();
                                let model_hdr = active.resolved_model.clone();
                                let msg_id_hdr = msg_id.clone();
                                let sse_stream = CutThroughSseStream::new(
                                    active,
                                    state,
                                    principal,
                                    conc_guard,
                                    budget_guard,
                                    CutThroughWireFormat::AnthropicMessages,
                                    IngressProtocol::AnthropicMessages,
                                    msg_id,
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
                                    &msg_id_hdr,
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
                                pre_reserved = Some((conc_guard, budget_guard));
                            }
                            Err(err) => {
                                state.telemetry.dec_active();
                                drop(conc_guard);
                                finalize_cut_through_accounting(
                                    &state,
                                    &principal,
                                    budget_guard,
                                    IngressProtocol::AnthropicMessages,
                                    msg_id,
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
                            .execute_anthropic_cut_through_json(
                                plan,
                                &raw_body,
                                est_in,
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
                                    &msg_id,
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
                                    IngressProtocol::AnthropicMessages,
                                    msg_id,
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
                                pre_reserved = Some((conc_guard, budget_guard));
                            }
                            Err(err) => {
                                state.telemetry.dec_active();
                                drop(conc_guard);
                                finalize_cut_through_accounting(
                                    &state,
                                    &principal,
                                    budget_guard,
                                    IngressProtocol::AnthropicMessages,
                                    msg_id,
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
    }

    // ── Transcoding / Feature-Rich Lane ──────────────────────────────────────
    let body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| QuixigateError::Validation(format!("invalid JSON body: {e}")))?;
    let req = parse_anthropic_canonical(&body, &headers)?;
    let requested_model = req.requested_model.clone();
    let est_in = estimate_request_tokens(&req);

    let (conc_guard, budget_guard) = match pre_reserved {
        Some(guards) => guards,
        None => {
            let max_out = u64::from(req.max_output_tokens.unwrap_or(4096));
            let (est_total, est_cost) =
                state
                    .router
                    .estimate_preflight_budget(&requested_model, est_in, max_out)?;
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
    let msg_id = TelemetryHub::fast_prefixed_id("msg_");
    let (oi, raw_req) = prepare_telemetry_capture(&state, &principal, &req, &body, None);
    let started = Instant::now();

    state.telemetry.inc_active();

    if !stream_requested {
        let res = state.router.execute_text(req).await;
        state.telemetry.dec_active();
        drop(conc_guard);

        match res {
            Ok(gen_out) => {
                let mut content_blocks = Vec::new();
                if let Some(thinking) = &gen_out.reasoning
                    && !thinking.is_empty()
                {
                    content_blocks.push(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": "quixigate-thinking"
                    }));
                }
                if !gen_out.text.is_empty() || gen_out.tool_calls.is_empty() {
                    content_blocks.push(json!({
                        "type": "text",
                        "text": gen_out.text,
                    }));
                }
                for tc in &gen_out.tool_calls {
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.tool.id,
                        "name": tc.tool.name,
                        "input": tc.input,
                    }));
                }

                let stop_reason = if !gen_out.tool_calls.is_empty() {
                    "tool_use"
                } else if gen_out.finish_reason.as_deref() == Some("length") {
                    "max_tokens"
                } else {
                    "end_turn"
                };

                let resp_json = json!({
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": gen_out.resolved_model,
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": gen_out.input_tokens,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": gen_out.cached_input_tokens,
                        "output_tokens": gen_out.output_tokens,
                    }
                });

                let raw_resp_str = capture_payloads.then(|| resp_json.to_string());
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
                    msg_id.clone(),
                    IngressProtocol::AnthropicMessages,
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
                    raw_req,
                    raw_resp_str,
                    oi,
                    need_deep_telemetry.then_some(gen_out.text),
                    oi_tools,
                );

                let mut resp = Json(resp_json).into_response();
                attach_gateway_headers(
                    resp.headers_mut(),
                    &msg_id,
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
                    msg_id,
                    IngressProtocol::AnthropicMessages,
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
                finalize_request_accounting(
                    &state,
                    &principal,
                    budget_guard,
                    msg_id,
                    IngressProtocol::AnthropicMessages,
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
                    raw_req,
                    None,
                    oi,
                    None,
                    Vec::new(),
                );
                return Err(ApiError(err));
            }
        };

        let resolved_hdr = routed.resolved_model.clone();
        let provider_hdr = routed.provider_id.clone();
        let cache_hit = routed.cache_hit;
        let hdr_msg_id = msg_id.clone();

        let sse_stream = AnthropicTranscodeSseStream::new(
            routed,
            state,
            principal,
            conc_guard,
            budget_guard,
            msg_id,
            requested_model,
            est_in,
            started,
            need_deep_telemetry,
            raw_req,
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
            &hdr_msg_id,
            &provider_hdr,
            &resolved_hdr,
            cache_hit,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        Ok(resp)
    }
}

// ── Zero-Spawn `AnthropicTranscodeSseStream` ─────────────────────────────────

const TRANSCODE_BATCH_CAP: usize = 32 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AnthropicOpenBlock {
    Thinking,
    Text,
    ToolUse,
}

struct AnthropicTranscodeSseStream {
    events: Pin<Box<dyn Stream<Item = RoutedStreamEvent> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
    msg_id: String,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    fallback_attempts: u32,
    cache_hit: bool,
    started: Instant,
    need_deep_telemetry: bool,
    raw_req: Option<String>,
    oi: Option<OpenInferencePayload>,
    block_idx: usize,
    open_block: Option<AnthropicOpenBlock>,
    tool_idx_by_id: HashMap<String, usize>,
    buf: BytesMut,
    completed_summary: Option<StreamCompletionSummary>,
    stream_err: Option<String>,
    eof: bool,
}

impl AnthropicTranscodeSseStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        routed: RoutedStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        msg_id: String,
        requested_model: String,
        est_in: u64,
        started: Instant,
        need_deep_telemetry: bool,
        raw_req: Option<String>,
        oi: OpenInferencePayload,
    ) -> Self {
        let mut buf = BytesMut::with_capacity(1024);
        buf.extend_from_slice(
            b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":",
        );
        put_json_escaped_str(&mut buf, &msg_id);
        buf.extend_from_slice(b",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
        put_json_escaped_str(&mut buf, &routed.resolved_model);
        buf.extend_from_slice(
            b",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":",
        );
        put_u64_dec(&mut buf, est_in);
        buf.extend_from_slice(
            b",\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":1}}}\n\n",
        );

        Self {
            events: routed.events,
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
            msg_id,
            requested_model,
            resolved_model: routed.resolved_model,
            provider_id: routed.provider_id,
            credential_id: routed.credential_id,
            fallback_attempts: routed.fallback_attempts,
            cache_hit: routed.cache_hit,
            started,
            need_deep_telemetry,
            raw_req,
            oi: Some(oi),
            block_idx: 0,
            open_block: None,
            tool_idx_by_id: HashMap::new(),
            buf,
            completed_summary: None,
            stream_err: None,
            eof: false,
        }
    }

    #[inline]
    fn write_close_open(&mut self) {
        self.buf.extend_from_slice(
            b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":",
        );
        put_usize_dec(&mut self.buf, self.block_idx);
        self.buf.extend_from_slice(b"}\n\n");
    }

    fn encode_event(&mut self, ev: RoutedStreamEvent) {
        match ev {
            RoutedStreamEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                if self.open_block != Some(AnthropicOpenBlock::Thinking) {
                    if self.open_block.is_some() {
                        self.write_close_open();
                        self.block_idx += 1;
                    }
                    self.buf.extend_from_slice(
                        b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":",
                    );
                    put_usize_dec(&mut self.buf, self.block_idx);
                    self.buf.extend_from_slice(
                        b",\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
                    );
                    self.open_block = Some(AnthropicOpenBlock::Thinking);
                }
                self.buf.extend_from_slice(
                    b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":",
                );
                put_usize_dec(&mut self.buf, self.block_idx);
                self.buf
                    .extend_from_slice(b",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":");
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}}\n\n");
            }
            RoutedStreamEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                if self.open_block != Some(AnthropicOpenBlock::Text) {
                    if self.open_block.is_some() {
                        self.write_close_open();
                        self.block_idx += 1;
                    }
                    self.buf.extend_from_slice(
                        b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":",
                    );
                    put_usize_dec(&mut self.buf, self.block_idx);
                    self.buf.extend_from_slice(
                        b",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                    );
                    self.open_block = Some(AnthropicOpenBlock::Text);
                }
                self.buf.extend_from_slice(
                    b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":",
                );
                put_usize_dec(&mut self.buf, self.block_idx);
                self.buf
                    .extend_from_slice(b",\"delta\":{\"type\":\"text_delta\",\"text\":");
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}}\n\n");
            }
            RoutedStreamEvent::ToolCallStart { id, name } => {
                if self.open_block.is_some() {
                    self.write_close_open();
                    self.block_idx += 1;
                }
                self.tool_idx_by_id.insert(id.clone(), self.block_idx);
                self.open_block = Some(AnthropicOpenBlock::ToolUse);
                self.buf.extend_from_slice(
                    b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":",
                );
                put_usize_dec(&mut self.buf, self.block_idx);
                self.buf
                    .extend_from_slice(b",\"content_block\":{\"type\":\"tool_use\",\"id\":");
                put_json_escaped_str(&mut self.buf, &id);
                self.buf.extend_from_slice(b",\"name\":");
                put_json_escaped_str(&mut self.buf, &name);
                self.buf.extend_from_slice(b",\"input\":{}}}\n\n");
            }
            RoutedStreamEvent::ToolCallDelta { id, delta } => {
                let idx = self
                    .tool_idx_by_id
                    .get(&id)
                    .copied()
                    .unwrap_or(self.block_idx);
                self.buf.extend_from_slice(
                    b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":",
                );
                put_usize_dec(&mut self.buf, idx);
                self.buf.extend_from_slice(
                    b",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                );
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}}\n\n");
            }
            RoutedStreamEvent::ToolCallReady(tc) => {
                if !self.tool_idx_by_id.contains_key(&tc.tool.id) {
                    if self.open_block.is_some() {
                        self.write_close_open();
                        self.block_idx += 1;
                    }
                    self.tool_idx_by_id
                        .insert(tc.tool.id.clone(), self.block_idx);
                    self.open_block = Some(AnthropicOpenBlock::ToolUse);
                    self.buf.extend_from_slice(
                        b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":",
                    );
                    put_usize_dec(&mut self.buf, self.block_idx);
                    self.buf
                        .extend_from_slice(b",\"content_block\":{\"type\":\"tool_use\",\"id\":");
                    put_json_escaped_str(&mut self.buf, &tc.tool.id);
                    self.buf.extend_from_slice(b",\"name\":");
                    put_json_escaped_str(&mut self.buf, &tc.tool.name);
                    self.buf.extend_from_slice(b",\"input\":{}}}\n\n");

                    let partial = tc.input.to_string();
                    self.buf.extend_from_slice(
                        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":",
                    );
                    put_usize_dec(&mut self.buf, self.block_idx);
                    self.buf.extend_from_slice(
                        b",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                    );
                    put_json_escaped_str(&mut self.buf, &partial);
                    self.buf.extend_from_slice(b"}}\n\n");
                }
            }
            RoutedStreamEvent::Finish(sum) => {
                if self.open_block.is_some() {
                    self.write_close_open();
                    self.open_block = None;
                }
                let stop_reason = if !sum.tool_calls.is_empty() {
                    b"\"tool_use\"".as_slice()
                } else {
                    b"\"end_turn\"".as_slice()
                };
                self.buf.extend_from_slice(
                    b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":",
                );
                self.buf.extend_from_slice(stop_reason);
                self.buf
                    .extend_from_slice(b",\"stop_sequence\":null},\"usage\":{\"input_tokens\":");
                put_u64_dec(&mut self.buf, sum.input_tokens);
                self.buf.extend_from_slice(b",\"cache_read_input_tokens\":");
                put_u64_dec(&mut self.buf, sum.cached_input_tokens);
                self.buf.extend_from_slice(b",\"output_tokens\":");
                put_u64_dec(&mut self.buf, sum.output_tokens);
                self.buf.extend_from_slice(
                    b"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                );
                self.completed_summary = Some(sum);
                self.eof = true;
            }
            RoutedStreamEvent::Error(err_msg) => {
                self.buf.extend_from_slice(
                    b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":",
                );
                put_json_escaped_str(&mut self.buf, &err_msg);
                self.buf.extend_from_slice(b"}}\n\n");
                self.stream_err = Some(err_msg);
                self.eof = true;
            }
            _ => {}
        }
    }
}

impl Stream for AnthropicTranscodeSseStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.eof {
                if !this.buf.is_empty() {
                    return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                }
                return Poll::Ready(None);
            }
            match this.events.as_mut().poll_next(cx) {
                Poll::Ready(Some(ev)) => {
                    this.encode_event(ev);
                    if this.buf.len() >= TRANSCODE_BATCH_CAP {
                        return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                    }
                }
                Poll::Ready(None) => {
                    this.eof = true;
                    if !this.buf.is_empty() {
                        return Poll::Ready(Some(Ok(this.buf.split().freeze())));
                    }
                    return Poll::Ready(None);
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

impl Drop for AnthropicTranscodeSseStream {
    fn drop(&mut self) {
        self.state.telemetry.dec_active();
        drop(self.conc_guard.take());
        let Some(budget_guard) = self.budget_guard.take() else {
            return;
        };
        let oi = self.oi.take().unwrap_or_default();
        let raw_req = self.raw_req.take();

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
            finalize_request_accounting(
                &self.state,
                &self.principal,
                budget_guard,
                std::mem::take(&mut self.msg_id),
                IngressProtocol::AnthropicMessages,
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
                raw_req,
                None,
                oi,
                self.need_deep_telemetry.then_some(sum.text),
                oi_tools,
            );
        } else {
            finalize_request_accounting(
                &self.state,
                &self.principal,
                budget_guard,
                std::mem::take(&mut self.msg_id),
                IngressProtocol::AnthropicMessages,
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
                raw_req,
                None,
                oi,
                None,
                Vec::new(),
            );
        }
    }
}
