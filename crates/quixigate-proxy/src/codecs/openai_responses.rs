//! `POST /v1/responses` (JSON + SSE) & `GET /v1/responses` (WebSocket mode) —
//! Two-Lane OpenAI Responses API Codec compatible with OpenAI SDKs and Codex CLI:
//! 1. **Zero-DOM Cut-Through Lane**: Borrowed `ResponsesEnvelopePeek` + raw-`Bytes` Codex splicer
//!    and `CutThroughSseStream` (`0` `tokio::spawn`, `0` `mpsc`, `0` per-token JSON DOM) when routing
//!    streaming `/v1/responses` directly to ChatGPT Codex (`/responses`).
//! 2. **Zero-Spawn Polyglot Transcoding Lane**: Full `CanonicalRequest` normalization and direct
//!    `BytesMut` `ResponsesTranscodeSseStream` state machine (`0` `tokio::spawn`, `0` `mpsc`, `0` `json!`)
//!    for OpenRouter, Anthropic, Gemini, local runtimes, and WebSocket `/v1/responses`.

#[cfg(feature = "subscription-oauth")]
use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use aisdk::core::language_model::LanguageModelResponseContentType;
use aisdk::core::tools::ToolDetails;
use aisdk::core::{
    AssistantMessage, Message, SystemMessage, ToolCallInfo, ToolResultInfo, UserMessage,
};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::BytesMut;
use futures::{Stream, StreamExt};
#[cfg(feature = "subscription-oauth")]
use quixigate_core::SecretScannerMode;
use quixigate_core::{IngressProtocol, NanoUsd, QuixigateError, now_ms};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::{
    CanonicalRequest, RoutedStream, RoutedStreamEvent, StreamCompletionSummary,
};
#[cfg(feature = "subscription-oauth")]
use quixigate_router::{CutThroughStreamOutcome, CutThroughWireFormat};
use quixigate_telemetry::{OpenInferencePayload, TelemetryHub};
use serde::Deserialize;
#[cfg(feature = "subscription-oauth")]
use serde::de::IgnoredAny;
use serde_json::{Map, Value, json};

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

#[derive(Debug, Default, Deserialize)]
pub struct WsQuery {
    pub key: Option<String>,
    pub model: Option<String>,
}

#[cfg(feature = "subscription-oauth")]
#[derive(Deserialize)]
struct ResponsesEnvelopePeek<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    input: Option<IgnoredAny>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(borrow, default)]
    user: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    end_user_id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    tags: Option<TagsPeek<'a>>,
    #[serde(borrow, default)]
    metadata: Option<MetadataPeek<'a>>,
}

fn parse_responses_canonical(
    body: &Value,
    headers: &HeaderMap,
) -> quixigate_core::Result<CanonicalRequest> {
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            QuixigateError::Validation("missing 'model' in Responses request".to_owned())
        })?
        .to_owned();

    let mut system_parts = Vec::new();
    if let Some(instr) = body["instructions"].as_str()
        && !instr.trim().is_empty()
    {
        system_parts.push(instr.to_owned());
    }

    let mut messages = Vec::new();
    let mut call_names: HashMap<String, String> = HashMap::new();

    match &body["input"] {
        Value::String(s) => {
            messages.push(Message::User(UserMessage::new(s.clone())));
        }
        Value::Array(items) => {
            for item in items {
                let item_type = item["type"].as_str().unwrap_or("");
                if item_type == "function_call" {
                    let id = item["call_id"]
                        .as_str()
                        .or_else(|| item["id"].as_str())
                        .unwrap_or("call_1")
                        .to_owned();
                    let name = item["name"].as_str().unwrap_or("tool").to_owned();
                    let args = match &item["arguments"] {
                        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                        v if v.is_object() => v.clone(),
                        _ => json!({}),
                    };
                    call_names.insert(id.clone(), name.clone());
                    messages.push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                            tool: ToolDetails { id, name },
                            input: args,
                            extensions: Default::default(),
                        }),
                        None,
                    )));
                    continue;
                }
                if item_type == "function_call_output" {
                    let id = item["call_id"].as_str().unwrap_or("call_1").to_owned();
                    let name = call_names
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| "tool".to_owned());
                    let out_str = extract_content_text(&item["output"]);
                    messages.push(Message::Tool(ToolResultInfo {
                        tool: ToolDetails { id, name },
                        output: Ok(Value::String(out_str)),
                    }));
                    continue;
                }

                let role = item["role"].as_str().unwrap_or("user");
                let text = if item.get("content").is_some() {
                    extract_content_text(&item["content"])
                } else {
                    extract_content_text(item)
                };
                match role {
                    "system" => {
                        if !text.trim().is_empty() {
                            system_parts.push(text.clone());
                            messages.push(Message::System(SystemMessage::new(text)));
                        }
                    }
                    "developer" => {
                        if !text.trim().is_empty() {
                            system_parts.push(text.clone());
                            messages.push(Message::Developer(text));
                        }
                    }
                    "assistant" => {
                        messages.push(Message::Assistant(AssistantMessage::new(
                            LanguageModelResponseContentType::Text(text),
                            None,
                        )));
                    }
                    _ => {
                        messages.push(Message::User(UserMessage::new(text)));
                    }
                }
            }
        }
        _ => {
            if let Some(raw_msgs) = body["messages"].as_array() {
                let (sys, parsed) = parse_openai_messages(raw_msgs);
                if let Some(s) = sys {
                    system_parts.push(s);
                }
                messages.extend(parsed);
            }
        }
    }

    let client_tools = parse_openai_tools(body);
    let schema = match body["text"]["format"]["type"].as_str() {
        Some("json_schema") => body["text"]["format"].get("schema").cloned(),
        Some("json_object") => Some(json!({ "type": "object" })),
        _ => None,
    };

    let max_out = body["max_output_tokens"]
        .as_u64()
        .or_else(|| body["max_tokens"].as_u64())
        .map(|v| v as u32);

    Ok(CanonicalRequest {
        requested_model,
        system: (!system_parts.is_empty()).then(|| system_parts.join("\n\n")),
        messages,
        client_tools,
        schema,
        temperature: scale_temperature(body["temperature"].as_f64()),
        top_p: scale_top_p(body["top_p"].as_f64()),
        top_k: None,
        max_output_tokens: max_out,
        stop_sequences: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        reasoning_effort: parse_reasoning_effort(body["reasoning"]["effort"].as_str()),
        extra_headers: extract_passthrough_headers(headers),
        extra_body: Map::new(),
        openrouter_body: extract_openrouter_body(body),
        cache_preference: parse_cache_preference(headers),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_completed_response_json(
    response_id: &str,
    created_sec: i64,
    resolved_model: &str,
    text: &str,
    reasoning: Option<&str>,
    tool_calls: &[ToolCallInfo],
    input_tokens: u64,
    cached_tokens: u64,
    reasoning_tokens: u64,
    output_tokens: u64,
) -> Value {
    let mut output_items = Vec::new();
    if let Some(r) = reasoning
        && !r.is_empty()
    {
        output_items.push(json!({
            "id": TelemetryHub::fast_prefixed_id("rs_"),
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": r }]
        }));
    }
    if !text.is_empty() || tool_calls.is_empty() {
        output_items.push(json!({
            "id": TelemetryHub::fast_prefixed_id("msg_"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": []
            }]
        }));
    }
    for tc in tool_calls {
        output_items.push(json!({
            "id": TelemetryHub::fast_prefixed_id("fc_"),
            "type": "function_call",
            "status": "completed",
            "call_id": tc.tool.id,
            "name": tc.tool.name,
            "arguments": tc.input.to_string()
        }));
    }

    json!({
        "id": response_id,
        "object": "response",
        "created_at": created_sec,
        "status": "completed",
        "model": resolved_model,
        "output": output_items,
        "output_text": text,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": { "cached_tokens": cached_tokens },
            "output_tokens": output_tokens,
            "output_tokens_details": { "reasoning_tokens": reasoning_tokens },
            "total_tokens": input_tokens + output_tokens
        }
    })
}

pub async fn handle_responses_post(
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

    #[cfg_attr(not(feature = "subscription-oauth"), allow(unused_mut))]
    let mut pre_reserved: Option<(ConcurrencyGuard, BudgetGuard)> = None;

    // ── Fast Lane Gate: Zero-DOM `/v1/responses` -> Codex Cut-Through ────────
    #[cfg(feature = "subscription-oauth")]
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
            && let Ok(peek) = serde_json::from_slice::<ResponsesEnvelopePeek<'_>>(&raw_body)
        {
            let requested_model_str = peek
                .model
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    QuixigateError::Validation("missing 'model' in Responses request".to_owned())
                })?;
            let cache_wanted = cache_pref.unwrap_or(settings.cache_enabled_default);
            if !cache_wanted && peek.stream == Some(true) && peek.input.is_some() {
                let est_in = ((raw_body.len() / 4) as u64).max(1) + 8;
                let max_out = peek.max_output_tokens.or(peek.max_tokens).unwrap_or(1024);
                let plan =
                    state
                        .router
                        .prepare_cut_through_plan(requested_model_str, est_in, max_out)?;

                if !plan.spec().online
                    && plan
                        .candidates
                        .first()
                        .is_some_and(|c| c.provider_id == "openai-codex")
                {
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

                    let passthrough_headers = extract_passthrough_headers_vec(&headers);
                    let response_id = TelemetryHub::fast_prefixed_id("resp_");
                    state.telemetry.inc_active();
                    let started = Instant::now();

                    match state
                        .router
                        .execute_responses_cut_through_stream(
                            plan,
                            &raw_body,
                            &passthrough_headers,
                            &response_id,
                        )
                        .await
                    {
                        Ok(CutThroughStreamOutcome::Stream(active)) => {
                            let provider_hdr = active.provider_id.clone();
                            let model_hdr = active.resolved_model.clone();
                            let resp_id_hdr = response_id.clone();
                            let sse_stream = CutThroughSseStream::new(
                                active,
                                state,
                                principal,
                                conc_guard,
                                budget_guard,
                                CutThroughWireFormat::OpenAiResponses,
                                IngressProtocol::Responses,
                                response_id,
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
                                &resp_id_hdr,
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
                                IngressProtocol::Responses,
                                response_id,
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
                }
            }
        }
    }

    // ── Transcoding / Feature-Rich Lane ──────────────────────────────────────
    let body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| QuixigateError::Validation(format!("invalid JSON body: {e}")))?;
    let req = parse_responses_canonical(&body, &headers)?;
    let requested_model = req.requested_model.clone();

    let (conc_guard, budget_guard) = match pre_reserved {
        Some(guards) => guards,
        None => {
            let max_out = u64::from(req.max_output_tokens.unwrap_or(1024));
            let est_in = estimate_request_tokens(&req);
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
    let response_id = TelemetryHub::fast_prefixed_id("resp_");
    let created_sec = now_ms() / 1000;
    let (oi, raw_req) = prepare_telemetry_capture(&state, &principal, &req, &body, None);
    let started = Instant::now();

    state.telemetry.inc_active();

    if !stream_requested {
        let res = state.router.execute_text(req).await;
        state.telemetry.dec_active();
        drop(conc_guard);

        match res {
            Ok(gen_out) => {
                let resp_json = build_completed_response_json(
                    &response_id,
                    created_sec,
                    &gen_out.resolved_model,
                    &gen_out.text,
                    gen_out.reasoning.as_deref(),
                    &gen_out.tool_calls,
                    gen_out.input_tokens,
                    gen_out.cached_input_tokens,
                    gen_out.reasoning_tokens,
                    gen_out.output_tokens,
                );
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
                    response_id.clone(),
                    IngressProtocol::Responses,
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
                    &response_id,
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
                    response_id,
                    IngressProtocol::Responses,
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
                    response_id,
                    IngressProtocol::Responses,
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
        let header_resp_id = response_id.clone();

        let sse_stream = ResponsesTranscodeSseStream::new(
            routed,
            state,
            principal,
            conc_guard,
            budget_guard,
            response_id,
            created_sec,
            requested_model,
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
            &header_resp_id,
            &provider_hdr,
            &resolved_hdr,
            cache_hit,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        Ok(resp)
    }
}

// ── Zero-Spawn `ResponsesTranscodeSseStream` ─────────────────────────────────

const TRANSCODE_BATCH_CAP: usize = 32 * 1024;

struct ResponsesTranscodeSseStream {
    events: Pin<Box<dyn Stream<Item = RoutedStreamEvent> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
    response_id: String,
    created_sec: i64,
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
    text_delta_prefix: Bytes,
    reasoning_delta_prefix: Bytes,
    buf: BytesMut,
    completed_summary: Option<StreamCompletionSummary>,
    stream_err: Option<String>,
    eof: bool,
}

impl ResponsesTranscodeSseStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        routed: RoutedStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        response_id: String,
        created_sec: i64,
        requested_model: String,
        started: Instant,
        need_deep_telemetry: bool,
        raw_req: Option<String>,
        oi: OpenInferencePayload,
    ) -> Self {
        let msg_item_id = TelemetryHub::fast_prefixed_id("msg_");

        let mut td = BytesMut::with_capacity(144);
        td.extend_from_slice(
            b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":",
        );
        put_json_escaped_str(&mut td, &msg_item_id);
        td.extend_from_slice(b",\"output_index\":0,\"content_index\":0,\"delta\":");
        let text_delta_prefix = td.freeze();

        let mut rd = BytesMut::with_capacity(168);
        rd.extend_from_slice(
            b"event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":",
        );
        put_json_escaped_str(&mut rd, &msg_item_id);
        rd.extend_from_slice(b",\"output_index\":0,\"summary_index\":0,\"delta\":");
        let reasoning_delta_prefix = rd.freeze();

        let mut buf = BytesMut::with_capacity(1024);
        // 1. `response.created`
        buf.extend_from_slice(
            b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":",
        );
        put_json_escaped_str(&mut buf, &response_id);
        buf.extend_from_slice(b",\"object\":\"response\",\"created_at\":");
        put_i64_dec(&mut buf, created_sec);
        buf.extend_from_slice(b",\"status\":\"in_progress\",\"model\":");
        put_json_escaped_str(&mut buf, &routed.resolved_model);
        buf.extend_from_slice(b",\"output\":[]}}\n\n");

        // 2. `response.output_item.added`
        buf.extend_from_slice(
            b"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":",
        );
        put_json_escaped_str(&mut buf, &msg_item_id);
        buf.extend_from_slice(
            b",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}\n\n",
        );

        Self {
            events: routed.events,
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
            response_id,
            created_sec,
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
            text_delta_prefix,
            reasoning_delta_prefix,
            buf,
            completed_summary: None,
            stream_err: None,
            eof: false,
        }
    }

    fn encode_event(&mut self, ev: RoutedStreamEvent) {
        match ev {
            RoutedStreamEvent::TextDelta(delta) => {
                self.buf.extend_from_slice(&self.text_delta_prefix);
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::ReasoningDelta(delta) => {
                self.buf.extend_from_slice(&self.reasoning_delta_prefix);
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::ToolCallDelta { id, delta } => {
                self.buf.extend_from_slice(
                    b"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":",
                );
                put_json_escaped_str(&mut self.buf, &id);
                self.buf
                    .extend_from_slice(b",\"output_index\":1,\"delta\":");
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::Finish(sum) => {
                let full_resp = build_completed_response_json(
                    &self.response_id,
                    self.created_sec,
                    &sum.resolved_model,
                    &sum.text,
                    sum.reasoning.as_deref(),
                    &sum.tool_calls,
                    sum.input_tokens,
                    sum.cached_input_tokens,
                    sum.reasoning_tokens,
                    sum.output_tokens,
                );
                self.buf.extend_from_slice(
                    b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":",
                );
                put_json_value(&mut self.buf, &full_resp);
                self.buf.extend_from_slice(b"}\n\n");
                self.completed_summary = Some(sum);
                self.eof = true;
            }
            RoutedStreamEvent::Error(err_msg) => {
                self.buf.extend_from_slice(
                    b"event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":",
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

impl Stream for ResponsesTranscodeSseStream {
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

impl Drop for ResponsesTranscodeSseStream {
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
                std::mem::take(&mut self.response_id),
                IngressProtocol::Responses,
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
                std::mem::take(&mut self.response_id),
                IngressProtocol::Responses,
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

/// `GET /v1/responses` — OpenAI Realtime/Responses WebSocket mode (`response.create` -> event frames).
pub async fn handle_responses_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<WsQuery>,
    headers: HeaderMap,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, query.key.as_deref())?;
    let default_model = query.model.clone();
    let req_headers = headers.clone();
    Ok(ws.on_upgrade(move |socket| {
        run_responses_websocket(socket, state, principal, req_headers, default_model)
    }))
}

async fn run_responses_websocket(
    mut socket: WebSocket,
    state: AppState,
    principal: AuthenticatedPrincipal,
    headers: HeaderMap,
    default_model: Option<String>,
) {
    while let Some(Ok(frame)) = socket.next().await {
        let text = match frame {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Close(_) => break,
            _ => continue,
        };
        let Ok(mut payload) = serde_json::from_str::<Value>(&text) else {
            let _ = socket
                .send(WsMessage::Text(
                    json!({ "type": "error", "error": { "message": "invalid JSON frame" } })
                        .to_string()
                        .into(),
                ))
                .await;
            continue;
        };

        // Support both `{"type": "response.create", "response": {...}}` and top-level `{"type": "response.create", "model": ..., "input": ...}`
        let mut inner = if payload.get("response").is_some_and(Value::is_object) {
            payload["response"].take()
        } else {
            payload
        };
        if inner.get("model").is_none()
            && let Some(dm) = &default_model
        {
            inner["model"] = json!(dm);
        }

        let req = match parse_responses_canonical(&inner, &headers) {
            Ok(r) => r,
            Err(err) => {
                let _ = socket
                    .send(WsMessage::Text(
                        json!({ "type": "error", "error": { "message": err.to_string() } })
                            .to_string()
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        let requested_model = req.requested_model.clone();
        let est_in = estimate_request_tokens(&req);
        let Ok((est_total, est_cost)) = state.router.estimate_preflight_budget(
            &requested_model,
            est_in,
            u64::from(req.max_output_tokens.unwrap_or(1024)),
        ) else {
            continue;
        };
        let (_conc_guard, budget_guard) = match authorize_inference_request(
            &state,
            &principal,
            &headers,
            Some(&inner),
            &requested_model,
            est_total,
            est_cost,
        ) {
            Ok(g) => g,
            Err(err) => {
                let _ = socket
                    .send(WsMessage::Text(
                        json!({ "type": "error", "error": { "message": err.to_string() } })
                            .to_string()
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        let response_id = TelemetryHub::fast_prefixed_id("resp_");
        let created_sec = now_ms() / 1000;
        let (oi, raw_req) = prepare_telemetry_capture(&state, &principal, &req, &inner, None);

        state.telemetry.inc_active();
        let routed = match state.router.execute_stream(req).await {
            Ok(s) => s,
            Err(err) => {
                state.telemetry.dec_active();
                let _ = budget_guard.commit(NanoUsd::ZERO);
                let _ = socket
                    .send(WsMessage::Text(
                        json!({ "type": "error", "error": { "message": err.to_string() } })
                            .to_string()
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        let created_frame = json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_sec,
                "status": "in_progress",
                "model": routed.resolved_model,
            }
        });
        let _ = socket
            .send(WsMessage::Text(created_frame.to_string().into()))
            .await;

        let mut events = routed.events;
        let mut completed = None;
        while let Some(ev) = events.next().await {
            match ev {
                RoutedStreamEvent::TextDelta(delta) => {
                    let frame = json!({
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "delta": delta,
                    });
                    if socket
                        .send(WsMessage::Text(frame.to_string().into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                RoutedStreamEvent::ReasoningDelta(delta) => {
                    let frame = json!({
                        "type": "response.reasoning_summary_text.delta",
                        "response_id": response_id,
                        "delta": delta,
                    });
                    let _ = socket.send(WsMessage::Text(frame.to_string().into())).await;
                }
                RoutedStreamEvent::Finish(sum) => {
                    let full = build_completed_response_json(
                        &response_id,
                        created_sec,
                        &sum.resolved_model,
                        &sum.text,
                        sum.reasoning.as_deref(),
                        &sum.tool_calls,
                        sum.input_tokens,
                        sum.cached_input_tokens,
                        sum.reasoning_tokens,
                        sum.output_tokens,
                    );
                    let done_frame = json!({
                        "type": "response.completed",
                        "response": full,
                    });
                    let _ = socket
                        .send(WsMessage::Text(done_frame.to_string().into()))
                        .await;
                    completed = Some(sum);
                }
                RoutedStreamEvent::Error(err_msg) => {
                    let _ = socket
                        .send(WsMessage::Text(
                            json!({ "type": "error", "error": { "message": err_msg } })
                                .to_string()
                                .into(),
                        ))
                        .await;
                    break;
                }
                _ => {}
            }
        }
        state.telemetry.dec_active();

        if let Some(sum) = completed {
            let oi_tools = sum
                .tool_calls
                .iter()
                .map(|tc| {
                    (
                        tc.tool.id.clone(),
                        tc.tool.name.clone(),
                        tc.input.to_string(),
                    )
                })
                .collect();
            finalize_request_accounting(
                &state,
                &principal,
                budget_guard,
                response_id,
                IngressProtocol::ResponsesWs,
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
                None,
                raw_req,
                None,
                oi,
                Some(sum.text),
                oi_tools,
            );
        }
    }
}
