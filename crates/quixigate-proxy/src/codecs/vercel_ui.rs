//! `POST /api/chat` — Vercel AI SDK v5/v6 `UIMessageStream` (`x-vercel-ai-ui-message-stream: v1`)
//! powering QuixiGate's `@ai-sdk/react` `useChat` Playground and Multi-Model Arena with a
//! zero-spawn `BytesMut` streaming state machine (`VercelUiSseStream`).

use std::collections::HashSet;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use aisdk::core::language_model::LanguageModelResponseContentType;
use aisdk::core::{AssistantMessage, Message, SystemMessage, UserMessage};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::Response;
use bytes::BytesMut;
use futures::Stream;
use quixigate_core::{IngressProtocol, NanoUsd, QuixigateError};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::{
    CanonicalRequest, RoutedStream, RoutedStreamEvent, StreamCompletionSummary,
};
use quixigate_telemetry::{OpenInferencePayload, TelemetryHub};
use serde_json::{Map, Value, json};

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

const HDR_VERCEL_UI_STREAM: HeaderName = HeaderName::from_static("x-vercel-ai-ui-message-stream");

fn extract_ui_message_text(msg: &Value) -> String {
    if let Some(parts) = msg["parts"].as_array() {
        let mut out = Vec::new();
        for p in parts {
            if p["type"].as_str() == Some("text")
                && let Some(t) = p["text"].as_str()
            {
                out.push(t.to_owned());
            }
        }
        if !out.is_empty() {
            return out.join("\n");
        }
    }
    extract_content_text(&msg["content"])
}

pub async fn handle_vercel_ui_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.settings_arc();
    let capture_payloads = principal
        .virtual_key
        .as_ref()
        .and_then(|k| k.log_payloads)
        .unwrap_or(settings.capture_payloads_default);
    let need_deep_telemetry = capture_payloads || settings.otlp_enabled;

    let base_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            QuixigateError::Validation("missing 'model' in /api/chat request".to_owned())
        })?;
    let online_flag = body["online"].as_bool().unwrap_or(false);
    let requested_model = if online_flag && !base_model.contains(":online") {
        format!("{base_model}:online")
    } else {
        base_model.to_owned()
    };

    let mut system_parts = Vec::new();
    if let Some(sys) = body["system"].as_str()
        && !sys.trim().is_empty()
    {
        system_parts.push(sys.to_owned());
    }

    let mut messages = Vec::new();
    if let Some(ui_msgs) = body["messages"].as_array() {
        for m in ui_msgs {
            let role = m["role"].as_str().unwrap_or("user");
            let txt = extract_ui_message_text(m);
            match role {
                "system" => {
                    if !txt.trim().is_empty() {
                        system_parts.push(txt.clone());
                        messages.push(Message::System(SystemMessage::new(txt)));
                    }
                }
                "assistant" => {
                    if !txt.is_empty() {
                        messages.push(Message::Assistant(AssistantMessage::new(
                            LanguageModelResponseContentType::Text(txt),
                            None,
                        )));
                    }
                }
                _ => {
                    messages.push(Message::User(UserMessage::new(txt)));
                }
            }
        }
    } else if let Some(prompt) = body["prompt"].as_str() {
        messages.push(Message::User(UserMessage::new(prompt.to_owned())));
    }

    let max_out = body["maxOutputTokens"]
        .as_u64()
        .or_else(|| body["max_tokens"].as_u64())
        .map(|v| v as u32);

    let req = CanonicalRequest {
        requested_model: requested_model.clone(),
        system: (!system_parts.is_empty()).then(|| system_parts.join("\n\n")),
        messages,
        client_tools: parse_openai_tools(&body),
        schema: None,
        temperature: scale_temperature(body["temperature"].as_f64()),
        top_p: scale_top_p(body["topP"].as_f64()),
        top_k: None,
        max_output_tokens: max_out,
        stop_sequences: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        reasoning_effort: parse_reasoning_effort(body["reasoningEffort"].as_str()),
        extra_headers: extract_passthrough_headers(&headers),
        extra_body: Map::new(),
        openrouter_body: extract_openrouter_body(&body),
        cache_preference: parse_cache_preference(&headers),
    };

    let est_in = estimate_request_tokens(&req);
    let (est_total, est_cost) = state.router.estimate_preflight_budget(
        &requested_model,
        est_in,
        u64::from(max_out.unwrap_or(2048)),
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

    let message_id = TelemetryHub::fast_prefixed_id("msg_");
    let (oi, raw_req) = prepare_telemetry_capture(
        &state,
        &principal,
        &req,
        &body,
        body["id"].as_str().map(str::to_owned),
    );
    let started = Instant::now();

    state.telemetry.inc_active();

    let routed = match state.router.execute_stream(req).await {
        Ok(s) => s,
        Err(err) => {
            state.telemetry.dec_active();
            drop(conc_guard);
            finalize_request_accounting(
                &state,
                &principal,
                budget_guard,
                message_id,
                IngressProtocol::VercelUi,
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
    let hdr_msg_id = message_id.clone();

    let sse_stream = VercelUiSseStream::new(
        routed,
        state,
        principal,
        conc_guard,
        budget_guard,
        message_id,
        requested_model,
        started,
        capture_payloads,
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
    hdrs.insert(HDR_VERCEL_UI_STREAM, HeaderValue::from_static("v1"));
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

// ── Zero-Spawn `VercelUiSseStream` ───────────────────────────────────────────

const TRANSCODE_BATCH_CAP: usize = 32 * 1024;

struct VercelUiSseStream {
    events: Pin<Box<dyn Stream<Item = RoutedStreamEvent> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
    message_id: String,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    fallback_attempts: u32,
    cache_hit: bool,
    started: Instant,
    capture_payloads: bool,
    need_deep_telemetry: bool,
    raw_req: Option<String>,
    oi: Option<OpenInferencePayload>,
    text_part_id: String,
    rsn_part_id: String,
    text_open: bool,
    rsn_open: bool,
    started_tool_ids: HashSet<String>,
    buf: BytesMut,
    completed_summary: Option<StreamCompletionSummary>,
    stream_err: Option<String>,
    eof: bool,
}

impl VercelUiSseStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        routed: RoutedStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        message_id: String,
        requested_model: String,
        started: Instant,
        capture_payloads: bool,
        need_deep_telemetry: bool,
        raw_req: Option<String>,
        oi: OpenInferencePayload,
    ) -> Self {
        let mut buf = BytesMut::with_capacity(1024);
        buf.extend_from_slice(b"data: {\"type\":\"start\",\"messageId\":");
        put_json_escaped_str(&mut buf, &message_id);
        buf.extend_from_slice(b",\"messageMetadata\":{\"model\":");
        put_json_escaped_str(&mut buf, &routed.resolved_model);
        buf.extend_from_slice(b",\"provider\":");
        put_json_escaped_str(&mut buf, &routed.provider_id);
        buf.extend_from_slice(if routed.cache_hit {
            b",\"cacheHit\":true}}\n\ndata: {\"type\":\"start-step\"}\n\n"
        } else {
            b",\"cacheHit\":false}}\n\ndata: {\"type\":\"start-step\"}\n\n"
        });

        Self {
            events: routed.events,
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
            message_id,
            requested_model,
            resolved_model: routed.resolved_model,
            provider_id: routed.provider_id,
            credential_id: routed.credential_id,
            fallback_attempts: routed.fallback_attempts,
            cache_hit: routed.cache_hit,
            started,
            capture_payloads,
            need_deep_telemetry,
            raw_req,
            oi: Some(oi),
            text_part_id: TelemetryHub::fast_prefixed_id("txt_"),
            rsn_part_id: TelemetryHub::fast_prefixed_id("rsn_"),
            text_open: false,
            rsn_open: false,
            started_tool_ids: HashSet::new(),
            buf,
            completed_summary: None,
            stream_err: None,
            eof: false,
        }
    }

    #[inline]
    fn write_close_rsn(&mut self) {
        if self.rsn_open {
            self.buf
                .extend_from_slice(b"data: {\"type\":\"reasoning-end\",\"id\":");
            put_json_escaped_str(&mut self.buf, &self.rsn_part_id);
            self.buf.extend_from_slice(b"}\n\n");
            self.rsn_open = false;
        }
    }

    #[inline]
    fn write_close_text(&mut self) {
        if self.text_open {
            self.buf
                .extend_from_slice(b"data: {\"type\":\"text-end\",\"id\":");
            put_json_escaped_str(&mut self.buf, &self.text_part_id);
            self.buf.extend_from_slice(b"}\n\n");
            self.text_open = false;
        }
    }

    fn encode_event(&mut self, ev: RoutedStreamEvent) {
        match ev {
            RoutedStreamEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                if !self.rsn_open {
                    self.buf
                        .extend_from_slice(b"data: {\"type\":\"reasoning-start\",\"id\":");
                    put_json_escaped_str(&mut self.buf, &self.rsn_part_id);
                    self.buf.extend_from_slice(b"}\n\n");
                    self.rsn_open = true;
                }
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"reasoning-delta\",\"id\":");
                put_json_escaped_str(&mut self.buf, &self.rsn_part_id);
                self.buf.extend_from_slice(b",\"delta\":");
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::ReasoningEnd => {
                self.write_close_rsn();
            }
            RoutedStreamEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                self.write_close_rsn();
                if !self.text_open {
                    self.buf
                        .extend_from_slice(b"data: {\"type\":\"text-start\",\"id\":");
                    put_json_escaped_str(&mut self.buf, &self.text_part_id);
                    self.buf.extend_from_slice(b"}\n\n");
                    self.text_open = true;
                }
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"text-delta\",\"id\":");
                put_json_escaped_str(&mut self.buf, &self.text_part_id);
                self.buf.extend_from_slice(b",\"delta\":");
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::ToolCallStart { id, name } => {
                self.started_tool_ids.insert(id.clone());
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"tool-input-start\",\"toolCallId\":");
                put_json_escaped_str(&mut self.buf, &id);
                self.buf.extend_from_slice(b",\"toolName\":");
                put_json_escaped_str(&mut self.buf, &name);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::ToolCallDelta { id, delta } => {
                if self.started_tool_ids.contains(&id) {
                    self.buf
                        .extend_from_slice(b"data: {\"type\":\"tool-input-delta\",\"toolCallId\":");
                    put_json_escaped_str(&mut self.buf, &id);
                    self.buf.extend_from_slice(b",\"inputTextDelta\":");
                    put_json_escaped_str(&mut self.buf, &delta);
                    self.buf.extend_from_slice(b"}\n\n");
                }
            }
            RoutedStreamEvent::ToolCallReady(tc) => {
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"tool-input-available\",\"toolCallId\":");
                put_json_escaped_str(&mut self.buf, &tc.tool.id);
                self.buf.extend_from_slice(b",\"toolName\":");
                put_json_escaped_str(&mut self.buf, &tc.tool.name);
                self.buf.extend_from_slice(b",\"input\":");
                put_json_value(&mut self.buf, &tc.input);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::ServerToolResult {
                id,
                name,
                input,
                output,
            } => {
                self.write_close_rsn();
                self.write_close_text();
                let parsed_output =
                    serde_json::from_str::<Value>(&output).unwrap_or(Value::String(output));
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"tool-input-available\",\"toolCallId\":");
                put_json_escaped_str(&mut self.buf, &id);
                self.buf.extend_from_slice(b",\"toolName\":");
                put_json_escaped_str(&mut self.buf, &name);
                self.buf.extend_from_slice(b",\"input\":");
                put_json_value(&mut self.buf, &input);
                self.buf.extend_from_slice(
                    b",\"providerExecuted\":true}\n\ndata: {\"type\":\"tool-output-available\",\"toolCallId\":",
                );
                put_json_escaped_str(&mut self.buf, &id);
                self.buf.extend_from_slice(b",\"output\":");
                put_json_value(&mut self.buf, &parsed_output);
                self.buf.extend_from_slice(
                    b",\"providerExecuted\":true}\n\ndata: {\"type\":\"finish-step\"}\n\ndata: {\"type\":\"start-step\"}\n\n",
                );
                self.text_part_id = TelemetryHub::fast_prefixed_id("txt_");
                self.rsn_part_id = TelemetryHub::fast_prefixed_id("rsn_");
            }
            RoutedStreamEvent::Finish(sum) => {
                self.write_close_rsn();
                self.write_close_text();
                let meta = json!({
                    "type": "message-metadata",
                    "messageMetadata": {
                        "model": sum.resolved_model,
                        "provider": sum.provider_id,
                        "ttftMs": sum.ttft_ms,
                        "durationMs": sum.duration_ms,
                        "inputTokens": sum.input_tokens,
                        "cachedTokens": sum.cached_input_tokens,
                        "reasoningTokens": sum.reasoning_tokens,
                        "outputTokens": sum.output_tokens,
                        "costNanos": sum.cost_nanos.nanos(),
                        "costFormatted": sum.cost_nanos.format_usd(),
                        "apiEquivNanos": sum.api_equiv_nanos.nanos(),
                        "apiEquivFormatted": sum.api_equiv_nanos.format_usd(),
                        "cacheHit": self.cache_hit,
                        "fallbackAttempts": sum.fallback_attempts,
                    }
                });
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"finish-step\"}\n\ndata: ");
                put_json_value(&mut self.buf, &meta);
                self.buf.extend_from_slice(
                    b"\n\ndata: {\"type\":\"finish\",\"finishReason\":\"stop\"}\n\ndata: [DONE]\n\n",
                );
                self.completed_summary = Some(sum);
                self.eof = true;
            }
            RoutedStreamEvent::Error(err_msg) => {
                self.buf
                    .extend_from_slice(b"data: {\"type\":\"error\",\"errorText\":");
                put_json_escaped_str(&mut self.buf, &err_msg);
                self.buf.extend_from_slice(b"}\n\ndata: [DONE]\n\n");
                self.stream_err = Some(err_msg);
                self.eof = true;
            }
            _ => {}
        }
    }
}

impl Stream for VercelUiSseStream {
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
                    if !this.eof {
                        this.buf.extend_from_slice(b"data: [DONE]\n\n");
                        this.eof = true;
                    }
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

impl Drop for VercelUiSseStream {
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
                std::mem::take(&mut self.message_id),
                IngressProtocol::VercelUi,
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
                std::mem::take(&mut self.message_id),
                IngressProtocol::VercelUi,
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
