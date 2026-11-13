//! `POST /v1beta/models/{*model_action}` — Google Gemini `:generateContent` &
//! `:streamGenerateContent` codec (native Gemini CLI & `@google/genai` compatibility)
//! with zero-spawn `BytesMut` streaming state machine (`GeminiTranscodeSseStream`).

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use aisdk::core::language_model::{LanguageModelResponseContentType, ReasoningEffort};
use aisdk::core::tools::ToolDetails;
use aisdk::core::{AssistantMessage, Message, ToolCallInfo, ToolResultInfo, UserMessage};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::BytesMut;
use futures::Stream;
use quixigate_core::{IngressProtocol, NanoUsd, QuixigateError};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::{
    CanonicalRequest, ClientToolSpec, RoutedStream, RoutedStreamEvent, StreamCompletionSummary,
};
use quixigate_telemetry::{OpenInferencePayload, TelemetryHub};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

#[derive(Debug, Default, Deserialize)]
pub struct GeminiQuery {
    pub alt: Option<String>,
    pub key: Option<String>,
}

fn parse_gemini_canonical(
    model_name: &str,
    body: &Value,
    headers: &HeaderMap,
) -> quixigate_core::Result<CanonicalRequest> {
    let system = body["systemInstruction"]["parts"]
        .as_array()
        .or_else(|| body["system_instruction"]["parts"].as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p["text"].as_str().map(str::to_owned))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|s| !s.trim().is_empty());

    let mut messages = Vec::new();
    if let Some(contents) = body["contents"].as_array() {
        for item in contents {
            let role = item["role"].as_str().unwrap_or("user");
            let Some(parts) = item["parts"].as_array() else {
                continue;
            };
            let mut text_acc = Vec::new();
            for part in parts {
                if let Some(fc) = part.get("functionCall") {
                    if !text_acc.is_empty() {
                        messages.push(Message::Assistant(AssistantMessage::new(
                            LanguageModelResponseContentType::Text(text_acc.join("\n")),
                            None,
                        )));
                        text_acc.clear();
                    }
                    let name = fc["name"].as_str().unwrap_or("tool").to_owned();
                    let id = fc["id"]
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| TelemetryHub::fast_prefixed_id("call_"));
                    let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    messages.push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                            tool: ToolDetails { id, name },
                            input: args,
                            extensions: Default::default(),
                        }),
                        None,
                    )));
                } else if let Some(fr) = part.get("functionResponse") {
                    if !text_acc.is_empty() {
                        messages.push(Message::User(UserMessage::new(text_acc.join("\n"))));
                        text_acc.clear();
                    }
                    let name = fr["name"].as_str().unwrap_or("tool").to_owned();
                    let id = fr["id"].as_str().unwrap_or(&name).to_owned();
                    let out_val = fr.get("response").cloned().unwrap_or(Value::Null);
                    messages.push(Message::Tool(ToolResultInfo {
                        tool: ToolDetails { id, name },
                        output: Ok(out_val),
                    }));
                } else if let Some(t) = part["text"].as_str() {
                    text_acc.push(t.to_owned());
                } else if part.get("inlineData").is_some() {
                    let mime = part["inlineData"]["mimeType"]
                        .as_str()
                        .unwrap_or("application/octet-stream");
                    text_acc.push(format!("[inlineData: {mime}]"));
                }
            }
            if !text_acc.is_empty() {
                if role == "model" || role == "assistant" {
                    messages.push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::Text(text_acc.join("\n")),
                        None,
                    )));
                } else {
                    messages.push(Message::User(UserMessage::new(text_acc.join("\n"))));
                }
            }
        }
    }

    let mut client_tools = Vec::new();
    if let Some(tool_groups) = body["tools"].as_array() {
        for grp in tool_groups {
            if let Some(decls) = grp["functionDeclarations"].as_array() {
                for decl in decls {
                    if let Some(name) = decl["name"].as_str() {
                        client_tools.push(ClientToolSpec {
                            name: name.to_owned(),
                            description: decl["description"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                            parameters: decl
                                .get("parameters")
                                .cloned()
                                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                        });
                    }
                }
            }
        }
    }

    let gen_cfg = &body["generationConfig"];
    let schema = gen_cfg
        .get("responseSchema")
        .or_else(|| gen_cfg.get("responseJsonSchema"))
        .cloned();
    let reasoning_effort = gen_cfg["thinkingConfig"]["thinkingBudget"]
        .as_u64()
        .map(|b| {
            if b == 0 {
                ReasoningEffort::None
            } else if b >= 8192 {
                ReasoningEffort::High
            } else {
                ReasoningEffort::Medium
            }
        });

    Ok(CanonicalRequest {
        requested_model: model_name.to_owned(),
        system,
        messages,
        client_tools,
        schema,
        temperature: scale_temperature(gen_cfg["temperature"].as_f64()),
        top_p: scale_top_p(gen_cfg["topP"].as_f64()),
        top_k: gen_cfg["topK"].as_u64().map(|v| v as u32),
        max_output_tokens: gen_cfg["maxOutputTokens"].as_u64().map(|v| v as u32),
        stop_sequences: gen_cfg["stopSequences"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        }),
        presence_penalty: None,
        frequency_penalty: None,
        seed: gen_cfg["seed"].as_u64().map(|v| v as u32),
        reasoning_effort,
        extra_headers: extract_passthrough_headers(headers),
        extra_body: Map::new(),
        openrouter_body: extract_openrouter_body(body),
        cache_preference: parse_cache_preference(headers),
    })
}

pub async fn handle_gemini_model_action(
    State(state): State<AppState>,
    Path(model_action): Path<String>,
    Query(query): Query<GeminiQuery>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, query.key.as_deref())?;
    let trimmed = model_action.trim_start_matches('/');
    let (raw_model, action) = trimmed.rsplit_once(':').ok_or_else(|| {
        QuixigateError::Validation(format!(
            "expected 'models/{{model}}:generateContent', got '{trimmed}'"
        ))
    })?;
    let model_name = raw_model.strip_prefix("models/").unwrap_or(raw_model);

    if action == "countTokens" {
        let req = parse_gemini_canonical(model_name, &body, &headers)?;
        let tok = estimate_request_tokens(&req);
        return Ok(Json(json!({ "totalTokens": tok })).into_response());
    }

    let is_stream = action == "streamGenerateContent";
    if action != "generateContent" && !is_stream {
        return Err(ApiError(QuixigateError::Validation(format!(
            "unsupported Gemini action '{action}'"
        ))));
    }

    let settings = state.db.settings_arc();
    let capture_payloads = principal
        .virtual_key
        .as_ref()
        .and_then(|k| k.log_payloads)
        .unwrap_or(settings.capture_payloads_default);
    let need_deep_telemetry = capture_payloads || settings.otlp_enabled;

    let req = parse_gemini_canonical(model_name, &body, &headers)?;
    let requested_model = req.requested_model.clone();
    let max_out = u64::from(req.max_output_tokens.unwrap_or(2048));
    let est_in = estimate_request_tokens(&req);
    let (est_total, est_cost) =
        state
            .router
            .estimate_preflight_budget(&requested_model, est_in, max_out)?;
    let (conc_guard, budget_guard) = authorize_inference_request(
        &state,
        &principal,
        &headers,
        Some(&body),
        &requested_model,
        est_total,
        est_cost,
    )?;

    let include_thoughts = body["generationConfig"]["thinkingConfig"]["includeThoughts"]
        .as_bool()
        .unwrap_or(false);
    let request_id = TelemetryHub::fast_prefixed_id("gem_");
    let (oi, raw_req) = prepare_telemetry_capture(&state, &principal, &req, &body, None);
    let started = Instant::now();

    state.telemetry.inc_active();

    if !is_stream {
        let res = state.router.execute_text(req).await;
        state.telemetry.dec_active();
        drop(conc_guard);

        match res {
            Ok(gen_out) => {
                let mut parts = Vec::new();
                if include_thoughts
                    && let Some(r) = &gen_out.reasoning
                    && !r.is_empty()
                {
                    parts.push(json!({ "text": r, "thought": true }));
                }
                if !gen_out.text.is_empty() || gen_out.tool_calls.is_empty() {
                    parts.push(json!({ "text": gen_out.text }));
                }
                for tc in &gen_out.tool_calls {
                    parts.push(json!({
                        "functionCall": {
                            "id": tc.tool.id,
                            "name": tc.tool.name,
                            "args": tc.input,
                        }
                    }));
                }

                let finish_reason = match gen_out.finish_reason.as_deref() {
                    Some("length") => "MAX_TOKENS",
                    Some("content_filter") => "SAFETY",
                    _ => "STOP",
                };
                let resp_json = json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": parts },
                        "finishReason": finish_reason,
                        "index": 0
                    }],
                    "usageMetadata": {
                        "promptTokenCount": gen_out.input_tokens,
                        "cachedContentTokenCount": gen_out.cached_input_tokens,
                        "candidatesTokenCount": gen_out.output_tokens,
                        "thoughtsTokenCount": gen_out.reasoning_tokens,
                        "totalTokenCount": gen_out.input_tokens + gen_out.output_tokens
                    },
                    "modelVersion": gen_out.resolved_model,
                    "responseId": request_id
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
                    request_id.clone(),
                    IngressProtocol::Gemini,
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
                    IngressProtocol::Gemini,
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
                    request_id,
                    IngressProtocol::Gemini,
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
        let hdr_req_id = request_id.clone();

        let sse_stream = GeminiTranscodeSseStream::new(
            routed,
            state,
            principal,
            conc_guard,
            budget_guard,
            request_id,
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
            &hdr_req_id,
            &provider_hdr,
            &resolved_hdr,
            cache_hit,
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        Ok(resp)
    }
}

// ── Zero-Spawn `GeminiTranscodeSseStream` ────────────────────────────────────

const TRANSCODE_BATCH_CAP: usize = 32 * 1024;

struct GeminiTranscodeSseStream {
    events: Pin<Box<dyn Stream<Item = RoutedStreamEvent> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
    request_id: String,
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
    text_suffix: Bytes,
    thought_suffix: Bytes,
    buf: BytesMut,
    completed_summary: Option<StreamCompletionSummary>,
    stream_err: Option<String>,
    eof: bool,
}

impl GeminiTranscodeSseStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        routed: RoutedStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        request_id: String,
        requested_model: String,
        started: Instant,
        need_deep_telemetry: bool,
        raw_req: Option<String>,
        oi: OpenInferencePayload,
    ) -> Self {
        let mut ts = BytesMut::with_capacity(64 + routed.resolved_model.len());
        ts.extend_from_slice(b"}]},\"index\":0}],\"modelVersion\":");
        put_json_escaped_str(&mut ts, &routed.resolved_model);
        ts.extend_from_slice(b"}\n\n");
        let text_suffix = ts.freeze();

        let mut ths = BytesMut::with_capacity(80 + routed.resolved_model.len());
        ths.extend_from_slice(b",\"thought\":true}]},\"index\":0}],\"modelVersion\":");
        put_json_escaped_str(&mut ths, &routed.resolved_model);
        ths.extend_from_slice(b"}\n\n");
        let thought_suffix = ths.freeze();

        Self {
            events: routed.events,
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
            request_id,
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
            text_suffix,
            thought_suffix,
            buf: BytesMut::with_capacity(1024),
            completed_summary: None,
            stream_err: None,
            eof: false,
        }
    }

    fn encode_event(&mut self, ev: RoutedStreamEvent) {
        match ev {
            RoutedStreamEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                self.buf.extend_from_slice(
                    b"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":",
                );
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(&self.thought_suffix);
            }
            RoutedStreamEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                self.buf.extend_from_slice(
                    b"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":",
                );
                put_json_escaped_str(&mut self.buf, &delta);
                self.buf.extend_from_slice(&self.text_suffix);
            }
            RoutedStreamEvent::ToolCallReady(tc) => {
                self.buf.extend_from_slice(
                    b"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":",
                );
                put_json_escaped_str(&mut self.buf, &tc.tool.id);
                self.buf.extend_from_slice(b",\"name\":");
                put_json_escaped_str(&mut self.buf, &tc.tool.name);
                self.buf.extend_from_slice(b",\"args\":");
                put_json_value(&mut self.buf, &tc.input);
                self.buf
                    .extend_from_slice(b"}}]},\"index\":0}],\"modelVersion\":");
                put_json_escaped_str(&mut self.buf, &self.resolved_model);
                self.buf.extend_from_slice(b"}\n\n");
            }
            RoutedStreamEvent::Finish(sum) => {
                self.buf.extend_from_slice(
                    b"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":",
                );
                put_u64_dec(&mut self.buf, sum.input_tokens);
                self.buf.extend_from_slice(b",\"cachedContentTokenCount\":");
                put_u64_dec(&mut self.buf, sum.cached_input_tokens);
                self.buf.extend_from_slice(b",\"candidatesTokenCount\":");
                put_u64_dec(&mut self.buf, sum.output_tokens);
                self.buf.extend_from_slice(b",\"thoughtsTokenCount\":");
                put_u64_dec(&mut self.buf, sum.reasoning_tokens);
                self.buf.extend_from_slice(b",\"totalTokenCount\":");
                put_u64_dec(&mut self.buf, sum.input_tokens + sum.output_tokens);
                self.buf.extend_from_slice(b"},\"modelVersion\":");
                put_json_escaped_str(&mut self.buf, &sum.resolved_model);
                self.buf.extend_from_slice(b"}\n\n");
                self.completed_summary = Some(sum);
                self.eof = true;
            }
            RoutedStreamEvent::Error(err_msg) => {
                self.stream_err = Some(err_msg);
                self.eof = true;
            }
            _ => {}
        }
    }
}

impl Stream for GeminiTranscodeSseStream {
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

impl Drop for GeminiTranscodeSseStream {
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
                std::mem::take(&mut self.request_id),
                IngressProtocol::Gemini,
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
                std::mem::take(&mut self.request_id),
                IngressProtocol::Gemini,
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
