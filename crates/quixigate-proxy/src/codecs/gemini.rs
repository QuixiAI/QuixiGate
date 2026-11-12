//! `POST /v1beta/models/{*model_action}` — Google Gemini `:generateContent` &
//! `:streamGenerateContent` codec (native Gemini CLI & `@google/genai` compatibility).

use std::convert::Infallible;
use std::time::Instant;

use aisdk::core::language_model::{LanguageModelResponseContentType, ReasoningEffort};
use aisdk::core::tools::ToolDetails;
use aisdk::core::{AssistantMessage, Message, ToolCallInfo, ToolResultInfo, UserMessage};
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use quixigate_core::{IngressProtocol, QuixigateError};
use quixigate_router::{CanonicalRequest, ClientToolSpec, RoutedStreamEvent};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

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
                        .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
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
                            description: decl["description"].as_str().unwrap_or_default().to_owned(),
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
    let (raw_model, action) = trimmed
        .rsplit_once(':')
        .ok_or_else(|| {
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

    let request_id = format!("gem_{}", Uuid::new_v4().simple());
    let oi = build_openinference_inputs(&req, None);
    let raw_req = Some(body.to_string());
    let started = Instant::now();

    state.telemetry.inc_active();

    if !is_stream {
        let res = state.router.execute_text(req).await;
        state.telemetry.dec_active();
        drop(conc_guard);

        match res {
            Ok(gen_out) => {
                let mut parts = Vec::new();
                if let Some(r) = &gen_out.reasoning
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

                let oi_tools = gen_out
                    .tool_calls
                    .iter()
                    .map(|tc| (tc.tool.id.clone(), tc.tool.name.clone(), tc.input.to_string()))
                    .collect();

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
                    Some(resp_json.to_string()),
                    oi,
                    Some(gen_out.text),
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
                    quixigate_core::NanoUsd::ZERO,
                    quixigate_core::NanoUsd::ZERO,
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
                    quixigate_core::NanoUsd::ZERO,
                    quixigate_core::NanoUsd::ZERO,
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

        let resolved_model = routed.resolved_model.clone();
        let provider_id = routed.provider_id.clone();
        let cache_hit = routed.cache_hit;
        let hdr_req_id = request_id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(64);
        let state_bg = state.clone();

        tokio::spawn(async move {
            let _conc = conc_guard;
            let mut events = routed.events;
            let mut completed_summary = None;
            let mut stream_err = None;

            while let Some(ev) = events.next().await {
                match ev {
                    RoutedStreamEvent::ReasoningDelta(delta) => {
                        if delta.is_empty() {
                            continue;
                        }
                        let chunk = json!({
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{ "text": delta, "thought": true }]
                                },
                                "index": 0
                            }],
                            "modelVersion": resolved_model
                        });
                        let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    }
                    RoutedStreamEvent::TextDelta(delta) => {
                        if delta.is_empty() {
                            continue;
                        }
                        let chunk = json!({
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{ "text": delta }]
                                },
                                "index": 0
                            }],
                            "modelVersion": resolved_model
                        });
                        if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                            break;
                        }
                    }
                    RoutedStreamEvent::ToolCallReady(tc) => {
                        let chunk = json!({
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{
                                        "functionCall": {
                                            "id": tc.tool.id,
                                            "name": tc.tool.name,
                                            "args": tc.input
                                        }
                                    }]
                                },
                                "index": 0
                            }],
                            "modelVersion": resolved_model
                        });
                        let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    }
                    RoutedStreamEvent::Finish(sum) => {
                        let final_chunk = json!({
                            "candidates": [{
                                "content": { "role": "model", "parts": [] },
                                "finishReason": "STOP",
                                "index": 0
                            }],
                            "usageMetadata": {
                                "promptTokenCount": sum.input_tokens,
                                "cachedContentTokenCount": sum.cached_input_tokens,
                                "candidatesTokenCount": sum.output_tokens,
                                "thoughtsTokenCount": sum.reasoning_tokens,
                                "totalTokenCount": sum.input_tokens + sum.output_tokens
                            },
                            "modelVersion": sum.resolved_model
                        });
                        let _ = tx
                            .send(Ok(Event::default().data(final_chunk.to_string())))
                            .await;
                        completed_summary = Some(sum);
                    }
                    RoutedStreamEvent::Error(err_msg) => {
                        stream_err = Some(err_msg);
                        break;
                    }
                    _ => {}
                }
            }

            state_bg.telemetry.dec_active();
            if let Some(sum) = completed_summary {
                let oi_tools = sum
                    .tool_calls
                    .iter()
                    .map(|tc| (tc.tool.id.clone(), tc.tool.name.clone(), tc.input.to_string()))
                    .collect();
                finalize_request_accounting(
                    &state_bg,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::Gemini,
                    requested_model,
                    sum.resolved_model,
                    sum.provider_id,
                    sum.credential_id,
                    200,
                    true,
                    sum.fallback_attempts,
                    cache_hit,
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
            } else {
                finalize_request_accounting(
                    &state_bg,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::Gemini,
                    requested_model,
                    resolved_model,
                    provider_id,
                    routed.credential_id,
                    if stream_err.is_some() { 502 } else { 200 },
                    true,
                    routed.fallback_attempts,
                    cache_hit,
                    None,
                    started.elapsed().as_millis() as u64,
                    0,
                    0,
                    0,
                    0,
                    quixigate_core::NanoUsd::ZERO,
                    quixigate_core::NanoUsd::ZERO,
                    stream_err,
                    raw_req,
                    None,
                    oi,
                    None,
                    Vec::new(),
                );
            }
        });

        let mut resp = Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response();
        attach_gateway_headers(
            resp.headers_mut(),
            &hdr_req_id,
            &routed.provider_id,
            &routed.resolved_model,
            cache_hit,
            quixigate_core::NanoUsd::ZERO,
            quixigate_core::NanoUsd::ZERO,
        );
        Ok(resp)
    }
}
