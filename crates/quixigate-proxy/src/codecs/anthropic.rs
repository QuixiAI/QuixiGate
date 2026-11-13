//! `POST /v1/messages` & `POST /v1/messages/count_tokens` —
//! Anthropic Messages JSON & SSE codec (native Claude Code compatibility).

use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Instant;

use aisdk::core::language_model::{LanguageModelResponseContentType, ReasoningEffort};
use aisdk::core::tools::ToolDetails;
use aisdk::core::{AssistantMessage, Message, ToolCallInfo, ToolResultInfo, UserMessage};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use quixigate_core::{IngressProtocol, QuixigateError};
use quixigate_router::{CanonicalRequest, ClientToolSpec, RoutedStreamEvent};
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

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
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let req = parse_anthropic_canonical(&body, &headers)?;
    let requested_model = req.requested_model.clone();
    let max_out = u64::from(req.max_output_tokens.unwrap_or(4096));
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

    let stream_requested = body["stream"].as_bool().unwrap_or(false);
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());
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

                let oi_tools = gen_out
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
                    Some(resp_json.to_string()),
                    oi,
                    Some(gen_out.text),
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
        let hdr_msg_id = msg_id.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(64);
        let state_bg = state.clone();

        tokio::spawn(async move {
            let _conc = conc_guard;
            let mut events = routed.events;

            let start_ev = json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": resolved_model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": est_in,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 1
                    }
                }
            });
            let _ = tx
                .send(Ok(Event::default()
                    .event("message_start")
                    .data(start_ev.to_string())))
                .await;

            let mut block_idx: usize = 0;
            let mut open_block: Option<&'static str> = None;
            let mut tool_idx_by_id: HashMap<String, usize> = HashMap::new();
            let mut completed_summary = None;
            let mut stream_err = None;

            let close_open =
                |tx: &tokio::sync::mpsc::Sender<std::result::Result<Event, Infallible>>,
                 idx: usize| {
                    let stop = json!({ "type": "content_block_stop", "index": idx });
                    let tx = tx.clone();
                    async move {
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("content_block_stop")
                                .data(stop.to_string())))
                            .await;
                    }
                };

            while let Some(ev) = events.next().await {
                match ev {
                    RoutedStreamEvent::ReasoningDelta(delta) => {
                        if delta.is_empty() {
                            continue;
                        }
                        if open_block != Some("thinking") {
                            if open_block.is_some() {
                                close_open(&tx, block_idx).await;
                                block_idx += 1;
                            }
                            let blk = json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": { "type": "thinking", "thinking": "" }
                            });
                            let _ = tx
                                .send(Ok(Event::default()
                                    .event("content_block_start")
                                    .data(blk.to_string())))
                                .await;
                            open_block = Some("thinking");
                        }
                        let d = json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": { "type": "thinking_delta", "thinking": delta }
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("content_block_delta")
                                .data(d.to_string())))
                            .await;
                    }
                    RoutedStreamEvent::TextDelta(delta) => {
                        if delta.is_empty() {
                            continue;
                        }
                        if open_block != Some("text") {
                            if open_block.is_some() {
                                close_open(&tx, block_idx).await;
                                block_idx += 1;
                            }
                            let blk = json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": { "type": "text", "text": "" }
                            });
                            let _ = tx
                                .send(Ok(Event::default()
                                    .event("content_block_start")
                                    .data(blk.to_string())))
                                .await;
                            open_block = Some("text");
                        }
                        let d = json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": { "type": "text_delta", "text": delta }
                        });
                        if tx
                            .send(Ok(Event::default()
                                .event("content_block_delta")
                                .data(d.to_string())))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    RoutedStreamEvent::ToolCallStart { id, name } => {
                        if open_block.is_some() {
                            close_open(&tx, block_idx).await;
                            block_idx += 1;
                        }
                        tool_idx_by_id.insert(id.clone(), block_idx);
                        open_block = Some("tool_use");
                        let blk = json!({
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {}
                            }
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("content_block_start")
                                .data(blk.to_string())))
                            .await;
                    }
                    RoutedStreamEvent::ToolCallDelta { id, delta } => {
                        let idx = tool_idx_by_id.get(&id).copied().unwrap_or(block_idx);
                        let d = json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": { "type": "input_json_delta", "partial_json": delta }
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("content_block_delta")
                                .data(d.to_string())))
                            .await;
                    }
                    RoutedStreamEvent::ToolCallReady(tc) => {
                        if !tool_idx_by_id.contains_key(&tc.tool.id) {
                            if open_block.is_some() {
                                close_open(&tx, block_idx).await;
                                block_idx += 1;
                            }
                            tool_idx_by_id.insert(tc.tool.id.clone(), block_idx);
                            open_block = Some("tool_use");
                            let blk = json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.tool.id,
                                    "name": tc.tool.name,
                                    "input": {}
                                }
                            });
                            let _ = tx
                                .send(Ok(Event::default()
                                    .event("content_block_start")
                                    .data(blk.to_string())))
                                .await;
                            let d = json!({
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": tc.input.to_string()
                                }
                            });
                            let _ = tx
                                .send(Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(d.to_string())))
                                .await;
                        }
                    }
                    RoutedStreamEvent::Finish(sum) => {
                        if open_block.is_some() {
                            close_open(&tx, block_idx).await;
                            open_block = None;
                        }
                        let stop_reason = if !sum.tool_calls.is_empty() {
                            "tool_use"
                        } else {
                            "end_turn"
                        };
                        let m_delta = json!({
                            "type": "message_delta",
                            "delta": { "stop_reason": stop_reason, "stop_sequence": Value::Null },
                            "usage": {
                                "input_tokens": sum.input_tokens,
                                "cache_read_input_tokens": sum.cached_input_tokens,
                                "output_tokens": sum.output_tokens
                            }
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("message_delta")
                                .data(m_delta.to_string())))
                            .await;
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("message_stop")
                                .data(json!({ "type": "message_stop" }).to_string())))
                            .await;
                        completed_summary = Some(sum);
                    }
                    RoutedStreamEvent::Error(err_msg) => {
                        stream_err = Some(err_msg.clone());
                        let err_ev = json!({
                            "type": "error",
                            "error": { "type": "api_error", "message": err_msg }
                        });
                        let _ = tx
                            .send(Ok(Event::default().event("error").data(err_ev.to_string())))
                            .await;
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
                    .map(|tc| {
                        (
                            tc.tool.id.clone(),
                            tc.tool.name.clone(),
                            tc.input.to_string(),
                        )
                    })
                    .collect();
                finalize_request_accounting(
                    &state_bg,
                    &principal,
                    budget_guard,
                    msg_id,
                    IngressProtocol::AnthropicMessages,
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
                    msg_id,
                    IngressProtocol::AnthropicMessages,
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
            &hdr_msg_id,
            &routed.provider_id,
            &routed.resolved_model,
            cache_hit,
            quixigate_core::NanoUsd::ZERO,
            quixigate_core::NanoUsd::ZERO,
        );
        Ok(resp)
    }
}
