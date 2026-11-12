//! `POST /v1/responses` (JSON + SSE) & `GET /v1/responses` (WebSocket mode) —
//! OpenAI Responses API codec compatible with OpenAI SDKs and Codex CLI.

use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Instant;

use aisdk::core::language_model::LanguageModelResponseContentType;
use aisdk::core::tools::ToolDetails;
use aisdk::core::{AssistantMessage, Message, SystemMessage, ToolCallInfo, ToolResultInfo, UserMessage};
use axum::Json;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use quixigate_core::{IngressProtocol, QuixigateError, now_ms};
use quixigate_db::AuthenticatedPrincipal;
use quixigate_router::{CanonicalRequest, RoutedStreamEvent};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

#[derive(Debug, Default, Deserialize)]
pub struct WsQuery {
    pub key: Option<String>,
    pub model: Option<String>,
}

fn parse_responses_canonical(body: &Value, headers: &HeaderMap) -> quixigate_core::Result<CanonicalRequest> {
    let requested_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("missing 'model' in Responses request".to_owned()))?
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
            "id": format!("rs_{}", Uuid::new_v4().simple()),
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": r }]
        }));
    }
    if !text.is_empty() || tool_calls.is_empty() {
        output_items.push(json!({
            "id": format!("msg_{}", Uuid::new_v4().simple()),
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
            "id": format!("fc_{}", Uuid::new_v4().simple()),
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
    Json(body): Json<Value>,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let req = parse_responses_canonical(&body, &headers)?;
    let requested_model = req.requested_model.clone();
    let max_out = u64::from(req.max_output_tokens.unwrap_or(1024));
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
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let created_sec = now_ms() / 1000;
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
                let oi_tools = gen_out
                    .tool_calls
                    .iter()
                    .map(|tc| (tc.tool.id.clone(), tc.tool.name.clone(), tc.input.to_string()))
                    .collect();

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
                    Some(resp_json.to_string()),
                    oi,
                    Some(gen_out.text),
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
        let header_resp_id = response_id.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(64);
        let state_bg = state.clone();

        tokio::spawn(async move {
            let _conc = conc_guard;
            let mut events = routed.events;
            let msg_item_id = format!("msg_{}", Uuid::new_v4().simple());

            let created_ev = json!({
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_sec,
                    "status": "in_progress",
                    "model": resolved_model,
                    "output": []
                }
            });
            let _ = tx
                .send(Ok(Event::default()
                    .event("response.created")
                    .data(created_ev.to_string())))
                .await;

            let added_ev = json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": msg_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            });
            let _ = tx
                .send(Ok(Event::default()
                    .event("response.output_item.added")
                    .data(added_ev.to_string())))
                .await;

            let mut completed_summary = None;
            let mut stream_err = None;

            while let Some(ev) = events.next().await {
                match ev {
                    RoutedStreamEvent::TextDelta(delta) => {
                        let payload = json!({
                            "type": "response.output_text.delta",
                            "item_id": msg_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta
                        });
                        if tx
                            .send(Ok(Event::default()
                                .event("response.output_text.delta")
                                .data(payload.to_string())))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    RoutedStreamEvent::ReasoningDelta(delta) => {
                        let payload = json!({
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": msg_item_id,
                            "output_index": 0,
                            "summary_index": 0,
                            "delta": delta
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("response.reasoning_summary_text.delta")
                                .data(payload.to_string())))
                            .await;
                    }
                    RoutedStreamEvent::ToolCallDelta { id, delta } => {
                        let payload = json!({
                            "type": "response.function_call_arguments.delta",
                            "item_id": id,
                            "output_index": 1,
                            "delta": delta
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("response.function_call_arguments.delta")
                                .data(payload.to_string())))
                            .await;
                    }
                    RoutedStreamEvent::Finish(sum) => {
                        let full_resp = build_completed_response_json(
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
                        let done_ev = json!({
                            "type": "response.completed",
                            "response": full_resp
                        });
                        let _ = tx
                            .send(Ok(Event::default()
                                .event("response.completed")
                                .data(done_ev.to_string())))
                            .await;
                        completed_summary = Some(sum);
                    }
                    RoutedStreamEvent::Error(err_msg) => {
                        stream_err = Some(err_msg.clone());
                        let fail_ev = json!({
                            "type": "error",
                            "error": { "message": err_msg }
                        });
                        let _ = tx
                            .send(Ok(Event::default().event("error").data(fail_ev.to_string())))
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
                    .map(|tc| (tc.tool.id.clone(), tc.tool.name.clone(), tc.input.to_string()))
                    .collect();
                finalize_request_accounting(
                    &state_bg,
                    &principal,
                    budget_guard,
                    response_id,
                    IngressProtocol::Responses,
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
                    response_id,
                    IngressProtocol::Responses,
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
            &header_resp_id,
            &routed.provider_id,
            &routed.resolved_model,
            cache_hit,
            quixigate_core::NanoUsd::ZERO,
            quixigate_core::NanoUsd::ZERO,
        );
        Ok(resp)
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

        let response_id = format!("resp_{}", Uuid::new_v4().simple());
        let created_sec = now_ms() / 1000;
        let oi = build_openinference_inputs(&req, None);
        let raw_req = Some(inner.to_string());

        state.telemetry.inc_active();
        let routed = match state.router.execute_stream(req).await {
            Ok(s) => s,
            Err(err) => {
                state.telemetry.dec_active();
                let _ = budget_guard.commit(quixigate_core::NanoUsd::ZERO);
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
                .map(|tc| (tc.tool.id.clone(), tc.tool.name.clone(), tc.input.to_string()))
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
