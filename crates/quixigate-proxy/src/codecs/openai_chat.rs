//! `POST /v1/chat/completions` — OpenAI Chat Completions JSON & SSE codec.

use std::convert::Infallible;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use quixigate_core::{IngressProtocol, QuixigateError, now_ms};
use quixigate_router::{CanonicalRequest, RoutedStreamEvent};
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

pub async fn handle_chat_completions(
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
        cache_preference: parse_cache_preference(&headers),
    };

    let est_in = estimate_request_tokens(&req);
    let (est_total, est_cost) = state.router.estimate_preflight_budget(
        &requested_model,
        est_in,
        u64::from(max_out.unwrap_or(1024)),
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
    let request_id = format!("chatcmpl_{}", Uuid::new_v4().simple());
    let created_sec = now_ms() / 1000;

    // Zero-cost instant `mock_response` short-circuit (`good.md` §21) for fast integration testing
    if let Some(mock_text) = body["mock_response"].as_str() {
        let mut guard_req = req.clone();
        quixigate_router::guardrails::apply_request_guardrails(
            &mut guard_req,
            &state.db.get_settings()?,
        )?;
        let out_tokens = ((mock_text.len() / 4) as u64).max(1);
        let zero = quixigate_core::NanoUsd(0);
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
                axum::http::StatusCode::OK,
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

    let session_id = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let oi = build_openinference_inputs(&req, session_id);
    let raw_req_str = Some(body.to_string());
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
                    quixigate_core::NanoUsd::ZERO,
                    quixigate_core::NanoUsd::ZERO,
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
                    quixigate_core::NanoUsd::ZERO,
                    quixigate_core::NanoUsd::ZERO,
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

        let resolved_model = routed.resolved_model.clone();
        let provider_id = routed.provider_id.clone();
        let cache_hit = routed.cache_hit;
        let req_id_for_headers = request_id.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(64);
        let state_bg = state.clone();

        tokio::spawn(async move {
            let _conc = conc_guard;
            let mut events = routed.events;
            let mut tool_index_by_id = std::collections::HashMap::<String, usize>::new();
            let mut next_tool_idx = 0usize;
            let mut completed_summary = None;
            let mut stream_err = None;

            // Emit initial assistant role chunk
            let init_chunk = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_sec,
                "model": resolved_model,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant" },
                    "finish_reason": Value::Null
                }]
            });
            let _ = tx
                .send(Ok(Event::default().data(init_chunk.to_string())))
                .await;

            while let Some(ev) = events.next().await {
                match ev {
                    RoutedStreamEvent::TextDelta(delta) => {
                        if delta.is_empty() {
                            continue;
                        }
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_sec,
                            "model": resolved_model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": delta },
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
                    RoutedStreamEvent::ReasoningDelta(delta) => {
                        if delta.is_empty() {
                            continue;
                        }
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_sec,
                            "model": resolved_model,
                            "choices": [{
                                "index": 0,
                                "delta": { "reasoning_content": delta },
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
                    RoutedStreamEvent::ToolCallStart { id, name } => {
                        let idx = *tool_index_by_id.entry(id.clone()).or_insert_with(|| {
                            let i = next_tool_idx;
                            next_tool_idx += 1;
                            i
                        });
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_sec,
                            "model": resolved_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": idx,
                                        "id": id,
                                        "type": "function",
                                        "function": { "name": name, "arguments": "" }
                                    }]
                                },
                                "finish_reason": Value::Null
                            }]
                        });
                        let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    }
                    RoutedStreamEvent::ToolCallDelta { id, delta } => {
                        let idx = *tool_index_by_id.entry(id.clone()).or_insert_with(|| {
                            let i = next_tool_idx;
                            next_tool_idx += 1;
                            i
                        });
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_sec,
                            "model": resolved_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": idx,
                                        "function": { "arguments": delta }
                                    }]
                                },
                                "finish_reason": Value::Null
                            }]
                        });
                        let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    }
                    RoutedStreamEvent::ToolCallReady(tc) => {
                        if !tool_index_by_id.contains_key(&tc.tool.id) {
                            let idx = next_tool_idx;
                            next_tool_idx += 1;
                            tool_index_by_id.insert(tc.tool.id.clone(), idx);
                            let chunk = json!({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_sec,
                                "model": resolved_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": idx,
                                            "id": tc.tool.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.tool.name,
                                                "arguments": tc.input.to_string()
                                            }
                                        }]
                                    },
                                    "finish_reason": Value::Null
                                }]
                            });
                            let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                        }
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
                        let final_chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_sec,
                            "model": summary.resolved_model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }],
                            "usage": {
                                "prompt_tokens": summary.input_tokens,
                                "completion_tokens": summary.output_tokens,
                                "total_tokens": summary.input_tokens + summary.output_tokens,
                                "prompt_tokens_details": {
                                    "cached_tokens": summary.cached_input_tokens,
                                },
                                "completion_tokens_details": {
                                    "reasoning_tokens": summary.reasoning_tokens,
                                }
                            }
                        });
                        let _ = tx
                            .send(Ok(Event::default().data(final_chunk.to_string())))
                            .await;
                        completed_summary = Some(summary);
                    }
                    RoutedStreamEvent::Error(err_msg) => {
                        stream_err = Some(err_msg.clone());
                        let err_chunk = json!({
                            "error": { "message": err_msg, "type": "upstream_error" }
                        });
                        let _ = tx
                            .send(Ok(Event::default().data(err_chunk.to_string())))
                            .await;
                        break;
                    }
                    _ => {}
                }
            }

            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
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
                let synth_resp = json!({
                    "content": sum.text,
                    "reasoning": sum.reasoning,
                })
                .to_string();
                finalize_request_accounting(
                    &state_bg,
                    &principal,
                    budget_guard,
                    request_id,
                    IngressProtocol::ChatCompletions,
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
                    raw_req_str,
                    Some(synth_resp),
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
                    IngressProtocol::ChatCompletions,
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
                    raw_req_str,
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
            &req_id_for_headers,
            &routed.provider_id,
            &routed.resolved_model,
            cache_hit,
            quixigate_core::NanoUsd::ZERO,
            quixigate_core::NanoUsd::ZERO,
        );
        Ok(resp)
    }
}
