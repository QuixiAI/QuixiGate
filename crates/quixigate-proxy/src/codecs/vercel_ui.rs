//! `POST /api/chat` — Vercel AI SDK v5/v6 `UIMessageStream` (`x-vercel-ai-ui-message-stream: v1`)
//! powering QuixiGate's `@ai-sdk/react` `useChat` Playground and Multi-Model Arena.

use std::convert::Infallible;
use std::time::Instant;

use aisdk::core::language_model::LanguageModelResponseContentType;
use aisdk::core::{AssistantMessage, Message, SystemMessage, UserMessage};
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use quixigate_core::{IngressProtocol, QuixigateError};
use quixigate_router::{CanonicalRequest, RoutedStreamEvent};
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

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

    let base_model = body["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Validation("missing 'model' in /api/chat request".to_owned()))?;
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

    let message_id = format!("msg_{}", Uuid::new_v4().simple());
    let oi = build_openinference_inputs(&req, body["id"].as_str().map(str::to_owned));
    let raw_req = Some(body.to_string());
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
    let hdr_msg_id = message_id.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(64);
    let state_bg = state.clone();

    tokio::spawn(async move {
        let _conc = conc_guard;
        let mut events = routed.events;
        let mut text_part_id = format!("txt_{}", Uuid::new_v4().simple());
        let mut rsn_part_id = format!("rsn_{}", Uuid::new_v4().simple());
        let mut text_open = false;
        let mut rsn_open = false;
        let mut started_tool_ids = std::collections::HashSet::<String>::new();
        let mut completed_summary = None;
        let mut stream_err = None;

        let send_json = |tx: &tokio::sync::mpsc::Sender<std::result::Result<Event, Infallible>>,
                         val: Value| {
            let tx = tx.clone();
            async move { tx.send(Ok(Event::default().data(val.to_string()))).await }
        };

        let _ = send_json(
            &tx,
            json!({
                "type": "start",
                "messageId": message_id,
                "messageMetadata": {
                    "model": resolved_model,
                    "provider": provider_id,
                    "cacheHit": cache_hit,
                }
            }),
        )
        .await;
        let _ = send_json(&tx, json!({ "type": "start-step" })).await;

        while let Some(ev) = events.next().await {
            match ev {
                RoutedStreamEvent::ReasoningDelta(delta) => {
                    if delta.is_empty() {
                        continue;
                    }
                    if !rsn_open {
                        let _ = send_json(
                            &tx,
                            json!({ "type": "reasoning-start", "id": rsn_part_id }),
                        )
                        .await;
                        rsn_open = true;
                    }
                    let _ = send_json(
                        &tx,
                        json!({
                            "type": "reasoning-delta",
                            "id": rsn_part_id,
                            "delta": delta
                        }),
                    )
                    .await;
                }
                RoutedStreamEvent::ReasoningEnd => {
                    if rsn_open {
                        let _ = send_json(
                            &tx,
                            json!({ "type": "reasoning-end", "id": rsn_part_id }),
                        )
                        .await;
                        rsn_open = false;
                    }
                }
                RoutedStreamEvent::TextDelta(delta) => {
                    if delta.is_empty() {
                        continue;
                    }
                    if rsn_open {
                        let _ = send_json(
                            &tx,
                            json!({ "type": "reasoning-end", "id": rsn_part_id }),
                        )
                        .await;
                        rsn_open = false;
                    }
                    if !text_open {
                        let _ =
                            send_json(&tx, json!({ "type": "text-start", "id": text_part_id }))
                                .await;
                        text_open = true;
                    }
                    if send_json(
                        &tx,
                        json!({
                            "type": "text-delta",
                            "id": text_part_id,
                            "delta": delta
                        }),
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                }
                RoutedStreamEvent::ToolCallStart { id, name } => {
                    started_tool_ids.insert(id.clone());
                    let _ = send_json(
                        &tx,
                        json!({
                            "type": "tool-input-start",
                            "toolCallId": id,
                            "toolName": name
                        }),
                    )
                    .await;
                }
                RoutedStreamEvent::ToolCallDelta { id, delta } => {
                    if started_tool_ids.contains(&id) {
                        let _ = send_json(
                            &tx,
                            json!({
                                "type": "tool-input-delta",
                                "toolCallId": id,
                                "inputTextDelta": delta
                            }),
                        )
                        .await;
                    }
                }
                RoutedStreamEvent::ToolCallReady(tc) => {
                    let _ = send_json(
                        &tx,
                        json!({
                            "type": "tool-input-available",
                            "toolCallId": tc.tool.id,
                            "toolName": tc.tool.name,
                            "input": tc.input
                        }),
                    )
                    .await;
                }
                RoutedStreamEvent::ServerToolResult {
                    id,
                    name,
                    input,
                    output,
                } => {
                    if rsn_open {
                        let _ = send_json(
                            &tx,
                            json!({ "type": "reasoning-end", "id": rsn_part_id }),
                        )
                        .await;
                        rsn_open = false;
                    }
                    if text_open {
                        let _ =
                            send_json(&tx, json!({ "type": "text-end", "id": text_part_id })).await;
                        text_open = false;
                    }
                    let parsed_output = serde_json::from_str::<Value>(&output)
                        .unwrap_or(Value::String(output));
                    let _ = send_json(
                        &tx,
                        json!({
                            "type": "tool-input-available",
                            "toolCallId": id,
                            "toolName": name,
                            "input": input,
                            "providerExecuted": true
                        }),
                    )
                    .await;
                    let _ = send_json(
                        &tx,
                        json!({
                            "type": "tool-output-available",
                            "toolCallId": id,
                            "output": parsed_output,
                            "providerExecuted": true
                        }),
                    )
                    .await;
                    let _ = send_json(&tx, json!({ "type": "finish-step" })).await;
                    let _ = send_json(&tx, json!({ "type": "start-step" })).await;
                    text_part_id = format!("txt_{}", Uuid::new_v4().simple());
                    rsn_part_id = format!("rsn_{}", Uuid::new_v4().simple());
                }
                RoutedStreamEvent::Finish(sum) => {
                    if rsn_open {
                        let _ = send_json(
                            &tx,
                            json!({ "type": "reasoning-end", "id": rsn_part_id }),
                        )
                        .await;
                        rsn_open = false;
                    }
                    if text_open {
                        let _ =
                            send_json(&tx, json!({ "type": "text-end", "id": text_part_id })).await;
                        text_open = false;
                    }
                    let _ = send_json(&tx, json!({ "type": "finish-step" })).await;
                    let _ = send_json(
                        &tx,
                        json!({
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
                                "cacheHit": cache_hit,
                                "fallbackAttempts": sum.fallback_attempts,
                            }
                        }),
                    )
                    .await;
                    let _ = send_json(&tx, json!({ "type": "finish", "finishReason": "stop" })).await;
                    completed_summary = Some(sum);
                }
                RoutedStreamEvent::Error(err_msg) => {
                    stream_err = Some(err_msg.clone());
                    let _ =
                        send_json(&tx, json!({ "type": "error", "errorText": err_msg })).await;
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
                .map(|tc| (tc.tool.id.clone(), tc.tool.name.clone(), tc.input.to_string()))
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
                message_id,
                IngressProtocol::VercelUi,
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
                message_id,
                IngressProtocol::VercelUi,
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
    resp.headers_mut().insert(
        "x-vercel-ai-ui-message-stream",
        HeaderValue::from_static("v1"),
    );
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
