//! Pure-Rust local multi-vendor upstream simulator + OpenInference OTLP `/v1/traces` collector
//! used for deterministic local demos and full-stack interactive UI verification (zero Python).

use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use parking_lot::Mutex;
use prost::Message as ProstMessage;
use quixigate_core::{
    AuthKind, CredentialSecret, ModelRoute, PoolStrategy, QuotaSnapshot, QuotaWindow, RouteTarget,
    now_ms,
};
use quixigate_db::Database;
use quixigate_vault::Vault;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;

#[derive(Clone, Default)]
pub struct DemoCollectorState {
    spans: Arc<Mutex<Vec<Value>>>,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    q: Option<String>,
}

/// Seeds a data directory with realistic multi-seat OAuth + API credentials and routes
/// pointing to `http://127.0.0.1:{upstream_port}` so every UI screen can execute live
/// streaming completions, reasoning traces, `:online` tool loops, quota polls, and failovers.
pub fn seed_demo_data_dir(data_dir: &Path, upstream_port: u16) -> anyhow::Result<()> {
    let vault = Vault::load_or_init(data_dir)?;
    let db = Database::open(&data_dir.join("quixigate.db"), vault)?;
    let base = format!("http://127.0.0.1:{upstream_port}");
    let now = now_ms();

    // Seat 1: ChatGPT Plus / Pro Codex OAuth seat (priority 10, $0 marginal cost)
    let codex = db.upsert_credential(
        Some("seat_codex_primary"),
        "openai-codex",
        "ChatGPT Pro Seat #1 (Primary)",
        AuthKind::Oauth,
        Some(&format!("{base}/codex")),
        10,
        true,
        false,
        &CredentialSecret::Codex {
            access: "eyJhbGciOiJSUzI1NiJ9.demo-codex-seat-1".to_owned(),
            refresh: "rt_demo_codex_1".to_owned(),
            expires_ms: now + 86_400_000,
            account_id: "acct_quixigate_pro_01".to_owned(),
        },
    )?;
    db.save_quota_snapshot(
        &codex.id,
        &QuotaSnapshot {
            windows: vec![
                QuotaWindow {
                    label: "5h Window".to_owned(),
                    used_percent: 34.0,
                    reset_at_ms: Some(now + 3 * 3_600_000),
                },
                QuotaWindow {
                    label: "7d Window".to_owned(),
                    used_percent: 19.5,
                    reset_at_ms: Some(now + 5 * 86_400_000),
                },
            ],
            plan: Some("pro".to_owned()),
            balance_credits: None,
            fetched_at_ms: now,
            error: None,
        },
    )?;

    // Seat 2: Claude Code Setup Token seat (priority 15, $0 marginal cost)
    let claude = db.upsert_credential(
        Some("seat_claude_max"),
        "anthropic-token",
        "Claude Max Setup Token #1",
        AuthKind::SetupToken,
        Some(&base),
        15,
        true,
        true,
        &CredentialSecret::AnthropicSetupToken {
            token: format!("sk-ant-oat01-{}", "a".repeat(84)),
        },
    )?;
    db.save_quota_snapshot(
        &claude.id,
        &QuotaSnapshot {
            windows: vec![
                QuotaWindow {
                    label: "5h Session".to_owned(),
                    used_percent: 48.0,
                    reset_at_ms: Some(now + 2 * 3_600_000),
                },
                QuotaWindow {
                    label: "7d Sonnet".to_owned(),
                    used_percent: 26.0,
                    reset_at_ms: Some(now + 4 * 86_400_000),
                },
            ],
            plan: Some("claude_max".to_owned()),
            balance_credits: None,
            fetched_at_ms: now,
            error: None,
        },
    )?;

    // Seat 3: Paid OpenAI API Overflow Key (priority 200)
    db.upsert_credential(
        Some("seat_openai_paid"),
        "openaicompatible",
        "OpenAI Paid API Key (Overflow)",
        AuthKind::ApiKey,
        Some(&base),
        200,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: "sk-proj-demo-paid-overflow-key".to_owned(),
        },
    )?;

    // Seat 4: Paid Anthropic API Key (priority 210)
    db.upsert_credential(
        Some("seat_anthropic_paid"),
        "anthropic",
        "Anthropic Console API Key",
        AuthKind::ApiKey,
        Some(&base),
        210,
        true,
        true,
        &CredentialSecret::ApiKey {
            key: "sk-ant-api03-demo-paid-overflow".to_owned(),
        },
    )?;

    // Pre-seed a flagship route alias `fast-reasoner`
    db.upsert_model_route(&ModelRoute {
        alias: "fast-reasoner".to_owned(),
        targets: vec![
            RouteTarget {
                model: "anthropic/claude-sonnet-4-5".to_owned(),
                provider_id: None,
                credential_id: None,
            },
            RouteTarget {
                model: "openai/gpt-5".to_owned(),
                provider_id: None,
                credential_id: None,
            },
        ],
        pool_strategy: PoolStrategy::FillFirst,
        allow_paid_overflow: true,
        blocked: false,
        input_nanos_per_m: None,
        output_nanos_per_m: None,
        cache_read_nanos_per_m: None,
        team_id: None,
        created_by: Some("system".to_owned()),
        updated_by: Some("system".to_owned()),
        created_at_ms: Some(now),
        updated_at_ms: now,
    })?;

    Ok(())
}

pub async fn run_demo_servers(
    upstream_port: u16,
    otlp_port: u16,
    seed_dir: Option<&Path>,
) -> anyhow::Result<()> {
    if let Some(dir) = seed_dir {
        seed_demo_data_dir(dir, upstream_port)?;
        println!("Seeded demo data directory at {}", dir.display());
    }

    let collector_state = DemoCollectorState::default();

    let upstream_app = Router::new()
        .route("/v1/chat/completions", post(handle_mock_openai_chat))
        .route("/chat/completions", post(handle_mock_openai_chat))
        .route("/v1/messages", post(handle_mock_anthropic_messages))
        .route("/messages", post(handle_mock_anthropic_messages))
        .route("/v1/embeddings", post(handle_mock_embeddings))
        .route("/embeddings", post(handle_mock_embeddings))
        .route("/v1/responses", post(handle_mock_codex_responses))
        .route("/responses", post(handle_mock_codex_responses))
        .route("/codex/responses", post(handle_mock_codex_responses))
        .route("/wham/usage", get(handle_mock_wham_usage))
        .route("/api/oauth/usage", get(handle_mock_anthropic_usage))
        .route(
            "/v1internal:retrieveUserQuota",
            post(handle_mock_gemini_quota),
        )
        .route("/api/v1/credits", get(handle_mock_openrouter_credits))
        .route("/api/tags", get(handle_mock_ollama_tags))
        .route("/api/ps", get(handle_mock_ollama_ps))
        .route("/v1/models", get(handle_mock_ollama_v1_models))
        .route("/models", get(handle_mock_ollama_v1_models))
        .route("/search", get(handle_mock_web_search));

    let otlp_app = Router::new()
        .route("/", get(handle_phoenix_home))
        .route("/v1/traces", post(handle_otlp_traces))
        .route("/api/spans", get(handle_list_captured_spans))
        .with_state(collector_state);

    let upstream_listener = TcpListener::bind(("127.0.0.1", upstream_port)).await?;
    let otlp_listener = TcpListener::bind(("127.0.0.1", otlp_port)).await?;

    println!(
        "QuixiGate pure-Rust demo upstreams listening on http://127.0.0.1:{upstream_port} | OTLP Phoenix collector on http://127.0.0.1:{otlp_port}"
    );

    tokio::select! {
        res = axum::serve(upstream_listener, upstream_app) => { res?; }
        res = axum::serve(otlp_listener, otlp_app) => { res?; }
        _ = tokio::signal::ctrl_c() => {}
    }

    Ok(())
}

fn extract_last_prompt(body: &Value) -> (String, bool) {
    let mut last_user = String::new();
    let mut has_tool_result = false;

    if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
        for m in msgs {
            let role = m.get("role").and_then(Value::as_str).unwrap_or_default();
            if role == "tool" {
                has_tool_result = true;
            }
            if role == "user" {
                if let Some(s) = m.get("content").and_then(Value::as_str) {
                    last_user = s.to_owned();
                } else if let Some(arr) = m.get("content").and_then(Value::as_array) {
                    for part in arr {
                        if part.get("type").and_then(Value::as_str) == Some("tool_result") {
                            has_tool_result = true;
                        }
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            last_user = t.to_owned();
                        }
                    }
                }
            }
        }
    }
    if let Some(inp) = body.get("input").and_then(Value::as_array) {
        for item in inp {
            if item.get("type").and_then(Value::as_str) == Some("function_call_output") {
                has_tool_result = true;
            }
            if item.get("role").and_then(Value::as_str) == Some("user")
                && let Some(arr) = item.get("content").and_then(Value::as_array)
            {
                for part in arr {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        last_user = t.to_owned();
                    }
                }
            }
        }
    }

    (last_user, has_tool_result)
}

async fn handle_mock_openai_chat(Json(body): Json<Value>) -> Response {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("gpt-5")
        .to_owned();
    let stream_flag = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let (prompt, has_tool_result) = extract_last_prompt(&body);

    let offered_web_search = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter().any(|t| {
                t.pointer("/function/name")
                    .and_then(Value::as_str)
                    .or_else(|| t.get("name").and_then(Value::as_str))
                    == Some("quixigate_web_search")
            })
        })
        .unwrap_or(false);

    if offered_web_search && !has_tool_result {
        let q = if prompt.is_empty() {
            "QuixiGate Rust AI gateway".to_owned()
        } else {
            prompt.clone()
        };
        let args_str = json!({ "query": q }).to_string();
        if stream_flag {
            let chunks = vec![
                json!({
                    "id": "chatcmpl-demo-tool",
                    "object": "chat.completion.chunk",
                    "created": 1735689600,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_web_1",
                                "type": "function",
                                "function": { "name": "quixigate_web_search", "arguments": args_str }
                            }]
                        },
                        "finish_reason": null
                    }]
                })
                .to_string(),
                json!({
                    "id": "chatcmpl-demo-tool",
                    "object": "chat.completion.chunk",
                    "created": 1735689600,
                    "model": model,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "tool_calls" }],
                    "usage": { "prompt_tokens": 38, "completion_tokens": 16, "total_tokens": 54 }
                })
                .to_string(),
                "[DONE]".to_owned(),
            ];
            let s = stream::iter(
                chunks
                    .into_iter()
                    .map(|d| Ok::<Event, Infallible>(Event::default().data(d))),
            );
            return Sse::new(s).into_response();
        }
    }

    let reply = if has_tool_result {
        format!(
            "[{model} + :online search] Grounded synthesis for \"{prompt}\": QuixiGate routes across 9 OAuth seats, 73 aisdk providers, and local VRAM with exact NanoUsd accounting."
        )
    } else {
        format!(
            "[{model}] Verified live response for \"{prompt}\" — routed cleanly through QuixiGate's pure-Rust aisdk engine with zero marginal seat cost."
        )
    };

    if !stream_flag {
        return Json(json!({
            "id": "chatcmpl-demo-1",
            "object": "chat.completion",
            "created": 1735689600,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": format!("Evaluating route and token budget for '{prompt}'..."),
                    "content": reply
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 34,
                "total_tokens": 76
            }
        }))
        .into_response();
    }

    let words: Vec<String> = reply.split_whitespace().map(str::to_owned).collect();
    let s = async_stream_openai_words(model, prompt, words);
    Sse::new(s).into_response()
}

fn async_stream_openai_words(
    model: String,
    prompt: String,
    words: Vec<String>,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    stream::unfold(
        (0usize, model, prompt, words),
        |(step, model, prompt, words)| async move {
            tokio::time::sleep(Duration::from_millis(12)).await;
            if step == 0 {
                let d = json!({
                    "id": "chatcmpl-demo-stream",
                    "object": "chat.completion.chunk",
                    "created": 1735689600,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": format!("Inspecting sticky seat pool & NanoUsd ledger for \"{prompt}\".")
                        },
                        "finish_reason": null
                    }]
                })
                .to_string();
                return Some((Ok(Event::default().data(d)), (1, model, prompt, words)));
            }
            let word_idx = step - 1;
            if word_idx < words.len() {
                let piece = if word_idx == 0 {
                    words[word_idx].clone()
                } else {
                    format!(" {}", words[word_idx])
                };
                let d = json!({
                    "id": "chatcmpl-demo-stream",
                    "object": "chat.completion.chunk",
                    "created": 1735689600,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": piece },
                        "finish_reason": null
                    }]
                })
                .to_string();
                return Some((
                    Ok(Event::default().data(d)),
                    (step + 1, model, prompt, words),
                ));
            }
            if word_idx == words.len() {
                let d = json!({
                    "id": "chatcmpl-demo-stream",
                    "object": "chat.completion.chunk",
                    "created": 1735689600,
                    "model": model,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                    "usage": {
                        "prompt_tokens": 44,
                        "completion_tokens": (words.len() as u64) + 12,
                        "total_tokens": 56 + (words.len() as u64),
                        "prompt_tokens_details": { "cached_tokens": 12 },
                        "completion_tokens_details": { "reasoning_tokens": 12 }
                    }
                })
                .to_string();
                return Some((
                    Ok(Event::default().data(d)),
                    (step + 1, model, prompt, words),
                ));
            }
            if word_idx == words.len() + 1 {
                return Some((
                    Ok(Event::default().data("[DONE]")),
                    (step + 1, model, prompt, words),
                ));
            }
            None
        },
    )
}

async fn handle_mock_codex_responses(Json(body): Json<Value>) -> Response {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("gpt-5")
        .to_owned();
    let (prompt, has_tool_result) = extract_last_prompt(&body);

    let offered_web_search = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .any(|t| t.get("name").and_then(Value::as_str) == Some("quixigate_web_search"))
        })
        .unwrap_or(false);

    if offered_web_search && !has_tool_result {
        let args = json!({ "query": prompt }).to_string();
        let events = vec![
            (
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "item": { "type": "function_call", "call_id": "call_codex_web", "name": "quixigate_web_search" }
                }),
            ),
            (
                "response.function_call_arguments.delta",
                json!({
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_codex_web",
                    "delta": args
                }),
            ),
            (
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_codex_web",
                        "name": "quixigate_web_search",
                        "arguments": args
                    }
                }),
            ),
            (
                "response.completed",
                json!({
                    "type": "response.completed",
                    "response": {
                        "id": "resp_codex_tool",
                        "status": "completed",
                        "usage": { "input_tokens": 40, "output_tokens": 18, "total_tokens": 58 }
                    }
                }),
            ),
        ];
        let s = stream::iter(events.into_iter().map(|(ev, data)| {
            Ok::<Event, Infallible>(Event::default().event(ev).data(data.to_string()))
        }));
        return Sse::new(s).into_response();
    }

    let reply = if has_tool_result {
        format!(
            "[Codex OAuth {model} + :online] Live web-grounded answer for \"{prompt}\" via ChatGPT Pro subscription ($0 marginal API cost)."
        )
    } else {
        format!(
            "[Codex OAuth {model}] Streamed from ChatGPT Pro seat #1 for \"{prompt}\" — $0.00 billed, API-equivalent savings credited."
        )
    };

    let events = vec![
        (
            "response.reasoning_summary_text.delta",
            json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": format!("Routing '{prompt}' via ChatGPT Pro Codex Responses seat (priority #10).")
            }),
        ),
        (
            "response.output_text.delta",
            json!({ "type": "response.output_text.delta", "delta": reply.clone() }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_codex_demo",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{ "type": "output_text", "text": reply }]
                        }
                    ],
                    "usage": {
                        "input_tokens": 48,
                        "output_tokens": 36,
                        "total_tokens": 84,
                        "input_tokens_details": { "cached_tokens": 16 },
                        "output_tokens_details": { "reasoning_tokens": 14 }
                    }
                }
            }),
        ),
    ];

    let s = stream::unfold(events.into_iter(), |mut iter| async move {
        tokio::time::sleep(Duration::from_millis(18)).await;
        iter.next().map(|(ev, d)| {
            (
                Ok::<Event, Infallible>(Event::default().event(ev).data(d.to_string())),
                iter,
            )
        })
    });
    Sse::new(s).into_response()
}

async fn handle_mock_anthropic_messages(Json(body): Json<Value>) -> Response {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("claude-sonnet-4-5")
        .to_owned();
    let (prompt, has_tool_result) = extract_last_prompt(&body);

    let offered_web_search = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .any(|t| t.get("name").and_then(Value::as_str) == Some("quixigate_web_search"))
        })
        .unwrap_or(false);

    if offered_web_search && !has_tool_result {
        let args = json!({ "query": prompt }).to_string();
        let events = vec![
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": { "id": "msg_ant_tool", "model": model, "usage": { "input_tokens": 42, "output_tokens": 1 } }
                }),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "tool_use", "id": "toolu_web_1", "name": "quixigate_web_search", "input": {} }
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "input_json_delta", "partial_json": args }
                }),
            ),
            (
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "tool_use" },
                    "usage": { "output_tokens": 19 }
                }),
            ),
            ("message_stop", json!({ "type": "message_stop" })),
        ];
        let s = stream::iter(events.into_iter().map(|(ev, d)| {
            Ok::<Event, Infallible>(Event::default().event(ev).data(d.to_string()))
        }));
        return Sse::new(s).into_response();
    }

    let stream_flag = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let reply = if has_tool_result {
        format!(
            "[Claude Max {model} + :online] Synthesized live search results for \"{prompt}\" over Claude Code setup-token seat."
        )
    } else {
        format!(
            "[Claude Max {model}] Streamed live response for \"{prompt}\" via Claude Code OAuth seat (#15) with extended thinking & $0 marginal cost."
        )
    };

    if !stream_flag {
        return Json(json!({
            "id": "msg_demo_claude_sync",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {
                    "type": "thinking",
                    "thinking": format!("Planning structured answer for \"{prompt}\"...")
                },
                {
                    "type": "text",
                    "text": reply
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 52,
                "cache_read_input_tokens": 18,
                "output_tokens": 39
            }
        }))
        .into_response();
    }

    let events = vec![
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_demo_claude",
                    "model": model,
                    "usage": { "input_tokens": 52, "cache_read_input_tokens": 18, "output_tokens": 1 }
                }
            }),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "thinking", "thinking": "" }
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": format!("Planning structured answer for \"{prompt}\" using Claude Max subscription seat...")
                }
            }),
        ),
        (
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 0 }),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": { "type": "text", "text": "" }
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "text_delta", "text": reply }
            }),
        ),
        (
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 1 }),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 39 }
            }),
        ),
        ("message_stop", json!({ "type": "message_stop" })),
    ];

    let s = stream::unfold(events.into_iter(), |mut iter| async move {
        tokio::time::sleep(Duration::from_millis(16)).await;
        iter.next().map(|(ev, d)| {
            (
                Ok::<Event, Infallible>(Event::default().event(ev).data(d.to_string())),
                iter,
            )
        })
    });
    Sse::new(s).into_response()
}

async fn handle_mock_embeddings(Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("text-embedding-3-small");
    let count = match body.get("input") {
        Some(Value::Array(a)) => a.len().max(1),
        _ => 1,
    };
    let data: Vec<Value> = (0..count)
        .map(|i| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": [0.125, -0.375, 0.625, 0.875, -0.25, 0.5, 0.1, -0.2]
            })
        })
        .collect();
    Json(json!({
        "object": "list",
        "model": model,
        "data": data,
        "usage": { "prompt_tokens": count * 6, "total_tokens": count * 6 }
    }))
}

async fn handle_mock_wham_usage() -> Json<Value> {
    let now_sec = now_ms() / 1000;
    Json(json!({
        "plan_type": "pro",
        "rate_limit": {
            "primary_window": {
                "used_percent": 36.5,
                "limit_window_seconds": 18000,
                "reset_at": now_sec + 9800
            },
            "secondary_window": {
                "used_percent": 21.0,
                "limit_window_seconds": 604800,
                "reset_at": now_sec + 410000
            }
        }
    }))
}

async fn handle_mock_anthropic_usage() -> Json<Value> {
    let reset_5h = (chrono::Utc::now() + chrono::Duration::hours(2)).to_rfc3339();
    let reset_7d = (chrono::Utc::now() + chrono::Duration::days(4)).to_rfc3339();
    Json(json!({
        "five_hour": { "utilization": 49.0, "resets_at": reset_5h },
        "seven_day": { "utilization": 27.5, "resets_at": reset_7d }
    }))
}

async fn handle_mock_gemini_quota() -> Json<Value> {
    let reset = (chrono::Utc::now() + chrono::Duration::hours(12)).to_rfc3339();
    Json(json!({
        "buckets": [
            { "modelId": "gemini-2.5-pro", "remainingFraction": 0.82, "resetTime": reset }
        ]
    }))
}

async fn handle_mock_openrouter_credits() -> Json<Value> {
    Json(json!({
        "data": { "total_credits": 50.0, "total_usage": 12.4 }
    }))
}

async fn handle_mock_ollama_tags() -> Json<Value> {
    Json(json!({
        "models": [
            { "name": "qwen2.5-coder:32b" },
            { "name": "deepseek-r1:14b" }
        ]
    }))
}

async fn handle_mock_ollama_ps() -> Json<Value> {
    Json(json!({
        "models": [
            { "name": "qwen2.5-coder:32b" }
        ]
    }))
}

async fn handle_mock_ollama_v1_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            { "id": "qwen2.5-coder:32b", "object": "model" },
            { "id": "deepseek-r1:14b", "object": "model" }
        ]
    }))
}

async fn handle_mock_web_search(Query(params): Query<SearchParams>) -> Json<Value> {
    let q = params.q.unwrap_or_else(|| "QuixiGate".to_owned());
    Json(json!({
        "Heading": format!("Live Search: {q}"),
        "AbstractText": format!(
            "QuixiGate is a 100% pure-Rust AI gateway built on aisdk 0.6.4 with 9 OAuth flows, NanoUsd accounting, and Arize Phoenix OpenInference telemetry (query: {q})."
        ),
        "AbstractURL": "https://quixigate.local/docs",
        "RelatedTopics": [
            {
                "Text": "Supports OpenAI Chat/Responses (SSE + WebSocket), Anthropic Messages, Gemini v1beta, and Vercel AI SDK UIMessageStream.",
                "FirstURL": "https://quixigate.local/docs/codecs"
            }
        ]
    }))
}

async fn handle_phoenix_home(State(state): State<DemoCollectorState>) -> Json<Value> {
    let spans = state.spans.lock();
    Json(json!({
        "service": "arize-phoenix-otlp-collector",
        "status": "ok",
        "captured_spans": spans.len(),
    }))
}

async fn handle_otlp_traces(
    State(state): State<DemoCollectorState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();

    if ct.contains("protobuf") {
        match ExportTraceServiceRequest::decode(body.as_ref()) {
            Ok(req) => {
                let mut guard = state.spans.lock();
                for rs in req.resource_spans {
                    for ss in rs.scope_spans {
                        for sp in ss.spans {
                            let attrs: serde_json::Map<String, Value> = sp
                                .attributes
                                .iter()
                                .map(|kv| (kv.key.clone(), json!(format!("{:?}", kv.value))))
                                .collect();
                            guard.push(json!({
                                "name": sp.name,
                                "trace_id_len": sp.trace_id.len(),
                                "attributes": attrs,
                            }));
                        }
                    }
                }
                (StatusCode::OK, Json(json!({ "partialSuccess": {} })))
            }
            Err(e) => (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid OTLP protobuf: {e}") })),
            ),
        }
    } else {
        let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
        state.spans.lock().push(parsed);
        (StatusCode::OK, Json(json!({ "partialSuccess": {} })))
    }
}

async fn handle_list_captured_spans(State(state): State<DemoCollectorState>) -> Json<Value> {
    let spans = state.spans.lock().clone();
    Json(json!({
        "count": spans.len(),
        "spans": spans,
    }))
}
