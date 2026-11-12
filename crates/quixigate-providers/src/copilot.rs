//! Custom `aisdk::core::LanguageModel` for GitHub Copilot OAuth sessions
//! (`https://api.individual.githubcopilot.com/chat/completions`) and custom
//! OpenAI-Chat-Completions SSE endpoints requiring custom headers or optional auth.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use aisdk::core::capabilities::{
    ImageInputSupport, ReasoningSupport, StructuredOutputSupport, TextInputSupport,
    TextOutputSupport, ToolCallSupport,
};
use aisdk::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream, ReasoningEffort, Usage,
};
use aisdk::core::tools::ToolDetails;
use aisdk::core::{AssistantMessage, LanguageModel, Message, Tool, ToolCallInfo};
use aisdk::extensions::Extensions;
use aisdk::{Error, Result};
use async_trait::async_trait;
use quixigate_core::NanoUsd;
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use crate::sse::{self, SseDecoder};

#[cfg(feature = "subscription-oauth")]
pub const DEFAULT_COPILOT_BASE_URL: &str = "https://api.individual.githubcopilot.com";

/// Concrete upstream resolution metadata captured from OpenRouter / OpenAI-compatible responses
/// (e.g. concrete `model` resolved by `@preset/<slug>`, `openrouter/auto`, or `~*-latest`, and
/// exact `usage.cost` reported by OpenRouter).
#[derive(Debug, Clone, Default)]
pub struct UpstreamResponseMeta {
    pub resolved_model: Option<String>,
    pub exact_cost_nanos: Option<NanoUsd>,
    pub finish_reason: Option<String>,
}

#[derive(Clone)]
pub struct GitHubCopilotModel {
    model: String,
    session_token: String,
    base_url: String,
    tools: Vec<Tool>,
    copilot_headers: bool,
    default_headers: HashMap<String, String>,
    default_body: Option<Map<String, Value>>,
    upstream_meta: Arc<Mutex<UpstreamResponseMeta>>,
    http: reqwest::Client,
}

impl GitHubCopilotModel {
    #[cfg(feature = "subscription-oauth")]
    pub fn new(model: impl Into<String>, session_token: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            session_token: session_token.into(),
            base_url: DEFAULT_COPILOT_BASE_URL.to_owned(),
            tools: Vec::new(),
            copilot_headers: true,
            default_headers: HashMap::new(),
            default_body: None,
            upstream_meta: Arc::new(Mutex::new(UpstreamResponseMeta::default())),
            http: crate::build_inference_http_client(),
        }
    }

    /// Creates an OpenAI-compatible Chat Completions `LanguageModel` without Copilot-specific headers
    /// (used for local runtimes like Ollama / LM Studio / llama.cpp / vLLM and for OpenRouter so
    /// reasoning deltas, `@preset/*` concrete model resolution, and exact `usage.cost` are preserved).
    pub fn openai_compatible(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            model: model.into(),
            session_token: api_key.unwrap_or_default(),
            base_url: base_url.into(),
            tools: Vec::new(),
            copilot_headers: false,
            default_headers: HashMap::new(),
            default_body: None,
            upstream_meta: Arc::new(Mutex::new(UpstreamResponseMeta::default())),
            http: crate::build_inference_http_client(),
        }
    }

    pub fn with_http(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let u = base_url.into();
        if !u.trim().is_empty() {
            self.base_url = u;
        }
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_default_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.default_headers = headers;
        self
    }

    pub fn with_default_body(mut self, body: Value) -> Self {
        if let Value::Object(map) = body {
            self.default_body = Some(map);
        }
        self
    }

    pub fn upstream_meta_handle(&self) -> Arc<Mutex<UpstreamResponseMeta>> {
        self.upstream_meta.clone()
    }

    fn completions_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_owned()
        } else {
            format!("{base}/chat/completions")
        }
    }

    async fn send(&self, options: &LanguageModelOptions, stream: bool) -> Result<reqwest::Response> {
        let has_agent_turns = options
            .messages()
            .iter()
            .any(|m| matches!(m, Message::Assistant(_) | Message::Tool(_)));
        let mut body = build_chat_completions_body(&self.model, options, &self.tools, stream);
        if let (Some(defaults), Some(obj)) = (&self.default_body, body.as_object_mut()) {
            for (k, v) in defaults {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        let url = self.completions_url();
        let mut last_send_err = None;
        let mut resp_opt = None;
        for attempt in 0..2 {
            let mut req = self
                .http
                .post(&url)
                .header("Content-Type", "application/json");

            if !self.session_token.trim().is_empty() {
                req = req.bearer_auth(&self.session_token);
            }
            #[cfg(feature = "subscription-oauth")]
            if self.copilot_headers {
                req = req
                    .header("User-Agent", "GitHubCopilotChat/0.35.0")
                    .header("Editor-Version", "vscode/1.107.0")
                    .header("Editor-Plugin-Version", "copilot-chat/0.35.0")
                    .header("Copilot-Integration-Id", "vscode-chat")
                    .header("Openai-Intent", "conversation-edits")
                    .header(
                        "X-Initiator",
                        if has_agent_turns { "agent" } else { "user" },
                    );
            }
            #[cfg(not(feature = "subscription-oauth"))]
            let _ = (has_agent_turns, self.copilot_headers);
            if stream {
                req = req.header("Accept", "text/event-stream");
            }
            for (k, v) in &self.default_headers {
                req = req.header(k, v);
            }
            for (k, v) in options.headers.iter().flatten() {
                req = req.header(k, v);
            }

            match req.json(&body).send().await {
                Ok(r) => {
                    resp_opt = Some(r);
                    break;
                }
                Err(e) if attempt == 0 && crate::is_retryable_stale_socket_error(&e) => {
                    continue;
                }
                Err(e) => {
                    last_send_err = Some(e);
                    break;
                }
            }
        }

        let resp = match resp_opt {
            Some(r) => r,
            None => {
                let e = last_send_err.expect("send loop terminated without response or error");
                let (status_code, why) = crate::describe_reqwest_error(&e);
                return Err(Error::ApiError {
                    status_code,
                    details: format!("Chat completions upstream request failed: {why}"),
                });
            }
        };
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let text = resp.text().await.unwrap_or_default();
        let body_json: Value = serde_json::from_str(&text).unwrap_or_default();
        let code = body_json["error"]["code"]
            .as_str()
            .or(body_json["error"]["type"].as_str())
            .unwrap_or_default();
        let msg = body_json["error"]["message"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| text.chars().take(500).collect());
        let mut details = if !code.is_empty() {
            format!("{code}: {msg}")
        } else {
            format!("HTTP {}: {msg}", status.as_u16())
        };
        if let Some(ra) = retry_after {
            details.push_str(&format!(" [retry_after={ra}]"));
        }
        Err(Error::ApiError {
            status_code: Some(status),
            details,
        })
    }
}

impl fmt::Debug for GitHubCopilotModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GitHubCopilotModel")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("copilot_headers", &self.copilot_headers)
            .field("tools_count", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl TextInputSupport for GitHubCopilotModel {}
impl TextOutputSupport for GitHubCopilotModel {}
impl ToolCallSupport for GitHubCopilotModel {}
impl StructuredOutputSupport for GitHubCopilotModel {}
impl ReasoningSupport for GitHubCopilotModel {}
impl ImageInputSupport for GitHubCopilotModel {}

#[async_trait]
impl LanguageModel for GitHubCopilotModel {
    fn name(&self) -> String {
        self.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let resp = self.send(&options, false).await?;
        let data: Value = resp.json().await.map_err(|e| {
            let (status_code, why) = crate::describe_reqwest_error(&e);
            Error::ApiError {
                status_code,
                details: format!("invalid chat completions JSON response: {why}"),
            }
        })?;
        if data["error"].is_object() {
            let msg = data["error"]["message"]
                .as_str()
                .unwrap_or("Upstream completion error");
            return Err(Error::ApiError {
                status_code: None,
                details: msg.to_owned(),
            });
        }
        record_upstream_meta(&self.upstream_meta, &data);
        let msg = &data["choices"][0]["message"];
        let raw_text = msg["content"].as_str().unwrap_or_default();
        let explicit_reasoning = msg["reasoning_content"]
            .as_str()
            .or(msg["reasoning"].as_str())
            .filter(|s| !s.is_empty());

        let (reasoning_opt, text) = if let Some(r) = explicit_reasoning {
            (Some(r.to_owned()), raw_text.to_owned())
        } else if let Some((inline_r, cleaned_t)) = split_inline_think_block(raw_text) {
            (Some(inline_r), cleaned_t)
        } else {
            (None, raw_text.to_owned())
        };

        let mut contents = Vec::new();
        if let Some(reasoning) = &reasoning_opt {
            contents.push(LanguageModelResponseContentType::Reasoning {
                content: reasoning.clone(),
                extensions: Extensions::default(),
            });
        }
        let tool_calls = msg["tool_calls"].as_array().cloned().unwrap_or_default();
        if !text.is_empty() || tool_calls.is_empty() {
            contents.push(LanguageModelResponseContentType::Text(text.clone()));
        }
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("call_0").to_owned();
            let name = tc["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
            let input = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
            contents.push(LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                tool: ToolDetails { id, name },
                input,
                extensions: Extensions::default(),
            }));
        }
        let raw_usage = parse_openai_usage_with_timings(&data["usage"], &data["timings"]);
        let usage = reconcile_reasoning_usage(
            raw_usage,
            reasoning_opt.as_deref().unwrap_or_default(),
            &text,
        );
        Ok(LanguageModelResponse { contents, usage })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let response = self.send(&options, true).await?;
        let state = CopilotStreamState {
            response,
            decoder: SseDecoder::default(),
            text_started: false,
            reasoning_started: false,
            acc_text: String::new(),
            acc_reasoning: String::new(),
            tool_calls: BTreeMap::new(),
            usage: None,
            stream_cost: None,
            upstream_meta: self.upstream_meta.clone(),
            ready: VecDeque::new(),
            ended: false,
        };
        Ok(Box::pin(futures::stream::unfold(
            state,
            |mut state| async move { state.next().await.map(|item| (item, state)) },
        )))
    }
}

#[derive(Default)]
struct AccToolCall {
    id: String,
    name: String,
    args: String,
    started: bool,
}

struct CopilotStreamState {
    response: reqwest::Response,
    decoder: SseDecoder,
    text_started: bool,
    reasoning_started: bool,
    acc_text: String,
    acc_reasoning: String,
    tool_calls: BTreeMap<usize, AccToolCall>,
    usage: Option<Usage>,
    stream_cost: Option<NanoUsd>,
    upstream_meta: Arc<Mutex<UpstreamResponseMeta>>,
    ready: VecDeque<Result<Vec<LanguageModelStreamChunk>>>,
    ended: bool,
}

impl CopilotStreamState {
    async fn next(&mut self) -> Option<Result<Vec<LanguageModelStreamChunk>>> {
        loop {
            if let Some(item) = self.ready.pop_front() {
                return Some(item);
            }
            if self.ended {
                return None;
            }
            match self.response.chunk().await {
                Ok(Some(bytes)) => match self.decoder.push(&bytes) {
                    Ok(events) => events.iter().for_each(|d| self.on_event(d)),
                    Err(sse::FrameTooLarge) => {
                        self.fail("Chat completions backend sent oversized SSE frame")
                    }
                },
                Ok(None) => {
                    if let Some(last) = self.decoder.finish() {
                        self.on_event(&last);
                    }
                    if !self.ended {
                        self.finish_stream();
                        self.ended = true;
                    }
                }
                Err(err) => {
                    let (_, why) = crate::describe_reqwest_error(&err);
                    self.fail(&format!("Chat completions stream error: {why}"));
                }
            }
        }
    }

    fn on_event(&mut self, data: &str) {
        if self.ended {
            return;
        }
        if data.trim() == "[DONE]" {
            self.finish_stream();
            self.ended = true;
            return;
        }
        let Ok(payload) = serde_json::from_str::<Value>(data) else {
            return;
        };
        if payload["error"].is_object() {
            let msg = payload["error"]["message"]
                .as_str()
                .unwrap_or("Upstream streaming error");
            self.fail(msg);
            return;
        }
        if let Some(m) = payload["model"].as_str().filter(|s| !s.trim().is_empty()) {
            self.upstream_meta.lock().resolved_model = Some(m.to_owned());
        }
        if let Some(fr) = payload["choices"][0]["finish_reason"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
        {
            self.upstream_meta.lock().finish_reason = Some(fr.to_owned());
        }
        if let Some(c) = parse_exact_cost_nanos(&payload["usage"]["cost"]) {
            self.stream_cost = Some(c);
        }
        if let Some(u) = parse_openai_usage_with_timings(&payload["usage"], &payload["timings"]) {
            self.usage = Some(u);
        }
        let delta = &payload["choices"][0]["delta"];
        if !delta.is_object() {
            return;
        }
        let mut chunks = Vec::new();
        if let Some(r) = delta["reasoning_content"]
            .as_str()
            .or(delta["reasoning"].as_str())
            .filter(|s| !s.is_empty())
        {
            if !self.reasoning_started {
                self.reasoning_started = true;
                chunks.push(LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::ReasoningStart,
                ));
            }
            self.acc_reasoning.push_str(r);
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::ReasoningDelta(r.to_owned()),
            ));
        }
        if let Some(c) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            if self.reasoning_started {
                self.reasoning_started = false;
                chunks.push(LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::ReasoningEnd,
                ));
            }
            if !self.text_started {
                self.text_started = true;
                chunks.push(LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::TextStart,
                ));
            }
            self.acc_text.push_str(c);
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::TextDelta(c.to_owned()),
            ));
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let idx = tc["index"].as_u64().unwrap_or(0) as usize;
            let entry = self.tool_calls.entry(idx).or_default();
            if let Some(id) = tc["id"].as_str().filter(|s| !s.is_empty()) {
                entry.id = id.to_owned();
            }
            if let Some(name) = tc["function"]["name"].as_str().filter(|s| !s.is_empty()) {
                entry.name = name.to_owned();
            }
            if !entry.started && !entry.name.is_empty() {
                if entry.id.is_empty() {
                    entry.id = format!("call_{idx}");
                }
                entry.started = true;
                chunks.push(LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::ToolCallStart(ToolDetails {
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                    }),
                ));
            }
            if let Some(arg_delta) = tc["function"]["arguments"]
                .as_str()
                .filter(|s| !s.is_empty())
            {
                entry.args.push_str(arg_delta);
                if entry.started {
                    chunks.push(LanguageModelStreamChunk::Delta(
                        LanguageModelStreamChunkType::ToolCallDelta {
                            id: entry.id.clone(),
                            delta: arg_delta.to_owned(),
                        },
                    ));
                }
            }
        }
        if !chunks.is_empty() {
            self.ready.push_back(Ok(chunks));
        }
    }

    fn finish_stream(&mut self) {
        if let Some(c) = self.stream_cost.take() {
            let mut guard = self.upstream_meta.lock();
            guard.exact_cost_nanos = Some(match guard.exact_cost_nanos {
                Some(prev) => prev.saturating_add(c),
                None => c,
            });
        }
        let mut chunks = Vec::new();
        if self.reasoning_started {
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::ReasoningEnd,
            ));
        }
        if self.text_started {
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::TextEnd,
            ));
        }
        if self.acc_reasoning.is_empty()
            && let Some((inline_r, cleaned_t)) = split_inline_think_block(&self.acc_text)
        {
            self.acc_reasoning = inline_r;
            self.acc_text = cleaned_t;
        }
        let usage = reconcile_reasoning_usage(
            self.usage.take(),
            &self.acc_reasoning,
            &self.acc_text,
        )
        .unwrap_or_default();
        let mut items = Vec::new();
        if !self.acc_reasoning.is_empty() {
            items.push(LanguageModelResponseContentType::Reasoning {
                content: std::mem::take(&mut self.acc_reasoning),
                extensions: Extensions::default(),
            });
        }
        if !self.acc_text.is_empty() || self.tool_calls.is_empty() {
            items.push(LanguageModelResponseContentType::Text(std::mem::take(
                &mut self.acc_text,
            )));
        }
        for (_, tc) in std::mem::take(&mut self.tool_calls) {
            let input = if tc.args.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&tc.args).unwrap_or_else(|_| json!({}))
            };
            items.push(LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                tool: ToolDetails {
                    id: if tc.id.is_empty() {
                        "call_0".to_owned()
                    } else {
                        tc.id
                    },
                    name: tc.name,
                },
                input,
                extensions: Extensions::default(),
            }));
        }
        for (idx, content) in items.into_iter().enumerate() {
            chunks.push(LanguageModelStreamChunk::Done(AssistantMessage {
                content,
                usage: if idx == 0 { Some(usage.clone()) } else { None },
            }));
        }
        if !chunks.is_empty() {
            self.ready.push_back(Ok(chunks));
        }
    }

    fn fail(&mut self, details: &str) {
        self.ready.push_back(Err(Error::ApiError {
            status_code: None,
            details: details.to_owned(),
        }));
        self.ended = true;
    }
}

fn record_upstream_meta(sink: &Mutex<UpstreamResponseMeta>, payload: &Value) {
    let model = payload["model"].as_str().filter(|s| !s.trim().is_empty());
    let cost = parse_exact_cost_nanos(&payload["usage"]["cost"]);
    let finish_reason = payload["choices"][0]["finish_reason"]
        .as_str()
        .filter(|s| !s.trim().is_empty());
    if model.is_none() && cost.is_none() && finish_reason.is_none() {
        return;
    }
    let mut guard = sink.lock();
    if let Some(m) = model {
        guard.resolved_model = Some(m.to_owned());
    }
    if let Some(fr) = finish_reason {
        guard.finish_reason = Some(fr.to_owned());
    }
    if let Some(c) = cost {
        guard.exact_cost_nanos = Some(match guard.exact_cost_nanos {
            Some(prev) => prev.saturating_add(c),
            None => c,
        });
    }
}

pub fn parse_exact_cost_nanos(val: &Value) -> Option<NanoUsd> {
    if val.is_null() {
        return None;
    }
    if let Some(s) = val.as_str() {
        return NanoUsd::parse_usd(s).ok();
    }
    if val.is_number() {
        let raw = val.to_string();
        if let Ok(parsed) = NanoUsd::parse_usd(&raw) {
            return Some(parsed);
        }
        if let Some(f) = val.as_f64().filter(|f| f.is_finite() && *f >= 0.0) {
            return Some(NanoUsd((f * 1_000_000_000.0).round() as i64));
        }
    }
    None
}

/// Splits a leading `<think>...</think>` reasoning block out of `content` when an OpenAI-compatible
/// local server (`llama.cpp`, `vllm`, `lmstudio`) leaves thoughts inline instead of populating
/// `reasoning_content` / `reasoning`.
pub fn split_inline_think_block(raw: &str) -> Option<(String, String)> {
    let trimmed = raw.trim_start();
    let rest = trimmed.strip_prefix("<think>")?;
    let (think_body, after) = rest.split_once("</think>")?;
    let reasoning = think_body.trim().to_owned();
    if reasoning.is_empty() {
        return None;
    }
    let content = after.trim_start_matches(['\r', '\n']).to_owned();
    Some((reasoning, content))
}

/// Reconciles `Usage::reasoning_tokens` when local runtimes (like Ollama `lfm2.5` / `qwen3` / `deepseek-r1`)
/// return non-empty `reasoning` + `completion_tokens` without populating `completion_tokens_details.reasoning_tokens`.
pub fn reconcile_reasoning_usage(
    usage: Option<Usage>,
    reasoning: &str,
    text: &str,
) -> Option<Usage> {
    if reasoning.is_empty() {
        return usage;
    }
    let mut u = usage?;
    if u.reasoning_tokens.unwrap_or(0) > 0 {
        return Some(u);
    }
    if let Some(out) = u.output_tokens.filter(|&o| o > 0) {
        let r_chars = reasoning.chars().count();
        let t_chars = text.chars().count();
        let total_chars = (r_chars + t_chars).max(1);
        let raw_rt = ((out as f64) * (r_chars as f64) / (total_chars as f64)).round() as usize;
        let rt = if !text.is_empty() && out > 1 {
            raw_rt.clamp(1, out - 1)
        } else {
            raw_rt.clamp(1, out)
        };
        u.reasoning_tokens = Some(rt);
    }
    Some(u)
}

fn parse_openai_usage_with_timings(u: &Value, timings: &Value) -> Option<Usage> {
    if !u.is_object() && !timings.is_object() {
        return None;
    }
    let count = |v: &Value| v.as_u64().map(|n| n as usize);
    let prompt_n = count(&timings["prompt_n"]);
    let cache_n = count(&timings["cache_n"]).filter(|&n| n > 0);
    let input_tokens = count(&u["prompt_tokens"])
        .or_else(|| count(&u["input_tokens"]))
        .or_else(|| match (prompt_n, cache_n) {
            (Some(p), Some(c)) => Some(p + c),
            (Some(p), None) => Some(p),
            _ => None,
        });
    let output_tokens = count(&u["completion_tokens"])
        .or_else(|| count(&u["output_tokens"]))
        .or_else(|| count(&timings["predicted_n"]));
    if input_tokens.is_none() && output_tokens.is_none() {
        return None;
    }
    Some(Usage {
        input_tokens,
        output_tokens,
        cached_tokens: count(&u["prompt_tokens_details"]["cached_tokens"])
            .or_else(|| count(&u["input_tokens_details"]["cached_tokens"]))
            .or(cache_n),
        reasoning_tokens: count(&u["completion_tokens_details"]["reasoning_tokens"])
            .or_else(|| count(&u["output_tokens_details"]["reasoning_tokens"]))
            .or_else(|| count(&u["reasoning_tokens"]))
            .or_else(|| count(&timings["reasoning_n"])),
    })
}

fn flush_assistant_turn(
    messages: &mut Vec<Value>,
    reasoning: &mut String,
    text: &mut String,
    tool_calls: &mut Vec<Value>,
) {
    if reasoning.is_empty() && text.is_empty() && tool_calls.is_empty() {
        return;
    }
    let mut obj = Map::new();
    obj.insert("role".into(), json!("assistant"));
    if text.is_empty() && !tool_calls.is_empty() {
        obj.insert("content".into(), Value::Null);
    } else {
        obj.insert("content".into(), json!(std::mem::take(text)));
    }
    if !reasoning.is_empty() {
        obj.insert(
            "reasoning_content".into(),
            json!(std::mem::take(reasoning)),
        );
    }
    if !tool_calls.is_empty() {
        obj.insert(
            "tool_calls".into(),
            Value::Array(std::mem::take(tool_calls)),
        );
    }
    messages.push(Value::Object(obj));
}

pub fn build_chat_completions_body(
    model: &str,
    options: &LanguageModelOptions,
    tools: &[Tool],
    stream: bool,
) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    let mut seen_system = Vec::new();
    if let Some(sys) = &options.system
        && !sys.trim().is_empty()
    {
        seen_system.push(sys.clone());
        messages.push(json!({ "role": "system", "content": sys }));
    }

    let mut acc_reasoning = String::new();
    let mut acc_text = String::new();
    let mut acc_tools: Vec<Value> = Vec::new();

    for msg in options.messages() {
        if !matches!(msg, Message::Assistant(_)) {
            flush_assistant_turn(
                &mut messages,
                &mut acc_reasoning,
                &mut acc_text,
                &mut acc_tools,
            );
        }
        match msg {
            Message::System(s) => {
                if !s.content.trim().is_empty() && !seen_system.contains(&s.content) {
                    seen_system.push(s.content.clone());
                    messages.push(json!({ "role": "system", "content": s.content }));
                }
            }
            Message::Developer(d) => {
                messages.push(json!({ "role": "developer", "content": d }));
            }
            Message::User(u) => {
                messages.push(json!({ "role": "user", "content": u.content }));
            }
            Message::Assistant(a) => match a.content {
                LanguageModelResponseContentType::Reasoning { content, .. } => {
                    if !acc_text.is_empty() || !acc_tools.is_empty() {
                        flush_assistant_turn(
                            &mut messages,
                            &mut acc_reasoning,
                            &mut acc_text,
                            &mut acc_tools,
                        );
                    }
                    if !content.is_empty() {
                        if !acc_reasoning.is_empty() {
                            acc_reasoning.push('\n');
                        }
                        acc_reasoning.push_str(&content);
                    }
                }
                LanguageModelResponseContentType::Text(t) => {
                    if !acc_tools.is_empty() {
                        flush_assistant_turn(
                            &mut messages,
                            &mut acc_reasoning,
                            &mut acc_text,
                            &mut acc_tools,
                        );
                    }
                    acc_text.push_str(&t);
                }
                LanguageModelResponseContentType::ToolCall(tc) => {
                    acc_tools.push(json!({
                        "id": tc.tool.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool.name,
                            "arguments": tc.input.to_string(),
                        }
                    }));
                }
                LanguageModelResponseContentType::NotSupported(_) => {}
            },
            Message::Tool(tr) => {
                let content = match tr.output {
                    Ok(Value::String(s)) => s,
                    Ok(other) => other.to_string(),
                    Err(e) => format!("Error: {e}"),
                };
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tr.tool.id,
                    "content": content,
                }));
            }
        }
    }
    flush_assistant_turn(
        &mut messages,
        &mut acc_reasoning,
        &mut acc_text,
        &mut acc_tools,
    );

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), json!(stream));
    if stream {
        body.insert(
            "stream_options".into(),
            json!({ "include_usage": true }),
        );
    }
    if let Some(temp) = options.temperature {
        body.insert("temperature".into(), json!(temp as f64 / 100.0));
    }
    if let Some(top_p) = options.top_p {
        body.insert("top_p".into(), json!(top_p as f64 / 100.0));
    }
    if let Some(top_k) = options.top_k {
        body.insert("top_k".into(), json!(top_k));
    }
    if let Some(max_tok) = options.max_output_tokens {
        body.insert("max_tokens".into(), json!(max_tok));
    }
    if let Some(seed) = options.seed {
        body.insert("seed".into(), json!(seed));
    }
    if let Some(pp) = options.presence_penalty {
        body.insert("presence_penalty".into(), json!(pp));
    }
    if let Some(fp) = options.frequency_penalty {
        body.insert("frequency_penalty".into(), json!(fp));
    }
    if let Some(stops) = &options.stop_sequences
        && !stops.is_empty()
    {
        body.insert("stop".into(), json!(stops));
    }
    if let Some(schema) = &options.schema {
        let mut sv = serde_json::to_value(schema).unwrap_or_default();
        if let Some(obj) = sv.as_object_mut() {
            obj.remove("$schema");
            obj.entry("additionalProperties").or_insert(json!(false));
        }
        body.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": sv["title"].as_str().unwrap_or("Response"),
                    "schema": sv,
                    "strict": false,
                }
            }),
        );
    }
    if !tools.is_empty() {
        let encoded: Vec<Value> = tools
            .iter()
            .map(|t| {
                let mut params = t.input_schema.clone().to_value();
                if let Some(obj) = params.as_object_mut() {
                    obj.remove("$schema");
                }
                if !params["properties"].is_object() {
                    params["properties"] = json!({});
                }
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": params,
                    }
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(encoded));
    }
    if let Some(effort) = options.reasoning_effort {
        let s = match effort {
            ReasoningEffort::None => "none",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
        };
        body.insert("reasoning_effort".into(), json!(s));
    }
    if let Some(extra) = &options.body {
        body.extend(extra.clone());
    }
    Value::Object(body)
}
