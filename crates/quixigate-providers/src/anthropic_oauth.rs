//! Custom `aisdk::core::LanguageModel` for Anthropic Setup-Token OAuth
//! (`sk-ant-oat01-...`) and full-fidelity Anthropic `/v1/messages` streaming.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

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
use serde_json::{Map, Value, json};

use crate::sse::{self, SseDecoder, SseFrame};

pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_OAUTH_BETA: &str = "oauth-2025-04-20";

#[derive(Clone)]
pub struct AnthropicOAuthModel {
    model: String,
    token: String,
    use_x_api_key: bool,
    base_url: String,
    tools: Vec<Tool>,
    default_headers: Option<HashMap<String, String>>,
    default_body: Option<Value>,
    http: reqwest::Client,
}

impl AnthropicOAuthModel {
    pub fn new(model: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            token: token.into(),
            use_x_api_key: false,
            base_url: DEFAULT_ANTHROPIC_BASE_URL.to_owned(),
            tools: Vec::new(),
            default_headers: None,
            default_body: None,
            http: crate::shared_inference_http_client(),
        }
    }

    /// Constructs a warm-pooled Anthropic `/v1/messages` client authenticating via `x-api-key`.
    pub fn api_key(model: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            model: model.into(),
            token: api_key.unwrap_or_default(),
            use_x_api_key: true,
            base_url: DEFAULT_ANTHROPIC_BASE_URL.to_owned(),
            tools: Vec::new(),
            default_headers: None,
            default_body: None,
            http: crate::shared_inference_http_client(),
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
        if !headers.is_empty() {
            self.default_headers = Some(headers);
        }
        self
    }

    pub fn with_default_body(mut self, body: Value) -> Self {
        self.default_body = Some(body);
        self
    }

    pub fn resolve_messages_url(base_url: &str) -> String {
        let trimmed = base_url.trim_end_matches('/');
        if trimmed.ends_with("/v1/messages") || trimmed.ends_with("/messages") {
            trimmed.to_owned()
        } else if trimmed.ends_with("/v1") {
            format!("{trimmed}/messages")
        } else {
            format!("{trimmed}/v1/messages")
        }
    }

    fn messages_url(&self) -> String {
        Self::resolve_messages_url(&self.base_url)
    }

    async fn send(
        &self,
        options: &LanguageModelOptions,
        stream: bool,
    ) -> Result<reqwest::Response> {
        let mut body = build_anthropic_body(&self.model, options, &self.tools, stream);
        if let (Some(Value::Object(extra)), Some(obj)) = (&self.default_body, body.as_object_mut())
        {
            for (k, v) in extra {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        let mut req = self
            .http
            .post(self.messages_url())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json");

        if self.use_x_api_key {
            if !self.token.trim().is_empty() {
                req = req.header("x-api-key", &self.token);
            }
        } else {
            req = req
                .bearer_auth(&self.token)
                .header("anthropic-beta", ANTHROPIC_OAUTH_BETA);
        }

        if stream {
            req = req.header("Accept", "text/event-stream");
        }
        for (k, v) in self.default_headers.iter().flatten() {
            req = req.header(k, v);
        }
        for (k, v) in options.headers.iter().flatten() {
            req = req.header(k, v);
        }

        let resp = req.json(&body).send().await.map_err(|e| {
            let (status_code, why) = crate::describe_reqwest_error(&e);
            Error::ApiError {
                status_code,
                details: format!("Anthropic OAuth request failed: {why}"),
            }
        })?;
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
        let err_type = body_json["error"]["type"].as_str().unwrap_or_default();
        let err_msg = body_json["error"]["message"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| text.chars().take(500).collect());
        let mut details = if !err_type.is_empty() {
            format!("{err_type}: {err_msg}")
        } else {
            format!("HTTP {}: {err_msg}", status.as_u16())
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

impl fmt::Debug for AnthropicOAuthModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicOAuthModel")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("tools_count", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl TextInputSupport for AnthropicOAuthModel {}
impl TextOutputSupport for AnthropicOAuthModel {}
impl ToolCallSupport for AnthropicOAuthModel {}
impl StructuredOutputSupport for AnthropicOAuthModel {}
impl ReasoningSupport for AnthropicOAuthModel {}
impl ImageInputSupport for AnthropicOAuthModel {}

#[async_trait]
impl LanguageModel for AnthropicOAuthModel {
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
                details: format!("invalid Anthropic JSON response: {why}"),
            }
        })?;
        let mut contents = Vec::new();
        let mut acc_thinking = String::new();
        let mut acc_text = String::new();
        for block in data["content"].as_array().into_iter().flatten() {
            match block["type"].as_str().unwrap_or_default() {
                "text" => {
                    let t = block["text"].as_str().unwrap_or_default().to_owned();
                    acc_text.push_str(&t);
                    contents.push(LanguageModelResponseContentType::Text(t));
                }
                "thinking" => {
                    let th = block["thinking"].as_str().unwrap_or_default().to_owned();
                    acc_thinking.push_str(&th);
                    contents.push(LanguageModelResponseContentType::Reasoning {
                        content: th,
                        extensions: Extensions::default(),
                    });
                }
                "tool_use" => {
                    contents.push(LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                        tool: ToolDetails {
                            id: block["id"].as_str().unwrap_or("toolu_0").to_owned(),
                            name: block["name"].as_str().unwrap_or_default().to_owned(),
                        },
                        input: block["input"].clone(),
                        extensions: Extensions::default(),
                    }));
                }
                _ => {}
            }
        }
        let u = &data["usage"];
        let input_tokens = u["input_tokens"].as_u64().unwrap_or(0) as usize;
        let output_tokens = u["output_tokens"].as_u64().unwrap_or(0) as usize;
        let cached = (u["cache_read_input_tokens"].as_u64().unwrap_or(0)
            + u["cache_creation_input_tokens"].as_u64().unwrap_or(0)) as usize;
        let usage = crate::copilot::reconcile_reasoning_usage(
            Some(Usage {
                input_tokens: Some(input_tokens),
                output_tokens: Some(output_tokens),
                cached_tokens: Some(cached),
                reasoning_tokens: None,
            }),
            &acc_thinking,
            &acc_text,
        );
        Ok(LanguageModelResponse { contents, usage })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let response = self.send(&options, true).await?;
        let state = AnthropicStreamState {
            response,
            decoder: SseDecoder::default(),
            blocks: BTreeMap::new(),
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            ready: VecDeque::new(),
            finished: false,
            ended: false,
        };
        Ok(Box::pin(futures::stream::unfold(
            state,
            |mut state| async move { state.next().await.map(|item| (item, state)) },
        )))
    }
}

enum AnthropicAccBlock {
    Text(String),
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
    },
}

#[derive(serde::Deserialize)]
struct FastAnthropicDeltaEvent<'a> {
    #[serde(borrow, rename = "type", default)]
    event_type: Option<Cow<'a, str>>,
    #[serde(default)]
    index: usize,
    #[serde(borrow, default)]
    delta: Option<FastAnthropicDelta<'a>>,
}

#[derive(serde::Deserialize)]
struct FastAnthropicDelta<'a> {
    #[serde(borrow, rename = "type", default)]
    delta_type: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    text: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    thinking: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    partial_json: Option<Cow<'a, str>>,
}

struct AnthropicStreamState {
    response: reqwest::Response,
    decoder: SseDecoder,
    blocks: BTreeMap<usize, AnthropicAccBlock>,
    input_tokens: usize,
    output_tokens: usize,
    cached_tokens: usize,
    ready: VecDeque<Result<Vec<LanguageModelStreamChunk>>>,
    finished: bool,
    ended: bool,
}

impl AnthropicStreamState {
    async fn next(&mut self) -> Option<Result<Vec<LanguageModelStreamChunk>>> {
        loop {
            if let Some(item) = self.ready.pop_front() {
                return Some(item);
            }
            if self.ended {
                return None;
            }
            match self.response.chunk().await {
                Ok(Some(bytes)) => match self.decoder.push_frames(&bytes) {
                    Ok(frames) => {
                        frames.into_iter().for_each(|f| self.on_frame(f));
                        self.coalesce_ready_ok();
                    }
                    Err(sse::FrameTooLarge) => {
                        self.fail("Anthropic backend sent oversized SSE frame")
                    }
                },
                Ok(None) => {
                    if let Some(frame) = self.decoder.finish_frame() {
                        self.on_frame(frame);
                    }
                    if !self.finished && !self.blocks.is_empty() {
                        self.emit_done();
                    }
                    self.coalesce_ready_ok();
                    self.ended = true;
                }
                Err(err) => {
                    let (_, why) = crate::describe_reqwest_error(&err);
                    self.fail(&format!("Anthropic stream error: {why}"));
                }
            }
        }
    }

    #[inline]
    fn coalesce_ready_ok(&mut self) {
        if self.ready.len() <= 1 || self.ready.iter().any(std::result::Result::is_err) {
            return;
        }
        let total_chunks: usize = self
            .ready
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(Vec::len)
            .sum();
        let mut merged = Vec::with_capacity(total_chunks);
        while let Some(Ok(mut v)) = self.ready.pop_front() {
            merged.append(&mut v);
        }
        if !merged.is_empty() {
            self.ready.push_back(Ok(merged));
        }
    }

    fn on_frame(&mut self, frame: SseFrame) {
        if self.ended || frame.data == "[DONE]" {
            return;
        }
        // Fast zero-DOM path for `content_block_delta` (99%+ of Anthropic SSE frames).
        if (frame.event.as_deref() == Some("content_block_delta")
            || memchr::memmem::find(frame.data.as_bytes(), b"\"content_block_delta\"").is_some())
            && let Ok(fast) = serde_json::from_str::<FastAnthropicDeltaEvent<'_>>(&frame.data)
            && fast.event_type.as_deref().or(frame.event.as_deref()) == Some("content_block_delta")
            && let Some(delta) = fast.delta
        {
            let Some(block) = self.blocks.get_mut(&fast.index) else {
                return;
            };
            match (block, delta.delta_type.as_deref().unwrap_or_default()) {
                (AnthropicAccBlock::Text(buf), "text_delta") => {
                    let d = delta.text.unwrap_or_default().into_owned();
                    buf.push_str(&d);
                    self.ready
                        .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::TextDelta(d),
                        )]));
                }
                (AnthropicAccBlock::Thinking(buf), "thinking_delta") => {
                    let d = delta.thinking.unwrap_or_default().into_owned();
                    buf.push_str(&d);
                    self.ready
                        .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::ReasoningDelta(d),
                        )]));
                }
                (AnthropicAccBlock::ToolUse { id, json_buf, .. }, "input_json_delta") => {
                    let d = delta.partial_json.unwrap_or_default().into_owned();
                    json_buf.push_str(&d);
                    self.ready
                        .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::ToolCallDelta {
                                id: id.clone(),
                                delta: d,
                            },
                        )]));
                }
                _ => {}
            }
            return;
        }

        let Ok(payload) = serde_json::from_str::<Value>(&frame.data) else {
            return;
        };
        let event_type = payload["type"]
            .as_str()
            .or(frame.event.as_deref())
            .unwrap_or_default();

        match event_type {
            "message_start" => {
                let u = &payload["message"]["usage"];
                if let Some(inp) = u["input_tokens"].as_u64() {
                    self.input_tokens = inp as usize;
                }
                if let Some(out) = u["output_tokens"].as_u64() {
                    self.output_tokens = out as usize;
                }
                self.cached_tokens += (u["cache_read_input_tokens"].as_u64().unwrap_or(0)
                    + u["cache_creation_input_tokens"].as_u64().unwrap_or(0))
                    as usize;
            }
            "content_block_start" => {
                let index = payload["index"].as_u64().unwrap_or(0) as usize;
                let cb = &payload["content_block"];
                match cb["type"].as_str().unwrap_or_default() {
                    "text" => {
                        let init = cb["text"].as_str().unwrap_or_default().to_owned();
                        let mut chunks = vec![LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::TextStart,
                        )];
                        if !init.is_empty() {
                            chunks.push(LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::TextDelta(init.clone()),
                            ));
                        }
                        self.blocks.insert(index, AnthropicAccBlock::Text(init));
                        self.ready.push_back(Ok(chunks));
                    }
                    "thinking" => {
                        self.blocks
                            .insert(index, AnthropicAccBlock::Thinking(String::new()));
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::ReasoningStart,
                            )]));
                    }
                    "tool_use" => {
                        let id = cb["id"].as_str().unwrap_or("toolu_0").to_owned();
                        let name = cb["name"].as_str().unwrap_or_default().to_owned();
                        self.blocks.insert(
                            index,
                            AnthropicAccBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                json_buf: String::new(),
                            },
                        );
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::ToolCallStart(ToolDetails {
                                    name,
                                    id,
                                }),
                            )]));
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = payload["index"].as_u64().unwrap_or(0) as usize;
                let delta = &payload["delta"];
                let Some(block) = self.blocks.get_mut(&index) else {
                    return;
                };
                match (block, delta["type"].as_str().unwrap_or_default()) {
                    (AnthropicAccBlock::Text(buf), "text_delta") => {
                        let d = delta["text"].as_str().unwrap_or_default().to_owned();
                        buf.push_str(&d);
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::TextDelta(d),
                            )]));
                    }
                    (AnthropicAccBlock::Thinking(buf), "thinking_delta") => {
                        let d = delta["thinking"].as_str().unwrap_or_default().to_owned();
                        buf.push_str(&d);
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::ReasoningDelta(d),
                            )]));
                    }
                    (AnthropicAccBlock::ToolUse { id, json_buf, .. }, "input_json_delta") => {
                        let d = delta["partial_json"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned();
                        json_buf.push_str(&d);
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::ToolCallDelta {
                                    id: id.clone(),
                                    delta: d,
                                },
                            )]));
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = payload["index"].as_u64().unwrap_or(0) as usize;
                match self.blocks.get(&index) {
                    Some(AnthropicAccBlock::Text(_)) => {
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::TextEnd,
                            )]));
                    }
                    Some(AnthropicAccBlock::Thinking(_)) => {
                        self.ready
                            .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::ReasoningEnd,
                            )]));
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                let u = &payload["usage"];
                if let Some(inp) = u["input_tokens"].as_u64() {
                    self.input_tokens = inp as usize;
                }
                if let Some(out) = u["output_tokens"].as_u64() {
                    self.output_tokens = out as usize;
                }
            }
            "message_stop" => {
                self.finished = true;
                self.emit_done();
                self.ended = true;
            }
            "error" => {
                self.finished = true;
                let msg = payload["error"]["message"]
                    .as_str()
                    .unwrap_or("Anthropic stream error");
                self.ready
                    .push_back(Ok(vec![LanguageModelStreamChunk::Delta(
                        LanguageModelStreamChunkType::Failed(msg.to_owned()),
                    )]));
                self.ended = true;
            }
            _ => {}
        }
    }

    fn emit_done(&mut self) {
        let usage = Usage {
            input_tokens: Some(self.input_tokens),
            output_tokens: Some(self.output_tokens),
            cached_tokens: Some(self.cached_tokens),
            reasoning_tokens: None,
        };
        let mut chunks = Vec::new();
        for (idx, (_, block)) in std::mem::take(&mut self.blocks).into_iter().enumerate() {
            let content = match block {
                AnthropicAccBlock::Text(t) => LanguageModelResponseContentType::Text(t),
                AnthropicAccBlock::Thinking(t) => LanguageModelResponseContentType::Reasoning {
                    content: t,
                    extensions: Extensions::default(),
                },
                AnthropicAccBlock::ToolUse { id, name, json_buf } => {
                    let parsed = if json_buf.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&json_buf).unwrap_or_else(|_| json!({}))
                    };
                    LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                        tool: ToolDetails { id, name },
                        input: parsed,
                        extensions: Extensions::default(),
                    })
                }
            };
            chunks.push(LanguageModelStreamChunk::Done(AssistantMessage {
                content,
                usage: if idx == 0 { Some(usage.clone()) } else { None },
            }));
        }
        if chunks.is_empty() {
            chunks.push(LanguageModelStreamChunk::Done(AssistantMessage {
                content: LanguageModelResponseContentType::Text(String::new()),
                usage: Some(usage),
            }));
        }
        self.ready.push_back(Ok(chunks));
    }

    fn fail(&mut self, details: &str) {
        self.ready.push_back(Err(Error::ApiError {
            status_code: None,
            details: details.to_owned(),
        }));
        self.ended = true;
    }
}

pub fn build_anthropic_body(
    model: &str,
    options: &LanguageModelOptions,
    tools: &[Tool],
    stream: bool,
) -> Value {
    let mut system_parts: Vec<String> = options.system.iter().cloned().collect();
    let mut messages: Vec<Value> = Vec::new();

    for msg in options.messages() {
        match msg {
            Message::System(s) => {
                if !s.content.trim().is_empty() && !system_parts.contains(&s.content) {
                    system_parts.push(s.content);
                }
            }
            Message::Developer(dev) => {
                messages.push(json!({
                    "role": "user",
                    "content": format!("<developer>\n{dev}\n</developer>"),
                }));
            }
            Message::User(u) => {
                messages.push(json!({
                    "role": "user",
                    "content": u.content,
                }));
            }
            Message::Assistant(a) => match a.content {
                LanguageModelResponseContentType::Text(text) => {
                    messages.push(json!({
                        "role": "assistant",
                        "content": [{ "type": "text", "text": text }],
                    }));
                }
                LanguageModelResponseContentType::ToolCall(tc) => {
                    messages.push(json!({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": tc.tool.id,
                            "name": tc.tool.name,
                            "input": tc.input,
                        }],
                    }));
                }
                LanguageModelResponseContentType::Reasoning { .. }
                | LanguageModelResponseContentType::NotSupported(_) => {}
            },
            Message::Tool(tr) => {
                let text = match tr.output {
                    Ok(Value::String(s)) => s,
                    Ok(other) => other.to_string(),
                    Err(e) => format!("Error: {e}"),
                };
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tr.tool.id,
                        "content": text,
                    }],
                }));
            }
        }
    }

    let max_tokens = options.max_output_tokens.unwrap_or(8_192);
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("max_tokens".into(), json!(max_tokens));
    body.insert("messages".into(), Value::Array(messages));
    if stream {
        body.insert("stream".into(), json!(true));
    }

    system_parts.retain(|p| !p.trim().is_empty());
    if !system_parts.is_empty() {
        body.insert("system".into(), json!(system_parts.join("\n\n")));
    }
    if let Some(temp) = options.temperature
        && matches!(options.reasoning_effort, None | Some(ReasoningEffort::None))
    {
        body.insert("temperature".into(), json!(temp as f64 / 100.0));
    }
    if let Some(top_p) = options.top_p {
        body.insert("top_p".into(), json!(top_p as f64 / 100.0));
    }
    if let Some(top_k) = options.top_k {
        body.insert("top_k".into(), json!(top_k));
    }
    if let Some(stop) = &options.stop_sequences
        && !stop.is_empty()
    {
        body.insert("stop_sequences".into(), json!(stop));
    }
    if !tools.is_empty() {
        let encoded_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                let mut schema = t.input_schema.clone().to_value();
                if let Some(obj) = schema.as_object_mut() {
                    obj.remove("$schema");
                }
                if !schema["properties"].is_object() {
                    schema["properties"] = json!({});
                }
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": schema,
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(encoded_tools));
    }
    if let Some(effort) = options.reasoning_effort {
        match effort {
            ReasoningEffort::None => {
                body.insert("thinking".into(), json!({ "type": "disabled" }));
            }
            ReasoningEffort::Low
            | ReasoningEffort::Medium
            | ReasoningEffort::High
            | ReasoningEffort::XHigh => {
                let budget = match effort {
                    ReasoningEffort::Low => (max_tokens / 4).max(1024),
                    ReasoningEffort::Medium => (max_tokens / 2).max(1024),
                    ReasoningEffort::High => ((max_tokens * 3) / 4).max(1024),
                    ReasoningEffort::XHigh => ((max_tokens * 9) / 10).max(1024),
                    ReasoningEffort::None => 1024,
                };
                if max_tokens <= budget {
                    body.insert("max_tokens".into(), json!(budget + 2048));
                }
                body.insert(
                    "thinking".into(),
                    json!({ "type": "enabled", "budget_tokens": budget }),
                );
            }
        }
    }
    if let Some(extra) = &options.body {
        body.extend(extra.clone());
    }
    Value::Object(body)
}
