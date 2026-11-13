//! Custom `aisdk::core::LanguageModel` for the ChatGPT Codex backend
//! (`https://chatgpt.com/backend-api/codex/responses`).

use std::collections::VecDeque;
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
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use futures::StreamExt;
use reqwest::StatusCode;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::sse::{self, SseDecoder};

pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const ORIGINATOR: &str = "pi";
const DEFAULT_INSTRUCTIONS: &str = "You are a helpful assistant.";

#[derive(Clone)]
pub struct ChatGptCodexModel {
    model: String,
    access_token: String,
    account_id: String,
    base_url: String,
    tools: Vec<Tool>,
    http: reqwest::Client,
}

impl ChatGptCodexModel {
    pub fn new(
        model: impl Into<String>,
        access_token: impl Into<String>,
        account_id: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            access_token: access_token.into(),
            account_id: account_id.into(),
            base_url: DEFAULT_CODEX_BASE_URL.to_owned(),
            tools: Vec::new(),
            http: crate::build_inference_http_client(),
        }
    }

    pub fn with_http(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let url = base_url.into();
        if !url.trim().is_empty() {
            self.base_url = url;
        }
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    async fn send(&self, options: &LanguageModelOptions) -> Result<reqwest::Response> {
        let body = build_codex_body(&self.model, options, &self.tools);
        let base = self.base_url.trim_end_matches('/');
        let url = if base.ends_with("/responses") {
            base.to_owned()
        } else {
            format!("{base}/responses")
        };
        let request_id = Uuid::new_v4().to_string();
        let mut request = self
            .http
            .post(&url)
            .bearer_auth(&self.access_token)
            .header("ChatGPT-Account-ID", &self.account_id)
            .header("originator", ORIGINATOR)
            .header("OpenAI-Beta", "responses=experimental")
            .header("Accept", "text/event-stream")
            .header(
                "User-Agent",
                concat!("quixigate/", env!("CARGO_PKG_VERSION")),
            )
            .header("session_id", &request_id)
            .header("x-client-request-id", &request_id);

        if let Some(res) = extract_residency(&self.access_token) {
            request = request.header("x-openai-internal-codex-residency", res);
        }
        for (name, value) in options.headers.iter().flatten() {
            request = request.header(name, value);
        }

        let response = request.json(&body).send().await.map_err(|err| {
            let (status_code, why) = crate::describe_reqwest_error(&err);
            Error::ApiError {
                status_code,
                details: format!("could not reach ChatGPT Codex backend: {why}"),
            }
        })?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let text = response.text().await.unwrap_or_default();
        let details = format_codex_failure(status, &text, retry_after.as_deref());
        Err(Error::ApiError {
            status_code: Some(status),
            details,
        })
    }
}

impl fmt::Debug for ChatGptCodexModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatGptCodexModel")
            .field("model", &self.model)
            .field("account_id", &self.account_id)
            .field("base_url", &self.base_url)
            .field("tools_count", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl TextInputSupport for ChatGptCodexModel {}
impl TextOutputSupport for ChatGptCodexModel {}
impl ToolCallSupport for ChatGptCodexModel {}
impl StructuredOutputSupport for ChatGptCodexModel {}
impl ReasoningSupport for ChatGptCodexModel {}
impl ImageInputSupport for ChatGptCodexModel {}

#[async_trait]
impl LanguageModel for ChatGptCodexModel {
    fn name(&self) -> String {
        self.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let mut stream = self.stream_text(options).await?;
        let mut response = LanguageModelResponse {
            contents: Vec::new(),
            usage: None,
        };
        while let Some(chunks) = stream.next().await {
            for chunk in chunks? {
                match chunk {
                    LanguageModelStreamChunk::Done(message) => {
                        response.usage = message.usage.or(response.usage);
                        response.contents.push(message.content);
                    }
                    LanguageModelStreamChunk::Delta(LanguageModelStreamChunkType::Failed(why)) => {
                        return Err(Error::ApiError {
                            status_code: None,
                            details: why,
                        });
                    }
                    LanguageModelStreamChunk::Delta(LanguageModelStreamChunkType::Incomplete(
                        why,
                    )) => {
                        return Err(Error::ApiError {
                            status_code: None,
                            details: format!("ChatGPT Codex response incomplete: {why}"),
                        });
                    }
                    LanguageModelStreamChunk::Delta(_) => {}
                }
            }
        }
        Ok(response)
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let response = self.send(&options).await?;
        let state = CodexStreamState {
            response,
            decoder: SseDecoder::default(),
            mapper: CodexEventMapper::default(),
            ready: VecDeque::new(),
            ended: false,
        };
        Ok(Box::pin(futures::stream::unfold(
            state,
            |mut state| async move { state.next().await.map(|item| (item, state)) },
        )))
    }
}

struct CodexStreamState {
    response: reqwest::Response,
    decoder: SseDecoder,
    mapper: CodexEventMapper,
    ready: VecDeque<Result<Vec<LanguageModelStreamChunk>>>,
    ended: bool,
}

impl CodexStreamState {
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
                    Ok(events) => events.iter().for_each(|data| self.on_event(data)),
                    Err(sse::FrameTooLarge) => {
                        self.fail("ChatGPT Codex backend sent an oversized SSE frame")
                    }
                },
                Ok(None) => {
                    if let Some(data) = self.decoder.finish() {
                        self.on_event(&data);
                    }
                    if !self.mapper.finished {
                        self.fail("ChatGPT Codex stream closed before response.completed");
                    }
                    self.ended = true;
                }
                Err(err) => {
                    let (_, why) = crate::describe_reqwest_error(&err);
                    self.fail(&format!("ChatGPT Codex stream error: {why}"));
                }
            }
        }
    }

    fn on_event(&mut self, data: &str) {
        if self.ended || data == "[DONE]" {
            return;
        }
        if let Ok(event) = serde_json::from_str::<Value>(data) {
            let chunks = self.mapper.map(&event);
            if !chunks.is_empty() {
                self.ready.push_back(Ok(chunks));
            }
            self.ended = self.mapper.finished;
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

pub fn build_codex_body(model: &str, options: &LanguageModelOptions, tools: &[Tool]) -> Value {
    let mut instructions: Vec<String> = options.system.iter().cloned().collect();
    let mut input = Vec::new();
    for message in options.messages() {
        match message {
            Message::System(system) => {
                if !instructions.contains(&system.content) {
                    instructions.push(system.content);
                }
            }
            Message::Developer(text) => input.push(text_message("developer", &text)),
            Message::User(user) => input.push(text_message("user", &user.content)),
            Message::Assistant(assistant) => match assistant.content {
                LanguageModelResponseContentType::Text(text) => input.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text, "annotations": [] }],
                })),
                LanguageModelResponseContentType::ToolCall(call) => input.push(json!({
                    "type": "function_call",
                    "call_id": call.tool.id,
                    "name": call.tool.name,
                    "arguments": call.input.to_string(),
                })),
                LanguageModelResponseContentType::Reasoning { .. }
                | LanguageModelResponseContentType::NotSupported(_) => {}
            },
            Message::Tool(result) => input.push(json!({
                "type": "function_call_output",
                "call_id": result.tool.id,
                "output": match result.output {
                    Ok(Value::String(text)) => text,
                    Ok(other) => other.to_string(),
                    Err(err) => format!("Error: {err}"),
                },
            })),
        }
    }
    instructions.retain(|part| !part.trim().is_empty());

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("store".into(), json!(false));
    body.insert("stream".into(), json!(true));
    body.insert(
        "instructions".into(),
        json!(if instructions.is_empty() {
            DEFAULT_INSTRUCTIONS.to_owned()
        } else {
            instructions.join("\n\n")
        }),
    );
    body.insert("input".into(), Value::Array(input));

    if let Some(schema) = &options.schema {
        let schema_val = serde_json::to_value(schema).unwrap_or_default();
        body.insert(
            "text".into(),
            json!({
                "format": {
                    "type": "json_schema",
                    "name": sanitize_format_name(schema_val["title"].as_str().unwrap_or("response")),
                    "schema": schema_val,
                    "strict": false,
                }
            }),
        );
    }
    if !tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(tools.iter().map(encode_codex_tool).collect()),
        );
        body.insert("tool_choice".into(), json!("auto"));
        body.insert("parallel_tool_calls".into(), json!(true));
    }
    if let Some(effort) = options.reasoning_effort {
        let effort_str = match effort {
            ReasoningEffort::None => "none",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
        };
        body.insert(
            "reasoning".into(),
            if effort_str == "none" {
                json!({ "effort": effort_str })
            } else {
                json!({ "effort": effort_str, "summary": "auto" })
            },
        );
    }

    if let Some(extra) = &options.body {
        body.extend(extra.clone());
    }
    Value::Object(body)
}

fn text_message(role: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": [{ "type": "input_text", "text": text }],
    })
}

fn encode_codex_tool(tool: &Tool) -> Value {
    let mut parameters = tool.input_schema.clone().to_value();
    if let Some(obj) = parameters.as_object_mut() {
        obj.remove("$schema");
    }
    if !parameters["properties"].is_object() {
        parameters["properties"] = json!({});
    }
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
        "strict": false,
    })
}

fn sanitize_format_name(title: &str) -> String {
    let name: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if name.is_empty() {
        "response".to_owned()
    } else {
        name
    }
}

fn extract_residency(jwt: &str) -> Option<String> {
    let payload_seg = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload_seg).ok()?;
    let val: Value = serde_json::from_slice(&bytes).ok()?;
    val["https://api.openai.com/auth"]["residency_region"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn format_codex_failure(status: StatusCode, text: &str, retry_after: Option<&str>) -> String {
    let body: Value = serde_json::from_str(text).unwrap_or_default();
    let error = &body["error"];
    let code = error["code"]
        .as_str()
        .or(error["type"].as_str())
        .unwrap_or_default();
    let message = body["detail"]
        .as_str()
        .or(error["message"].as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| text.chars().take(500).collect());

    let mut out = if !code.is_empty() {
        format!("{code}: {message}")
    } else {
        format!("HTTP {}: {message}", status.as_u16())
    };
    if let Some(ra) = retry_after {
        out.push_str(&format!(" [retry_after={ra}]"));
    }
    out
}

#[derive(Debug, Default)]
pub struct CodexEventMapper {
    items: Vec<Value>,
    call_ids_by_item: std::collections::HashMap<String, String>,
    accumulated_text: String,
    accumulated_reasoning: String,
    pub finished: bool,
}

impl CodexEventMapper {
    pub fn map(&mut self, event: &Value) -> Vec<LanguageModelStreamChunk> {
        let delta =
            |kind: LanguageModelStreamChunkType| vec![LanguageModelStreamChunk::Delta(kind)];
        let text = |key: &str| event[key].as_str().unwrap_or_default().to_owned();
        match event["type"].as_str().unwrap_or_default() {
            "response.output_item.added" => match event["item"]["type"].as_str() {
                Some("message") => delta(LanguageModelStreamChunkType::TextStart),
                Some("reasoning") => delta(LanguageModelStreamChunkType::ReasoningStart),
                Some("function_call") => {
                    let item_id = event["item"]["id"].as_str().unwrap_or_default().to_owned();
                    let call_id = event["item"]["call_id"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .unwrap_or(&item_id)
                        .to_owned();
                    if !item_id.is_empty() {
                        self.call_ids_by_item.insert(item_id, call_id.clone());
                    }
                    delta(LanguageModelStreamChunkType::ToolCallStart(ToolDetails {
                        name: event["item"]["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned(),
                        id: call_id,
                    }))
                }
                _ => Vec::new(),
            },
            "response.output_item.done" => {
                self.items.push(event["item"].clone());
                Vec::new()
            }
            "response.output_text.delta" => {
                let d = text("delta");
                self.accumulated_text.push_str(&d);
                delta(LanguageModelStreamChunkType::TextDelta(d))
            }
            "response.output_text.done" => delta(LanguageModelStreamChunkType::TextEnd),
            "response.reasoning_summary_text.delta" => {
                let d = text("delta");
                self.accumulated_reasoning.push_str(&d);
                delta(LanguageModelStreamChunkType::ReasoningDelta(d))
            }
            "response.reasoning_summary_text.done" => {
                delta(LanguageModelStreamChunkType::ReasoningEnd)
            }
            "response.function_call_arguments.delta" => {
                let raw_id = text("item_id");
                let id = self
                    .call_ids_by_item
                    .get(&raw_id)
                    .cloned()
                    .unwrap_or(raw_id);
                delta(LanguageModelStreamChunkType::ToolCallDelta {
                    id,
                    delta: text("delta"),
                })
            }
            "response.completed" | "response.done" => {
                self.finished = true;
                self.completed(&event["response"])
            }
            "response.incomplete" => {
                self.finished = true;
                let reason = event["response"]["incomplete_details"]["reason"]
                    .as_str()
                    .unwrap_or("unknown");
                delta(LanguageModelStreamChunkType::Incomplete(reason.to_owned()))
            }
            "response.failed" => {
                self.finished = true;
                delta(LanguageModelStreamChunkType::Failed(codex_error_text(
                    &event["response"]["error"],
                )))
            }
            "error" => {
                self.finished = true;
                let err_obj = if event["error"].is_object() {
                    &event["error"]
                } else {
                    event
                };
                delta(LanguageModelStreamChunkType::Failed(codex_error_text(
                    err_obj,
                )))
            }
            _ => Vec::new(),
        }
    }

    fn completed(&mut self, response: &Value) -> Vec<LanguageModelStreamChunk> {
        let usage = parse_codex_usage(&response["usage"]);
        let listed = response["output"]
            .as_array()
            .filter(|items| !items.is_empty())
            .cloned();
        let items = listed.unwrap_or_else(|| std::mem::take(&mut self.items));
        let mut out: Vec<LanguageModelStreamChunk> = items
            .iter()
            .filter_map(parse_codex_item)
            .enumerate()
            .map(|(idx, content)| {
                LanguageModelStreamChunk::Done(AssistantMessage {
                    content,
                    // Attach usage only to the first Done message so Step::usage() does not double-count!
                    usage: if idx == 0 { Some(usage.clone()) } else { None },
                })
            })
            .collect();
        if out.is_empty() {
            let fallback_text = std::mem::take(&mut self.accumulated_text);
            let fallback_reasoning = std::mem::take(&mut self.accumulated_reasoning);
            if !fallback_reasoning.is_empty() {
                out.push(LanguageModelStreamChunk::Done(AssistantMessage {
                    content: LanguageModelResponseContentType::Reasoning {
                        content: fallback_reasoning,
                        extensions: Extensions::default(),
                    },
                    usage: None,
                }));
            }
            out.push(LanguageModelStreamChunk::Done(AssistantMessage {
                content: LanguageModelResponseContentType::Text(fallback_text),
                usage: Some(usage),
            }));
        }
        out
    }
}

fn parse_codex_item(item: &Value) -> Option<LanguageModelResponseContentType> {
    let joined = |parts: &Value, kind: &str| -> String {
        parts
            .as_array()
            .into_iter()
            .flatten()
            .filter(|part| part["type"] == kind)
            .filter_map(|part| part["text"].as_str())
            .collect()
    };
    match item["type"].as_str()? {
        "message" => Some(LanguageModelResponseContentType::Text(joined(
            &item["content"],
            "output_text",
        ))),
        "function_call" => {
            let mut call = ToolCallInfo::new(item["name"].as_str()?);
            let call_id = item["call_id"]
                .as_str()
                .or(item["id"].as_str())
                .unwrap_or("call_0");
            call.id(call_id);
            let arguments = item["arguments"].as_str().unwrap_or("{}");
            call.input(serde_json::from_str(arguments).unwrap_or_else(|_| json!({})));
            Some(LanguageModelResponseContentType::ToolCall(call))
        }
        "reasoning" => {
            let summary = joined(&item["summary"], "summary_text");
            (!summary.is_empty()).then(|| LanguageModelResponseContentType::Reasoning {
                content: summary,
                extensions: Extensions::default(),
            })
        }
        _ => None,
    }
}

fn parse_codex_usage(usage: &Value) -> Usage {
    let count = |v: &Value| v.as_u64().map(|n| n as usize);
    Usage {
        input_tokens: count(&usage["input_tokens"]),
        output_tokens: count(&usage["output_tokens"]),
        cached_tokens: count(&usage["input_tokens_details"]["cached_tokens"]),
        reasoning_tokens: count(&usage["output_tokens_details"]["reasoning_tokens"]),
    }
}

fn codex_error_text(error: &Value) -> String {
    let message = error["message"]
        .as_str()
        .unwrap_or("Codex model call failed");
    match error["code"].as_str().or(error["type"].as_str()) {
        Some(code) => format!("{code}: {message}"),
        None => message.to_owned(),
    }
}
