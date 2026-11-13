//! Custom `aisdk::core::LanguageModel` for Google Cloud Code Assist
//! (`https://cloudcode-pa.googleapis.com/v1internal/{model}:streamGenerateContent?alt=sse`).

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
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::sse::{self, SseDecoder};

pub const DEFAULT_CODE_ASSIST_BASE_URL: &str = "https://cloudcode-pa.googleapis.com";
const GEMINI_CLI_USER_AGENT: &str = "google-api-nodejs-client/9.15.1";
const GEMINI_CLI_API_CLIENT: &str = "google-cloud-sdk";
const GEMINI_CLIENT_METADATA: &str =
    r#"{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}"#;

#[derive(Clone)]
pub struct GoogleCodeAssistModel {
    model: String,
    access_token: String,
    project_id: String,
    base_url: String,
    tools: Vec<Tool>,
    http: reqwest::Client,
}

impl GoogleCodeAssistModel {
    pub fn new(
        model: impl Into<String>,
        access_token: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            access_token: access_token.into(),
            project_id: project_id.into(),
            base_url: DEFAULT_CODE_ASSIST_BASE_URL.to_owned(),
            tools: Vec::new(),
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

    fn endpoint_url(&self, stream: bool) -> String {
        let base = self.base_url.trim_end_matches('/');
        let model = self.model.trim_start_matches("models/");
        let method = if stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        if base.ends_with("/v1internal") {
            format!("{base}/{model}:{method}")
        } else {
            format!("{base}/v1internal/{model}:{method}")
        }
    }

    async fn send(
        &self,
        options: &LanguageModelOptions,
        stream: bool,
    ) -> Result<reqwest::Response> {
        let body = build_code_assist_body(options, &self.tools);
        let mut req = self
            .http
            .post(self.endpoint_url(stream))
            .bearer_auth(&self.access_token)
            .header("User-Agent", GEMINI_CLI_USER_AGENT)
            .header("X-Goog-Api-Client", GEMINI_CLI_API_CLIENT)
            .header("Client-Metadata", GEMINI_CLIENT_METADATA)
            .header("Content-Type", "application/json");

        if !self.project_id.trim().is_empty() {
            req = req.header("X-Goog-User-Project", &self.project_id);
        }
        if stream {
            req = req.header("Accept", "text/event-stream");
        }
        for (k, v) in options.headers.iter().flatten() {
            req = req.header(k, v);
        }

        let resp = req.json(&body).send().await.map_err(|e| {
            let (status_code, why) = crate::describe_reqwest_error(&e);
            Error::ApiError {
                status_code,
                details: format!("Google Code Assist request failed: {why}"),
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
        let err_msg = body_json["error"]["message"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| text.chars().take(500).collect());
        let mut details = format!("HTTP {}: {err_msg}", status.as_u16());
        if let Some(ra) = retry_after {
            details.push_str(&format!(" [retry_after={ra}]"));
        }
        Err(Error::ApiError {
            status_code: Some(status),
            details,
        })
    }
}

impl fmt::Debug for GoogleCodeAssistModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleCodeAssistModel")
            .field("model", &self.model)
            .field("project_id", &self.project_id)
            .field("base_url", &self.base_url)
            .field("tools_count", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl TextInputSupport for GoogleCodeAssistModel {}
impl TextOutputSupport for GoogleCodeAssistModel {}
impl ToolCallSupport for GoogleCodeAssistModel {}
impl StructuredOutputSupport for GoogleCodeAssistModel {}
impl ReasoningSupport for GoogleCodeAssistModel {}
impl ImageInputSupport for GoogleCodeAssistModel {}

#[async_trait]
impl LanguageModel for GoogleCodeAssistModel {
    fn name(&self) -> String {
        self.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let resp = self.send(&options, false).await?;
        let data: Value = resp.json().await.map_err(|e| Error::ApiError {
            status_code: None,
            details: format!("invalid Google Code Assist JSON response: {e}"),
        })?;
        let payload = unwrap_code_assist_envelope(&data);
        let (text, reasoning, calls) = extract_gemini_parts(payload);
        let mut contents = Vec::new();
        if !reasoning.is_empty() {
            contents.push(LanguageModelResponseContentType::Reasoning {
                content: reasoning,
                extensions: Extensions::default(),
            });
        }
        if !text.is_empty() || calls.is_empty() {
            contents.push(LanguageModelResponseContentType::Text(text));
        }
        for call in calls {
            contents.push(LanguageModelResponseContentType::ToolCall(call));
        }
        Ok(LanguageModelResponse {
            contents,
            usage: parse_gemini_usage(&payload["usageMetadata"]),
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let response = self.send(&options, true).await?;
        let state = CodeAssistStreamState {
            response,
            decoder: SseDecoder::default(),
            text_started: false,
            reasoning_started: false,
            acc_text: String::new(),
            acc_reasoning: String::new(),
            acc_calls: Vec::new(),
            usage: None,
            ready: VecDeque::new(),
            ended: false,
        };
        Ok(Box::pin(futures::stream::unfold(
            state,
            |mut state| async move { state.next().await.map(|item| (item, state)) },
        )))
    }
}

struct CodeAssistStreamState {
    response: reqwest::Response,
    decoder: SseDecoder,
    text_started: bool,
    reasoning_started: bool,
    acc_text: String,
    acc_reasoning: String,
    acc_calls: Vec<ToolCallInfo>,
    usage: Option<Usage>,
    ready: VecDeque<Result<Vec<LanguageModelStreamChunk>>>,
    ended: bool,
}

impl CodeAssistStreamState {
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
                        self.fail("Google Code Assist sent oversized SSE frame")
                    }
                },
                Ok(None) => {
                    if let Some(last) = self.decoder.finish() {
                        self.on_event(&last);
                    }
                    self.finish_stream();
                    self.ended = true;
                }
                Err(err) => {
                    let (_, why) = crate::describe_reqwest_error(&err);
                    self.fail(&format!("Google Code Assist stream error: {why}"));
                }
            }
        }
    }

    fn on_event(&mut self, data: &str) {
        if self.ended || data == "[DONE]" {
            return;
        }
        let Ok(raw) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let payload = unwrap_code_assist_envelope(&raw);
        if let Some(u) = parse_gemini_usage(&payload["usageMetadata"]) {
            self.usage = Some(u);
        }
        let (text, reasoning, calls) = extract_gemini_parts(payload);
        let mut chunks = Vec::new();
        if !reasoning.is_empty() {
            if !self.reasoning_started {
                self.reasoning_started = true;
                chunks.push(LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::ReasoningStart,
                ));
            }
            self.acc_reasoning.push_str(&reasoning);
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::ReasoningDelta(reasoning),
            ));
        }
        if !text.is_empty() {
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
            self.acc_text.push_str(&text);
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::TextDelta(text),
            ));
        }
        for call in calls {
            let args_str = call.input.to_string();
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::ToolCallStart(call.tool.clone()),
            ));
            chunks.push(LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::ToolCallDelta {
                    id: call.tool.id.clone(),
                    delta: args_str,
                },
            ));
            self.acc_calls.push(call);
        }
        if !chunks.is_empty() {
            self.ready.push_back(Ok(chunks));
        }
    }

    fn finish_stream(&mut self) {
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

        let usage = self.usage.take().unwrap_or_default();
        let mut items = Vec::new();
        if !self.acc_reasoning.is_empty() {
            items.push(LanguageModelResponseContentType::Reasoning {
                content: std::mem::take(&mut self.acc_reasoning),
                extensions: Extensions::default(),
            });
        }
        if !self.acc_text.is_empty() || self.acc_calls.is_empty() {
            items.push(LanguageModelResponseContentType::Text(std::mem::take(
                &mut self.acc_text,
            )));
        }
        for call in std::mem::take(&mut self.acc_calls) {
            items.push(LanguageModelResponseContentType::ToolCall(call));
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

fn unwrap_code_assist_envelope(v: &Value) -> &Value {
    if v["response"].is_object() && v["response"]["candidates"].is_array() {
        &v["response"]
    } else {
        v
    }
}

fn extract_gemini_parts(payload: &Value) -> (String, String, Vec<ToolCallInfo>) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();

    for cand in payload["candidates"].as_array().into_iter().flatten() {
        for part in cand["content"]["parts"].as_array().into_iter().flatten() {
            let is_thought = part["thought"].as_bool().unwrap_or(false);
            if let Some(t) = part["text"].as_str() {
                if is_thought {
                    reasoning.push_str(t);
                } else {
                    text.push_str(t);
                }
            }
            if part["functionCall"].is_object() {
                let fc = &part["functionCall"];
                let name = fc["name"].as_str().unwrap_or_default().to_owned();
                let id = fc["id"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
                let args = if fc["args"].is_object() {
                    fc["args"].clone()
                } else {
                    json!({})
                };
                calls.push(ToolCallInfo {
                    tool: ToolDetails { name, id },
                    input: args,
                    extensions: Extensions::default(),
                });
            }
        }
    }
    (text, reasoning, calls)
}

fn parse_gemini_usage(meta: &Value) -> Option<Usage> {
    if !meta.is_object() {
        return None;
    }
    let count = |k: &str| meta[k].as_u64().map(|n| n as usize);
    Some(Usage {
        input_tokens: count("promptTokenCount"),
        output_tokens: count("candidatesTokenCount"),
        cached_tokens: count("cachedContentTokenCount"),
        reasoning_tokens: count("thoughtsTokenCount"),
    })
}

pub fn build_code_assist_body(options: &LanguageModelOptions, tools: &[Tool]) -> Value {
    let mut system_parts: Vec<String> = options.system.iter().cloned().collect();
    let mut contents: Vec<Value> = Vec::new();

    for msg in options.messages() {
        match msg {
            Message::System(s) => {
                if !s.content.trim().is_empty() && !system_parts.contains(&s.content) {
                    system_parts.push(s.content);
                }
            }
            Message::Developer(d) => {
                contents.push(json!({
                    "role": "user",
                    "parts": [{ "text": format!("<developer>\n{d}\n</developer>") }],
                }));
            }
            Message::User(u) => {
                contents.push(json!({
                    "role": "user",
                    "parts": [{ "text": u.content }],
                }));
            }
            Message::Assistant(a) => match a.content {
                LanguageModelResponseContentType::Text(t) => {
                    contents.push(json!({
                        "role": "model",
                        "parts": [{ "text": t }],
                    }));
                }
                LanguageModelResponseContentType::ToolCall(tc) => {
                    contents.push(json!({
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": tc.tool.name,
                                "args": tc.input,
                            }
                        }],
                    }));
                }
                LanguageModelResponseContentType::Reasoning { .. }
                | LanguageModelResponseContentType::NotSupported(_) => {}
            },
            Message::Tool(tr) => {
                let resp_obj = match tr.output {
                    Ok(Value::Object(map)) => Value::Object(map),
                    Ok(other) => json!({ "result": other }),
                    Err(e) => json!({ "error": e.to_string() }),
                };
                contents.push(json!({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": if tr.tool.name.is_empty() { "tool".to_owned() } else { tr.tool.name },
                            "response": resp_obj,
                        }
                    }],
                }));
            }
        }
    }

    let mut body = Map::new();
    body.insert("contents".into(), Value::Array(contents));

    system_parts.retain(|p| !p.trim().is_empty());
    if !system_parts.is_empty() {
        body.insert(
            "systemInstruction".into(),
            json!({ "parts": [{ "text": system_parts.join("\n\n") }] }),
        );
    }

    if !tools.is_empty() {
        let decls: Vec<Value> = tools
            .iter()
            .map(|t| {
                let mut params = t.input_schema.clone().to_value();
                if let Some(obj) = params.as_object_mut() {
                    obj.remove("$schema");
                }
                json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": params,
                })
            })
            .collect();
        body.insert("tools".into(), json!([{ "functionDeclarations": decls }]));
    }

    let mut gen_cfg = Map::new();
    gen_cfg.insert("candidateCount".into(), json!(1));
    if let Some(temp) = options.temperature {
        gen_cfg.insert("temperature".into(), json!(temp as f64 / 100.0));
    }
    if let Some(top_p) = options.top_p {
        gen_cfg.insert("topP".into(), json!(top_p as f64 / 100.0));
    }
    if let Some(top_k) = options.top_k {
        gen_cfg.insert("topK".into(), json!(top_k));
    }
    if let Some(max_tok) = options.max_output_tokens {
        gen_cfg.insert("maxOutputTokens".into(), json!(max_tok));
    }
    if let Some(stops) = &options.stop_sequences
        && !stops.is_empty()
    {
        gen_cfg.insert("stopSequences".into(), json!(stops));
    }
    if let Some(schema) = &options.schema {
        let mut sv = serde_json::to_value(schema).unwrap_or_default();
        if let Some(obj) = sv.as_object_mut() {
            obj.remove("$schema");
        }
        gen_cfg.insert("responseMimeType".into(), json!("application/json"));
        gen_cfg.insert("responseSchema".into(), sv);
    }
    if let Some(effort) = options.reasoning_effort {
        let budget = match effort {
            ReasoningEffort::None => 0,
            ReasoningEffort::Low => 1024,
            ReasoningEffort::Medium => 4096,
            ReasoningEffort::High => 8192,
            ReasoningEffort::XHigh => 16384,
        };
        gen_cfg.insert(
            "thinkingConfig".into(),
            json!({ "thinkingBudget": budget, "includeThoughts": budget > 0 }),
        );
    }
    body.insert("generationConfig".into(), Value::Object(gen_cfg));

    if let Some(extra) = &options.body {
        body.extend(extra.clone());
    }
    Value::Object(body)
}
