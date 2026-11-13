//! Shared message/tool normalization, token estimation, header extraction,
//! and request-completion accounting across all polyglot ingress routes.

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use aisdk::core::language_model::{LanguageModelResponseContentType, ReasoningEffort};
use aisdk::core::tools::ToolDetails;
use aisdk::core::{
    AssistantMessage, Message, SystemMessage, ToolCallInfo, ToolResultInfo, UserMessage,
};
use axum::body::Bytes;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::{BufMut, BytesMut};
use futures::Stream;
use quixigate_core::{
    AuthKind, GatewaySettings, IngressProtocol, ModelRoute, NanoUsd, QuixigateError, RequestLog,
    SecretScannerMode, now_ms,
};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::guardrails::redact_secrets_in_str;
use quixigate_router::{
    CanonicalRequest, ClientToolSpec, CutThroughActiveStream, CutThroughWireFormat, FlybyUsage,
    parse_flyby_anthropic_usage, parse_flyby_responses_usage, parse_flyby_sse_usage,
};
use quixigate_telemetry::{OpenInferencePayload, TelemetryHub};
use serde::Deserialize;
use serde::de::{Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde_json::{Map, Value, json};

use crate::state::AppState;

// ── Shared Zero-DOM Envelope Peek Types ──────────────────────────────────────

pub struct MessagesArrayMarker;

impl<'de> Deserialize<'de> for MessagesArrayMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ArraySkipper;
        impl<'de> Visitor<'de> for ArraySkipper {
            type Value = MessagesArrayMarker;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON array of messages")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(MessagesArrayMarker)
            }
        }
        deserializer.deserialize_seq(ArraySkipper)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum TagsPeek<'a> {
    #[serde(borrow)]
    List(Vec<Cow<'a, str>>),
    #[serde(borrow)]
    Csv(Cow<'a, str>),
    Other(IgnoredAny),
}

#[derive(Deserialize, Default)]
pub struct MetadataPeek<'a> {
    #[serde(borrow, default)]
    pub tags: Option<TagsPeek<'a>>,
    #[serde(borrow, default)]
    pub user_id: Option<Cow<'a, str>>,
}

#[inline]
pub fn collect_peek_tags(
    headers: &HeaderMap,
    metadata: Option<&MetadataPeek<'_>>,
    top_tags: Option<&TagsPeek<'_>>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |raw: &str| {
        let t = raw.trim();
        if !t.is_empty() && !out.iter().any(|e| e == t) {
            out.push(t.to_owned());
        }
    };
    for hdr_name in [&HDR_QG_TAGS, &HDR_LITELLM_TAGS] {
        if let Some(val) = headers.get(hdr_name).and_then(|v| v.to_str().ok()) {
            for item in val.split(',') {
                push(item);
            }
        }
    }
    for peek_opt in [metadata.and_then(|m| m.tags.as_ref()), top_tags]
        .into_iter()
        .flatten()
    {
        match peek_opt {
            TagsPeek::List(arr) => {
                for s in arr {
                    push(s);
                }
            }
            TagsPeek::Csv(csv) => {
                for item in csv.split(',') {
                    push(item);
                }
            }
            TagsPeek::Other(_) => {}
        }
    }
    out
}

#[inline]
pub fn peek_end_user<'b>(
    headers: &'b HeaderMap,
    user: Option<&'b str>,
    end_user_id: Option<&'b str>,
) -> Option<&'b str> {
    for cand in [user, end_user_id] {
        if let Some(s) = cand.map(str::trim)
            && !s.is_empty()
        {
            return Some(s);
        }
    }
    for hdr_name in [&HDR_LITELLM_END_USER, &HDR_X_END_USER] {
        if let Some(s) = headers
            .get(hdr_name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            && !s.is_empty()
        {
            return Some(s);
        }
    }
    None
}

struct ByteLenWriter(usize);

impl std::io::Write for ByteLenWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[inline]
fn json_serialized_len(val: &Value) -> usize {
    let mut w = ByteLenWriter(0);
    let _ = serde_json::to_writer(&mut w, val);
    w.0
}

// ── Zero-AST `BytesMut` SSE Formatting Helpers ───────────────────────────────

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Appends a JSON-quoted, RFC 8259-escaped string (`"..."`) directly into `buf` with zero heap
/// allocations. Fast-paths clean UTF-8 token deltas via a single slice copy.
#[inline]
pub fn put_json_escaped_str(buf: &mut BytesMut, s: &str) {
    let bytes = s.as_bytes();
    buf.reserve(bytes.len() + 2);
    buf.put_u8(b'"');
    if !bytes.iter().any(|&b| b < 0x20 || b == b'"' || b == b'\\') {
        buf.extend_from_slice(bytes);
        buf.put_u8(b'"');
        return;
    }
    let mut last = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        let esc: &[u8] = match b {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            0x08 => b"\\b",
            0x0c => b"\\f",
            0x00..=0x1f => {
                if i > last {
                    buf.extend_from_slice(&bytes[last..i]);
                }
                buf.extend_from_slice(&[
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX_DIGITS[(b >> 4) as usize],
                    HEX_DIGITS[(b & 0x0f) as usize],
                ]);
                last = i + 1;
                continue;
            }
            _ => continue,
        };
        if i > last {
            buf.extend_from_slice(&bytes[last..i]);
        }
        buf.extend_from_slice(esc);
        last = i + 1;
    }
    if last < bytes.len() {
        buf.extend_from_slice(&bytes[last..]);
    }
    buf.put_u8(b'"');
}

#[inline]
pub fn put_u64_dec(buf: &mut BytesMut, n: u64) {
    let mut it = itoa::Buffer::new();
    buf.extend_from_slice(it.format(n).as_bytes());
}

#[inline]
pub fn put_i64_dec(buf: &mut BytesMut, n: i64) {
    let mut it = itoa::Buffer::new();
    buf.extend_from_slice(it.format(n).as_bytes());
}

#[inline]
pub fn put_usize_dec(buf: &mut BytesMut, n: usize) {
    let mut it = itoa::Buffer::new();
    buf.extend_from_slice(it.format(n).as_bytes());
}

#[inline]
pub fn put_json_value(buf: &mut BytesMut, val: &Value) {
    let _ = serde_json::to_writer(buf.writer(), val);
}

// ── Smart Zero-Copy + Micro-Coalescing Cut-Through SSE Stream ────────────────

/// Chunks `>= 32 KiB` bypass `BytesMut` completely (`0` alloc, `0` `memcpy`), and single ready
/// chunks `< 32 KiB` also pass straight through with zero copy when no second chunk is queued.
const ZERO_COPY_BYPASS_LEN: usize = 32 * 1024;
/// Multiple ready chunks arriving in the same reactor wakeup coalesce up to `32 KiB` per `writev`.
const MICRO_COALESCE_CAP: usize = 32 * 1024;

pub struct CutThroughSseStream {
    upstream: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
    wire_format: CutThroughWireFormat,
    ingress_protocol: IngressProtocol,
    request_id: String,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    auth_kind: AuthKind,
    route: Option<Arc<ModelRoute>>,
    fallback_attempts: u32,
    est_input_tokens: u64,
    started: Instant,
    ttft_ms: Option<u64>,
    chunk_count: u64,
    stream_error: Option<String>,
    head_chunk: Option<Bytes>,
    prev_chunk: Option<Bytes>,
    last_chunk: Option<Bytes>,
    pending_chunk: Option<Bytes>,
    coalesce_buf: BytesMut,
    eof: bool,
}

impl CutThroughSseStream {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        active: CutThroughActiveStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        wire_format: CutThroughWireFormat,
        ingress_protocol: IngressProtocol,
        request_id: String,
        est_input_tokens: u64,
    ) -> Self {
        Self {
            upstream: Box::pin(active.response.bytes_stream()),
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
            wire_format,
            ingress_protocol,
            request_id,
            requested_model: active.requested_model,
            resolved_model: active.resolved_model,
            provider_id: active.provider_id,
            credential_id: active.credential_id,
            auth_kind: active.auth_kind,
            route: active.route,
            fallback_attempts: active.fallback_attempts,
            est_input_tokens,
            started: active.started,
            ttft_ms: None,
            chunk_count: 0,
            stream_error: None,
            head_chunk: None,
            prev_chunk: None,
            last_chunk: None,
            pending_chunk: None,
            coalesce_buf: BytesMut::new(),
            eof: false,
        }
    }

    #[inline]
    fn observe_chunk(&mut self, chunk: &Bytes) {
        if chunk.is_empty() {
            return;
        }
        if self.ttft_ms.is_none() {
            self.ttft_ms = Some(self.started.elapsed().as_millis() as u64);
        }
        if self.wire_format == CutThroughWireFormat::AnthropicMessages && self.head_chunk.is_none()
        {
            self.head_chunk = Some(chunk.slice(..chunk.len().min(1024)));
        }
        self.chunk_count += 1;
        self.prev_chunk = self.last_chunk.replace(chunk.clone());
    }

    fn extract_flyby_usage(&self) -> Option<FlybyUsage> {
        let head = self.head_chunk.as_deref();
        let parse_tail = |tail: &[u8]| -> Option<FlybyUsage> {
            match self.wire_format {
                CutThroughWireFormat::OpenAiChat => parse_flyby_sse_usage(tail),
                CutThroughWireFormat::OpenAiResponses => parse_flyby_responses_usage(tail),
                CutThroughWireFormat::AnthropicMessages => parse_flyby_anthropic_usage(head, tail),
            }
        };

        match (&self.prev_chunk, &self.last_chunk) {
            (_, None) => None,
            (None, Some(last)) => parse_tail(last),
            (Some(prev), Some(last)) => parse_tail(last).or_else(|| {
                let prev_tail = &prev[prev.len().saturating_sub(8192)..];
                let mut stitched = Vec::with_capacity(prev_tail.len() + last.len());
                stitched.extend_from_slice(prev_tail);
                stitched.extend_from_slice(last);
                parse_tail(&stitched)
            }),
        }
    }
}

impl Stream for CutThroughSseStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.eof {
            return Poll::Ready(None);
        }

        let mut first_chunk: Option<Bytes> = None;
        if let Some(pending) = this.pending_chunk.take() {
            if pending.len() >= ZERO_COPY_BYPASS_LEN {
                return Poll::Ready(Some(Ok(pending)));
            }
            first_chunk = Some(pending);
        }

        loop {
            match this.upstream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    if chunk.is_empty() {
                        continue;
                    }
                    this.observe_chunk(&chunk);
                    if first_chunk.is_none() && this.coalesce_buf.is_empty() {
                        if chunk.len() >= ZERO_COPY_BYPASS_LEN {
                            // Pure zero-copy fast path: pass large kernel TCP reads straight to Hyper!
                            return Poll::Ready(Some(Ok(chunk)));
                        }
                        first_chunk = Some(chunk);
                    } else if chunk.len() >= ZERO_COPY_BYPASS_LEN {
                        // Flush accumulated micro-frames first and preserve zero-copy for `chunk`!
                        let out = if !this.coalesce_buf.is_empty() {
                            this.coalesce_buf.split().freeze()
                        } else {
                            first_chunk.take().expect("first_chunk checked")
                        };
                        this.pending_chunk = Some(chunk);
                        return Poll::Ready(Some(Ok(out)));
                    } else {
                        if let Some(prev) = first_chunk.take() {
                            this.coalesce_buf
                                .reserve((prev.len() + chunk.len()).max(MICRO_COALESCE_CAP));
                            this.coalesce_buf.extend_from_slice(&prev);
                        }
                        this.coalesce_buf.extend_from_slice(&chunk);
                        if this.coalesce_buf.len() >= MICRO_COALESCE_CAP {
                            return Poll::Ready(Some(Ok(this.coalesce_buf.split().freeze())));
                        }
                    }
                }
                Poll::Ready(Some(Err(err))) => {
                    this.stream_error = Some(err.to_string());
                    this.eof = true;
                    if !this.coalesce_buf.is_empty() {
                        return Poll::Ready(Some(Ok(this.coalesce_buf.split().freeze())));
                    }
                    if let Some(prev) = first_chunk.take() {
                        return Poll::Ready(Some(Ok(prev)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Ready(None) => {
                    this.eof = true;
                    if !this.coalesce_buf.is_empty() {
                        return Poll::Ready(Some(Ok(this.coalesce_buf.split().freeze())));
                    }
                    if let Some(prev) = first_chunk.take() {
                        return Poll::Ready(Some(Ok(prev)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    if !this.coalesce_buf.is_empty() {
                        return Poll::Ready(Some(Ok(this.coalesce_buf.split().freeze())));
                    }
                    if let Some(prev) = first_chunk.take() {
                        return Poll::Ready(Some(Ok(prev)));
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

impl Drop for CutThroughSseStream {
    fn drop(&mut self) {
        self.state.telemetry.dec_active();
        drop(self.conc_guard.take());
        let Some(budget_guard) = self.budget_guard.take() else {
            return;
        };

        let duration_ms = self.started.elapsed().as_millis() as u64;
        let flyby = self.extract_flyby_usage();

        let final_resolved_model = flyby
            .as_ref()
            .and_then(|u| u.resolved_model.clone())
            .unwrap_or_else(|| self.resolved_model.clone());
        let input_tokens = flyby
            .as_ref()
            .map(|u| u.input_tokens)
            .unwrap_or(self.est_input_tokens);
        let cached_input_tokens = flyby.as_ref().map(|u| u.cached_input_tokens).unwrap_or(0);
        let reasoning_tokens = flyby.as_ref().map(|u| u.reasoning_tokens).unwrap_or(0);
        let output_tokens = flyby
            .as_ref()
            .map(|u| u.output_tokens)
            .unwrap_or(self.chunk_count);
        let exact_cost = flyby.and_then(|u| u.exact_cost_nanos);

        if self.auth_kind == AuthKind::LocalNoauth
            || matches!(
                self.provider_id.as_str(),
                "ollama" | "lmstudio" | "llamacpp" | "vllm" | "mlx"
            )
        {
            let tps =
                (duration_ms > 0).then(|| (output_tokens as f64) / (duration_ms as f64 / 1000.0));
            self.state.router.local_scanner().record_performance(
                &self.provider_id,
                self.ttft_ms,
                tps,
            );
        }

        let (cost_nanos, api_equiv_nanos) = self.state.router.compute_cut_through_cost(
            &self.provider_id,
            self.auth_kind,
            &final_resolved_model,
            input_tokens,
            cached_input_tokens,
            output_tokens,
            exact_cost,
            self.route.as_deref(),
        );

        let status_code = if self.stream_error.is_some() {
            502
        } else {
            200
        };
        finalize_cut_through_accounting(
            &self.state,
            &self.principal,
            budget_guard,
            self.ingress_protocol,
            std::mem::take(&mut self.request_id),
            std::mem::take(&mut self.requested_model),
            final_resolved_model,
            std::mem::take(&mut self.provider_id),
            self.credential_id.take(),
            status_code,
            true,
            self.fallback_attempts,
            self.ttft_ms,
            duration_ms,
            input_tokens,
            cached_input_tokens,
            reasoning_tokens,
            output_tokens,
            cost_nanos,
            api_equiv_nanos,
            self.stream_error.take(),
        );
    }
}

/// Rough fast token estimator (~4 chars per token + message framing) used for
/// pre-flight budget reservation and `/v1/messages/count_tokens`.
pub fn estimate_request_tokens(req: &CanonicalRequest) -> u64 {
    let mut chars = req.system.as_ref().map_or(0usize, String::len);
    for msg in &req.messages {
        chars += match msg {
            Message::System(s) => s.content.len() + 12,
            Message::Developer(d) => d.len() + 12,
            Message::User(u) => u.content.len() + 12,
            Message::Assistant(a) => match &a.content {
                LanguageModelResponseContentType::Text(t) => t.len() + 12,
                LanguageModelResponseContentType::ToolCall(tc) => {
                    tc.tool.name.len() + json_serialized_len(&tc.input) + 24
                }
                LanguageModelResponseContentType::Reasoning { content, .. } => content.len() + 16,
                LanguageModelResponseContentType::NotSupported(s) => s.len(),
            },
            Message::Tool(tr) => match &tr.output {
                Ok(v) => json_serialized_len(v) + 16,
                Err(e) => e.to_string().len() + 16,
            },
        };
    }
    for tool in &req.client_tools {
        chars += tool.name.len() + tool.description.len() + json_serialized_len(&tool.parameters);
    }
    ((chars / 4) as u64).max(1) + 8
}

/// Converts a float temperature (`0.0..=2.0`) to `aisdk`'s integer scale (`0..=100` / percentage).
pub fn scale_temperature(val: Option<f64>) -> Option<u32> {
    val.map(|v| (v.clamp(0.0, 2.0) * 100.0).round() as u32)
}

pub fn scale_top_p(val: Option<f64>) -> Option<u32> {
    val.map(|v| (v.clamp(0.0, 1.0) * 100.0).round() as u32)
}

pub fn parse_reasoning_effort(raw: Option<&str>) -> Option<ReasoningEffort> {
    match raw?.trim().to_ascii_lowercase().as_str() {
        "none" | "disabled" => Some(ReasoningEffort::None),
        "low" | "minimal" => Some(ReasoningEffort::Low),
        "medium" | "med" => Some(ReasoningEffort::Medium),
        "high" | "max" => Some(ReasoningEffort::High),
        _ => None,
    }
}

const HDR_QG_CACHE_CONTROL: HeaderName = HeaderName::from_static("x-quixigate-cache-control");
const HDR_ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");
const HDR_X_SESSION_ID: HeaderName = HeaderName::from_static("x-session-id");
const HDR_PROMPT_CACHE_KEY: HeaderName = HeaderName::from_static("prompt_cache_key");
const HDR_HTTP_REFERER: HeaderName = HeaderName::from_static("http-referer");
const HDR_X_TITLE: HeaderName = HeaderName::from_static("x-title");

/// Reads per-request cache preference from `Cache-Control` or `x-quixigate-cache-control`.
#[inline]
pub fn parse_cache_preference(headers: &HeaderMap) -> Option<bool> {
    if let Some(v) = headers
        .get(&HDR_QG_CACHE_CONTROL)
        .and_then(|h| h.to_str().ok())
    {
        let lower = v.to_ascii_lowercase();
        if lower.contains("no-cache") || lower.contains("bypass") || lower == "false" {
            return Some(false);
        }
        if lower.contains("enable") || lower.contains("force") || lower == "true" {
            return Some(true);
        }
    }
    if let Some(cc) = headers
        .get(axum::http::header::CACHE_CONTROL)
        .and_then(|h| h.to_str().ok())
    {
        let lower = cc.to_ascii_lowercase();
        if lower.contains("no-cache") || lower.contains("no-store") {
            return Some(false);
        }
    }
    None
}

/// Extracts prompt-cache affinity headers (`anthropic-beta`, `x-session-id`, etc.) and optional
/// OpenRouter attribution overrides (`HTTP-Referer`, `X-Title`) as a zero-alloc-when-empty `Vec`.
#[inline]
pub fn extract_passthrough_headers_vec(headers: &HeaderMap) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (hdr, key) in [
        (&HDR_ANTHROPIC_BETA, "anthropic-beta"),
        (&HDR_X_SESSION_ID, "x-session-id"),
        (&HDR_PROMPT_CACHE_KEY, "prompt_cache_key"),
    ] {
        if let Some(val) = headers.get(hdr).and_then(|v| v.to_str().ok()) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                out.push((key.to_owned(), trimmed.to_owned()));
            }
        }
    }
    if let Some(val) = headers.get(&HDR_HTTP_REFERER).and_then(|v| v.to_str().ok()) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            out.push(("HTTP-Referer".to_owned(), trimmed.to_owned()));
        }
    }
    if let Some(val) = headers.get(&HDR_X_TITLE).and_then(|v| v.to_str().ok()) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            out.push(("X-Title".to_owned(), trimmed.to_owned()));
        }
    }
    out
}

/// Extracts prompt-cache affinity headers (`anthropic-beta`, `x-session-id`, etc.) and optional
/// OpenRouter attribution overrides (`HTTP-Referer`, `X-Title`) to forward upstream.
pub fn extract_passthrough_headers(headers: &HeaderMap) -> HashMap<String, String> {
    extract_passthrough_headers_vec(headers)
        .into_iter()
        .collect()
}

/// Extracts OpenRouter-specific request parameters (`provider`, `transforms`, `models`,
/// `plugins`, `route`) so `GatewayRouter` forwards them exclusively when dispatching to an
/// `"openrouter"` seat (never polluting Local Runtimes or OAuth Subscription upstreams).
pub fn extract_openrouter_body(body: &Value) -> Map<String, Value> {
    let mut map = Map::new();
    for key in ["provider", "transforms", "models", "plugins", "route"] {
        if let Some(val) = body.get(key)
            && !val.is_null()
        {
            map.insert(key.to_owned(), val.clone());
        }
    }
    map
}

/// Returns `true` when `m` has at least one configured & enabled upstream seat or healthy local runtime.
/// Strictly gates OpenRouter-exclusive entries (`@preset/*`, `~*`, `openrouter/auto`, `openrouter/free`)
/// on `configured_providers.contains("openrouter")` so OpenRouter is never assumed when disconnected.
pub fn is_catalog_model_routable(
    m: &quixigate_router::catalog::CatalogModel,
    configured_providers: &std::collections::HashSet<String>,
) -> bool {
    if m.is_local {
        return true;
    }
    let is_openrouter_exclusive = m.is_preset
        || m.canonical_id.starts_with("@preset/")
        || m.canonical_id.starts_with('~')
        || m.canonical_id == "openrouter/auto"
        || m.canonical_id == "openrouter/free";
    if is_openrouter_exclusive {
        return configured_providers.contains("openrouter")
            && m.providers.contains_key("openrouter");
    }
    configured_providers.contains("openaicompatible")
        || configured_providers.contains("vercel")
        || configured_providers.contains("cloudflare-ai-gateway")
        || configured_providers.contains("helicone")
        || (configured_providers.contains("qwen-portal") && m.author == "qwen")
        || (configured_providers.contains("minimax-portal") && m.author == "minimax")
        || m.providers.keys().any(|p| {
            configured_providers.contains(p)
                || (p == "openai" && configured_providers.contains("openai-codex"))
                || (p == "anthropic" && configured_providers.contains("anthropic-token"))
                || (p == "google" && configured_providers.contains("google-gemini-cli"))
                || (p == "xai" && configured_providers.contains("xai-oauth"))
                || (p == "alibaba" && configured_providers.contains("qwen-portal"))
                || (p == "minimax" && configured_providers.contains("minimax-portal"))
                || (matches!(p.as_str(), "openai" | "anthropic" | "google")
                    && configured_providers.contains("github-copilot"))
        })
}

/// Parses OpenAI-style message content (`"string"` or `[{"type":"text","text":"..."}, {"type":"image_url",...}]`)
/// into consolidated text (preserving image URLs/descriptions cleanly so multimodal context is never lost).
pub fn extract_content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = Vec::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        if let Some(t) = part["text"].as_str() {
                            out.push(t.to_owned());
                        }
                    }
                    Some("image_url") | Some("input_image") => {
                        let url = part["image_url"]["url"]
                            .as_str()
                            .or_else(|| part["image_url"].as_str())
                            .or_else(|| part["url"].as_str())
                            .unwrap_or("<image>");
                        let preview = if url.len() > 96 {
                            format!("{}...", &url[..96])
                        } else {
                            url.to_owned()
                        };
                        out.push(format!("[image: {preview}]"));
                    }
                    _ => {
                        if let Some(t) = part["text"].as_str().or_else(|| part["content"].as_str())
                        {
                            out.push(t.to_owned());
                        }
                    }
                }
            }
            out.join("\n")
        }
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Normalizes an OpenAI-style `messages` JSON array into `(system_prompt, Vec<aisdk::core::Message>)`.
pub fn parse_openai_messages(raw_messages: &[Value]) -> (Option<String>, Vec<Message>) {
    let mut system_parts = Vec::new();
    let mut messages = Vec::new();
    let mut call_id_to_name: HashMap<String, String> = HashMap::new();

    for msg in raw_messages {
        let role = msg["role"].as_str().unwrap_or("user");
        match role {
            "system" => {
                let txt = extract_content_text(&msg["content"]);
                if !txt.trim().is_empty() {
                    system_parts.push(txt.clone());
                    messages.push(Message::System(SystemMessage::new(txt)));
                }
            }
            "developer" => {
                let txt = extract_content_text(&msg["content"]);
                if !txt.trim().is_empty() {
                    system_parts.push(txt.clone());
                    messages.push(Message::Developer(txt));
                }
            }
            "user" => {
                messages.push(Message::User(UserMessage::new(extract_content_text(
                    &msg["content"],
                ))));
            }
            "assistant" => {
                if let Some(reasoning) = msg["reasoning_content"]
                    .as_str()
                    .or_else(|| msg["reasoning"].as_str())
                    .filter(|s| !s.is_empty())
                {
                    messages.push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::Reasoning {
                            content: reasoning.to_owned(),
                            extensions: Default::default(),
                        },
                        None,
                    )));
                }
                let txt = extract_content_text(&msg["content"]);
                if !txt.is_empty() {
                    messages.push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::Text(txt),
                        None,
                    )));
                }
                if let Some(tool_calls) = msg["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let id = tc["id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                            .unwrap_or_else(|| TelemetryHub::fast_prefixed_id("call_"));
                        let name = tc["function"]["name"]
                            .as_str()
                            .or_else(|| tc["name"].as_str())
                            .unwrap_or("tool")
                            .to_owned();
                        let args_val = match &tc["function"]["arguments"] {
                            Value::String(s) => serde_json::from_str::<Value>(s)
                                .unwrap_or_else(|_| json!({ "raw": s })),
                            v if v.is_object() => v.clone(),
                            _ => json!({}),
                        };
                        call_id_to_name.insert(id.clone(), name.clone());
                        messages.push(Message::Assistant(AssistantMessage::new(
                            LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                                tool: ToolDetails { id, name },
                                input: args_val,
                                extensions: Default::default(),
                            }),
                            None,
                        )));
                    }
                }
            }
            "tool" | "function" => {
                let call_id = msg["tool_call_id"]
                    .as_str()
                    .or_else(|| msg["id"].as_str())
                    .unwrap_or("call_result")
                    .to_owned();
                let name = msg["name"]
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| call_id_to_name.get(&call_id).cloned())
                    .unwrap_or_else(|| "tool".to_owned());
                let out_str = extract_content_text(&msg["content"]);
                let output_val =
                    serde_json::from_str::<Value>(&out_str).unwrap_or(Value::String(out_str));
                messages.push(Message::Tool(ToolResultInfo {
                    tool: ToolDetails { id: call_id, name },
                    output: Ok(output_val),
                }));
            }
            _ => {
                messages.push(Message::User(UserMessage::new(extract_content_text(
                    &msg["content"],
                ))));
            }
        }
    }

    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    (system, messages)
}

/// Parses OpenAI `tools` or legacy `functions` array into `Vec<ClientToolSpec>`.
pub fn parse_openai_tools(body: &Value) -> Vec<ClientToolSpec> {
    let mut out = Vec::new();
    if let Some(tools) = body["tools"].as_array() {
        for tool in tools {
            if let Some(func) = tool.get("function") {
                if let Some(name) = func["name"].as_str() {
                    out.push(ClientToolSpec {
                        name: name.to_owned(),
                        description: func["description"].as_str().unwrap_or_default().to_owned(),
                        parameters: func
                            .get("parameters")
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                    });
                }
            } else if let Some(name) = tool["name"].as_str() {
                out.push(ClientToolSpec {
                    name: name.to_owned(),
                    description: tool["description"].as_str().unwrap_or_default().to_owned(),
                    parameters: tool
                        .get("parameters")
                        .or_else(|| tool.get("input_schema"))
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                });
            }
        }
    }
    out
}

/// Builds OpenInference input message pairs & tool schema strings from `CanonicalRequest`.
pub fn build_openinference_inputs(
    req: &CanonicalRequest,
    session_id: Option<String>,
) -> OpenInferencePayload {
    let mut input_messages = Vec::new();
    if let Some(sys) = &req.system
        && !sys.trim().is_empty()
    {
        input_messages.push(("system".to_owned(), sys.clone()));
    }
    for msg in &req.messages {
        match msg {
            Message::System(_) => {}
            Message::Developer(d) => input_messages.push(("developer".to_owned(), d.clone())),
            Message::User(u) => input_messages.push(("user".to_owned(), u.content.clone())),
            Message::Assistant(a) => match &a.content {
                LanguageModelResponseContentType::Text(t) => {
                    input_messages.push(("assistant".to_owned(), t.clone()));
                }
                LanguageModelResponseContentType::ToolCall(tc) => {
                    input_messages.push((
                        "assistant".to_owned(),
                        format!("tool_call:{}({})", tc.tool.name, tc.input),
                    ));
                }
                _ => {}
            },
            Message::Tool(tr) => {
                let val = match &tr.output {
                    Ok(v) => v.to_string(),
                    Err(e) => e.to_string(),
                };
                input_messages.push(("tool".to_owned(), val));
            }
        }
    }

    let tool_schemas_json = req
        .client_tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
            .to_string()
        })
        .collect();

    let mut params = Map::new();
    if let Some(t) = req.temperature {
        params.insert("temperature".into(), json!((t as f64) / 100.0));
    }
    if let Some(m) = req.max_output_tokens {
        params.insert("max_output_tokens".into(), json!(m));
    }

    OpenInferencePayload {
        invocation_parameters_json: (!params.is_empty()).then(|| Value::Object(params).to_string()),
        input_messages,
        output_text: None,
        output_tool_calls: Vec::new(),
        tool_schemas_json,
        session_id,
        tags: vec!["quixigate".to_owned()],
    }
}

/// Lazily constructs `OpenInferencePayload` and serialized `raw_request_json` only when
/// `capture_payloads` or `otlp_enabled` is active, avoiding DOM re-serialization on normal traffic.
pub fn prepare_telemetry_capture(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    req: &CanonicalRequest,
    body: &Value,
    session_id: Option<String>,
) -> (OpenInferencePayload, Option<String>) {
    let settings = state.db.settings_arc();
    let capture_payloads = principal
        .virtual_key
        .as_ref()
        .and_then(|k| k.log_payloads)
        .unwrap_or(settings.capture_payloads_default);
    let oi = if capture_payloads || settings.otlp_enabled {
        build_openinference_inputs(req, session_id)
    } else {
        OpenInferencePayload::default()
    };
    let raw_req = capture_payloads.then(|| body.to_string());
    (oi, raw_req)
}

const HDR_QG_REQ_ID: HeaderName = HeaderName::from_static("x-quixigate-request-id");
const HDR_QG_PROVIDER: HeaderName = HeaderName::from_static("x-quixigate-provider");
const HDR_QG_MODEL: HeaderName = HeaderName::from_static("x-quixigate-model");
const HDR_QG_CACHE: HeaderName = HeaderName::from_static("x-quixigate-cache");
const HDR_QG_COST_NANOS: HeaderName = HeaderName::from_static("x-quixigate-cost-nanos");
const HDR_LITELLM_RESP_COST: HeaderName = HeaderName::from_static("x-litellm-response-cost");
const HDR_QG_API_EQUIV: HeaderName = HeaderName::from_static("x-quixigate-api-equiv-nanos");
const HDR_QG_TAGS: HeaderName = HeaderName::from_static("x-quixigate-tags");
const HDR_LITELLM_TAGS: HeaderName = HeaderName::from_static("x-litellm-tags");
const HDR_QG_NO_LOG: HeaderName = HeaderName::from_static("x-quixigate-no-log");
const HDR_LITELLM_NO_LOG: HeaderName = HeaderName::from_static("x-litellm-disable-callbacks");
const HDR_LITELLM_END_USER: HeaderName = HeaderName::from_static("x-litellm-end-user-id");
const HDR_X_END_USER: HeaderName = HeaderName::from_static("x-end-user-id");

#[inline]
pub fn attach_gateway_headers(
    headers: &mut HeaderMap,
    request_id: &str,
    provider_id: &str,
    resolved_model: &str,
    cache_hit: bool,
    cost_nanos: NanoUsd,
    api_equiv_nanos: NanoUsd,
) {
    let set = |headers: &mut HeaderMap, k: HeaderName, v: &str| {
        if let Ok(hv) = HeaderValue::from_str(v) {
            headers.insert(k, hv);
        }
    };
    set(headers, HDR_QG_REQ_ID, request_id);
    set(headers, HDR_QG_PROVIDER, provider_id);
    set(headers, HDR_QG_MODEL, resolved_model);
    headers.insert(
        HDR_QG_CACHE,
        if cache_hit {
            HeaderValue::from_static("HIT")
        } else {
            HeaderValue::from_static("MISS")
        },
    );
    if cost_nanos.is_zero() {
        headers.insert(HDR_QG_COST_NANOS, HeaderValue::from_static("0"));
        headers.insert(HDR_LITELLM_RESP_COST, HeaderValue::from_static("0.00"));
    } else {
        let mut it = itoa::Buffer::new();
        set(headers, HDR_QG_COST_NANOS, it.format(cost_nanos.nanos()));
        set(
            headers,
            HDR_LITELLM_RESP_COST,
            &cost_nanos.to_decimal_string(),
        );
    }
    if api_equiv_nanos.is_zero() {
        headers.insert(HDR_QG_API_EQUIV, HeaderValue::from_static("0"));
    } else {
        let mut it = itoa::Buffer::new();
        set(
            headers,
            HDR_QG_API_EQUIV,
            it.format(api_equiv_nanos.nanos()),
        );
    }
}

/// Extracts the concrete upstream `provider_id` when a request fails upstream so SQLite/OTel
/// trace rows attribute failures to `"ollama"`, `"openrouter"`, etc. instead of `"error"`.
pub fn error_provider_id(err: &quixigate_core::QuixigateError) -> String {
    match err {
        quixigate_core::QuixigateError::Upstream { provider, .. } => provider.clone(),
        _ => "gateway".to_owned(),
    }
}

/// Extracts per-request attribution tags from `x-quixigate-tags` / `x-litellm-tags` headers
/// and `body["metadata"]["tags"]` / `body["tags"]`.
pub fn extract_request_tags(headers: &HeaderMap, body: Option<&Value>) -> Vec<String> {
    let mut tags = Vec::new();
    let mut push_tag = |raw: &str| {
        let t = raw.trim();
        if !t.is_empty() && !tags.iter().any(|existing| existing == t) {
            tags.push(t.to_owned());
        }
    };
    for hdr_name in [&HDR_QG_TAGS, &HDR_LITELLM_TAGS] {
        if let Some(val) = headers.get(hdr_name).and_then(|v| v.to_str().ok()) {
            for item in val.split(',') {
                push_tag(item);
            }
        }
    }
    if let Some(b) = body {
        for candidate in [b.get("metadata").and_then(|m| m.get("tags")), b.get("tags")]
            .into_iter()
            .flatten()
        {
            match candidate {
                Value::Array(arr) => {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            push_tag(s);
                        }
                    }
                }
                Value::String(s) => {
                    for item in s.split(',') {
                        push_tag(item);
                    }
                }
                _ => {}
            }
        }
    }
    tags
}

/// Checks whether the client requested opt-out trace logging (`x-quixigate-no-log: true` or
/// `x-litellm-disable-callbacks: true`) and `settings.allow_no_log_header` permits it.
#[inline]
pub fn should_suppress_request_log(headers: &HeaderMap, settings: &GatewaySettings) -> bool {
    if !settings.allow_no_log_header {
        return false;
    }
    for hdr_name in [&HDR_QG_NO_LOG, &HDR_LITELLM_NO_LOG] {
        if let Some(v) = headers.get(hdr_name).and_then(|h| h.to_str().ok()) {
            let t = v.trim();
            if t == "1"
                || t.eq_ignore_ascii_case("true")
                || t.eq_ignore_ascii_case("yes")
                || t.eq_ignore_ascii_case("on")
            {
                return true;
            }
        }
    }
    false
}

/// Enforces `settings.enforced_params` (e.g. `["user", "metadata.tags", "header:x-session-id"]`)
/// on incoming JSON inference payloads.
pub fn validate_enforced_params(
    settings: &GatewaySettings,
    headers: &HeaderMap,
    body: &Value,
) -> quixigate_core::Result<()> {
    for spec in &settings.enforced_params {
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(hdr_name) = trimmed
            .strip_prefix("header:")
            .or_else(|| trimmed.strip_prefix("Header:"))
        {
            let h = hdr_name.trim();
            let present = headers
                .get(h)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|s| !s.trim().is_empty());
            if !present {
                return Err(QuixigateError::Validation(format!(
                    "missing required header '{h}' enforced by gateway policy (enforced_params)"
                )));
            }
            continue;
        }

        let mut cursor = Some(body);
        for segment in trimmed.split('.') {
            cursor = cursor.and_then(|v| v.get(segment));
        }
        let non_empty = match cursor {
            None | Some(Value::Null) => false,
            Some(Value::String(s)) => !s.trim().is_empty(),
            Some(Value::Array(arr)) => !arr.is_empty(),
            Some(Value::Object(map)) => !map.is_empty(),
            Some(_) => true,
        };
        if !non_empty {
            return Err(QuixigateError::Validation(format!(
                "missing required request parameter '{trimmed}' enforced by gateway policy (enforced_params)"
            )));
        }
    }
    Ok(())
}

/// Extracts optional end-user / customer identifier (`body["user"]`, `body["end_user_id"]`,
/// or `x-litellm-end-user-id` / `x-end-user-id` header).
pub fn extract_end_user_id(headers: &HeaderMap, body: Option<&Value>) -> Option<String> {
    if let Some(b) = body {
        for key in ["user", "end_user_id"] {
            if let Some(s) = b.get(key).and_then(Value::as_str).map(str::trim)
                && !s.is_empty()
            {
                return Some(s.to_owned());
            }
        }
    }
    for hdr_name in [&HDR_LITELLM_END_USER, &HDR_X_END_USER] {
        if let Some(s) = headers
            .get(hdr_name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            && !s.is_empty()
        {
            return Some(s.to_owned());
        }
    }
    None
}

/// Unified pre-flight inference authorization: validates `enforced_params`, extracts request `tags`
/// and `end_user_id`, enforces team/model/end-user blocks + key/team/user/end-user/model/tag `NanoUsd`
/// budgets + RPM/TPM/concurrency, and records per-request privacy header state (`x-quixigate-no-log`).
pub fn authorize_inference_request(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    headers: &HeaderMap,
    body: Option<&Value>,
    requested_model: &str,
    estimated_tokens: u64,
    estimated_paid_nanos: NanoUsd,
) -> quixigate_core::Result<(ConcurrencyGuard, BudgetGuard)> {
    let settings = state.db.settings_arc();
    if let Some(json_body) = body {
        validate_enforced_params(&settings, headers, json_body)?;
    }
    let req_tags = extract_request_tags(headers, body);
    let end_user_id = extract_end_user_id(headers, body);
    let (conc, mut budget_guard) = state.db.authorize_and_reserve_full(
        principal,
        requested_model,
        estimated_tokens,
        estimated_paid_nanos,
        &req_tags,
        end_user_id.as_deref(),
    )?;
    if should_suppress_request_log(headers, &settings) {
        budget_guard.set_suppress_log(true);
    }
    Ok((conc, budget_guard))
}

/// Fast zero-DOM completion accounting for Cut-Through streams and JSON responses across
/// `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`.
#[allow(clippy::too_many_arguments)]
pub fn finalize_cut_through_accounting(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    budget_guard: BudgetGuard,
    ingress_protocol: IngressProtocol,
    request_id: String,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    status_code: u16,
    stream: bool,
    fallback_attempts: u32,
    ttft_ms: Option<u64>,
    duration_ms: u64,
    input_tokens: u64,
    cached_input_tokens: u64,
    reasoning_tokens: u64,
    output_tokens: u64,
    cost_nanos: NanoUsd,
    api_equiv_nanos: NanoUsd,
    error_message: Option<String>,
) {
    let tags = budget_guard.effective_tags().to_vec();
    let suppress_log = budget_guard.suppress_log();
    let _ = budget_guard.commit(cost_nanos);
    if suppress_log {
        return;
    }

    let settings = state.db.settings_arc();
    let trace_hex = TelemetryHub::fast_trace_id_hex();
    let log = RequestLog {
        id: request_id,
        timestamp_ms: now_ms(),
        ingress_protocol,
        requested_model,
        resolved_model,
        provider_id,
        credential_id,
        user_id: Some(principal.user.id.clone()),
        team_id: principal.effective_team_id(),
        virtual_key_id: principal.virtual_key.as_ref().map(|k| k.id.clone()),
        tags,
        status_code,
        stream,
        fallback_attempts,
        cache_hit: false,
        ttft_ms,
        duration_ms,
        input_tokens,
        cached_input_tokens,
        reasoning_tokens,
        output_tokens,
        cost_nanos,
        api_equiv_nanos,
        otel_trace_id: Some(trace_hex),
        error_message,
        request_body_json: None,
        response_body_json: None,
    };

    state
        .telemetry
        .record_completed_request(&log, &settings, false, None);
    let _ = state.db.insert_request_log_owned(log);
}

#[allow(clippy::too_many_arguments)]
pub fn finalize_request_accounting(
    state: &AppState,
    principal: &AuthenticatedPrincipal,
    budget_guard: BudgetGuard,
    request_id: String,
    ingress_protocol: IngressProtocol,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    status_code: u16,
    stream: bool,
    fallback_attempts: u32,
    cache_hit: bool,
    ttft_ms: Option<u64>,
    duration_ms: u64,
    input_tokens: u64,
    cached_input_tokens: u64,
    reasoning_tokens: u64,
    output_tokens: u64,
    cost_nanos: NanoUsd,
    api_equiv_nanos: NanoUsd,
    error_message: Option<String>,
    raw_request_json: Option<String>,
    raw_response_json: Option<String>,
    mut oi: OpenInferencePayload,
    output_text: Option<String>,
    output_tool_calls: Vec<(String, String, String)>,
) {
    let tags = budget_guard.effective_tags().to_vec();
    let suppress_log = budget_guard.suppress_log();
    let _ = budget_guard.commit(cost_nanos);
    if suppress_log {
        return;
    }

    let settings = state.db.settings_arc();
    let capture_payloads = principal
        .virtual_key
        .as_ref()
        .and_then(|k| k.log_payloads)
        .unwrap_or(settings.capture_payloads_default);

    let redact_secrets = settings.secret_scanner_mode == SecretScannerMode::Redact;
    let scrubbed_request_json = raw_request_json.map(|raw| {
        if redact_secrets {
            redact_secrets_in_str(&raw)
        } else {
            raw
        }
    });
    if redact_secrets {
        for (_, content) in &mut oi.input_messages {
            *content = redact_secrets_in_str(content);
        }
    }
    for t in &tags {
        if !oi.tags.contains(t) {
            oi.tags.push(t.clone());
        }
    }

    let trace_hex = if settings.otlp_enabled {
        TelemetryHub::generate_trace_id().1
    } else {
        TelemetryHub::fast_trace_id_hex()
    };
    oi.output_text = output_text;
    oi.output_tool_calls = output_tool_calls;

    let log = RequestLog {
        id: request_id,
        timestamp_ms: now_ms(),
        ingress_protocol,
        requested_model,
        resolved_model,
        provider_id,
        credential_id,
        user_id: Some(principal.user.id.clone()),
        team_id: principal.effective_team_id(),
        virtual_key_id: principal.virtual_key.as_ref().map(|k| k.id.clone()),
        tags,
        status_code,
        stream,
        fallback_attempts,
        cache_hit,
        ttft_ms,
        duration_ms,
        input_tokens,
        cached_input_tokens,
        reasoning_tokens,
        output_tokens,
        cost_nanos,
        api_equiv_nanos,
        otel_trace_id: Some(trace_hex),
        error_message,
        request_body_json: if capture_payloads {
            scrubbed_request_json
        } else {
            None
        },
        response_body_json: if capture_payloads {
            raw_response_json
        } else {
            None
        },
    };

    state
        .telemetry
        .record_completed_request(&log, &settings, capture_payloads, Some(oi));
    let _ = state.db.insert_request_log_owned(log);
}
