//! Zero-DOM Cut-Through Engine for wire-compatible OpenAI `/v1/chat/completions` traffic.
//!
//! When ingress is OpenAI Chat Completions and the selected upstream candidate natively speaks
//! OpenAI Chat Completions (`openrouter`, `openai`, `groq`, `deepseek`, `mistral`, `together`,
//! `fireworks`, `cerebras`, `xai`, `ollama`, `vllm`, `llamacpp`, `lmstudio`, `github-copilot`,
//! `openaicompatible`, etc.) with no server-side tool loop or payload-mutating guardrail active:
//! 1. Top-level `"model"` and `"stream_options":{"include_usage":true}` are spliced directly onto
//!    raw `Bytes` using SIMD `memchr` structural scanning (zero `Message` or `serde_json::Value` DOM).
//! 2. Multi-seat/multi-provider pre-stream failover & cooldown benching run identically to the
//!    transcoding lane; if a fallback candidate requires protocol translation (e.g. Anthropic or
//!    Codex Responses), execution seamlessly hands off via `FallbackTranscode`.
//! 3. Upstream SSE `Bytes` flow straight to Hyper's socket writer with flyby `memchr::memmem`
//!    usage extraction on the final frame.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use bytes::{BufMut, Bytes, BytesMut};
use quixigate_core::{
    AuthKind, CredentialRecord, CredentialSecret, GatewaySettings, ModelRoute, ModelSpec, NanoUsd,
    PoolStrategy, QuixigateError, Result, now_ms,
};
use quixigate_db::dataplane::ControlSnapshot;
use quixigate_providers::{
    describe_reqwest_error, is_retryable_stale_socket_error, openai_compatible_default_base_url,
    parse_exact_cost_nanos,
};
use serde::Deserialize;

use crate::GatewayRouter;
use crate::pool::{DispatchCandidate, record_upstream_http_failure};

/// Immutable, generation-validated routing plan cached per `requested_model` across requests.
pub struct CachedModelPlan {
    pub snap_gen: u64,
    pub catalog_gen: u64,
    pub scanner_gen: u64,
    pub earliest_cooldown_expiry_ms: Option<i64>,
    pub requested_model: Arc<str>,
    pub spec: ModelSpec,
    pub route: Option<Arc<ModelRoute>>,
    pub creds: Arc<Vec<CredentialRecord>>,
    pub candidates: Arc<Vec<DispatchCandidate>>,
    pub first_cost_rates: Option<(NanoUsd, NanoUsd, bool)>,
}

/// Single-pass candidate & pre-flight budget plan shared between authorization and dispatch.
pub struct PreparedCutThroughPlan {
    pub requested_model: Arc<str>,
    pub cached: Arc<CachedModelPlan>,
    pub snap: Arc<ControlSnapshot>,
    pub route: Option<Arc<ModelRoute>>,
    pub creds: Arc<Vec<CredentialRecord>>,
    pub candidates: Arc<Vec<DispatchCandidate>>,
    pub est_total_tokens: u64,
    pub est_cost: NanoUsd,
}

impl PreparedCutThroughPlan {
    #[inline]
    pub fn spec(&self) -> &ModelSpec {
        &self.cached.spec
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutThroughWireFormat {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
}

#[allow(clippy::large_enum_variant)]
pub enum CutThroughStreamOutcome {
    Stream(CutThroughActiveStream),
    FallbackTranscode,
}

pub struct CutThroughActiveStream {
    pub response: reqwest::Response,
    pub requested_model: String,
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub auth_kind: AuthKind,
    pub route: Option<Arc<ModelRoute>>,
    pub fallback_attempts: u32,
    pub started: Instant,
}

#[allow(clippy::large_enum_variant)]
pub enum CutThroughJsonOutcome {
    Ready(CutThroughJsonResult),
    FallbackTranscode,
}

pub struct CutThroughJsonResult {
    pub body: Bytes,
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub auth_kind: AuthKind,
    pub fallback_attempts: u32,
    pub duration_ms: u64,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub output_tokens: u64,
    pub cost_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
}

#[derive(Debug, Clone, Default)]
pub struct FlybyUsage {
    pub resolved_model: Option<String>,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub output_tokens: u64,
    pub exact_cost_nanos: Option<NanoUsd>,
}

struct CutThroughEndpoint {
    url: String,
    bearer_token: Option<String>,
    copilot_headers: bool,
    is_openrouter: bool,
}

// ── SIMD Top-Level JSON Scanner & Raw Bytes Splicer ──────────────────────────

#[derive(Debug)]
struct TopLevelChatScan {
    model_val_span: Option<(usize, usize)>,
    stream_options_close_brace: Option<(usize, bool)>,
    stream_options_null_span: Option<(usize, usize)>,
    has_stream_options: bool,
    has_top_usage: bool,
    root_close_brace: usize,
    root_is_empty: bool,
}

#[inline]
fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// Uses SIMD `memchr2` to jump to the closing `"` of a JSON string starting at `open_quote_idx`.
/// Returns `(close_quote_idx, had_escapes)`.
#[inline]
fn skip_json_string(bytes: &[u8], open_quote_idx: usize) -> Option<(usize, bool)> {
    debug_assert_eq!(bytes.get(open_quote_idx).copied(), Some(b'"'));
    let mut pos = open_quote_idx + 1;
    let mut had_escapes = false;
    while pos < bytes.len() {
        let off = memchr::memchr2(b'"', b'\\', &bytes[pos..])?;
        let hit = pos + off;
        if bytes[hit] == b'"' {
            return Some((hit, had_escapes));
        }
        had_escapes = true;
        pos = hit + 2;
    }
    None
}

/// Skips a single JSON value starting at `start_idx` (already whitespace-trimmed), returning the
/// index immediately after the value.
fn skip_json_value(bytes: &[u8], start_idx: usize) -> Option<usize> {
    let i = skip_ws(bytes, start_idx);
    let b = *bytes.get(i)?;
    match b {
        b'"' => {
            let (close, _) = skip_json_string(bytes, i)?;
            Some(close + 1)
        }
        b'{' | b'[' => {
            let mut depth = 1usize;
            let mut pos = i + 1;
            while pos < bytes.len() {
                let rest = &bytes[pos..];
                let off = memchr::memchr3(b'"', b'{', b'}', rest)
                    .into_iter()
                    .chain(memchr::memchr2(b'[', b']', rest))
                    .min();
                let min_off = off?;
                let idx = pos + min_off;
                match bytes[idx] {
                    b'"' => {
                        let (close, _) = skip_json_string(bytes, idx)?;
                        pos = close + 1;
                    }
                    b'{' | b'[' => {
                        depth += 1;
                        pos = idx + 1;
                    }
                    b'}' | b']' => {
                        depth -= 1;
                        pos = idx + 1;
                        if depth == 0 {
                            return Some(pos);
                        }
                    }
                    _ => unreachable!(),
                }
            }
            None
        }
        _ => {
            let mut pos = i;
            while pos < bytes.len()
                && !matches!(
                    bytes[pos],
                    b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'
                )
            {
                pos += 1;
            }
            (pos > i).then_some(pos)
        }
    }
}

/// Scans root-level keys of a JSON object in a single zero-allocation pass, jumping over arbitrarily
/// large `messages` / `tools` arrays at SIMD `memchr` memory bandwidth.
fn scan_top_level_chat_json(bytes: &[u8]) -> Option<TopLevelChatScan> {
    let open = skip_ws(bytes, 0);
    if bytes.get(open).copied() != Some(b'{') {
        return None;
    }
    let mut scan = TopLevelChatScan {
        model_val_span: None,
        stream_options_close_brace: None,
        stream_options_null_span: None,
        has_stream_options: false,
        has_top_usage: false,
        root_close_brace: 0,
        root_is_empty: true,
    };

    let mut pos = skip_ws(bytes, open + 1);
    if bytes.get(pos).copied() == Some(b'}') {
        scan.root_close_brace = pos;
        return Some(scan);
    }

    loop {
        pos = skip_ws(bytes, pos);
        if bytes.get(pos).copied() != Some(b'"') {
            return None;
        }
        scan.root_is_empty = false;
        let (key_close, key_escaped) = skip_json_string(bytes, pos)?;
        let raw_key = &bytes[pos + 1..key_close];
        pos = skip_ws(bytes, key_close + 1);
        if bytes.get(pos).copied() != Some(b':') {
            return None;
        }
        let val_start = skip_ws(bytes, pos + 1);

        if !key_escaped && raw_key == b"model" {
            if bytes.get(val_start).copied() == Some(b'"') {
                let (val_close, _) = skip_json_string(bytes, val_start)?;
                scan.model_val_span = Some((val_start, val_close + 1));
                pos = val_close + 1;
            } else {
                pos = skip_json_value(bytes, val_start)?;
            }
        } else if !key_escaped && raw_key == b"stream_options" {
            scan.has_stream_options = true;
            if bytes.get(val_start).copied() == Some(b'{') {
                let val_end = skip_json_value(bytes, val_start)?;
                let close_brace = val_end - 1;
                let inner_non_empty = skip_ws(bytes, val_start + 1) < close_brace;
                scan.stream_options_close_brace = Some((close_brace, inner_non_empty));
                pos = val_end;
            } else if bytes.get(val_start..val_start + 4) == Some(b"null") {
                scan.stream_options_null_span = Some((val_start, val_start + 4));
                pos = val_start + 4;
            } else {
                pos = skip_json_value(bytes, val_start)?;
            }
        } else if !key_escaped && raw_key == b"usage" {
            scan.has_top_usage = true;
            pos = skip_json_value(bytes, val_start)?;
        } else {
            pos = skip_json_value(bytes, val_start)?;
        }

        pos = skip_ws(bytes, pos);
        match bytes.get(pos).copied()? {
            b',' => pos += 1,
            b'}' => {
                scan.root_close_brace = pos;
                return Some(scan);
            }
            _ => return None,
        }
    }
}

#[inline]
fn put_json_escaped_str(buf: &mut BytesMut, s: &str) {
    buf.put_u8(b'"');
    for b in s.bytes() {
        match b {
            b'"' => buf.extend_from_slice(b"\\\""),
            b'\\' => buf.extend_from_slice(b"\\\\"),
            b'\n' => buf.extend_from_slice(b"\\n"),
            b'\r' => buf.extend_from_slice(b"\\r"),
            b'\t' => buf.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                let esc = format!("\\u{:04x}", b);
                buf.extend_from_slice(esc.as_bytes());
            }
            _ => buf.put_u8(b),
        }
    }
    buf.put_u8(b'"');
}

/// Zero-DOM OpenAI Chat request splicer:
/// - Returns `raw_body.clone()` (0 bytes copied!) when `upstream_model == requested_model` and
///   required usage flags are already present.
/// - Otherwise rewrites top-level `"model"` and injects `"stream_options":{"include_usage":true}`
///   (and `"usage":{"include":true}` on OpenRouter) in a single contiguous `BytesMut` pass.
pub fn splice_openai_chat_request(
    raw_body: &Bytes,
    requested_model: &str,
    upstream_model: &str,
    is_stream: bool,
    has_include_usage: bool,
    is_openrouter: bool,
) -> Option<Bytes> {
    let need_model_rewrite = upstream_model != requested_model;
    let need_stream_usage = is_stream && !has_include_usage;

    if !need_model_rewrite && !need_stream_usage && !is_openrouter {
        return Some(raw_body.clone());
    }

    let scan = scan_top_level_chat_json(raw_body)?;
    let need_or_usage = is_openrouter && !scan.has_top_usage;
    if !need_model_rewrite && !need_stream_usage && !need_or_usage {
        return Some(raw_body.clone());
    }

    // Collect ordered byte edits `(start, end, replacement_kind)` prior to `root_close_brace`.
    enum Edit<'a> {
        Model(&'a str),
        Static(&'static [u8]),
    }

    let mut edits: Vec<(usize, usize, Edit<'_>)> = Vec::with_capacity(3);
    if need_model_rewrite {
        let (m_start, m_end) = scan.model_val_span?;
        edits.push((m_start, m_end, Edit::Model(upstream_model)));
    }

    let mut append_stream_options_at_end = false;
    if need_stream_usage {
        if let Some((so_close, non_empty)) = scan.stream_options_close_brace {
            let snippet: &'static [u8] = if non_empty {
                b",\"include_usage\":true"
            } else {
                b"\"include_usage\":true"
            };
            edits.push((so_close, so_close, Edit::Static(snippet)));
        } else if let Some((null_start, null_end)) = scan.stream_options_null_span {
            edits.push((
                null_start,
                null_end,
                Edit::Static(b"{\"include_usage\":true}"),
            ));
        } else if !scan.has_stream_options {
            append_stream_options_at_end = true;
        }
    }

    edits.sort_by_key(|e| e.0);

    let mut out = BytesMut::with_capacity(raw_body.len() + upstream_model.len() + 96);
    let mut cursor = 0usize;
    for (start, end, edit) in edits {
        out.extend_from_slice(&raw_body[cursor..start]);
        match edit {
            Edit::Model(m) => put_json_escaped_str(&mut out, m),
            Edit::Static(slice) => out.extend_from_slice(slice),
        }
        cursor = end;
    }

    out.extend_from_slice(&raw_body[cursor..scan.root_close_brace]);
    let mut has_fields = !scan.root_is_empty;
    if append_stream_options_at_end {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"stream_options\":{\"include_usage\":true}");
        has_fields = true;
    }
    if need_or_usage {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"usage\":{\"include\":true}");
    }
    out.extend_from_slice(&raw_body[scan.root_close_brace..]);
    Some(out.freeze())
}

// ── Polyglot Zero-DOM Splicers (`/v1/responses` Codex & `/v1/messages` Anthropic) ──

#[derive(Debug)]
struct TopLevelResponsesScan {
    model_val_span: Option<(usize, usize)>,
    input_string_span: Option<(usize, usize)>,
    has_input: bool,
    store_val_span: Option<(usize, usize, bool)>,
    stream_val_span: Option<(usize, usize, bool)>,
    has_instructions: bool,
    root_close_brace: usize,
    root_is_empty: bool,
}

fn scan_top_level_responses_json(bytes: &[u8]) -> Option<TopLevelResponsesScan> {
    let open = skip_ws(bytes, 0);
    if bytes.get(open).copied() != Some(b'{') {
        return None;
    }
    let mut scan = TopLevelResponsesScan {
        model_val_span: None,
        input_string_span: None,
        has_input: false,
        store_val_span: None,
        stream_val_span: None,
        has_instructions: false,
        root_close_brace: 0,
        root_is_empty: true,
    };

    let mut pos = skip_ws(bytes, open + 1);
    if bytes.get(pos).copied() == Some(b'}') {
        scan.root_close_brace = pos;
        return Some(scan);
    }

    loop {
        pos = skip_ws(bytes, pos);
        if bytes.get(pos).copied() != Some(b'"') {
            return None;
        }
        scan.root_is_empty = false;
        let (key_close, key_escaped) = skip_json_string(bytes, pos)?;
        let raw_key = &bytes[pos + 1..key_close];
        pos = skip_ws(bytes, key_close + 1);
        if bytes.get(pos).copied() != Some(b':') {
            return None;
        }
        let val_start = skip_ws(bytes, pos + 1);

        if !key_escaped && raw_key == b"model" {
            if bytes.get(val_start).copied() == Some(b'"') {
                let (val_close, _) = skip_json_string(bytes, val_start)?;
                scan.model_val_span = Some((val_start, val_close + 1));
                pos = val_close + 1;
            } else {
                pos = skip_json_value(bytes, val_start)?;
            }
        } else if !key_escaped && raw_key == b"input" {
            scan.has_input = true;
            if bytes.get(val_start).copied() == Some(b'"') {
                let (val_close, _) = skip_json_string(bytes, val_start)?;
                scan.input_string_span = Some((val_start, val_close + 1));
                pos = val_close + 1;
            } else {
                pos = skip_json_value(bytes, val_start)?;
            }
        } else if !key_escaped && raw_key == b"store" {
            let val_end = skip_json_value(bytes, val_start)?;
            let is_false = &bytes[val_start..val_end] == b"false";
            scan.store_val_span = Some((val_start, val_end, is_false));
            pos = val_end;
        } else if !key_escaped && raw_key == b"stream" {
            let val_end = skip_json_value(bytes, val_start)?;
            let is_true = &bytes[val_start..val_end] == b"true";
            scan.stream_val_span = Some((val_start, val_end, is_true));
            pos = val_end;
        } else if !key_escaped && raw_key == b"instructions" {
            scan.has_instructions = true;
            pos = skip_json_value(bytes, val_start)?;
        } else {
            pos = skip_json_value(bytes, val_start)?;
        }

        pos = skip_ws(bytes, pos);
        match bytes.get(pos).copied()? {
            b',' => pos += 1,
            b'}' => {
                scan.root_close_brace = pos;
                return Some(scan);
            }
            _ => return None,
        }
    }
}

/// Zero-DOM splicer for `POST /v1/responses` -> ChatGPT Codex (`/responses`):
/// - Rewrites top-level `"model"` when needed
/// - Wraps string `"input":"..."` into Codex's required array item envelope without unescaping
/// - Enforces `"store":false`, `"stream":true`, and default `"instructions"` when absent
pub fn splice_codex_responses_request(
    raw_body: &Bytes,
    requested_model: &str,
    upstream_model: &str,
) -> Option<Bytes> {
    let scan = scan_top_level_responses_json(raw_body)?;
    if !scan.has_input {
        return None;
    }
    let need_model = upstream_model != requested_model;
    let store_already_false = scan.store_val_span.is_some_and(|(_, _, f)| f);
    let stream_already_true = scan.stream_val_span.is_some_and(|(_, _, t)| t);

    if !need_model
        && scan.input_string_span.is_none()
        && store_already_false
        && stream_already_true
        && scan.has_instructions
    {
        return Some(raw_body.clone());
    }

    enum RespEdit<'a> {
        Model(&'a str),
        WrapInputString(usize, usize),
        Static(&'static [u8]),
    }

    let mut edits: Vec<(usize, usize, RespEdit<'_>)> = Vec::with_capacity(4);
    if need_model {
        let (ms, me) = scan.model_val_span?;
        edits.push((ms, me, RespEdit::Model(upstream_model)));
    }
    if let Some((is, ie)) = scan.input_string_span {
        edits.push((is, ie, RespEdit::WrapInputString(is, ie)));
    }
    if let Some((ss, se, false)) = scan.store_val_span {
        edits.push((ss, se, RespEdit::Static(b"false")));
    }
    if let Some((sts, ste, false)) = scan.stream_val_span {
        edits.push((sts, ste, RespEdit::Static(b"true")));
    }
    edits.sort_by_key(|e| e.0);

    let mut out = BytesMut::with_capacity(raw_body.len() + upstream_model.len() + 192);
    let mut cursor = 0usize;
    for (start, end, edit) in edits {
        out.extend_from_slice(&raw_body[cursor..start]);
        match edit {
            RespEdit::Model(m) => put_json_escaped_str(&mut out, m),
            RespEdit::WrapInputString(is, ie) => {
                out.extend_from_slice(
                    b"[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":",
                );
                out.extend_from_slice(&raw_body[is..ie]);
                out.extend_from_slice(b"}]}]");
            }
            RespEdit::Static(s) => out.extend_from_slice(s),
        }
        cursor = end;
    }

    out.extend_from_slice(&raw_body[cursor..scan.root_close_brace]);
    let mut has_fields = !scan.root_is_empty;
    if scan.store_val_span.is_none() {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"store\":false");
        has_fields = true;
    }
    if scan.stream_val_span.is_none() {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"stream\":true");
        has_fields = true;
    }
    if !scan.has_instructions {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"instructions\":\"You are a helpful assistant.\"");
    }
    out.extend_from_slice(&raw_body[scan.root_close_brace..]);
    Some(out.freeze())
}

#[derive(Debug)]
struct TopLevelAnthropicScan {
    model_val_span: Option<(usize, usize)>,
    stream_val_span: Option<(usize, usize, bool)>,
    has_max_tokens: bool,
    has_messages: bool,
    root_close_brace: usize,
    root_is_empty: bool,
}

fn scan_top_level_anthropic_json(bytes: &[u8]) -> Option<TopLevelAnthropicScan> {
    let open = skip_ws(bytes, 0);
    if bytes.get(open).copied() != Some(b'{') {
        return None;
    }
    let mut scan = TopLevelAnthropicScan {
        model_val_span: None,
        stream_val_span: None,
        has_max_tokens: false,
        has_messages: false,
        root_close_brace: 0,
        root_is_empty: true,
    };

    let mut pos = skip_ws(bytes, open + 1);
    if bytes.get(pos).copied() == Some(b'}') {
        scan.root_close_brace = pos;
        return Some(scan);
    }

    loop {
        pos = skip_ws(bytes, pos);
        if bytes.get(pos).copied() != Some(b'"') {
            return None;
        }
        scan.root_is_empty = false;
        let (key_close, key_escaped) = skip_json_string(bytes, pos)?;
        let raw_key = &bytes[pos + 1..key_close];
        pos = skip_ws(bytes, key_close + 1);
        if bytes.get(pos).copied() != Some(b':') {
            return None;
        }
        let val_start = skip_ws(bytes, pos + 1);

        if !key_escaped && raw_key == b"model" {
            if bytes.get(val_start).copied() == Some(b'"') {
                let (val_close, _) = skip_json_string(bytes, val_start)?;
                scan.model_val_span = Some((val_start, val_close + 1));
                pos = val_close + 1;
            } else {
                pos = skip_json_value(bytes, val_start)?;
            }
        } else if !key_escaped && raw_key == b"stream" {
            let val_end = skip_json_value(bytes, val_start)?;
            let is_true = &bytes[val_start..val_end] == b"true";
            scan.stream_val_span = Some((val_start, val_end, is_true));
            pos = val_end;
        } else if !key_escaped && raw_key == b"max_tokens" {
            scan.has_max_tokens = true;
            pos = skip_json_value(bytes, val_start)?;
        } else if !key_escaped && raw_key == b"messages" {
            scan.has_messages = bytes.get(val_start).copied() == Some(b'[');
            pos = skip_json_value(bytes, val_start)?;
        } else {
            pos = skip_json_value(bytes, val_start)?;
        }

        pos = skip_ws(bytes, pos);
        match bytes.get(pos).copied()? {
            b',' => pos += 1,
            b'}' => {
                scan.root_close_brace = pos;
                return Some(scan);
            }
            _ => return None,
        }
    }
}

/// Zero-DOM splicer for `POST /v1/messages` -> Anthropic (`/v1/messages`).
pub fn splice_anthropic_messages_request(
    raw_body: &Bytes,
    requested_model: &str,
    upstream_model: &str,
    is_stream: bool,
) -> Option<Bytes> {
    let scan = scan_top_level_anthropic_json(raw_body)?;
    if !scan.has_messages {
        return None;
    }
    let need_model = upstream_model != requested_model;
    let stream_ok = matches!(
        (is_stream, scan.stream_val_span),
        (true, Some((_, _, true))) | (false, None | Some((_, _, false)))
    );

    if !need_model && stream_ok && scan.has_max_tokens {
        return Some(raw_body.clone());
    }

    enum AnthEdit<'a> {
        Model(&'a str),
        Static(&'static [u8]),
    }

    let mut edits: Vec<(usize, usize, AnthEdit<'_>)> = Vec::with_capacity(2);
    if need_model {
        let (ms, me) = scan.model_val_span?;
        edits.push((ms, me, AnthEdit::Model(upstream_model)));
    }
    if let Some((ss, se, val_true)) = scan.stream_val_span
        && val_true != is_stream
    {
        edits.push((
            ss,
            se,
            AnthEdit::Static(if is_stream { b"true" } else { b"false" }),
        ));
    }
    edits.sort_by_key(|e| e.0);

    let mut out = BytesMut::with_capacity(raw_body.len() + upstream_model.len() + 64);
    let mut cursor = 0usize;
    for (start, end, edit) in edits {
        out.extend_from_slice(&raw_body[cursor..start]);
        match edit {
            AnthEdit::Model(m) => put_json_escaped_str(&mut out, m),
            AnthEdit::Static(s) => out.extend_from_slice(s),
        }
        cursor = end;
    }

    out.extend_from_slice(&raw_body[cursor..scan.root_close_brace]);
    let mut has_fields = !scan.root_is_empty;
    if !scan.has_max_tokens {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"max_tokens\":8192");
        has_fields = true;
    }
    if is_stream && scan.stream_val_span.is_none() {
        if has_fields {
            out.put_u8(b',');
        }
        out.extend_from_slice(b"\"stream\":true");
    }
    out.extend_from_slice(&raw_body[scan.root_close_brace..]);
    Some(out.freeze())
}

// ── Flyby SSE & JSON Response Usage Extractors ───────────────────────────────

#[derive(Deserialize)]
struct BorrowedResponseEnvelope<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    usage: Option<BorrowedUsage>,
    #[serde(default)]
    timings: Option<BorrowedTimings>,
}

#[derive(Deserialize)]
struct BorrowedTimings {
    #[serde(default)]
    cache_n: Option<u64>,
    #[serde(default)]
    prompt_n: Option<u64>,
    #[serde(default)]
    predicted_n: Option<u64>,
    #[serde(default)]
    reasoning_n: Option<u64>,
}

#[derive(Deserialize)]
struct BorrowedUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<BorrowedPromptDetails>,
    #[serde(default)]
    completion_tokens_details: Option<BorrowedCompletionDetails>,
    #[serde(default)]
    cost: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct BorrowedPromptDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct BorrowedCompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

fn extract_usage_from_envelope(env: BorrowedResponseEnvelope<'_>) -> Option<FlybyUsage> {
    if env.usage.is_none() && env.timings.is_none() {
        return None;
    }
    let u = env.usage.unwrap_or(BorrowedUsage {
        prompt_tokens: None,
        input_tokens: None,
        completion_tokens: None,
        output_tokens: None,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        cost: None,
    });
    let cache_n = env
        .timings
        .as_ref()
        .and_then(|t| t.cache_n)
        .filter(|&n| n > 0);
    let prompt_n = env.timings.as_ref().and_then(|t| t.prompt_n);
    let input_tokens =
        u.prompt_tokens
            .or(u.input_tokens)
            .or_else(|| match (prompt_n, cache_n) {
                (Some(p), Some(c)) => Some(p + c),
                (Some(p), None) => Some(p),
                _ => None,
            })?;
    let output_tokens = u
        .completion_tokens
        .or(u.output_tokens)
        .or_else(|| env.timings.as_ref().and_then(|t| t.predicted_n))
        .unwrap_or(0);
    let cached_input_tokens = u
        .prompt_tokens_details
        .and_then(|d| d.cached_tokens)
        .or(cache_n)
        .unwrap_or(0);
    let reasoning_tokens = u
        .completion_tokens_details
        .and_then(|d| d.reasoning_tokens)
        .or_else(|| env.timings.as_ref().and_then(|t| t.reasoning_n))
        .unwrap_or(0);
    let exact_cost_nanos = u.cost.as_ref().and_then(parse_exact_cost_nanos);
    Some(FlybyUsage {
        resolved_model: env
            .model
            .map(|c| c.into_owned())
            .filter(|s| !s.trim().is_empty()),
        input_tokens,
        cached_input_tokens,
        reasoning_tokens,
        output_tokens,
        exact_cost_nanos,
    })
}

/// Normalizes Ollama `"reasoning"` -> `"reasoning_content"`, splits inline `<think>...</think>`,
/// and reconciles missing `usage.completion_tokens_details.reasoning_tokens` in place when needed.
fn normalize_json_reasoning_in_place(data: &mut serde_json::Value) -> bool {
    let mut mutated = false;
    let Some(msg) = data
        .get_mut("choices")
        .and_then(|c| c.get_mut(0))
        .and_then(|c| c.get_mut("message"))
        .and_then(serde_json::Value::as_object_mut)
    else {
        return false;
    };

    if !msg.contains_key("reasoning_content")
        && let Some(r_val) = msg.get("reasoning").cloned()
        && r_val.as_str().is_some_and(|s| !s.is_empty())
    {
        msg.insert("reasoning_content".to_owned(), r_val);
        mutated = true;
    }

    if !msg
        .get("reasoning_content")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.is_empty())
        && let Some(raw_text) = msg.get("content").and_then(serde_json::Value::as_str)
        && let Some((inline_r, cleaned_t)) =
            quixigate_providers::copilot::split_inline_think_block(raw_text)
    {
        msg.insert(
            "reasoning_content".to_owned(),
            serde_json::Value::String(inline_r),
        );
        msg.insert("content".to_owned(), serde_json::Value::String(cleaned_t));
        mutated = true;
    }

    let reasoning_str = msg
        .get("reasoning_content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let text_str = msg
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();

    if !reasoning_str.is_empty()
        && let Some(usage_obj) = data
            .get_mut("usage")
            .and_then(serde_json::Value::as_object_mut)
    {
        let existing_rt = usage_obj
            .get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let out_tok = usage_obj
            .get("completion_tokens")
            .or_else(|| usage_obj.get("output_tokens"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if existing_rt == 0 && out_tok > 0 {
            let synth = aisdk::core::language_model::Usage {
                input_tokens: Some(0),
                output_tokens: Some(out_tok as usize),
                cached_tokens: None,
                reasoning_tokens: None,
            };
            if let Some(reconciled) = quixigate_providers::copilot::reconcile_reasoning_usage(
                Some(synth),
                &reasoning_str,
                &text_str,
            ) && let Some(rt) = reconciled.reasoning_tokens
            {
                let details = usage_obj
                    .entry("completion_tokens_details".to_owned())
                    .or_insert_with(|| serde_json::json!({}));
                if let Some(det_obj) = details.as_object_mut() {
                    det_obj.insert("reasoning_tokens".to_owned(), serde_json::json!(rt as u64));
                    mutated = true;
                }
            }
        }
    }

    mutated
}

/// Extracts exact token counts & OpenRouter cost from the trailing window of an SSE byte stream
/// using SIMD reverse substring search (`memchr::memmem::rfind`), scanning backward to the
/// enclosing `data:` frame boundary.
pub fn parse_flyby_sse_usage(tail: &[u8]) -> Option<FlybyUsage> {
    let mut search_end = tail.len();
    while let Some(usage_pos) = memchr::memmem::rfind(&tail[..search_end], b"\"usage\"") {
        // Locate the `data:` prefix starting this SSE line.
        let line_start = memchr::memrchr(b'\n', &tail[..usage_pos])
            .map(|i| i + 1)
            .unwrap_or(0);
        let line_end = memchr::memchr2(b'\n', b'\r', &tail[usage_pos..])
            .map(|off| usage_pos + off)
            .unwrap_or(tail.len());
        let line = &tail[line_start..line_end];
        let json_slice = line
            .strip_prefix(b"data: ")
            .or_else(|| line.strip_prefix(b"data:"))
            .unwrap_or(line);
        if let Ok(env) = serde_json::from_slice::<BorrowedResponseEnvelope<'_>>(json_slice)
            && let Some(parsed) = extract_usage_from_envelope(env)
        {
            return Some(parsed);
        }
        if usage_pos == 0 {
            break;
        }
        search_end = usage_pos;
    }
    None
}

#[derive(Deserialize)]
struct BorrowedResponsesCompletedEnvelope<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    usage: Option<BorrowedResponsesUsage>,
    #[serde(borrow, default)]
    response: Option<BorrowedNestedResponse<'a>>,
}

#[derive(Deserialize)]
struct BorrowedNestedResponse<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    usage: Option<BorrowedResponsesUsage>,
}

#[derive(Deserialize)]
struct BorrowedResponsesUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<BorrowedPromptDetails>,
    #[serde(default)]
    output_tokens_details: Option<BorrowedCompletionDetails>,
}

/// Flyby SSE usage extractor for OpenAI `/v1/responses` (`response.completed` frame).
pub fn parse_flyby_responses_usage(tail: &[u8]) -> Option<FlybyUsage> {
    let mut search_end = tail.len();
    while let Some(usage_pos) = memchr::memmem::rfind(&tail[..search_end], b"\"usage\"") {
        let line_start = memchr::memrchr(b'\n', &tail[..usage_pos])
            .map(|i| i + 1)
            .unwrap_or(0);
        let line_end = memchr::memchr2(b'\n', b'\r', &tail[usage_pos..])
            .map(|off| usage_pos + off)
            .unwrap_or(tail.len());
        let line = &tail[line_start..line_end];
        let json_slice = line
            .strip_prefix(b"data: ")
            .or_else(|| line.strip_prefix(b"data:"))
            .unwrap_or(line);
        if let Ok(env) =
            serde_json::from_slice::<BorrowedResponsesCompletedEnvelope<'_>>(json_slice)
        {
            let (model, u_opt) = match env.response {
                Some(r) => (r.model.or(env.model), r.usage.or(env.usage)),
                None => (env.model, env.usage),
            };
            if let Some(u) = u_opt
                && (u.input_tokens.is_some() || u.output_tokens.is_some())
            {
                return Some(FlybyUsage {
                    resolved_model: model
                        .map(|c| c.into_owned())
                        .filter(|s| !s.trim().is_empty()),
                    input_tokens: u.input_tokens.unwrap_or(0),
                    cached_input_tokens: u
                        .input_tokens_details
                        .and_then(|d| d.cached_tokens)
                        .unwrap_or(0),
                    reasoning_tokens: u
                        .output_tokens_details
                        .and_then(|d| d.reasoning_tokens)
                        .unwrap_or(0),
                    output_tokens: u.output_tokens.unwrap_or(0),
                    exact_cost_nanos: None,
                });
            }
        }
        if usage_pos == 0 {
            break;
        }
        search_end = usage_pos;
    }
    None
}

#[derive(Deserialize)]
struct BorrowedAnthropicSseEnvelope<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    message: Option<BorrowedAnthropicMessage<'a>>,
    #[serde(default)]
    usage: Option<BorrowedAnthropicUsage>,
}

#[derive(Deserialize)]
struct BorrowedAnthropicMessage<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    usage: Option<BorrowedAnthropicUsage>,
}

#[derive(Deserialize)]
struct BorrowedAnthropicUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

/// Flyby SSE usage extractor for Anthropic `/v1/messages` (`message_start` in `head` + `message_delta` in `tail`).
pub fn parse_flyby_anthropic_usage(head: Option<&[u8]>, tail: &[u8]) -> Option<FlybyUsage> {
    let mut out = FlybyUsage::default();
    let mut found_any = false;

    let mut apply_slice = |buf: &[u8], forward: bool| {
        if forward {
            let mut cursor = 0usize;
            while let Some(off) = memchr::memmem::find(&buf[cursor..], b"\"usage\"") {
                let pos = cursor + off;
                let ls = memchr::memrchr(b'\n', &buf[..pos])
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let le = memchr::memchr2(b'\n', b'\r', &buf[pos..])
                    .map(|o| pos + o)
                    .unwrap_or(buf.len());
                let line = buf[ls..le]
                    .strip_prefix(b"data: ")
                    .or_else(|| buf[ls..le].strip_prefix(b"data:"))
                    .unwrap_or(&buf[ls..le]);
                if let Ok(env) = serde_json::from_slice::<BorrowedAnthropicSseEnvelope<'_>>(line) {
                    if let Some(m) = env
                        .message
                        .as_ref()
                        .and_then(|m| m.model.as_deref())
                        .or(env.model.as_deref())
                        .filter(|s| !s.trim().is_empty())
                    {
                        out.resolved_model = Some(m.to_owned());
                    }
                    if let Some(u) = env.message.and_then(|m| m.usage).or(env.usage) {
                        if let Some(inp) = u.input_tokens {
                            out.input_tokens = inp;
                            found_any = true;
                        }
                        if let Some(ot) = u.output_tokens {
                            out.output_tokens = ot;
                            found_any = true;
                        }
                        let cache = u.cache_read_input_tokens.unwrap_or(0)
                            + u.cache_creation_input_tokens.unwrap_or(0);
                        if cache > 0 {
                            out.cached_input_tokens = cache;
                        }
                    }
                }
                cursor = le;
            }
        } else {
            let mut search_end = buf.len();
            while let Some(pos) = memchr::memmem::rfind(&buf[..search_end], b"\"usage\"") {
                let ls = memchr::memrchr(b'\n', &buf[..pos])
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let le = memchr::memchr2(b'\n', b'\r', &buf[pos..])
                    .map(|o| pos + o)
                    .unwrap_or(buf.len());
                let line = buf[ls..le]
                    .strip_prefix(b"data: ")
                    .or_else(|| buf[ls..le].strip_prefix(b"data:"))
                    .unwrap_or(&buf[ls..le]);
                if let Ok(env) = serde_json::from_slice::<BorrowedAnthropicSseEnvelope<'_>>(line)
                    && let Some(u) = env.usage.or_else(|| env.message.and_then(|m| m.usage))
                {
                    if let Some(ot) = u.output_tokens {
                        out.output_tokens = ot;
                        found_any = true;
                    }
                    if let Some(inp) = u.input_tokens {
                        out.input_tokens = inp;
                        found_any = true;
                    }
                    let cache = u.cache_read_input_tokens.unwrap_or(0)
                        + u.cache_creation_input_tokens.unwrap_or(0);
                    if cache > 0 {
                        out.cached_input_tokens = cache;
                    }
                    break;
                }
                if pos == 0 {
                    break;
                }
                search_end = pos;
            }
        }
    };

    if let Some(h) = head {
        apply_slice(h, true);
    } else {
        apply_slice(tail, true);
    }
    apply_slice(tail, false);
    out.input_tokens = out.input_tokens.saturating_add(out.cached_input_tokens);

    found_any.then_some(out)
}

// ── Endpoint Resolution & Cut-Through Execution on `GatewayRouter` ───────────

fn build_completions_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn resolve_cut_through_endpoint(
    provider_id: &str,
    base_override: Option<&str>,
    cred_base_url: Option<&str>,
    secret: &CredentialSecret,
) -> Option<CutThroughEndpoint> {
    let eff_override = base_override
        .filter(|s| !s.trim().is_empty())
        .or_else(|| cred_base_url.filter(|s| !s.trim().is_empty()));

    match (provider_id, secret) {
        #[cfg(feature = "subscription-oauth")]
        (
            _,
            CredentialSecret::GitHubCopilot {
                access, base_url, ..
            },
        ) => {
            let base = eff_override.unwrap_or(if base_url.trim().is_empty() {
                quixigate_providers::copilot::DEFAULT_COPILOT_BASE_URL
            } else {
                base_url.as_str()
            });
            Some(CutThroughEndpoint {
                url: build_completions_url(base),
                bearer_token: Some(access.clone()),
                copilot_headers: true,
                is_openrouter: false,
            })
        }
        #[cfg(feature = "subscription-oauth")]
        (
            _,
            CredentialSecret::QwenPortal {
                access,
                resource_url,
                ..
            },
        ) => {
            let base = eff_override
                .or(resource_url.as_deref().filter(|s| !s.trim().is_empty()))
                .unwrap_or("https://portal.qwen.ai/v1");
            Some(CutThroughEndpoint {
                url: build_completions_url(base),
                bearer_token: Some(access.clone()),
                copilot_headers: false,
                is_openrouter: false,
            })
        }
        #[cfg(feature = "subscription-oauth")]
        (_, CredentialSecret::Xai { access, .. }) => {
            let base = eff_override.unwrap_or("https://api.x.ai/v1");
            Some(CutThroughEndpoint {
                url: build_completions_url(base),
                bearer_token: Some(access.clone()),
                copilot_headers: false,
                is_openrouter: false,
            })
        }
        #[cfg(feature = "subscription-oauth")]
        (_, CredentialSecret::Chutes { access, .. }) => {
            let base = eff_override.unwrap_or("https://llm.chutes.ai/v1");
            Some(CutThroughEndpoint {
                url: build_completions_url(base),
                bearer_token: Some(access.clone()),
                copilot_headers: false,
                is_openrouter: false,
            })
        }
        ("openrouter", sec) | (_, sec @ CredentialSecret::OpenRouterOidc { .. }) => {
            let base = eff_override.unwrap_or("https://openrouter.ai/api/v1");
            let token = match sec {
                CredentialSecret::ApiKey { key } if !key.trim().is_empty() => Some(key.clone()),
                CredentialSecret::OpenRouterOidc { access, .. } if !access.trim().is_empty() => {
                    Some(access.clone())
                }
                _ => None,
            };
            Some(CutThroughEndpoint {
                url: build_completions_url(base),
                bearer_token: token,
                copilot_headers: false,
                is_openrouter: true,
            })
        }
        (pid, sec)
            if openai_compatible_default_base_url(pid).is_some()
                || matches!(sec, CredentialSecret::LocalNoauth) =>
        {
            // Non-OpenAI OAuth secrets (`Codex`, `AnthropicSetupToken`, `GeminiCli`, `MiniMaxPortal`)
            // use custom wire formats and must run on the transcoding lane.
            let token = match sec {
                CredentialSecret::ApiKey { key } if !key.trim().is_empty() => Some(key.clone()),
                CredentialSecret::ApiKey { .. } | CredentialSecret::LocalNoauth => None,
                _ => return None,
            };
            let default_url =
                openai_compatible_default_base_url(pid).unwrap_or("http://127.0.0.1:11434/v1");
            let base = eff_override.unwrap_or(default_url);
            Some(CutThroughEndpoint {
                url: build_completions_url(base),
                bearer_token: token,
                copilot_headers: false,
                is_openrouter: false,
            })
        }
        _ => None,
    }
}

impl GatewayRouter {
    /// Resolves (or retrieves from the lock-free `plan_cache`) the candidate list and pricing rates
    /// for `requested_model` in `~12 ns` with 0 heap allocations on steady-state traffic.
    pub fn resolve_cached_model_plan(
        &self,
        requested_model: &str,
        snap: &Arc<ControlSnapshot>,
    ) -> Result<Arc<CachedModelPlan>> {
        let snap_gen = snap.generation;
        let cat_gen = self.catalog.generation();
        let scan_gen = self.local_scanner.generation();

        if let Some(hit) = self.plan_cache.get(requested_model)
            && hit.snap_gen == snap_gen
            && hit.catalog_gen == cat_gen
            && hit.scanner_gen == scan_gen
            && hit
                .earliest_cooldown_expiry_ms
                .is_none_or(|exp| now_ms() < exp)
        {
            return Ok(Arc::clone(hit.value()));
        }

        let spec = ModelSpec::parse(requested_model);
        let route = snap.routes_by_alias.get(&spec.base_model).cloned();
        if route.as_ref().is_some_and(|r| r.blocked) {
            return Err(QuixigateError::Forbidden(format!(
                "model '{requested_model}' is currently blocked"
            )));
        }

        let now = now_ms();
        let mut earliest_cooldown: Option<i64> = None;
        let mut observe_exp = |ts: i64| {
            if ts > now {
                earliest_cooldown = Some(match earliest_cooldown {
                    Some(cur) => cur.min(ts),
                    None => ts,
                });
            }
        };

        let mut creds_vec = Vec::with_capacity(snap.credentials.len());
        for c in snap.credentials.iter() {
            if let Some(until) = c.record.cooldown_until_ms {
                observe_exp(until);
            }
            for &m_until in c.record.model_cooldowns.values() {
                observe_exp(m_until);
            }
            creds_vec.push(c.snapshot_record(now));
        }
        let creds = Arc::new(creds_vec);

        let candidates_vec = self.pool.resolve_candidates(
            &spec,
            route.as_deref(),
            &creds,
            &self.catalog,
            &self.local_scanner,
            &snap.settings,
        );
        let first_cost_rates = candidates_vec.first().map(|first| {
            let auth_kind = first
                .credential
                .as_ref()
                .map(|c| c.auth_kind)
                .unwrap_or(AuthKind::LocalNoauth);
            let stripped = first
                .upstream_model
                .strip_suffix(":free")
                .unwrap_or(&first.upstream_model);
            let pricing = self.catalog.lookup_pricing(stripped);
            let is_free_or = first.provider_id == "openrouter"
                && (first.upstream_model == "openrouter/free"
                    || first.upstream_model.ends_with(":free"));
            let is_zero = is_free_or
                || (auth_kind.is_zero_marginal_cost() && first.provider_id != "openrouter")
                || matches!(
                    first.provider_id.as_str(),
                    "ollama"
                        | "lmstudio"
                        | "lm-studio"
                        | "llamacpp"
                        | "llama.cpp"
                        | "llama-cpp"
                        | "llama-server"
                        | "vllm"
                        | "mlx"
                        | "local"
                );
            let in_rate = route
                .as_deref()
                .and_then(|r| r.input_nanos_per_m.map(NanoUsd))
                .or_else(|| pricing.map(|(i, _, _)| i).filter(|n| !n.is_zero()))
                .unwrap_or(if is_zero {
                    NanoUsd(500_000_000)
                } else {
                    NanoUsd(1_000_000_000)
                });
            let out_rate = route
                .as_deref()
                .and_then(|r| r.output_nanos_per_m.map(NanoUsd))
                .or_else(|| pricing.map(|(_, o, _)| o).filter(|n| !n.is_zero()))
                .unwrap_or(if is_zero {
                    NanoUsd(1_500_000_000)
                } else {
                    NanoUsd(3_000_000_000)
                });
            (in_rate, out_rate, is_zero)
        });

        let strategy = route
            .as_deref()
            .map(|r| r.pool_strategy)
            .unwrap_or(snap.settings.default_pool_strategy);
        let cacheable = candidates_vec.len() <= 1 || strategy == PoolStrategy::FillFirst;

        let plan_arc = Arc::new(CachedModelPlan {
            snap_gen,
            catalog_gen: cat_gen,
            scanner_gen: scan_gen,
            earliest_cooldown_expiry_ms: earliest_cooldown,
            requested_model: Arc::from(requested_model),
            spec,
            route,
            creds,
            candidates: Arc::new(candidates_vec),
            first_cost_rates,
        });

        if cacheable {
            if self.plan_cache.len() > 4096 {
                self.plan_cache.clear();
            }
            self.plan_cache
                .insert(requested_model.to_owned(), Arc::clone(&plan_arc));
        }

        Ok(plan_arc)
    }

    /// Resolves candidates and pre-flight worst-case budget in a single lock-free `ControlSnapshot` read.
    pub fn prepare_cut_through_plan(
        &self,
        requested_model: &str,
        est_input_tokens: u64,
        max_out: u64,
    ) -> Result<PreparedCutThroughPlan> {
        let snap = self.db.control_snapshot_arc();
        let cached = self.resolve_cached_model_plan(requested_model, &snap)?;
        let est_total_tokens = est_input_tokens.saturating_add(max_out);
        let est_cost = match cached.first_cost_rates {
            Some((_, _, true)) | None => NanoUsd::ZERO,
            Some((in_rate, out_rate, false)) => {
                let ic = NanoUsd::from_tokens_and_rate(est_input_tokens, in_rate.nanos());
                let oc = NanoUsd::from_tokens_and_rate(max_out, out_rate.nanos());
                ic.saturating_add(oc)
            }
        };
        Ok(PreparedCutThroughPlan {
            requested_model: Arc::clone(&cached.requested_model),
            route: cached.route.clone(),
            creds: Arc::clone(&cached.creds),
            candidates: Arc::clone(&cached.candidates),
            cached,
            snap,
            est_total_tokens,
            est_cost,
        })
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cut_through_cost(
        &self,
        provider_id: &str,
        auth_kind: AuthKind,
        resolved_model: &str,
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
        exact_cost_nanos: Option<NanoUsd>,
        route: Option<&ModelRoute>,
    ) -> (NanoUsd, NanoUsd) {
        let (computed_cost, computed_equiv) = self.catalog.compute_cost(
            provider_id,
            auth_kind,
            resolved_model,
            input_tokens,
            cached_input_tokens,
            output_tokens,
            route,
        );
        match exact_cost_nanos {
            Some(exact) => (
                exact,
                if exact.is_zero() {
                    computed_equiv
                } else {
                    exact
                },
            ),
            None => (computed_cost, computed_equiv),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn send_cut_through_candidate(
        &self,
        cand: &DispatchCandidate,
        plan: &PreparedCutThroughPlan,
        raw_body: &Bytes,
        is_stream: bool,
        has_include_usage: bool,
        has_openrouter_only_fields: bool,
        passthrough_headers: &[(String, String)],
    ) -> std::result::Result<Option<(reqwest::Response, AuthKind, Option<String>)>, QuixigateError>
    {
        if has_openrouter_only_fields && cand.provider_id != "openrouter" {
            return Ok(None);
        }

        // Check if OpenRouter catalog metadata explicitly forbids `reasoning_effort` on this model.
        if cand.provider_id == "openrouter"
            && memchr::memmem::find(raw_body, b"\"reasoning_effort\"").is_some()
        {
            let stripped_upstream = ModelSpec::parse(&cand.upstream_model).base_model;
            let cat = self
                .catalog
                .lookup(&stripped_upstream)
                .or_else(|| self.catalog.lookup(&plan.spec().base_model));
            if cat
                .as_ref()
                .and_then(|c| c.provider_supports_reasoning.get("openrouter").copied())
                == Some(false)
            {
                return Ok(None);
            }
        }

        let (endpoint, auth_kind, cred_id) = if let Some(cred) = &cand.credential {
            let (fresh_rec, secret_arc) = self.refresher.ensure_fresh_arc(&cred.id).await?;
            let Some(ep) = resolve_cut_through_endpoint(
                &cand.provider_id,
                cand.base_url_override.as_deref(),
                fresh_rec.base_url.as_deref(),
                &secret_arc,
            ) else {
                return Ok(None);
            };
            (ep, fresh_rec.auth_kind, Some(fresh_rec.id))
        } else {
            let local_sec = match self.local_scanner.api_key_for(&cand.provider_id) {
                Some(key) => CredentialSecret::ApiKey { key },
                None => CredentialSecret::LocalNoauth,
            };
            let Some(ep) = resolve_cut_through_endpoint(
                &cand.provider_id,
                cand.base_url_override.as_deref(),
                None,
                &local_sec,
            ) else {
                return Ok(None);
            };
            (ep, AuthKind::LocalNoauth, None)
        };

        let Some(spliced_body) = splice_openai_chat_request(
            raw_body,
            &plan.requested_model,
            &cand.upstream_model,
            is_stream,
            has_include_usage,
            endpoint.is_openrouter,
        ) else {
            return Ok(None);
        };

        let settings: &GatewaySettings = &plan.snap.settings;
        let http = self.inference_http();

        let build_req = || {
            let mut req = http
                .post(&endpoint.url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header(
                    reqwest::header::ACCEPT,
                    if is_stream {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                );
            if let Some(tok) = &endpoint.bearer_token {
                req = req.bearer_auth(tok);
            }
            if endpoint.copilot_headers {
                let initiator = if memchr::memmem::find(raw_body, b"\"tool\"").is_some() {
                    "agent"
                } else {
                    "user"
                };
                req = req
                    .header("User-Agent", "GitHubCopilotChat/0.35.0")
                    .header("Editor-Version", "vscode/1.107.0")
                    .header("Editor-Plugin-Version", "copilot-chat/0.35.0")
                    .header("Copilot-Integration-Id", "vscode-chat")
                    .header("X-Initiator", initiator)
                    .header("Openai-Intent", "conversation-edits");
                if memchr::memmem::find(raw_body, b"\"image_url\"").is_some() {
                    req = req.header("Copilot-Vision-Request", "true");
                }
            }
            if endpoint.is_openrouter {
                let has_hdr = |target: &str| {
                    passthrough_headers
                        .iter()
                        .any(|(k, _)| k.eq_ignore_ascii_case(target))
                };
                if !settings.openrouter_app_url.trim().is_empty() && !has_hdr("http-referer") {
                    req = req.header("HTTP-Referer", settings.openrouter_app_url.trim());
                }
                if !settings.openrouter_app_title.trim().is_empty() && !has_hdr("x-title") {
                    req = req.header("X-Title", settings.openrouter_app_title.trim());
                }
            }
            for (k, v) in passthrough_headers {
                req = req.header(k.as_str(), v.as_str());
            }
            req.body(spliced_body.clone())
        };

        let resp_res = match build_req().send().await {
            Err(first_err) if is_retryable_stale_socket_error(&first_err) => {
                build_req().send().await
            }
            other => other,
        };

        let resp = match resp_res {
            Ok(r) => r,
            Err(err) => {
                let (status_opt, why) = describe_reqwest_error(&err);
                return Err(self.record_cut_through_error(
                    cand,
                    &plan.creds,
                    status_opt.map(|s| s.as_u16()),
                    format!("Chat completions request failed: {why}"),
                ));
            }
        };

        if !resp.status().is_success() {
            let status_u16 = resp.status().as_u16();
            let retry_hdr = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            let body_txt = resp.text().await.unwrap_or_default();
            let details = match retry_hdr {
                Some(s) => format!("{body_txt} [retry_after={s}]"),
                None => body_txt,
            };
            return Err(self.record_cut_through_error(
                cand,
                &plan.creds,
                Some(status_u16),
                details,
            ));
        }

        Ok(Some((resp, auth_kind, cred_id)))
    }

    fn record_cut_through_error(
        &self,
        cand: &DispatchCandidate,
        all_creds: &[CredentialRecord],
        status_opt: Option<u16>,
        details: String,
    ) -> QuixigateError {
        if let Some(cred) = &cand.credential {
            let live_count = all_creds
                .iter()
                .filter(|c| c.provider_id == cred.provider_id && c.enabled)
                .count();
            let (status, msg, retry_after_secs) = record_upstream_http_failure(
                &self.db,
                cred,
                &cand.upstream_model,
                status_opt,
                details,
                live_count,
            );
            if status == 429 {
                return QuixigateError::RateLimited {
                    message: format!("{} ({}): {msg}", cand.provider_id, cred.label),
                    retry_after_secs,
                };
            }
            return QuixigateError::Upstream {
                status,
                provider: cand.provider_id.clone(),
                message: msg,
            };
        }
        QuixigateError::Upstream {
            status: status_opt.unwrap_or(502),
            provider: cand.provider_id.clone(),
            message: details,
        }
    }

    pub async fn execute_cut_through_stream(
        &self,
        plan: PreparedCutThroughPlan,
        raw_body: &Bytes,
        has_include_usage: bool,
        has_openrouter_only_fields: bool,
        passthrough_headers: &[(String, String)],
    ) -> Result<CutThroughStreamOutcome> {
        if plan.candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(
                plan.requested_model.as_ref().to_owned(),
            ));
        }
        let started = Instant::now();
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in plan.candidates.iter().enumerate() {
            match self
                .send_cut_through_candidate(
                    cand,
                    &plan,
                    raw_body,
                    true,
                    has_include_usage,
                    has_openrouter_only_fields,
                    passthrough_headers,
                )
                .await
            {
                Ok(Some((response, auth_kind, cred_id))) => {
                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    return Ok(CutThroughStreamOutcome::Stream(CutThroughActiveStream {
                        response,
                        requested_model: plan.requested_model.as_ref().to_owned(),
                        resolved_model: cand.upstream_model.clone(),
                        provider_id: cand.provider_id.clone(),
                        credential_id: cred_id,
                        auth_kind,
                        route: plan.route,
                        fallback_attempts: attempt_idx as u32,
                        started,
                    }));
                }
                Ok(None) => {
                    // Candidate requires protocol translation (e.g. Anthropic / Codex / Gemini CLI).
                    // Any earlier failed seats have already been benched in `ControlSnapshot`.
                    return Ok(CutThroughStreamOutcome::FallbackTranscode);
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            QuixigateError::NoAvailableRoute(plan.requested_model.as_ref().to_owned())
        }))
    }

    pub async fn execute_cut_through_json(
        &self,
        plan: PreparedCutThroughPlan,
        raw_body: &Bytes,
        est_input_tokens: u64,
        has_openrouter_only_fields: bool,
        passthrough_headers: &[(String, String)],
    ) -> Result<CutThroughJsonOutcome> {
        if plan.candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(
                plan.requested_model.as_ref().to_owned(),
            ));
        }
        let started = Instant::now();
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in plan.candidates.iter().enumerate() {
            match self
                .send_cut_through_candidate(
                    cand,
                    &plan,
                    raw_body,
                    false,
                    false,
                    has_openrouter_only_fields,
                    passthrough_headers,
                )
                .await
            {
                Ok(Some((response, auth_kind, cred_id))) => {
                    let mut resp_bytes = match response.bytes().await {
                        Ok(b) => b,
                        Err(err) => {
                            let (st, why) = describe_reqwest_error(&err);
                            last_err = Some(self.record_cut_through_error(
                                cand,
                                &plan.creds,
                                st.map(|s| s.as_u16()),
                                format!("Failed reading response body: {why}"),
                            ));
                            continue;
                        }
                    };

                    // SIMD fast-check (~100 ns): only materialize DOM if an HTTP-200 `"error"` envelope,
                    // raw `"reasoning"` alias (Ollama), or inline `<think>` block is present.
                    if (memchr::memmem::find(&resp_bytes, b"\"error\"").is_some()
                        || memchr::memmem::find(&resp_bytes, b"<think>").is_some()
                        || memchr::memmem::find(&resp_bytes, b"\"reasoning\"").is_some())
                        && let Ok(mut data) =
                            serde_json::from_slice::<serde_json::Value>(&resp_bytes)
                    {
                        if data["error"].is_object() {
                            let msg = data["error"]["message"]
                                .as_str()
                                .unwrap_or("Upstream completion error")
                                .to_owned();
                            last_err =
                                Some(self.record_cut_through_error(cand, &plan.creds, None, msg));
                            continue;
                        }
                        if normalize_json_reasoning_in_place(&mut data)
                            && let Ok(rewritten) = serde_json::to_vec(&data)
                        {
                            resp_bytes = Bytes::from(rewritten);
                        }
                    }

                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    let duration_ms = started.elapsed().as_millis() as u64;

                    let parsed =
                        serde_json::from_slice::<BorrowedResponseEnvelope<'_>>(&resp_bytes)
                            .ok()
                            .and_then(extract_usage_from_envelope);

                    let final_resolved_model = parsed
                        .as_ref()
                        .and_then(|p| p.resolved_model.clone())
                        .unwrap_or_else(|| cand.upstream_model.clone());
                    let input_tokens = parsed
                        .as_ref()
                        .map(|p| p.input_tokens)
                        .unwrap_or(est_input_tokens);
                    let cached_input_tokens =
                        parsed.as_ref().map(|p| p.cached_input_tokens).unwrap_or(0);
                    let reasoning_tokens = parsed.as_ref().map(|p| p.reasoning_tokens).unwrap_or(0);
                    let output_tokens = parsed
                        .as_ref()
                        .map(|p| p.output_tokens)
                        .unwrap_or_else(|| ((resp_bytes.len() / 12) as u64).max(1));
                    let exact_cost = parsed.and_then(|p| p.exact_cost_nanos);

                    if auth_kind == AuthKind::LocalNoauth
                        || matches!(
                            cand.provider_id.as_str(),
                            "ollama" | "lmstudio" | "llamacpp" | "vllm" | "mlx"
                        )
                    {
                        let tps = (duration_ms > 0)
                            .then(|| (output_tokens as f64) / (duration_ms as f64 / 1000.0));
                        self.local_scanner
                            .record_performance(&cand.provider_id, None, tps);
                    }

                    let (cost_nanos, api_equiv_nanos) = self.compute_cut_through_cost(
                        &cand.provider_id,
                        auth_kind,
                        &final_resolved_model,
                        input_tokens,
                        cached_input_tokens,
                        output_tokens,
                        exact_cost,
                        plan.route.as_deref(),
                    );

                    return Ok(CutThroughJsonOutcome::Ready(CutThroughJsonResult {
                        body: resp_bytes,
                        resolved_model: final_resolved_model,
                        provider_id: cand.provider_id.clone(),
                        credential_id: cred_id,
                        auth_kind,
                        fallback_attempts: attempt_idx as u32,
                        duration_ms,
                        input_tokens,
                        cached_input_tokens,
                        reasoning_tokens,
                        output_tokens,
                        cost_nanos,
                        api_equiv_nanos,
                    }));
                }
                Ok(None) => return Ok(CutThroughJsonOutcome::FallbackTranscode),
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            QuixigateError::NoAvailableRoute(plan.requested_model.as_ref().to_owned())
        }))
    }

    /// Zero-DOM cut-through for `POST /v1/responses` (`stream: true`) -> ChatGPT Codex (`/responses`).
    #[cfg(feature = "subscription-oauth")]
    pub async fn execute_responses_cut_through_stream(
        &self,
        plan: PreparedCutThroughPlan,
        raw_body: &Bytes,
        passthrough_headers: &[(String, String)],
        session_req_id: &str,
    ) -> Result<CutThroughStreamOutcome> {
        if plan.candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(
                plan.requested_model.as_ref().to_owned(),
            ));
        }
        let started = Instant::now();
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in plan.candidates.iter().enumerate() {
            let Some(cred) = &cand.credential else {
                return Ok(CutThroughStreamOutcome::FallbackTranscode);
            };
            let (fresh_rec, secret_arc) = self.refresher.ensure_fresh_arc(&cred.id).await?;
            let CredentialSecret::Codex {
                access, account_id, ..
            } = secret_arc.as_ref()
            else {
                return Ok(CutThroughStreamOutcome::FallbackTranscode);
            };

            let Some(spliced_body) = splice_codex_responses_request(
                raw_body,
                &plan.requested_model,
                &cand.upstream_model,
            ) else {
                return Ok(CutThroughStreamOutcome::FallbackTranscode);
            };

            let raw_base = cand
                .base_url_override
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .or_else(|| {
                    fresh_rec
                        .base_url
                        .as_deref()
                        .filter(|s| !s.trim().is_empty())
                })
                .unwrap_or(quixigate_providers::codex::DEFAULT_CODEX_BASE_URL)
                .trim_end_matches('/');
            let url = if raw_base.ends_with("/responses") {
                raw_base.to_owned()
            } else {
                format!("{raw_base}/responses")
            };
            let residency = quixigate_providers::codex::extract_residency(access);
            let http = self.inference_http();

            let build_req = || {
                let mut req = http
                    .post(&url)
                    .bearer_auth(access)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .header(reqwest::header::ACCEPT, "text/event-stream")
                    .header("ChatGPT-Account-ID", account_id)
                    .header("originator", "pi")
                    .header("OpenAI-Beta", "responses=experimental")
                    .header(
                        "User-Agent",
                        concat!("quixigate/", env!("CARGO_PKG_VERSION")),
                    )
                    .header("session_id", session_req_id)
                    .header("x-client-request-id", session_req_id);
                if let Some(res) = &residency {
                    req = req.header("x-openai-internal-codex-residency", res);
                }
                for (k, v) in passthrough_headers {
                    req = req.header(k.as_str(), v.as_str());
                }
                req.body(spliced_body.clone())
            };

            let resp_res = match build_req().send().await {
                Err(first_err) if is_retryable_stale_socket_error(&first_err) => {
                    build_req().send().await
                }
                other => other,
            };

            let resp = match resp_res {
                Ok(r) => r,
                Err(err) => {
                    let (st, why) = describe_reqwest_error(&err);
                    last_err = Some(self.record_cut_through_error(
                        cand,
                        &plan.creds,
                        st.map(|s| s.as_u16()),
                        format!("could not reach ChatGPT Codex backend: {why}"),
                    ));
                    continue;
                }
            };

            if !resp.status().is_success() {
                let status_u16 = resp.status().as_u16();
                let retry_hdr = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.trim().parse::<u64>().ok());
                let body_txt = resp.text().await.unwrap_or_default();
                let details = match retry_hdr {
                    Some(s) => format!("{body_txt} [retry_after={s}]"),
                    None => body_txt,
                };
                last_err = Some(self.record_cut_through_error(
                    cand,
                    &plan.creds,
                    Some(status_u16),
                    details,
                ));
                continue;
            }

            let _ = self.db.mark_credential_used(&fresh_rec.id);
            return Ok(CutThroughStreamOutcome::Stream(CutThroughActiveStream {
                response: resp,
                requested_model: plan.requested_model.as_ref().to_owned(),
                resolved_model: cand.upstream_model.clone(),
                provider_id: cand.provider_id.clone(),
                credential_id: Some(fresh_rec.id),
                auth_kind: fresh_rec.auth_kind,
                route: plan.route,
                fallback_attempts: attempt_idx as u32,
                started,
            }));
        }

        Err(last_err.unwrap_or_else(|| {
            QuixigateError::NoAvailableRoute(plan.requested_model.as_ref().to_owned())
        }))
    }

    #[cfg(feature = "subscription-oauth")]
    async fn send_anthropic_cut_through_candidate(
        &self,
        cand: &DispatchCandidate,
        plan: &PreparedCutThroughPlan,
        raw_body: &Bytes,
        is_stream: bool,
        passthrough_headers: &[(String, String)],
    ) -> std::result::Result<Option<(reqwest::Response, AuthKind, Option<String>)>, QuixigateError>
    {
        let Some(cred) = &cand.credential else {
            return Ok(None);
        };
        let (fresh_rec, secret_arc) = self.refresher.ensure_fresh_arc(&cred.id).await?;
        let eff_base = cand
            .base_url_override
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                fresh_rec
                    .base_url
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
            });

        let (url, token, use_x_api_key) = match (cand.provider_id.as_str(), secret_arc.as_ref()) {
            (_, CredentialSecret::AnthropicSetupToken { token }) => {
                let base = eff_base
                    .unwrap_or(quixigate_providers::anthropic_oauth::DEFAULT_ANTHROPIC_BASE_URL);
                (
                    quixigate_providers::AnthropicOAuthModel::resolve_messages_url(base),
                    token.clone(),
                    false,
                )
            }
            (
                _,
                CredentialSecret::MiniMaxPortal {
                    access,
                    resource_url,
                    ..
                },
            ) => {
                let base = eff_base
                    .or(resource_url.as_deref().filter(|s| !s.trim().is_empty()))
                    .unwrap_or("https://api.minimax.io/anthropic");
                (
                    quixigate_providers::AnthropicOAuthModel::resolve_messages_url(base),
                    access.clone(),
                    false,
                )
            }
            ("anthropic", CredentialSecret::ApiKey { key }) => {
                let base = eff_base
                    .unwrap_or(quixigate_providers::anthropic_oauth::DEFAULT_ANTHROPIC_BASE_URL);
                (
                    quixigate_providers::AnthropicOAuthModel::resolve_messages_url(base),
                    key.clone(),
                    true,
                )
            }
            _ => return Ok(None),
        };

        let Some(spliced_body) = splice_anthropic_messages_request(
            raw_body,
            &plan.requested_model,
            &cand.upstream_model,
            is_stream,
        ) else {
            return Ok(None);
        };

        let http = self.inference_http();
        let build_req = || {
            let mut req = http
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header(
                    reqwest::header::ACCEPT,
                    if is_stream {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )
                .header("anthropic-version", "2023-06-01");
            if use_x_api_key {
                if !token.trim().is_empty() {
                    req = req.header("x-api-key", &token);
                }
            } else {
                req = req
                    .bearer_auth(&token)
                    .header("anthropic-beta", "oauth-2025-04-20");
            }
            for (k, v) in passthrough_headers {
                req = req.header(k.as_str(), v.as_str());
            }
            req.body(spliced_body.clone())
        };

        let resp_res = match build_req().send().await {
            Err(first_err) if is_retryable_stale_socket_error(&first_err) => {
                build_req().send().await
            }
            other => other,
        };

        let resp = match resp_res {
            Ok(r) => r,
            Err(err) => {
                let (st, why) = describe_reqwest_error(&err);
                return Err(self.record_cut_through_error(
                    cand,
                    &plan.creds,
                    st.map(|s| s.as_u16()),
                    format!("Anthropic Messages request failed: {why}"),
                ));
            }
        };

        if !resp.status().is_success() {
            let status_u16 = resp.status().as_u16();
            let retry_hdr = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            let body_txt = resp.text().await.unwrap_or_default();
            let details = match retry_hdr {
                Some(s) => format!("{body_txt} [retry_after={s}]"),
                None => body_txt,
            };
            return Err(self.record_cut_through_error(
                cand,
                &plan.creds,
                Some(status_u16),
                details,
            ));
        }

        Ok(Some((resp, fresh_rec.auth_kind, Some(fresh_rec.id))))
    }

    /// Zero-DOM cut-through for `POST /v1/messages` (`stream: true`) -> Anthropic (`/v1/messages`).
    #[cfg(feature = "subscription-oauth")]
    pub async fn execute_anthropic_cut_through_stream(
        &self,
        plan: PreparedCutThroughPlan,
        raw_body: &Bytes,
        passthrough_headers: &[(String, String)],
    ) -> Result<CutThroughStreamOutcome> {
        if plan.candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(
                plan.requested_model.as_ref().to_owned(),
            ));
        }
        let started = Instant::now();
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in plan.candidates.iter().enumerate() {
            match self
                .send_anthropic_cut_through_candidate(
                    cand,
                    &plan,
                    raw_body,
                    true,
                    passthrough_headers,
                )
                .await
            {
                Ok(Some((response, auth_kind, cred_id))) => {
                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    return Ok(CutThroughStreamOutcome::Stream(CutThroughActiveStream {
                        response,
                        requested_model: plan.requested_model.as_ref().to_owned(),
                        resolved_model: cand.upstream_model.clone(),
                        provider_id: cand.provider_id.clone(),
                        credential_id: cred_id,
                        auth_kind,
                        route: plan.route,
                        fallback_attempts: attempt_idx as u32,
                        started,
                    }));
                }
                Ok(None) => return Ok(CutThroughStreamOutcome::FallbackTranscode),
                Err(err) => last_err = Some(err),
            }
        }

        Err(last_err.unwrap_or_else(|| {
            QuixigateError::NoAvailableRoute(plan.requested_model.as_ref().to_owned())
        }))
    }

    /// Zero-DOM cut-through for `POST /v1/messages` (`stream: false`) -> Anthropic (`/v1/messages`).
    #[cfg(feature = "subscription-oauth")]
    pub async fn execute_anthropic_cut_through_json(
        &self,
        plan: PreparedCutThroughPlan,
        raw_body: &Bytes,
        est_input_tokens: u64,
        passthrough_headers: &[(String, String)],
    ) -> Result<CutThroughJsonOutcome> {
        if plan.candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(
                plan.requested_model.as_ref().to_owned(),
            ));
        }
        let started = Instant::now();
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in plan.candidates.iter().enumerate() {
            match self
                .send_anthropic_cut_through_candidate(
                    cand,
                    &plan,
                    raw_body,
                    false,
                    passthrough_headers,
                )
                .await
            {
                Ok(Some((response, auth_kind, cred_id))) => {
                    let resp_bytes = match response.bytes().await {
                        Ok(b) => b,
                        Err(err) => {
                            let (st, why) = describe_reqwest_error(&err);
                            last_err = Some(self.record_cut_through_error(
                                cand,
                                &plan.creds,
                                st.map(|s| s.as_u16()),
                                format!("Failed reading Anthropic response body: {why}"),
                            ));
                            continue;
                        }
                    };
                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    let duration_ms = started.elapsed().as_millis() as u64;
                    let parsed =
                        serde_json::from_slice::<BorrowedAnthropicSseEnvelope<'_>>(&resp_bytes)
                            .ok();
                    let final_resolved_model = parsed
                        .as_ref()
                        .and_then(|e| e.model.as_deref())
                        .filter(|s| !s.trim().is_empty())
                        .map(str::to_owned)
                        .unwrap_or_else(|| cand.upstream_model.clone());
                    let u = parsed.and_then(|e| e.usage);
                    let cached_input_tokens = u
                        .as_ref()
                        .map(|x| {
                            x.cache_read_input_tokens.unwrap_or(0)
                                + x.cache_creation_input_tokens.unwrap_or(0)
                        })
                        .unwrap_or(0);
                    let input_tokens = u
                        .as_ref()
                        .and_then(|x| x.input_tokens)
                        .map(|inp| inp.saturating_add(cached_input_tokens))
                        .unwrap_or(est_input_tokens);
                    let output_tokens = u
                        .as_ref()
                        .and_then(|x| x.output_tokens)
                        .unwrap_or_else(|| ((resp_bytes.len() / 12) as u64).max(1));

                    let (cost_nanos, api_equiv_nanos) = self.compute_cut_through_cost(
                        &cand.provider_id,
                        auth_kind,
                        &final_resolved_model,
                        input_tokens,
                        cached_input_tokens,
                        output_tokens,
                        None,
                        plan.route.as_deref(),
                    );

                    return Ok(CutThroughJsonOutcome::Ready(CutThroughJsonResult {
                        body: resp_bytes,
                        resolved_model: final_resolved_model,
                        provider_id: cand.provider_id.clone(),
                        credential_id: cred_id,
                        auth_kind,
                        fallback_attempts: attempt_idx as u32,
                        duration_ms,
                        input_tokens,
                        cached_input_tokens,
                        reasoning_tokens: 0,
                        output_tokens,
                        cost_nanos,
                        api_equiv_nanos,
                    }));
                }
                Ok(None) => return Ok(CutThroughJsonOutcome::FallbackTranscode),
                Err(err) => last_err = Some(err),
            }
        }

        Err(last_err.unwrap_or_else(|| {
            QuixigateError::NoAvailableRoute(plan.requested_model.as_ref().to_owned())
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splicer_preserves_zero_copy_and_handles_unordered_decoy_keys() {
        // 1. Zero-copy identity when model matches and include_usage is already present
        let raw = Bytes::from_static(
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}"#,
        );
        let out = splice_openai_chat_request(&raw, "gpt-4o", "gpt-4o", true, true, false).unwrap();
        assert_eq!(raw.as_ptr(), out.as_ptr(), "expected zero-copy Bytes clone");

        // 2. Messages BEFORE model, containing escaped decoy `"model":"decoy"` inside prompt
        let tricky = Bytes::from_static(
            br#"{"messages":[{"role":"user","content":"ignore {\"model\":\"decoy\"} please"}],"model":"gpt-4o","stream":true}"#,
        );
        let spliced =
            splice_openai_chat_request(&tricky, "gpt-4o", "openai/gpt-4o", true, false, true)
                .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&spliced).unwrap();
        assert_eq!(val["model"], "openai/gpt-4o");
        assert_eq!(
            val["messages"][0]["content"],
            "ignore {\"model\":\"decoy\"} please"
        );
        assert_eq!(val["stream_options"]["include_usage"], true);
        assert_eq!(val["usage"]["include"], true);

        // 3. Existing non-empty `stream_options` without `include_usage`
        let existing_so = Bytes::from_static(
            br#"{"model":"llama3","stream":true,"stream_options":{"continuous_usage_stats":false},"messages":[]}"#,
        );
        let spliced2 =
            splice_openai_chat_request(&existing_so, "llama3", "llama3", true, false, false)
                .unwrap();
        let val2: serde_json::Value = serde_json::from_slice(&spliced2).unwrap();
        assert_eq!(val2["stream_options"]["continuous_usage_stats"], false);
        assert_eq!(val2["stream_options"]["include_usage"], true);
    }

    #[test]
    fn flyby_sse_usage_extracts_tokens_and_openrouter_cost() {
        let tail = br#"data: {"id":"c1","model":"openai/gpt-4o-2024-11-20","choices":[{"index":0,"delta":{"content":"!"}}]}

data: {"id":"c1","model":"openai/gpt-4o-2024-11-20","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":300,"total_tokens":342,"prompt_tokens_details":{"cached_tokens":10},"completion_tokens_details":{"reasoning_tokens":50},"cost":0.00125}}

data: [DONE]

"#;
        let u = parse_flyby_sse_usage(tail).unwrap();
        assert_eq!(
            u.resolved_model.as_deref(),
            Some("openai/gpt-4o-2024-11-20")
        );
        assert_eq!(u.input_tokens, 42);
        assert_eq!(u.output_tokens, 300);
        assert_eq!(u.cached_input_tokens, 10);
        assert_eq!(u.reasoning_tokens, 50);
        assert_eq!(u.exact_cost_nanos, Some(NanoUsd(1_250_000)));
    }

    #[test]
    fn polyglot_responses_and_anthropic_splicers_and_flyby_usage_work() {
        // 1. Codex `/v1/responses` splicer wraps string `input`, rewrites `model`, and injects `store:false` + `instructions`
        let raw_resp = Bytes::from_static(
            br#"{"model":"openai/gpt-5","stream":true,"input":"Hello \"Codex\"!"}"#,
        );
        let spliced_resp =
            splice_codex_responses_request(&raw_resp, "openai/gpt-5", "gpt-5").unwrap();
        let val_resp: serde_json::Value = serde_json::from_slice(&spliced_resp).unwrap();
        assert_eq!(val_resp["model"], "gpt-5");
        assert_eq!(val_resp["store"], false);
        assert_eq!(val_resp["stream"], true);
        assert_eq!(
            val_resp["input"][0]["content"][0]["text"],
            "Hello \"Codex\"!"
        );
        assert!(val_resp["instructions"].is_string());

        // Zero-copy identity when already normalized for Codex
        let ready_codex = Bytes::from_static(
            br#"{"model":"gpt-5","stream":true,"store":false,"instructions":"be brief","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#,
        );
        let zc_codex = splice_codex_responses_request(&ready_codex, "gpt-5", "gpt-5").unwrap();
        assert_eq!(ready_codex.as_ptr(), zc_codex.as_ptr());

        // 2. Anthropic `/v1/messages` splicer preserves zero-copy when `model` + `max_tokens` + `stream` match
        let ready_anth = Bytes::from_static(
            br#"{"model":"claude-sonnet-4-5","max_tokens":512,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let zc_anth = splice_anthropic_messages_request(
            &ready_anth,
            "claude-sonnet-4-5",
            "claude-sonnet-4-5",
            true,
        )
        .unwrap();
        assert_eq!(ready_anth.as_ptr(), zc_anth.as_ptr());

        // Rewrites `anthropic/claude-sonnet-4-5` -> `claude-sonnet-4-5` and injects default `max_tokens`
        let prefixed_anth = Bytes::from_static(
            br#"{"model":"anthropic/claude-sonnet-4-5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let spliced_anth = splice_anthropic_messages_request(
            &prefixed_anth,
            "anthropic/claude-sonnet-4-5",
            "claude-sonnet-4-5",
            true,
        )
        .unwrap();
        let val_anth: serde_json::Value = serde_json::from_slice(&spliced_anth).unwrap();
        assert_eq!(val_anth["model"], "claude-sonnet-4-5");
        assert_eq!(val_anth["max_tokens"], 8192);

        // 3. Flyby Responses SSE usage extractor
        let resp_tail = br#"event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5-2026","usage":{"input_tokens":28,"input_tokens_details":{"cached_tokens":8},"output_tokens":300,"output_tokens_details":{"reasoning_tokens":15}}}}

"#;
        let ru = parse_flyby_responses_usage(resp_tail).unwrap();
        assert_eq!(ru.resolved_model.as_deref(), Some("gpt-5-2026"));
        assert_eq!(ru.input_tokens, 28);
        assert_eq!(ru.cached_input_tokens, 8);
        assert_eq!(ru.output_tokens, 300);
        assert_eq!(ru.reasoning_tokens, 15);

        // 4. Flyby Anthropic SSE usage extractor (`message_start` head + `message_delta` tail)
        let anth_head = br#"event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5-20250929","usage":{"input_tokens":19,"cache_read_input_tokens":5,"cache_creation_input_tokens":2,"output_tokens":1}}}

"#;
        let anth_tail = br#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":300}}

event: message_stop
data: {"type":"message_stop"}

"#;
        let au = parse_flyby_anthropic_usage(Some(anth_head), anth_tail).unwrap();
        assert_eq!(
            au.resolved_model.as_deref(),
            Some("claude-sonnet-4-5-20250929")
        );
        assert_eq!(au.input_tokens, 26);
        assert_eq!(au.cached_input_tokens, 7);
        assert_eq!(au.output_tokens, 300);
    }
}
