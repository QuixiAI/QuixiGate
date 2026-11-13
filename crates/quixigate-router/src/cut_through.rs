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
    QuixigateError, Result, now_ms,
};
use quixigate_db::dataplane::ControlSnapshot;
use quixigate_providers::{
    describe_reqwest_error, is_retryable_stale_socket_error, openai_compatible_default_base_url,
    parse_exact_cost_nanos,
};
use serde::Deserialize;

use crate::GatewayRouter;
use crate::pool::{DispatchCandidate, record_upstream_http_failure};

/// Single-pass candidate & pre-flight budget plan shared between authorization and dispatch.
pub struct PreparedCutThroughPlan {
    pub requested_model: String,
    pub spec: ModelSpec,
    pub snap: Arc<ControlSnapshot>,
    pub route: Option<Arc<ModelRoute>>,
    pub creds: Vec<CredentialRecord>,
    pub candidates: Vec<DispatchCandidate>,
    pub est_total_tokens: u64,
    pub est_cost: NanoUsd,
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
    /// Resolves candidates and pre-flight worst-case budget in a single lock-free `ControlSnapshot` read.
    pub fn prepare_cut_through_plan(
        &self,
        requested_model: &str,
        est_input_tokens: u64,
        max_out: u64,
    ) -> Result<PreparedCutThroughPlan> {
        let spec = ModelSpec::parse(requested_model);
        let snap = self.db.control_snapshot_arc();
        let route = snap.routes_by_alias.get(&spec.base_model).cloned();
        if route.as_ref().is_some_and(|r| r.blocked) {
            return Err(QuixigateError::Forbidden(format!(
                "model '{requested_model}' is currently blocked"
            )));
        }
        let now = now_ms();
        let creds: Vec<CredentialRecord> = snap
            .credentials
            .iter()
            .map(|c| c.snapshot_record(now))
            .collect();
        let candidates = self.pool.resolve_candidates(
            &spec,
            route.as_deref(),
            &creds,
            &self.catalog,
            &self.local_scanner,
            &snap.settings,
        );
        let est_total_tokens = est_input_tokens.saturating_add(max_out);
        let est_cost = match candidates.first() {
            Some(first) => {
                let auth_kind = first
                    .credential
                    .as_ref()
                    .map(|c| c.auth_kind)
                    .unwrap_or(AuthKind::LocalNoauth);
                self.catalog
                    .compute_cost(
                        &first.provider_id,
                        auth_kind,
                        &first.upstream_model,
                        est_input_tokens,
                        0,
                        max_out,
                        route.as_deref(),
                    )
                    .0
            }
            None => NanoUsd::ZERO,
        };
        Ok(PreparedCutThroughPlan {
            requested_model: requested_model.to_owned(),
            spec,
            snap,
            route,
            creds,
            candidates,
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
                .or_else(|| self.catalog.lookup(&plan.spec.base_model));
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
            return Err(QuixigateError::NoAvailableRoute(plan.requested_model));
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
                        requested_model: plan.requested_model,
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

        Err(last_err.unwrap_or(QuixigateError::NoAvailableRoute(plan.requested_model)))
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
            return Err(QuixigateError::NoAvailableRoute(plan.requested_model));
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

        Err(last_err.unwrap_or(QuixigateError::NoAvailableRoute(plan.requested_model)))
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
}
