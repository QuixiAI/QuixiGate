//! `POST /v1/chat/completions` — Two-Lane OpenAI Chat Completions Codec:
//! 1. **Zero-DOM Cut-Through Lane**: Borrowed `ChatEnvelopePeek` + SIMD raw-`Bytes` model/usage
//!    splicer + `CutThroughSseStream` with Actix-style greedy `poll_next` write coalescing (`32 KB`
//!    watermark, `0 ms` timer delay, zero `tokio::spawn`, zero `mpsc` channel, zero per-token JSON DOM)
//!    + RAII `Drop` spend/telemetry accounting.
//! 2. **Polyglot Transcoding Lane**: Full `CanonicalRequest` normalization for non-OpenAI upstreams
//!    (Anthropic, ChatGPT Codex, Gemini Code Assist), `:online` / MCP server-side tool loops,
//!    exact BLAKE3 response caching, secret redaction/blocking guardrails, and payload/OTLP capture.

use std::borrow::Cow;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use bytes::BytesMut;
use futures::{Stream, StreamExt};
use quixigate_core::{
    AuthKind, IngressProtocol, ModelRoute, ModelSpec, NanoUsd, QuixigateError, SecretScannerMode,
    now_ms,
};
use quixigate_db::{AuthenticatedPrincipal, BudgetGuard, ConcurrencyGuard};
use quixigate_router::{
    CanonicalRequest, CutThroughActiveStream, CutThroughJsonOutcome, CutThroughStreamOutcome,
    RoutedStreamEvent, parse_flyby_sse_usage,
};
use quixigate_telemetry::OpenInferencePayload;
use serde::Deserialize;
use serde::de::{Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde_json::{Map, Value, json};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::authenticate_inference;
use crate::codecs::common::*;
use crate::state::{ApiError, ApiResult, AppState};

/// Maximum bytes coalesced in a single greedy `poll_next` drain before yielding to Hyper (`32 KiB`).
const POLL_COALESCE_CAP: usize = 32 * 1024;

// ── Zero-DOM Request Envelope Peek ───────────────────────────────────────────

struct MessagesArrayMarker;

impl<'de> Deserialize<'de> for MessagesArrayMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ArraySkipper;
        impl<'de> Visitor<'de> for ArraySkipper {
            type Value = MessagesArrayMarker;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON array of chat messages")
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

#[derive(Deserialize, Default)]
struct StreamOptionsPeek {
    #[serde(default)]
    include_usage: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TagsPeek<'a> {
    #[serde(borrow)]
    List(Vec<Cow<'a, str>>),
    #[serde(borrow)]
    Csv(Cow<'a, str>),
    Other(IgnoredAny),
}

#[derive(Deserialize, Default)]
struct MetadataPeek<'a> {
    #[serde(borrow, default)]
    tags: Option<TagsPeek<'a>>,
}

#[derive(Deserialize)]
struct ChatEnvelopePeek<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    messages: Option<MessagesArrayMarker>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(borrow, default)]
    mock_response: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    user: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    end_user_id: Option<Cow<'a, str>>,
    #[serde(default)]
    stream_options: Option<StreamOptionsPeek>,
    #[serde(borrow, default)]
    tags: Option<TagsPeek<'a>>,
    #[serde(borrow, default)]
    metadata: Option<MetadataPeek<'a>>,
    #[serde(default)]
    provider: Option<IgnoredAny>,
    #[serde(default)]
    transforms: Option<IgnoredAny>,
    #[serde(default)]
    models: Option<IgnoredAny>,
    #[serde(default)]
    plugins: Option<IgnoredAny>,
    #[serde(default)]
    route: Option<IgnoredAny>,
}

impl<'a> ChatEnvelopePeek<'a> {
    #[inline]
    fn has_openrouter_only_fields(&self) -> bool {
        self.provider.is_some()
            || self.transforms.is_some()
            || self.models.is_some()
            || self.plugins.is_some()
            || self.route.is_some()
    }

    fn collect_tags(&self, headers: &HeaderMap) -> Vec<String> {
        let mut out = Vec::new();
        let mut push = |raw: &str| {
            let t = raw.trim();
            if !t.is_empty() && !out.iter().any(|e| e == t) {
                out.push(t.to_owned());
            }
        };
        for hdr_name in ["x-quixigate-tags", "x-litellm-tags"] {
            if let Some(val) = headers.get(hdr_name).and_then(|v| v.to_str().ok()) {
                for item in val.split(',') {
                    push(item);
                }
            }
        }
        for peek_opt in [
            self.metadata.as_ref().and_then(|m| m.tags.as_ref()),
            self.tags.as_ref(),
        ]
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
    fn end_user<'b>(&'b self, headers: &'b HeaderMap) -> Option<&'b str> {
        for cand in [&self.user, &self.end_user_id] {
            if let Some(s) = cand.as_deref().map(str::trim)
                && !s.is_empty()
            {
                return Some(s);
            }
        }
        for hdr_name in ["x-litellm-end-user-id", "x-end-user-id"] {
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
}

// ── Actix-Style Greedy-Drain Cut-Through SSE Stream ──────────────────────────

struct CutThroughSseStream {
    upstream: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send>>,
    state: AppState,
    principal: AuthenticatedPrincipal,
    conc_guard: Option<ConcurrencyGuard>,
    budget_guard: Option<BudgetGuard>,
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
    prev_chunk: Option<Bytes>,
    last_chunk: Option<Bytes>,
    coalesce_buf: BytesMut,
    eof: bool,
}

impl CutThroughSseStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        active: CutThroughActiveStream,
        state: AppState,
        principal: AuthenticatedPrincipal,
        conc_guard: ConcurrencyGuard,
        budget_guard: BudgetGuard,
        request_id: String,
        est_input_tokens: u64,
    ) -> Self {
        Self {
            upstream: Box::pin(active.response.bytes_stream()),
            state,
            principal,
            conc_guard: Some(conc_guard),
            budget_guard: Some(budget_guard),
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
            prev_chunk: None,
            last_chunk: None,
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
        self.chunk_count += 1;
        // Zero-copy O(1) trailing window: retain refcounted handles to the last two chunks
        // (zero byte scans and zero `memmove`s during token streaming!).
        self.prev_chunk = self.last_chunk.replace(chunk.clone());
    }
}

impl Stream for CutThroughSseStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.eof {
            return Poll::Ready(None);
        }

        // Actix-style greedy poll drain: pull all currently ready chunks from the Tokio socket
        // up to `POLL_COALESCE_CAP` (`32 KiB`) in one wakeup — zero timer delay, zero copy when
        // 1 chunk is ready, and 3-10x fewer socket writes under high token throughput!
        let mut first_chunk: Option<Bytes> = None;

        loop {
            match this.upstream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    if chunk.is_empty() {
                        continue;
                    }
                    this.observe_chunk(&chunk);
                    if first_chunk.is_none() && this.coalesce_buf.is_empty() {
                        if chunk.len() >= POLL_COALESCE_CAP {
                            return Poll::Ready(Some(Ok(chunk)));
                        }
                        first_chunk = Some(chunk);
                    } else {
                        if let Some(prev) = first_chunk.take() {
                            this.coalesce_buf
                                .reserve((prev.len() + chunk.len()).max(4096));
                            this.coalesce_buf.extend_from_slice(&prev);
                        }
                        this.coalesce_buf.extend_from_slice(&chunk);
                        if this.coalesce_buf.len() >= POLL_COALESCE_CAP {
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
        let flyby = match (&self.prev_chunk, &self.last_chunk) {
            (_, None) => None,
            (None, Some(last)) => parse_flyby_sse_usage(last),
            (Some(prev), Some(last)) => parse_flyby_sse_usage(last).or_else(|| {
                let prev_tail = &prev[prev.len().saturating_sub(8192)..];
                let mut stitched = Vec::with_capacity(prev_tail.len() + last.len());
                stitched.extend_from_slice(prev_tail);
                stitched.extend_from_slice(last);
                parse_flyby_sse_usage(&stitched)
            }),
        };

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

// ── Route Entrypoint ─────────────────────────────────────────────────────────

pub async fn handle_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    raw_body: Bytes,
) -> ApiResult<Response> {
    let principal = authenticate_inference(&state, &headers, None)?;
    let settings = state.db.settings_arc();
    let capture_payloads = principal
        .virtual_key
        .as_ref()
        .and_then(|k| k.log_payloads)
        .unwrap_or(settings.capture_payloads_default);
    let cache_pref = parse_cache_preference(&headers);

    // ── Fast Lane Gate: Zero-DOM Peek + Cut-Through Bit-Shuffler ─────────────
    let can_attempt_peek = settings.enforced_params.is_empty()
        && settings.secret_scanner_mode == SecretScannerMode::Off
        && settings.banned_keywords.is_empty()
        && !capture_payloads
        && !settings.otlp_enabled
        && settings.mcp_server_urls.is_empty()
        && !state.router.has_mcp_urls();

    let mut pre_reserved: Option<(ConcurrencyGuard, BudgetGuard, String)> = None;

    if can_attempt_peek {
        // Quick structural check: if `messages` exists, isn't an array, let the slow-path validator
        // produce the exact `'messages' must be an array` 400 error message.
        if let Ok(peek) = serde_json::from_slice::<ChatEnvelopePeek<'_>>(&raw_body) {
            let requested_model_str = peek
                .model
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| QuixigateError::Validation("missing 'model' field".to_owned()))?;
            if peek.messages.is_none() {
                return Err(ApiError(QuixigateError::Validation(
                    "'messages' must be an array".to_owned(),
                )));
            }

            let spec = ModelSpec::parse(requested_model_str);
            let use_cache = cache_pref.unwrap_or(settings.cache_enabled_default) && !spec.online;

            if peek.mock_response.is_none() && !spec.online && !use_cache {
                let est_in = ((raw_body.len() / 4) as u64).max(1) + 8;
                let max_out = peek
                    .max_completion_tokens
                    .or(peek.max_tokens)
                    .unwrap_or(1024);
                let plan =
                    state
                        .router
                        .prepare_cut_through_plan(requested_model_str, est_in, max_out)?;

                let req_tags = peek.collect_tags(&headers);
                let end_user_id = peek.end_user(&headers);
                let (conc_guard, mut budget_guard) = state.db.authorize_and_reserve_full(
                    &principal,
                    requested_model_str,
                    plan.est_total_tokens,
                    plan.est_cost,
                    &req_tags,
                    end_user_id,
                )?;
                if should_suppress_request_log(&headers, &settings) {
                    budget_guard.set_suppress_log(true);
                }

                let stream_requested = peek.stream.unwrap_or(false);
                let has_include_usage = peek
                    .stream_options
                    .as_ref()
                    .and_then(|so| so.include_usage)
                    .unwrap_or(false);
                let has_or_only = peek.has_openrouter_only_fields();
                let passthrough_headers = extract_passthrough_headers_vec(&headers);
                let request_id = format!("chatcmpl_{}", Uuid::new_v4().simple());

                if stream_requested {
                    state.telemetry.inc_active();
                    let started = Instant::now();
                    match state
                        .router
                        .execute_cut_through_stream(
                            plan,
                            &raw_body,
                            has_include_usage,
                            has_or_only,
                            &passthrough_headers,
                        )
                        .await
                    {
                        Ok(CutThroughStreamOutcome::Stream(active)) => {
                            let provider_hdr = active.provider_id.clone();
                            let model_hdr = active.resolved_model.clone();
                            let req_id_hdr = request_id.clone();
                            let sse_stream = CutThroughSseStream::new(
                                active,
                                state,
                                principal,
                                conc_guard,
                                budget_guard,
                                request_id,
                                est_in,
                            );
                            let mut resp = Response::new(Body::from_stream(sse_stream));
                            *resp.status_mut() = StatusCode::OK;
                            let hdrs = resp.headers_mut();
                            hdrs.insert(
                                header::CONTENT_TYPE,
                                HeaderValue::from_static("text/event-stream"),
                            );
                            hdrs.insert(
                                header::CACHE_CONTROL,
                                HeaderValue::from_static("no-cache"),
                            );
                            attach_gateway_headers(
                                hdrs,
                                &req_id_hdr,
                                &provider_hdr,
                                &model_hdr,
                                false,
                                NanoUsd::ZERO,
                                NanoUsd::ZERO,
                            );
                            return Ok(resp);
                        }
                        Ok(CutThroughStreamOutcome::FallbackTranscode) => {
                            state.telemetry.dec_active();
                            pre_reserved =
                                Some((conc_guard, budget_guard, requested_model_str.to_owned()));
                        }
                        Err(err) => {
                            state.telemetry.dec_active();
                            drop(conc_guard);
                            finalize_cut_through_accounting(
                                &state,
                                &principal,
                                budget_guard,
                                request_id,
                                requested_model_str.to_owned(),
                                requested_model_str.to_owned(),
                                error_provider_id(&err),
                                None,
                                err.http_status(),
                                true,
                                0,
                                None,
                                started.elapsed().as_millis() as u64,
                                0,
                                0,
                                0,
                                0,
                                NanoUsd::ZERO,
                                NanoUsd::ZERO,
                                Some(err.to_string()),
                            );
                            return Err(ApiError(err));
                        }
                    }
                } else {
                    state.telemetry.inc_active();
                    let started = Instant::now();
                    match state
                        .router
                        .execute_cut_through_json(
                            plan,
                            &raw_body,
                            est_in,
                            has_or_only,
                            &passthrough_headers,
                        )
                        .await
                    {
                        Ok(CutThroughJsonOutcome::Ready(res)) => {
                            state.telemetry.dec_active();
                            drop(conc_guard);
                            let mut resp = Response::new(Body::from(res.body));
                            *resp.status_mut() = StatusCode::OK;
                            resp.headers_mut().insert(
                                header::CONTENT_TYPE,
                                HeaderValue::from_static("application/json"),
                            );
                            attach_gateway_headers(
                                resp.headers_mut(),
                                &request_id,
                                &res.provider_id,
                                &res.resolved_model,
                                false,
                                res.cost_nanos,
                                res.api_equiv_nanos,
                            );
                            finalize_cut_through_accounting(
                                &state,
                                &principal,
                                budget_guard,
                                request_id,
                                requested_model_str.to_owned(),
                                res.resolved_model,
                                res.provider_id,
                                res.credential_id,
                                200,
                                false,
                                res.fallback_attempts,
                                None,
                                res.duration_ms,
                                res.input_tokens,
                                res.cached_input_tokens,
                                res.reasoning_tokens,
                                res.output_tokens,
                                res.cost_nanos,
                                res.api_equiv_nanos,
                                None,
                            );
                            return Ok(resp);
                        }
                        Ok(CutThroughJsonOutcome::FallbackTranscode) => {
                            state.telemetry.dec_active();
                            pre_reserved =
                                Some((conc_guard, budget_guard, requested_model_str.to_owned()));
                        }
                        Err(err) => {
                            state.telemetry.dec_active();
                            drop(conc_guard);
                            finalize_cut_through_accounting(
                                &state,
                                &principal,
                                budget_guard,
                                request_id,
                                requested_model_str.to_owned(),
                                requested_model_str.to_owned(),
                                error_provider_id(&err),
                                None,
                                err.http_status(),
                                false,
                                0,
                                None,
                                started.elapsed().as_millis() as u64,
                                0,
                                0,
                                0,
                                0,
                                NanoUsd::ZERO,
                                NanoUsd::ZERO,
                                Some(err.to_string()),
                            );
                            return Err(ApiError(err));
                        }
                    }
                }
            }
        }
    }

    // ── Transcoding / Feature-Rich Lane (`serde_json::Value` + `aisdk`) ──────
    let body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| QuixigateError::Validation(format!("invalid JSON body: {e}")))?;

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
        cache_preference: cache_pref,
    };

    let est_in = estimate_request_tokens(&req);
    let (conc_guard, budget_guard) = match pre_reserved {
        Some((cg, bg, _)) => (cg, bg),
        None => {
            let (est_total, est_cost) = state.router.estimate_preflight_budget(
                &requested_model,
                est_in,
                u64::from(max_out.unwrap_or(1024)),
            )?;
            authorize_inference_request(
                &state,
                &principal,
                &headers,
                Some(&body),
                &requested_model,
                est_total,
                est_cost,
            )?
        }
    };

    let stream_requested = body["stream"].as_bool().unwrap_or(false);
    let request_id = format!("chatcmpl_{}", Uuid::new_v4().simple());
    let created_sec = now_ms() / 1000;

    // Zero-cost instant `mock_response` short-circuit (`good.md` §21) for fast integration testing
    if let Some(mock_text) = body["mock_response"].as_str() {
        let mut guard_req = req.clone();
        quixigate_router::guardrails::apply_request_guardrails(&mut guard_req, &settings)?;
        let out_tokens = ((mock_text.len() / 4) as u64).max(1);
        let zero = NanoUsd(0);
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
                StatusCode::OK,
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

    let need_deep_telemetry = capture_payloads || settings.otlp_enabled;
    let oi = if need_deep_telemetry {
        let session_id = headers
            .get("x-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        build_openinference_inputs(&req, session_id)
    } else {
        OpenInferencePayload::default()
    };
    let raw_req_str = capture_payloads.then(|| {
        std::str::from_utf8(&raw_body)
            .map(str::to_owned)
            .unwrap_or_else(|_| body.to_string())
    });
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

                let oi_tools = if need_deep_telemetry {
                    gen_out
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            (
                                tc.tool.id.clone(),
                                tc.tool.name.clone(),
                                tc.input.to_string(),
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                };

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
                    capture_payloads.then(|| resp_json.to_string()),
                    oi,
                    need_deep_telemetry.then_some(gen_out.text),
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
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
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
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
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
                let oi_tools = if need_deep_telemetry {
                    sum.tool_calls
                        .iter()
                        .map(|tc| {
                            (
                                tc.tool.id.clone(),
                                tc.tool.name.clone(),
                                tc.input.to_string(),
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let synth_resp = capture_payloads.then(|| {
                    json!({
                        "content": sum.text,
                        "reasoning": sum.reasoning,
                    })
                    .to_string()
                });
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
                    synth_resp,
                    oi,
                    need_deep_telemetry.then_some(sum.text),
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
                    NanoUsd::ZERO,
                    NanoUsd::ZERO,
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
            NanoUsd::ZERO,
            NanoUsd::ZERO,
        );
        Ok(resp)
    }
}
