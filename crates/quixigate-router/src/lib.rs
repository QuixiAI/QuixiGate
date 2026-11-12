//! Smart multi-seat router, `models.dev` catalog, exact BLAKE3 response cache,
//! `:online` / MCP server-side tool loop, and unified `aisdk` execution engine.

pub mod cache;
pub mod catalog;
pub mod guardrails;
pub mod pool;
pub mod tools;

use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use aisdk::core::language_model::{
    LanguageModelOptions, LanguageModelResponseContentType, LanguageModelStreamChunk,
    LanguageModelStreamChunkType, ProviderStream, ReasoningEffort, Usage,
};
use aisdk::core::tools::{ToolContext, ToolDetails, ToolExecute};
use aisdk::core::{
    AssistantMessage, LanguageModel, LanguageModelRequest, Message, Tool, ToolCallInfo,
    ToolResultInfo,
};
use cache::{CachedGeneration, CachedToolCall, ResponseCache};
use catalog::{CatalogModel, ModelCatalog};
use futures::{Stream, StreamExt};
use quixigate_auth::refresh::CredentialRefresher;
use quixigate_core::{
    AuthKind, CredentialSecret, ModelRoute, ModelSpec, NanoUsd, QuixigateError, Result,
};
use quixigate_db::Database;
use quixigate_providers::{
    AnyLanguageModel, EmbeddingOutput, LocalRuntimeScanner, ModelBuildRequest, embed_texts,
    instantiate_model,
};
use pool::{DispatchCandidate, PoolSelector, record_upstream_failure};
use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const MAX_SERVER_TOOL_STEPS: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Default)]
pub struct CanonicalRequest {
    pub requested_model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub client_tools: Vec<ClientToolSpec>,
    pub schema: Option<Value>,
    pub temperature: Option<u32>,
    pub top_p: Option<u32>,
    pub top_k: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub extra_headers: HashMap<String, String>,
    pub extra_body: Map<String, Value>,
    /// OpenRouter-native request extensions (`provider`, `transforms`, `models`, `plugins`, `route`)
    /// forwarded exclusively when dispatching to an `openrouter` candidate.
    pub openrouter_body: Map<String, Value>,
    /// `Some(true)` forces cache lookup/write, `Some(false)` bypasses cache (`no-cache`),
    /// `None` defers to `GatewaySettings.cache_enabled_default`.
    pub cache_preference: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct RoutedGeneration {
    pub requested_model: String,
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub auth_kind: AuthKind,
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallInfo>,
    pub finish_reason: Option<String>,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub output_tokens: u64,
    pub cost_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
    pub duration_ms: u64,
    pub fallback_attempts: u32,
    pub cache_hit: bool,
}

#[derive(Debug, Clone)]
pub enum RoutedStreamEvent {
    TextStart,
    TextDelta(String),
    TextEnd,
    ReasoningStart,
    ReasoningDelta(String),
    ReasoningEnd,
    ToolCallStart { id: String, name: String },
    ToolCallDelta { id: String, delta: String },
    ToolCallReady(ToolCallInfo),
    ServerToolResult {
        id: String,
        name: String,
        input: Value,
        output: String,
    },
    Finish(StreamCompletionSummary),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct StreamCompletionSummary {
    pub requested_model: String,
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub auth_kind: AuthKind,
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallInfo>,
    pub finish_reason: Option<String>,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub output_tokens: u64,
    pub cost_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
    pub ttft_ms: Option<u64>,
    pub duration_ms: u64,
    pub fallback_attempts: u32,
}

pub struct RoutedStream {
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub auth_kind: AuthKind,
    pub fallback_attempts: u32,
    pub cache_hit: bool,
    pub events: Pin<Box<dyn Stream<Item = RoutedStreamEvent> + Send>>,
}

#[derive(Debug, Clone)]
pub struct RoutedEmbeddings {
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub output: EmbeddingOutput,
    pub cost_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
    pub duration_ms: u64,
    pub fallback_attempts: u32,
}

#[derive(Clone)]
pub struct GatewayRouter {
    db: Database,
    refresher: CredentialRefresher,
    inference_http: reqwest::Client,
    catalog: ModelCatalog,
    pool: Arc<PoolSelector>,
    cache: ResponseCache,
    local_scanner: LocalRuntimeScanner,
    web_search_engine: Arc<tools::WebSearchEngine>,
    web_search_url_override: Option<String>,
    mcp_urls: Vec<String>,
}

impl GatewayRouter {
    pub fn new(
        db: Database,
        refresher: CredentialRefresher,
        catalog: ModelCatalog,
        local_scanner: LocalRuntimeScanner,
    ) -> Self {
        let cache = ResponseCache::new(db.clone());
        let env_mcp = std::env::var("QUIXIGATE_MCP_URLS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let web_search_engine = Arc::new(tools::WebSearchEngine::new(refresher.http().clone()));
        Self {
            db,
            refresher,
            inference_http: quixigate_providers::build_inference_http_client(),
            catalog,
            pool: Arc::new(PoolSelector::new()),
            cache,
            local_scanner,
            web_search_engine,
            web_search_url_override: None,
            mcp_urls: env_mcp,
        }
    }

    pub fn with_inference_http(mut self, http: reqwest::Client) -> Self {
        self.inference_http = http;
        self
    }

    pub fn with_web_search_url(mut self, url: impl Into<String>) -> Self {
        self.web_search_url_override = Some(url.into());
        self
    }

    pub fn with_mcp_urls(mut self, urls: Vec<String>) -> Self {
        self.mcp_urls = urls;
        self
    }

    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    pub fn cache(&self) -> &ResponseCache {
        &self.cache
    }

    pub fn local_scanner(&self) -> &LocalRuntimeScanner {
        &self.local_scanner
    }

    pub fn refresher(&self) -> &CredentialRefresher {
        &self.refresher
    }

    pub async fn execute_web_search(&self, query: &str, max_results: usize) -> std::result::Result<Value, String> {
        let creds = self.resolve_web_search_credentials().await;
        self.web_search_engine.search(query, max_results, &creds).await
    }

    pub async fn execute_web_fetch(&self, url: &str, max_chars: usize) -> std::result::Result<Value, String> {
        let creds = self.resolve_web_search_credentials().await;
        self.web_search_engine.fetch_url(url, max_chars, &creds).await
    }

    pub fn list_available_models(&self) -> Vec<CatalogModel> {
        let locals = self.local_scanner.list_statuses();
        let openrouter_connected = self
            .db
            .list_credentials()
            .map(|cs| cs.iter().any(|c| c.enabled && c.provider_id == "openrouter"))
            .unwrap_or(false);
        self.catalog
            .list_models_with_context(&locals, openrouter_connected)
    }

    /// Estimates worst-case paid `NanoUsd` before upstream dispatch for zero-race budget reservation.
    /// If the first healthy candidate is an OAuth subscription seat or local runtime, reserves `$0`.
    pub fn estimate_preflight_budget(&self, requested_model: &str, est_input_tokens: u64, max_out: u64) -> Result<(u64, NanoUsd)> {
        let spec = ModelSpec::parse(requested_model);
        let route = self.db.get_model_route(&spec.base_model)?;
        let creds = self.db.list_credentials()?;
        let settings = self.db.get_settings()?;
        let candidates = self.pool.resolve_candidates(
            &spec,
            route.as_ref(),
            &creds,
            &self.catalog,
            &self.local_scanner,
            &settings,
        );
        let est_total_tokens = est_input_tokens.saturating_add(max_out);
        let Some(first) = candidates.first() else {
            return Ok((est_total_tokens, NanoUsd::ZERO));
        };
        let auth_kind = first
            .credential
            .as_ref()
            .map(|c| c.auth_kind)
            .unwrap_or(AuthKind::LocalNoauth);
        let (cost, _) = self.catalog.compute_cost(
            &first.provider_id,
            auth_kind,
            &first.upstream_model,
            est_input_tokens,
            0,
            max_out,
            route.as_ref(),
        );
        Ok((est_total_tokens, cost))
    }

    /// Executes a non-streaming generation across the multi-seat pool with automatic
    /// single-flight OAuth refresh, server-side `:online` tool execution, exact caching,
    /// and multi-seat/multi-target failover.
    pub async fn execute_text(&self, mut req: CanonicalRequest) -> Result<RoutedGeneration> {
        let started = Instant::now();
        let spec = ModelSpec::parse(&req.requested_model);
        let settings = self.db.get_settings()?;
        guardrails::apply_request_guardrails(&mut req, &settings)?;
        let route = self.db.get_model_route(&spec.base_model)?;

        let use_cache = req
            .cache_preference
            .unwrap_or(settings.cache_enabled_default)
            && !spec.online;
        let cache_key = use_cache.then(|| ResponseCache::compute_key(&cache_fingerprint(&spec, &req)));

        if let Some(key) = &cache_key
            && let Some(hit) = self.cache.get(key).await
        {
            let (_, equiv) = self.catalog.compute_cost(
                &hit.provider_id,
                AuthKind::ApiKey,
                &hit.resolved_model,
                hit.input_tokens,
                hit.cached_input_tokens,
                hit.output_tokens,
                route.as_ref(),
            );
            let tool_calls = hit
                .tool_calls
                .into_iter()
                .map(|tc| ToolCallInfo {
                    tool: ToolDetails {
                        id: tc.id,
                        name: tc.name,
                    },
                    input: tc.arguments,
                    extensions: Default::default(),
                })
                .collect();
            return Ok(RoutedGeneration {
                requested_model: req.requested_model,
                resolved_model: hit.resolved_model,
                provider_id: hit.provider_id,
                credential_id: None,
                auth_kind: AuthKind::LocalNoauth,
                text: hit.text,
                reasoning: hit.reasoning,
                tool_calls,
                finish_reason: None,
                input_tokens: hit.input_tokens,
                cached_input_tokens: hit.cached_input_tokens,
                reasoning_tokens: hit.reasoning_tokens,
                output_tokens: hit.output_tokens,
                cost_nanos: NanoUsd::ZERO,
                api_equiv_nanos: equiv,
                duration_ms: started.elapsed().as_millis() as u64,
                fallback_attempts: 0,
                cache_hit: true,
            });
        }

        let creds = self.db.list_credentials()?;
        let candidates = self.pool.resolve_candidates(
            &spec,
            route.as_ref(),
            &creds,
            &self.catalog,
            &self.local_scanner,
            &settings,
        );
        if candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(req.requested_model.clone()));
        }

        let (all_tools, server_tools_map) = self
            .prepare_tools(&spec, &req.client_tools, &settings.mcp_server_urls)
            .await;
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in candidates.iter().enumerate() {
            let (cand_req, cand_tools, cand_server_tools) =
                self.adapt_for_candidate(cand, &spec, &req, &all_tools, &server_tools_map);
            let prepared = match self
                .instantiate_candidate(cand, cand_tools.clone(), &settings, &cand_req)
                .await
            {
                Ok(p) => p,
                Err(err) => {
                    last_err = Some(err);
                    continue;
                }
            };
            let (mut model, auth_kind, cred_id) = prepared;

            match run_non_streaming_steps(
                &mut model,
                &cand_req,
                &cand_tools,
                &cand_server_tools,
            )
            .await
            {
                Ok(step_out) => {
                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    let duration_ms = started.elapsed().as_millis() as u64;
                    if auth_kind == AuthKind::LocalNoauth
                        || matches!(
                            cand.provider_id.as_str(),
                            "ollama" | "lmstudio" | "llamacpp" | "vllm" | "mlx"
                        )
                    {
                        let tps = (duration_ms > 0)
                            .then(|| (step_out.output_tokens as f64) / (duration_ms as f64 / 1000.0));
                        self.local_scanner
                            .record_performance(&cand.provider_id, None, tps);
                    }
                    let meta = model.upstream_meta();
                    let final_resolved_model = meta
                        .resolved_model
                        .filter(|m| !m.trim().is_empty())
                        .unwrap_or_else(|| cand.upstream_model.clone());

                    let (computed_cost, computed_equiv) = self.catalog.compute_cost(
                        &cand.provider_id,
                        auth_kind,
                        &final_resolved_model,
                        step_out.input_tokens,
                        step_out.cached_input_tokens,
                        step_out.output_tokens,
                        route.as_ref(),
                    );
                    let (cost_nanos, api_equiv_nanos) = match meta.exact_cost_nanos {
                        Some(exact) => (
                            exact,
                            if exact.is_zero() {
                                computed_equiv
                            } else {
                                exact
                            },
                        ),
                        None => (computed_cost, computed_equiv),
                    };

                    if let Some(key) = &cache_key {
                        let cached = CachedGeneration {
                            resolved_model: final_resolved_model.clone(),
                            provider_id: cand.provider_id.clone(),
                            text: step_out.text.clone(),
                            reasoning: step_out.reasoning.clone(),
                            tool_calls: step_out
                                .tool_calls
                                .iter()
                                .map(|tc| CachedToolCall {
                                    id: tc.tool.id.clone(),
                                    name: tc.tool.name.clone(),
                                    arguments: tc.input.clone(),
                                })
                                .collect(),
                            input_tokens: step_out.input_tokens,
                            cached_input_tokens: step_out.cached_input_tokens,
                            reasoning_tokens: step_out.reasoning_tokens,
                            output_tokens: step_out.output_tokens,
                        };
                        self.cache
                            .put(key, &cached, settings.cache_ttl_secs.max(1))
                            .await;
                    }

                    return Ok(RoutedGeneration {
                        requested_model: req.requested_model,
                        resolved_model: final_resolved_model,
                        provider_id: cand.provider_id.clone(),
                        credential_id: cred_id,
                        auth_kind,
                        text: step_out.text,
                        reasoning: step_out.reasoning,
                        tool_calls: step_out.tool_calls,
                        finish_reason: meta.finish_reason,
                        input_tokens: step_out.input_tokens,
                        cached_input_tokens: step_out.cached_input_tokens,
                        reasoning_tokens: step_out.reasoning_tokens,
                        output_tokens: step_out.output_tokens,
                        cost_nanos,
                        api_equiv_nanos,
                        duration_ms,
                        fallback_attempts: attempt_idx as u32,
                        cache_hit: false,
                    });
                }
                Err(sdk_err) => {
                    last_err = Some(self.handle_candidate_error(cand, &creds, &sdk_err));
                }
            }
        }

        Err(last_err.unwrap_or(QuixigateError::NoAvailableRoute(req.requested_model)))
    }

    /// Executes a streaming generation with deterministic pre-stream multi-seat failover:
    /// if Seat #1 fails handshake (e.g. 429 or 5xx), records cooldown and immediately connects
    /// Seat #2 BEFORE downstream SSE bytes begin flowing.
    pub async fn execute_stream(&self, mut req: CanonicalRequest) -> Result<RoutedStream> {
        let started = Instant::now();
        let spec = ModelSpec::parse(&req.requested_model);
        let settings = self.db.get_settings()?;
        guardrails::apply_request_guardrails(&mut req, &settings)?;
        let route = self.db.get_model_route(&spec.base_model)?;

        let use_cache = req
            .cache_preference
            .unwrap_or(settings.cache_enabled_default)
            && !spec.online;
        let cache_key = use_cache.then(|| ResponseCache::compute_key(&cache_fingerprint(&spec, &req)));

        if let Some(key) = &cache_key
            && let Some(hit) = self.cache.get(key).await
        {
            let (_, equiv) = self.catalog.compute_cost(
                &hit.provider_id,
                AuthKind::ApiKey,
                &hit.resolved_model,
                hit.input_tokens,
                hit.cached_input_tokens,
                hit.output_tokens,
                route.as_ref(),
            );
            let mut events = Vec::new();
            if let Some(r) = &hit.reasoning
                && !r.is_empty()
            {
                events.push(RoutedStreamEvent::ReasoningStart);
                events.push(RoutedStreamEvent::ReasoningDelta(r.clone()));
                events.push(RoutedStreamEvent::ReasoningEnd);
            }
            if !hit.text.is_empty() {
                events.push(RoutedStreamEvent::TextStart);
                events.push(RoutedStreamEvent::TextDelta(hit.text.clone()));
                events.push(RoutedStreamEvent::TextEnd);
            }
            let mut ready_calls = Vec::new();
            for tc in &hit.tool_calls {
                let args_str = tc.arguments.to_string();
                events.push(RoutedStreamEvent::ToolCallStart {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                });
                events.push(RoutedStreamEvent::ToolCallDelta {
                    id: tc.id.clone(),
                    delta: args_str,
                });
                let info = ToolCallInfo {
                    tool: ToolDetails {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                    },
                    input: tc.arguments.clone(),
                    extensions: Default::default(),
                };
                events.push(RoutedStreamEvent::ToolCallReady(info.clone()));
                ready_calls.push(info);
            }
            events.push(RoutedStreamEvent::Finish(StreamCompletionSummary {
                requested_model: req.requested_model,
                resolved_model: hit.resolved_model.clone(),
                provider_id: hit.provider_id.clone(),
                credential_id: None,
                auth_kind: AuthKind::LocalNoauth,
                text: hit.text,
                reasoning: hit.reasoning,
                tool_calls: ready_calls,
                finish_reason: None,
                input_tokens: hit.input_tokens,
                cached_input_tokens: hit.cached_input_tokens,
                reasoning_tokens: hit.reasoning_tokens,
                output_tokens: hit.output_tokens,
                cost_nanos: NanoUsd::ZERO,
                api_equiv_nanos: equiv,
                ttft_ms: Some(0),
                duration_ms: 0,
                fallback_attempts: 0,
            }));
            return Ok(RoutedStream {
                resolved_model: hit.resolved_model,
                provider_id: hit.provider_id,
                credential_id: None,
                auth_kind: AuthKind::LocalNoauth,
                fallback_attempts: 0,
                cache_hit: true,
                events: Box::pin(futures::stream::iter(events)),
            });
        }

        let creds = self.db.list_credentials()?;
        let candidates = self.pool.resolve_candidates(
            &spec,
            route.as_ref(),
            &creds,
            &self.catalog,
            &self.local_scanner,
            &settings,
        );
        if candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(req.requested_model.clone()));
        }

        let (all_tools, server_tools_map) = self
            .prepare_tools(&spec, &req.client_tools, &settings.mcp_server_urls)
            .await;
        let mut last_err: Option<QuixigateError> = None;

        for (attempt_idx, cand) in candidates.iter().enumerate() {
            let (cand_req, cand_tools, cand_server_tools) =
                self.adapt_for_candidate(cand, &spec, &req, &all_tools, &server_tools_map);
            let prepared = match self
                .instantiate_candidate(cand, cand_tools.clone(), &settings, &cand_req)
                .await
            {
                Ok(p) => p,
                Err(err) => {
                    last_err = Some(err);
                    continue;
                }
            };
            let (mut model, auth_kind, cred_id) = prepared;
            let options = build_aisdk_options(&model, &cand_req, &cand_req.messages, &cand_tools);

            match model.stream_text(options).await {
                Ok(first_stream) => {
                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    let fallback_attempts = attempt_idx as u32;
                    let resolved_model = cand.upstream_model.clone();
                    let provider_id = cand.provider_id.clone();

                    let driver = StreamDriver {
                        router: self.clone(),
                        model,
                        req: cand_req,
                        all_tools: cand_tools,
                        server_tools_map: cand_server_tools,
                        conversation: req.messages.clone(),
                        current_stream: Some(first_stream),
                        step_count: 1,
                        requested_model: req.requested_model.clone(),
                        resolved_model: resolved_model.clone(),
                        provider_id: provider_id.clone(),
                        credential_id: cred_id.clone(),
                        auth_kind,
                        route: route.clone(),
                        cache_key: cache_key.clone(),
                        cache_ttl_secs: settings.cache_ttl_secs.max(1),
                        started,
                        ttft_ms: None,
                        fallback_attempts,
                        acc_text: String::new(),
                        acc_reasoning: String::new(),
                        client_tool_calls: Vec::new(),
                        pending_server_calls: Vec::new(),
                        server_tool_ids: HashSet::new(),
                        input_tokens: 0,
                        cached_input_tokens: 0,
                        reasoning_tokens: 0,
                        output_tokens: 0,
                        ready: VecDeque::new(),
                        finished: false,
                    };

                    let stream = futures::stream::unfold(driver, |mut drv| async move {
                        drv.next_event().await.map(|ev| (ev, drv))
                    });

                    return Ok(RoutedStream {
                        resolved_model,
                        provider_id,
                        credential_id: cred_id,
                        auth_kind,
                        fallback_attempts,
                        cache_hit: false,
                        events: Box::pin(stream),
                    });
                }
                Err(sdk_err) => {
                    last_err = Some(self.handle_candidate_error(cand, &creds, &sdk_err));
                }
            }
        }

        Err(last_err.unwrap_or(QuixigateError::NoAvailableRoute(req.requested_model)))
    }

    pub async fn execute_embeddings(
        &self,
        requested_model: &str,
        inputs: &[String],
        dimensions: Option<usize>,
    ) -> Result<RoutedEmbeddings> {
        let started = Instant::now();
        let spec = ModelSpec::parse(requested_model);
        let settings = self.db.get_settings()?;
        let route = self.db.get_model_route(&spec.base_model)?;
        let creds = self.db.list_credentials()?;
        let candidates = self.pool.resolve_candidates(
            &spec,
            route.as_ref(),
            &creds,
            &self.catalog,
            &self.local_scanner,
            &settings,
        );
        if candidates.is_empty() {
            return Err(QuixigateError::NoAvailableRoute(requested_model.to_owned()));
        }

        let mut last_err: Option<QuixigateError> = None;
        for (idx, cand) in candidates.iter().enumerate() {
            let (api_key, auth_kind, cred_id) = if let Some(cred) = &cand.credential {
                match self.refresher.ensure_fresh(&cred.id).await {
                    Ok((fresh_rec, secret)) => {
                        let key = match secret {
                            CredentialSecret::ApiKey { key } => Some(key),
                            CredentialSecret::Xai { access, .. }
                            | CredentialSecret::QwenPortal { access, .. }
                            | CredentialSecret::Chutes { access, .. }
                            | CredentialSecret::GitHubCopilot { access, .. }
                            | CredentialSecret::OpenRouterOidc { access, .. } => Some(access),
                            _ => None,
                        };
                        (key, fresh_rec.auth_kind, Some(fresh_rec.id))
                    }
                    Err(e) => {
                        last_err = Some(e);
                        continue;
                    }
                }
            } else {
                (
                    self.local_scanner.api_key_for(&cand.provider_id),
                    AuthKind::LocalNoauth,
                    None,
                )
            };

            match embed_texts(
                &self.inference_http,
                &cand.provider_id,
                &cand.upstream_model,
                api_key.as_deref(),
                cand.base_url_override.as_deref(),
                inputs,
                dimensions,
            )
            .await
            {
                Ok(output) => {
                    if let Some(cid) = &cred_id {
                        let _ = self.db.mark_credential_used(cid);
                    }
                    let (cost_nanos, api_equiv_nanos) = self.catalog.compute_cost(
                        &cand.provider_id,
                        auth_kind,
                        &cand.upstream_model,
                        output.prompt_tokens,
                        0,
                        0,
                        route.as_ref(),
                    );
                    return Ok(RoutedEmbeddings {
                        resolved_model: cand.upstream_model.clone(),
                        provider_id: cand.provider_id.clone(),
                        credential_id: cred_id,
                        output,
                        cost_nanos,
                        api_equiv_nanos,
                        duration_ms: started.elapsed().as_millis() as u64,
                        fallback_attempts: idx as u32,
                    });
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| QuixigateError::NoAvailableRoute(requested_model.to_owned())))
    }

    async fn prepare_tools(
        &self,
        spec: &ModelSpec,
        client_tools: &[ClientToolSpec],
        settings_mcp_urls: &[String],
    ) -> (Vec<Tool>, HashMap<String, Tool>) {
        let mut all = Vec::new();
        let mut server_map = HashMap::new();
        let mut seen = HashSet::new();

        for ct in client_tools {
            let mut schema_json = ct.parameters.clone();
            if !schema_json.is_object() {
                schema_json = json!({ "type": "object", "properties": {} });
            }
            if let Ok(input_schema) = serde_json::from_value::<Schema>(schema_json) {
                seen.insert(ct.name.clone());
                all.push(Tool {
                    name: ct.name.clone(),
                    description: ct.description.clone(),
                    input_schema,
                    execute: ToolExecute::default(),
                });
            }
        }

        if spec.online {
            let creds = self.resolve_web_search_credentials().await;
            if !seen.contains(tools::WEB_SEARCH_TOOL_NAME) {
                let search_tool = tools::build_web_search_tool_with_engine(
                    Arc::clone(&self.web_search_engine),
                    creds.clone(),
                );
                seen.insert(search_tool.name.clone());
                server_map.insert(search_tool.name.clone(), search_tool.clone());
                all.push(search_tool);
            }
            if !seen.contains(tools::WEB_FETCH_TOOL_NAME) {
                let fetch_tool = tools::build_web_fetch_tool_with_engine(
                    Arc::clone(&self.web_search_engine),
                    creds,
                );
                seen.insert(fetch_tool.name.clone());
                server_map.insert(fetch_tool.name.clone(), fetch_tool.clone());
                all.push(fetch_tool);
            }
        }

        let mut mcp_endpoints: Vec<&str> = self
            .mcp_urls
            .iter()
            .chain(settings_mcp_urls.iter())
            .map(String::as_str)
            .map(str::trim)
            .filter(|u| !u.is_empty())
            .collect();
        mcp_endpoints.dedup();

        for mcp_url in mcp_endpoints {
            let discovered = tools::fetch_mcp_http_tools(self.refresher.http(), mcp_url).await;
            for mcp_tool in discovered {
                if seen.insert(mcp_tool.name.clone()) {
                    server_map.insert(mcp_tool.name.clone(), mcp_tool.clone());
                    all.push(mcp_tool);
                }
            }
        }

        (all, server_map)
    }

    async fn resolve_web_search_credentials(&self) -> tools::WebSearchCredentials {
        let mut creds = tools::WebSearchCredentials::from_env(self.web_search_url_override.clone());
        let Ok(records) = self.db.list_credentials() else {
            return creds;
        };
        for rec in records
            .into_iter()
            .filter(|r| r.enabled && r.auth_kind == AuthKind::ApiKey)
        {
            let pid = rec.provider_id.to_ascii_lowercase();
            let needs_vault = match pid.as_str() {
                "tavily" => creds.tavily_api_key.is_none(),
                "exa" => creds.exa_api_key.is_none(),
                "brave" | "brave_search" => creds.brave_api_key.is_none(),
                "perplexity" => creds.perplexity_api_key.is_none(),
                "ollama" => creds.ollama_api_key.is_none(),
                _ => false,
            };
            if !needs_vault {
                continue;
            }
            if let Ok((_, CredentialSecret::ApiKey { key })) =
                self.refresher.ensure_fresh(&rec.id).await
                && !key.trim().is_empty()
            {
                let trimmed = Some(key.trim().to_owned());
                match pid.as_str() {
                    "tavily" => creds.tavily_api_key = trimmed,
                    "exa" => creds.exa_api_key = trimmed,
                    "brave" | "brave_search" => creds.brave_api_key = trimmed,
                    "perplexity" => creds.perplexity_api_key = trimmed,
                    "ollama" => creds.ollama_api_key = trimmed,
                    _ => {}
                }
            }
        }
        creds
    }

    async fn instantiate_candidate(
        &self,
        cand: &DispatchCandidate,
        tools: Vec<Tool>,
        settings: &quixigate_core::GatewaySettings,
        req: &CanonicalRequest,
    ) -> Result<(AnyLanguageModel, AuthKind, Option<String>)> {
        let (headers, body) = if cand.provider_id == "openrouter" {
            let mut hdrs = HashMap::new();
            let has_hdr = |target: &str| {
                req.extra_headers
                    .keys()
                    .any(|k| k.eq_ignore_ascii_case(target))
            };
            if !settings.openrouter_app_url.trim().is_empty() && !has_hdr("http-referer") {
                hdrs.insert(
                    "HTTP-Referer".to_owned(),
                    settings.openrouter_app_url.trim().to_owned(),
                );
            }
            if !settings.openrouter_app_title.trim().is_empty() && !has_hdr("x-title") {
                hdrs.insert(
                    "X-Title".to_owned(),
                    settings.openrouter_app_title.trim().to_owned(),
                );
            }
            let extra = (!req.openrouter_body.is_empty())
                .then(|| Value::Object(req.openrouter_body.clone()));
            ((!hdrs.is_empty()).then_some(hdrs), extra)
        } else {
            (None, None)
        };

        if let Some(cred) = &cand.credential {
            let (fresh_rec, secret) = self.refresher.ensure_fresh(&cred.id).await?;
            let base = cand.base_url_override.clone().or(fresh_rec.base_url.clone());
            let model = instantiate_model(ModelBuildRequest {
                provider_id: cand.provider_id.clone(),
                model_name: cand.upstream_model.clone(),
                secret: Some(secret),
                base_url: base,
                headers,
                body,
                tools,
                http: Some(self.inference_http.clone()),
            })?;
            Ok((model, fresh_rec.auth_kind, Some(fresh_rec.id)))
        } else {
            let secret = match self.local_scanner.api_key_for(&cand.provider_id) {
                Some(key) => CredentialSecret::ApiKey { key },
                None => CredentialSecret::LocalNoauth,
            };
            let model = instantiate_model(ModelBuildRequest {
                provider_id: cand.provider_id.clone(),
                model_name: cand.upstream_model.clone(),
                secret: Some(secret),
                base_url: cand.base_url_override.clone(),
                headers,
                body,
                tools,
                http: Some(self.inference_http.clone()),
            })?;
            Ok((model, AuthKind::LocalNoauth, None))
        }
    }

    /// Strips server-injected `:online` / MCP tools and unsupported reasoning parameters when
    /// live provider metadata (such as OpenRouter's `supported_parameters`) explicitly declares
    /// that the candidate model does not support them, preventing `404 No endpoints found that support tool use`.
    fn adapt_for_candidate(
        &self,
        cand: &DispatchCandidate,
        spec: &ModelSpec,
        req: &CanonicalRequest,
        all_tools: &[Tool],
        server_tools_map: &HashMap<String, Tool>,
    ) -> (CanonicalRequest, Vec<Tool>, HashMap<String, Tool>) {
        let stripped_upstream = ModelSpec::parse(&cand.upstream_model).base_model;
        let cat = if cand.upstream_model.ends_with(":free") || spec.free_only {
            self.catalog
                .lookup(&format!("{stripped_upstream}:free"))
                .or_else(|| self.catalog.lookup(&stripped_upstream))
                .or_else(|| self.catalog.lookup(&spec.base_model))
        } else {
            self.catalog
                .lookup(&stripped_upstream)
                .or_else(|| self.catalog.lookup(&spec.base_model))
        };

        let explicit_tools_support = cat
            .as_ref()
            .and_then(|c| c.provider_supports_tools.get(&cand.provider_id).copied());
        let explicit_reasoning_support = cat
            .as_ref()
            .and_then(|c| c.provider_supports_reasoning.get(&cand.provider_id).copied());

        let mut cand_req = req.clone();
        if cand.provider_id == "openrouter" && explicit_reasoning_support == Some(false) {
            cand_req.reasoning_effort = None;
        }

        if cand.provider_id == "openrouter" && explicit_tools_support == Some(false) {
            let client_only: Vec<Tool> = all_tools
                .iter()
                .filter(|t| !server_tools_map.contains_key(&t.name))
                .cloned()
                .collect();
            (cand_req, client_only, HashMap::new())
        } else {
            (cand_req, all_tools.to_vec(), server_tools_map.clone())
        }
    }

    fn handle_candidate_error(
        &self,
        cand: &DispatchCandidate,
        all_creds: &[quixigate_core::CredentialRecord],
        sdk_err: &aisdk::Error,
    ) -> QuixigateError {
        if let Some(cred) = &cand.credential {
            let live_count = all_creds
                .iter()
                .filter(|c| c.provider_id == cred.provider_id && c.enabled)
                .count();
            let (status, details, retry_after_secs) =
                record_upstream_failure(&self.db, cred, &cand.upstream_model, sdk_err, live_count);
            if status == 429 {
                return QuixigateError::RateLimited {
                    message: format!("{} ({}): {details}", cand.provider_id, cred.label),
                    retry_after_secs,
                };
            }
            return QuixigateError::Upstream {
                status,
                provider: cand.provider_id.clone(),
                message: details,
            };
        }
        let (status, message) = match sdk_err {
            aisdk::Error::ApiError {
                status_code,
                details,
            } => (status_code.map(|s| s.as_u16()).unwrap_or(502), details.clone()),
            other => (502, other.to_string()),
        };
        QuixigateError::Upstream {
            status,
            provider: cand.provider_id.clone(),
            message,
        }
    }
}

struct NonStreamStepOutput {
    text: String,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCallInfo>,
    input_tokens: u64,
    cached_input_tokens: u64,
    reasoning_tokens: u64,
    output_tokens: u64,
}

async fn run_non_streaming_steps(
    model: &mut AnyLanguageModel,
    req: &CanonicalRequest,
    all_tools: &[Tool],
    server_tools: &HashMap<String, Tool>,
) -> std::result::Result<NonStreamStepOutput, aisdk::Error> {
    let mut conversation = req.messages.clone();
    let mut total_in = 0u64;
    let mut total_cached = 0u64;
    let mut total_reasoning = 0u64;
    let mut total_out = 0u64;
    let mut final_text = String::new();
    let mut final_reasoning = String::new();

    for _step in 0..MAX_SERVER_TOOL_STEPS {
        let options = build_aisdk_options(model, req, &conversation, all_tools);
        let resp = model.generate_text(options.clone()).await?;
        if let Some(u) = &resp.usage {
            add_usage(
                u,
                &mut total_in,
                &mut total_cached,
                &mut total_reasoning,
                &mut total_out,
            );
        }

        let mut step_text = String::new();
        let mut step_reasoning = String::new();
        let mut step_client_calls = Vec::new();
        let mut step_server_calls = Vec::new();

        for item in resp.contents {
            match item {
                LanguageModelResponseContentType::Text(t) => {
                    step_text.push_str(&t);
                }
                LanguageModelResponseContentType::Reasoning { content, .. } => {
                    if !step_reasoning.is_empty() && !content.is_empty() {
                        step_reasoning.push('\n');
                    }
                    step_reasoning.push_str(&content);
                    if !final_reasoning.is_empty() && !content.is_empty() {
                        final_reasoning.push('\n');
                    }
                    final_reasoning.push_str(&content);
                }
                LanguageModelResponseContentType::ToolCall(tc) => {
                    if server_tools.contains_key(&tc.tool.name) {
                        step_server_calls.push(tc);
                    } else {
                        step_client_calls.push(tc);
                    }
                }
                LanguageModelResponseContentType::NotSupported(_) => {}
            }
        }

        final_text = step_text;
        if step_server_calls.is_empty() || !step_client_calls.is_empty() {
            return Ok(NonStreamStepOutput {
                text: final_text,
                reasoning: (!final_reasoning.is_empty()).then_some(final_reasoning),
                tool_calls: step_client_calls,
                input_tokens: total_in,
                cached_input_tokens: total_cached,
                reasoning_tokens: total_reasoning,
                output_tokens: total_out,
            });
        }

        // Execute server-side tools (`:online` web search / MCP) and loop to next step!
        let ctx = ToolContext::new(options);
        if !step_reasoning.is_empty() {
            conversation.push(Message::Assistant(AssistantMessage::new(
                LanguageModelResponseContentType::Reasoning {
                    content: step_reasoning,
                    extensions: Default::default(),
                },
                None,
            )));
        }
        for call in step_server_calls {
            conversation.push(Message::Assistant(AssistantMessage::new(
                LanguageModelResponseContentType::ToolCall(call.clone()),
                None,
            )));
            let result_val = if let Some(tool) = server_tools.get(&call.tool.name) {
                match tool.execute.call(ctx.clone(), call.input.clone()).await {
                    Ok(s) => serde_json::from_str::<Value>(&s).unwrap_or(Value::String(s)),
                    Err(e) => json!({ "error": e.to_string() }),
                }
            } else {
                json!({ "error": "server tool missing" })
            };
            let mut tr = ToolResultInfo::new(&call.tool.name);
            tr.id(&call.tool.id);
            tr.output(result_val);
            conversation.push(Message::Tool(tr));
        }
    }

    Ok(NonStreamStepOutput {
        text: final_text,
        reasoning: (!final_reasoning.is_empty()).then_some(final_reasoning),
        tool_calls: Vec::new(),
        input_tokens: total_in,
        cached_input_tokens: total_cached,
        reasoning_tokens: total_reasoning,
        output_tokens: total_out,
    })
}

struct StreamDriver {
    router: GatewayRouter,
    model: AnyLanguageModel,
    req: CanonicalRequest,
    all_tools: Vec<Tool>,
    server_tools_map: HashMap<String, Tool>,
    conversation: Vec<Message>,
    current_stream: Option<ProviderStream>,
    step_count: usize,
    requested_model: String,
    resolved_model: String,
    provider_id: String,
    credential_id: Option<String>,
    auth_kind: AuthKind,
    route: Option<ModelRoute>,
    cache_key: Option<String>,
    cache_ttl_secs: u64,
    started: Instant,
    ttft_ms: Option<u64>,
    fallback_attempts: u32,
    acc_text: String,
    acc_reasoning: String,
    client_tool_calls: Vec<ToolCallInfo>,
    pending_server_calls: Vec<ToolCallInfo>,
    server_tool_ids: HashSet<String>,
    input_tokens: u64,
    cached_input_tokens: u64,
    reasoning_tokens: u64,
    output_tokens: u64,
    ready: VecDeque<RoutedStreamEvent>,
    finished: bool,
}

impl StreamDriver {
    async fn next_event(&mut self) -> Option<RoutedStreamEvent> {
        loop {
            if let Some(ev) = self.ready.pop_front() {
                return Some(ev);
            }
            if self.finished {
                return None;
            }
            let Some(stream) = self.current_stream.as_mut() else {
                self.finish_all().await;
                continue;
            };

            match stream.next().await {
                Some(Ok(chunks)) => {
                    for chunk in chunks {
                        self.process_chunk(chunk);
                    }
                }
                Some(Err(err)) => {
                    self.ready.push_back(RoutedStreamEvent::Error(err.to_string()));
                    self.current_stream = None;
                    self.finish_all().await;
                }
                None => {
                    self.current_stream = None;
                    if !self.pending_server_calls.is_empty()
                        && self.client_tool_calls.is_empty()
                        && self.step_count < MAX_SERVER_TOOL_STEPS
                    {
                        self.step_count += 1;
                        let calls = std::mem::take(&mut self.pending_server_calls);
                        let opts = build_aisdk_options(
                            &self.model,
                            &self.req,
                            &self.conversation,
                            &self.all_tools,
                        );
                        let ctx = ToolContext::new(opts);
                        for call in calls {
                            self.conversation.push(Message::Assistant(AssistantMessage::new(
                                LanguageModelResponseContentType::ToolCall(call.clone()),
                                None,
                            )));
                            let out_str = if let Some(t) = self.server_tools_map.get(&call.tool.name)
                            {
                                t.execute
                                    .call(ctx.clone(), call.input.clone())
                                    .await
                                    .unwrap_or_else(|e| json!({ "error": e.to_string() }).to_string())
                            } else {
                                "{}".to_owned()
                            };
                            self.ready.push_back(RoutedStreamEvent::ServerToolResult {
                                id: call.tool.id.clone(),
                                name: call.tool.name.clone(),
                                input: call.input.clone(),
                                output: out_str.clone(),
                            });
                            let mut tr = ToolResultInfo::new(&call.tool.name);
                            tr.id(&call.tool.id);
                            tr.output(
                                serde_json::from_str::<Value>(&out_str)
                                    .unwrap_or(Value::String(out_str)),
                            );
                            self.conversation.push(Message::Tool(tr));
                        }
                        let next_opts = build_aisdk_options(
                            &self.model,
                            &self.req,
                            &self.conversation,
                            &self.all_tools,
                        );
                        match self.model.stream_text(next_opts).await {
                            Ok(next_s) => {
                                self.current_stream = Some(next_s);
                            }
                            Err(e) => {
                                self.ready.push_back(RoutedStreamEvent::Error(e.to_string()));
                                self.finish_all().await;
                            }
                        }
                    } else {
                        self.finish_all().await;
                    }
                }
            }
        }
    }

    fn record_ttft(&mut self) {
        if self.ttft_ms.is_none() {
            self.ttft_ms = Some(self.started.elapsed().as_millis() as u64);
        }
    }

    fn process_chunk(&mut self, chunk: LanguageModelStreamChunk) {
        match chunk {
            LanguageModelStreamChunk::Delta(delta) => match delta {
                LanguageModelStreamChunkType::TextStart => {
                    self.record_ttft();
                    self.ready.push_back(RoutedStreamEvent::TextStart);
                }
                LanguageModelStreamChunkType::TextDelta(t) => {
                    self.record_ttft();
                    self.acc_text.push_str(&t);
                    self.ready.push_back(RoutedStreamEvent::TextDelta(t));
                }
                LanguageModelStreamChunkType::TextEnd => {
                    self.ready.push_back(RoutedStreamEvent::TextEnd);
                }
                LanguageModelStreamChunkType::ReasoningStart => {
                    self.record_ttft();
                    self.ready.push_back(RoutedStreamEvent::ReasoningStart);
                }
                LanguageModelStreamChunkType::ReasoningDelta(r) => {
                    self.record_ttft();
                    self.acc_reasoning.push_str(&r);
                    self.ready.push_back(RoutedStreamEvent::ReasoningDelta(r));
                }
                LanguageModelStreamChunkType::ReasoningEnd => {
                    self.ready.push_back(RoutedStreamEvent::ReasoningEnd);
                }
                LanguageModelStreamChunkType::ToolCallStart(details) => {
                    self.record_ttft();
                    if self.server_tools_map.contains_key(&details.name) {
                        self.server_tool_ids.insert(details.id);
                    } else {
                        self.ready.push_back(RoutedStreamEvent::ToolCallStart {
                            id: details.id,
                            name: details.name,
                        });
                    }
                }
                LanguageModelStreamChunkType::ToolCallDelta { id, delta } => {
                    self.record_ttft();
                    if !self.server_tool_ids.contains(&id) {
                        self.ready
                            .push_back(RoutedStreamEvent::ToolCallDelta { id, delta });
                    }
                }
                LanguageModelStreamChunkType::Failed(why) => {
                    self.ready.push_back(RoutedStreamEvent::Error(why));
                }
                LanguageModelStreamChunkType::Incomplete(why) => {
                    self.ready
                        .push_back(RoutedStreamEvent::Error(format!("incomplete: {why}")));
                }
                _ => {}
            },
            LanguageModelStreamChunk::Done(msg) => {
                if let Some(u) = &msg.usage {
                    add_usage(
                        u,
                        &mut self.input_tokens,
                        &mut self.cached_input_tokens,
                        &mut self.reasoning_tokens,
                        &mut self.output_tokens,
                    );
                }
                match msg.content {
                    LanguageModelResponseContentType::Text(t) => {
                        if self.acc_text.is_empty() && !t.is_empty() {
                            self.record_ttft();
                            self.acc_text = t.clone();
                            self.ready.push_back(RoutedStreamEvent::TextStart);
                            self.ready.push_back(RoutedStreamEvent::TextDelta(t));
                            self.ready.push_back(RoutedStreamEvent::TextEnd);
                        }
                    }
                    LanguageModelResponseContentType::Reasoning { content, .. } => {
                        if self.acc_reasoning.is_empty() && !content.is_empty() {
                            self.acc_reasoning = content;
                        }
                    }
                    LanguageModelResponseContentType::ToolCall(tc) => {
                        if self.server_tools_map.contains_key(&tc.tool.name) {
                            self.pending_server_calls.push(tc);
                        } else {
                            self.ready.push_back(RoutedStreamEvent::ToolCallReady(tc.clone()));
                            self.client_tool_calls.push(tc);
                        }
                    }
                    LanguageModelResponseContentType::NotSupported(_) => {}
                }
            }
        }
    }

    async fn finish_all(&mut self) {
        self.finished = true;
        let duration_ms = self.started.elapsed().as_millis() as u64;
        if self.output_tokens == 0 && !self.acc_text.is_empty() {
            self.output_tokens = (self.acc_text.len() as u64).div_ceil(4).max(1);
        }
        if self.auth_kind == AuthKind::LocalNoauth
            || matches!(
                self.provider_id.as_str(),
                "ollama" | "lmstudio" | "llamacpp" | "vllm" | "mlx"
            )
        {
            let tps = (duration_ms > 0)
                .then(|| (self.output_tokens as f64) / (duration_ms as f64 / 1000.0));
            self.router
                .local_scanner
                .record_performance(&self.provider_id, self.ttft_ms, tps);
        }

        let meta = self.model.upstream_meta();
        if let Some(concrete) = meta.resolved_model.filter(|m| !m.trim().is_empty()) {
            self.resolved_model = concrete;
        }

        let (computed_cost, computed_equiv) = self.router.catalog.compute_cost(
            &self.provider_id,
            self.auth_kind,
            &self.resolved_model,
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.route.as_ref(),
        );
        let (cost_nanos, api_equiv_nanos) = match meta.exact_cost_nanos {
            Some(exact) => (
                exact,
                if exact.is_zero() {
                    computed_equiv
                } else {
                    exact
                },
            ),
            None => (computed_cost, computed_equiv),
        };

        if let Some(key) = &self.cache_key
            && (!self.acc_text.is_empty() || !self.client_tool_calls.is_empty())
        {
            let cached = CachedGeneration {
                resolved_model: self.resolved_model.clone(),
                provider_id: self.provider_id.clone(),
                text: self.acc_text.clone(),
                reasoning: (!self.acc_reasoning.is_empty()).then(|| self.acc_reasoning.clone()),
                tool_calls: self
                    .client_tool_calls
                    .iter()
                    .map(|tc| CachedToolCall {
                        id: tc.tool.id.clone(),
                        name: tc.tool.name.clone(),
                        arguments: tc.input.clone(),
                    })
                    .collect(),
                input_tokens: self.input_tokens,
                cached_input_tokens: self.cached_input_tokens,
                reasoning_tokens: self.reasoning_tokens,
                output_tokens: self.output_tokens,
            };
            self.router
                .cache
                .put(key, &cached, self.cache_ttl_secs)
                .await;
        }

        self.ready
            .push_back(RoutedStreamEvent::Finish(StreamCompletionSummary {
                requested_model: self.requested_model.clone(),
                resolved_model: self.resolved_model.clone(),
                provider_id: self.provider_id.clone(),
                credential_id: self.credential_id.clone(),
                auth_kind: self.auth_kind,
                text: self.acc_text.clone(),
                reasoning: (!self.acc_reasoning.is_empty()).then(|| self.acc_reasoning.clone()),
                tool_calls: self.client_tool_calls.clone(),
                finish_reason: meta.finish_reason,
                input_tokens: self.input_tokens,
                cached_input_tokens: self.cached_input_tokens,
                reasoning_tokens: self.reasoning_tokens,
                output_tokens: self.output_tokens,
                cost_nanos,
                api_equiv_nanos,
                ttft_ms: self.ttft_ms,
                duration_ms,
                fallback_attempts: self.fallback_attempts,
            }));
    }
}

fn add_usage(
    u: &Usage,
    total_in: &mut u64,
    total_cached: &mut u64,
    total_reasoning: &mut u64,
    total_out: &mut u64,
) {
    *total_in += u.input_tokens.unwrap_or(0) as u64;
    *total_cached += u.cached_tokens.unwrap_or(0) as u64;
    *total_reasoning += u.reasoning_tokens.unwrap_or(0) as u64;
    *total_out += u.output_tokens.unwrap_or(0) as u64;
}

/// Uses `LanguageModelRequest::builder()` to populate `LanguageModelOptions` (including `pub(crate)`
/// `messages` and `tools` fields) so all 73 built-in `aisdk` providers and all custom OAuth
/// `LanguageModel`s receive identical normalized options!
fn build_aisdk_options(
    model: &AnyLanguageModel,
    req: &CanonicalRequest,
    conversation: &[Message],
    tools: &[Tool],
) -> LanguageModelOptions {
    let msgs = if conversation.is_empty() {
        vec![Message::User("".into())]
    } else {
        conversation.to_vec()
    };

    let stage = LanguageModelRequest::builder().model(model.clone());
    let mut builder = if let Some(sys) = &req.system
        && !sys.trim().is_empty()
    {
        stage.system(sys.clone()).messages(msgs)
    } else {
        stage.messages(msgs)
    };

    if let Some(t) = req.temperature {
        builder = builder.temperature(t);
    }
    if let Some(tp) = req.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(tk) = req.top_k {
        builder = builder.top_k(tk);
    }
    if let Some(stops) = &req.stop_sequences {
        builder = builder.stop_sequences(stops.clone());
    }
    if let Some(fp) = req.frequency_penalty {
        builder = builder.frequency_penalty(fp);
    }
    if let Some(seed) = req.seed {
        builder = builder.seed(seed);
    }
    if let Some(effort) = req.reasoning_effort {
        builder = builder.reasoning_effort(effort);
    }
    if !req.extra_headers.is_empty() {
        builder = builder.headers(req.extra_headers.clone());
    }
    if !req.extra_body.is_empty() {
        builder = builder.body(Value::Object(req.extra_body.clone()));
    }
    for tool in tools {
        builder = builder.with_tool(tool.clone());
    }
    builder = builder.stop_when(|_| true);

    let mut built = builder.build();
    built.max_output_tokens = req.max_output_tokens;
    built.presence_penalty = req.presence_penalty;
    if let Some(schema_val) = &req.schema
        && let Ok(schema) = serde_json::from_value::<Schema>(schema_val.clone())
    {
        built.schema = Some(schema);
    }
    // Note: `LanguageModelRequest::generate_text` resolves `system` separately, whereas calling
    // `model.generate_text(options)` directly expects `options.system` set if provided.
    let mut opts: LanguageModelOptions = (*built).clone();
    if opts.system.is_none()
        && let Some(sys) = &req.system
        && !sys.trim().is_empty()
    {
        opts.system = Some(sys.clone());
    }
    opts
}

fn cache_fingerprint(spec: &ModelSpec, req: &CanonicalRequest) -> Value {
    let msgs_json: Vec<Value> = req
        .messages
        .iter()
        .map(|m| match m {
            Message::System(s) => json!({ "role": "system", "content": s.content }),
            Message::Developer(d) => json!({ "role": "developer", "content": d }),
            Message::User(u) => json!({ "role": "user", "content": u.content }),
            Message::Assistant(a) => match &a.content {
                LanguageModelResponseContentType::Text(t) => {
                    json!({ "role": "assistant", "text": t })
                }
                LanguageModelResponseContentType::ToolCall(tc) => {
                    json!({ "role": "assistant", "tool": tc.tool.name, "id": tc.tool.id, "input": tc.input })
                }
                LanguageModelResponseContentType::Reasoning { content, .. } => {
                    json!({ "role": "assistant", "reasoning": content })
                }
                LanguageModelResponseContentType::NotSupported(_) => json!({ "role": "assistant" }),
            },
            Message::Tool(tr) => json!({
                "role": "tool",
                "id": tr.tool.id,
                "name": tr.tool.name,
                "output": tr.output.as_ref().ok(),
            }),
        })
        .collect();
    let effort_str = req.reasoning_effort.as_ref().map(|e| match e {
        ReasoningEffort::None => "none",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
    });
    json!({
        "model": spec.base_model,
        "pinned_provider": spec.pinned_target,
        "nitro": spec.nitro,
        "floor": spec.floor,
        "free_only": spec.free_only,
        "system": req.system,
        "messages": msgs_json,
        "tools": req.client_tools,
        "schema": req.schema,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "max_output_tokens": req.max_output_tokens,
        "stop_sequences": req.stop_sequences,
        "seed": req.seed,
        "reasoning_effort": effort_str,
        "openrouter_body": req.openrouter_body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn sticky_fill_first_failover_model_scoped_429_online_search_and_exact_cache() {
        let seat1_hits = Arc::new(AtomicUsize::new(0));
        let seat2_hits = Arc::new(AtomicUsize::new(0));
        let search_hits = Arc::new(AtomicUsize::new(0));

        let s1 = Arc::clone(&seat1_hits);
        let s2 = Arc::clone(&seat2_hits);
        let sh = Arc::clone(&search_hits);

        let app = Router::new()
            .route(
                "/v1/messages",
                post(move |headers: HeaderMap, axum::Json(body): axum::Json<Value>| {
                    let s1 = Arc::clone(&s1);
                    let s2 = Arc::clone(&s2);
                    async move {
                        let auth = headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or_default()
                            .to_owned();
                        let model = body["model"].as_str().unwrap_or_default().to_owned();

                        // Seat 1 returns a 429 ONLY for claude-sonnet-4-5, while claude-haiku-4-5 succeeds!
                        if auth.contains("seat1") {
                            s1.fetch_add(1, Ordering::SeqCst);
                            if model == "claude-sonnet-4-5" {
                                return (
                                    StatusCode::TOO_MANY_REQUESTS,
                                    axum::Json(json!({
                                        "error": { "type": "rate_limit_error", "message": "model rate limit" }
                                    })),
                                )
                                    .into_response();
                            }
                            return axum::Json(json!({
                                "id": "msg_haiku",
                                "content": [{ "type": "text", "text": "Haiku from Seat 1" }],
                                "usage": { "input_tokens": 10, "output_tokens": 5 }
                            }))
                            .into_response();
                        }

                        // Seat 2 handles failover for claude-sonnet-4-5 AND `:online` search tool loop!
                        s2.fetch_add(1, Ordering::SeqCst);
                        let msgs = body["messages"].as_array().cloned().unwrap_or_default();
                        let has_tool_result = msgs.iter().any(|m| {
                            m["content"]
                                .as_array()
                                .is_some_and(|arr| arr.iter().any(|b| b["type"] == "tool_result"))
                        });
                        if body["tools"].is_array() && !has_tool_result {
                            return axum::Json(json!({
                                "id": "msg_tool",
                                "content": [{
                                    "type": "tool_use",
                                    "id": "toolu_search_1",
                                    "name": "quixigate_web_search",
                                    "input": { "query": "quixigate rust aisdk" }
                                }],
                                "usage": { "input_tokens": 15, "output_tokens": 8 }
                            }))
                            .into_response();
                        }

                        axum::Json(json!({
                            "id": "msg_seat2",
                            "content": [{ "type": "text", "text": "Sonnet from Seat 2" }],
                            "usage": { "input_tokens": 20, "output_tokens": 6 }
                        }))
                        .into_response()
                    }
                }),
            )
            .route(
                "/search",
                get(move || {
                    let sh = Arc::clone(&sh);
                    async move {
                        sh.fetch_add(1, Ordering::SeqCst);
                        axum::Json(json!({
                            "results": [{ "title": "QuixiGate", "snippet": "Pure Rust LLM Gateway" }]
                        }))
                    }
                }),
            );

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}");

        let db = Database::open_in_memory().unwrap();
        let mut settings = db.get_settings().unwrap();
        settings.cache_enabled_default = true;
        db.update_settings(&settings).unwrap();

        let c1 = db
            .upsert_credential(
                None,
                "anthropic-token",
                "Claude Max Seat #1",
                AuthKind::Oauth,
                Some(&base),
                1,
                true,
                false,
                &CredentialSecret::AnthropicSetupToken {
                    token: "sk-ant-oat01-seat1".to_owned(),
                },
            )
            .unwrap();
        let c2 = db
            .upsert_credential(
                None,
                "anthropic-token",
                "Claude Max Seat #2",
                AuthKind::Oauth,
                Some(&base),
                2,
                true,
                false,
                &CredentialSecret::AnthropicSetupToken {
                    token: "sk-ant-oat01-seat2".to_owned(),
                },
            )
            .unwrap();

        let refresher = CredentialRefresher::new(db.clone());
        let catalog = ModelCatalog::embedded();
        assert!(catalog.list_models(&[]).len() > 100);
        let scanner = LocalRuntimeScanner::new();
        let router = GatewayRouter::new(db.clone(), refresher, catalog, scanner)
            .with_web_search_url(format!("{base}/search"));

        // 1. Request `anthropic/claude-sonnet-4-5` -> Seat #1 returns 429, seamlessly fails over to Seat #2!
        let gen1 = router
            .execute_text(CanonicalRequest {
                requested_model: "anthropic/claude-sonnet-4-5".to_owned(),
                messages: vec![Message::User("hello".into())],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(gen1.text, "Sonnet from Seat 2");
        assert_eq!(gen1.credential_id.as_deref(), Some(c2.id.as_str()));
        assert_eq!(gen1.fallback_attempts, 1);
        assert_eq!(gen1.cost_nanos, NanoUsd::ZERO);
        assert!(gen1.api_equiv_nanos.nanos() > 0);
        assert!(!gen1.cache_hit);

        // 2. Identical request hits exact BLAKE3 cache (`cache_hit == true`, zero new upstream hits)!
        let s2_before = seat2_hits.load(Ordering::SeqCst);
        let gen_cached = router
            .execute_text(CanonicalRequest {
                requested_model: "anthropic/claude-sonnet-4-5".to_owned(),
                messages: vec![Message::User("hello".into())],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(gen_cached.cache_hit);
        assert_eq!(gen_cached.text, "Sonnet from Seat 2");
        assert_eq!(seat2_hits.load(Ordering::SeqCst), s2_before);

        // 3. Sibling model `anthropic/claude-haiku-4-5` STILL routes to Seat #1 (model-scoped cooldown!)
        let gen_haiku = router
            .execute_text(CanonicalRequest {
                requested_model: "anthropic/claude-haiku-4-5".to_owned(),
                messages: vec![Message::User("quick".into())],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(gen_haiku.text, "Haiku from Seat 1");
        assert_eq!(gen_haiku.credential_id.as_deref(), Some(c1.id.as_str()));
        assert_eq!(gen_haiku.fallback_attempts, 0);

        // 4. `:online` modifier triggers server-side `quixigate_web_search` multi-step loop!
        let gen_online = router
            .execute_text(CanonicalRequest {
                requested_model: "anthropic/claude-sonnet-4-5:online".to_owned(),
                messages: vec![Message::User("search quixigate".into())],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(gen_online.text, "Sonnet from Seat 2");
        assert_eq!(search_hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mcp_http_jsonrpc_server_side_tool_loop_discovers_and_executes_tools() {
        use axum::Router;
        use axum::response::IntoResponse;
        use axum::routing::post;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::net::TcpListener;

        let mcp_calls = Arc::new(AtomicUsize::new(0));
        let mc = Arc::clone(&mcp_calls);

        let app = Router::new()
            .route(
                "/mcp",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let mc = Arc::clone(&mc);
                    async move {
                        match body["method"].as_str().unwrap_or_default() {
                            "tools/list" => axum::Json(json!({
                                "jsonrpc": "2.0",
                                "id": body["id"],
                                "result": {
                                    "tools": [{
                                        "name": "mcp_lookup_ticket",
                                        "description": "Looks up a support ticket by ID",
                                        "inputSchema": {
                                            "type": "object",
                                            "properties": {
                                                "ticket_id": { "type": "string" }
                                            },
                                            "required": ["ticket_id"]
                                        }
                                    }]
                                }
                            })),
                            "tools/call" => {
                                mc.fetch_add(1, Ordering::SeqCst);
                                let tid = body["params"]["arguments"]["ticket_id"]
                                    .as_str()
                                    .unwrap_or("unknown");
                                axum::Json(json!({
                                    "jsonrpc": "2.0",
                                    "id": body["id"],
                                    "result": {
                                        "ticket": tid,
                                        "status": "resolved_in_rust"
                                    }
                                }))
                            }
                            _ => axum::Json(json!({ "jsonrpc": "2.0", "result": {} })),
                        }
                    }
                }),
            )
            .route(
                "/v1/messages",
                post(|axum::Json(body): axum::Json<Value>| async move {
                    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
                    let has_tool_result = msgs.iter().any(|m| {
                        m["content"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .any(|b| b["type"] == "tool_result")
                    });
                    if !has_tool_result {
                        axum::Json(json!({
                            "id": "msg_mcp_1",
                            "stop_reason": "tool_use",
                            "content": [{
                                "type": "tool_use",
                                "id": "toolu_mcp_1",
                                "name": "mcp_lookup_ticket",
                                "input": { "ticket_id": "QX-42" }
                            }],
                            "usage": { "input_tokens": 18, "output_tokens": 9 }
                        }))
                        .into_response()
                    } else {
                        axum::Json(json!({
                            "id": "msg_mcp_2",
                            "content": [{ "type": "text", "text": "Ticket QX-42 is resolved_in_rust." }],
                            "usage": { "input_tokens": 30, "output_tokens": 8 }
                        }))
                        .into_response()
                    }
                }),
            );

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}");

        let db = Database::open_in_memory().unwrap();
        let mut settings = db.get_settings().unwrap();
        settings.mcp_server_urls = vec![format!("{base}/mcp")];
        db.update_settings(&settings).unwrap();

        db.upsert_credential(
            None,
            "anthropic-token",
            "Claude MCP Seat",
            AuthKind::Oauth,
            Some(&base),
            1,
            true,
            false,
            &CredentialSecret::AnthropicSetupToken {
                token: "sk-ant-oat01-mcp".to_owned(),
            },
        )
        .unwrap();

        let router = GatewayRouter::new(
            db.clone(),
            CredentialRefresher::new(db),
            ModelCatalog::embedded(),
            LocalRuntimeScanner::new(),
        );

        let out = router
            .execute_text(CanonicalRequest {
                requested_model: "anthropic/claude-sonnet-4-5".to_owned(),
                messages: vec![Message::User("Check ticket QX-42".into())],
                cache_preference: Some(false),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(out.text, "Ticket QX-42 is resolved_in_rust.");
        assert_eq!(mcp_calls.load(Ordering::SeqCst), 1);
    }
}
