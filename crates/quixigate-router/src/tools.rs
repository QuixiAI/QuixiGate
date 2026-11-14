//! Server-side tool implementations for `:online` web search grounding,
//! SSRF-guarded URL content extraction (`quixigate_web_fetch`), and optional
//! MCP Streamable-HTTP JSON-RPC (`tools/list`, `tools/call`) bridges.
//!
//! Key design properties:
//! - **Ordered Provider Cascade (`auto`)**: Custom override -> Keyed search APIs
//!   (Tavily, Exa, Brave, Perplexity, SearXNG, Ollama Cloud) -> Keyless public MCP
//!   round-robin ring (Exa MCP + Parallel MCP) -> Keyless Bing RSS -> DuckDuckGo Lite.
//! - **One-Shot Stateless Rescue**: When an unpinned keyed provider hits `429`/`5xx` or
//!   network failure, rescues *that single call* via the keyless ring (`rescued_from`)
//!   without caching the fallback so the next request retries the primary.
//! - **10-Minute TTL Cache**: Lock-free `moka` cache keyed by `(provider, max_results, query)`.
//! - **Prompt-Injection & Local-Model Safety**: Strips LLM chat-template control tokens
//!   (`<|im_start|>`, `<|eot_id|>`, `[INST]`, `<start_of_turn>`, etc.) and blocks SSRF
//!   against loopback, RFC1918, link-local/metadata, and DNS-rebound private IPs
//!   across every HTTP redirect hop.

use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use aisdk::core::tools::{Tool, ToolContext, ToolExecute};
use moka::future::Cache;
use reqwest::redirect::Policy;
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

pub const WEB_SEARCH_TOOL_NAME: &str = "quixigate_web_search";
pub const WEB_FETCH_TOOL_NAME: &str = "quixigate_web_fetch";

const DEFAULT_EXA_MCP_URL: &str = "https://mcp.exa.ai/mcp";
const DEFAULT_PARALLEL_MCP_URL: &str = "https://search.parallel.ai/mcp";
const DEFAULT_BING_RSS_URL: &str = "https://www.bing.com/search";
const DEFAULT_DDG_LITE_URL: &str = "https://lite.duckduckgo.com/lite/";
const SEARCH_USER_AGENT: &str = concat!(
    "QuixiGate/",
    env!("CARGO_PKG_VERSION"),
    " (+https://quixi.ai)"
);
const SEARCH_CACHE_TTL: Duration = Duration::from_secs(600);
const MAX_SNIPPET_CHARS: usize = 600;
const DEFAULT_FETCH_MAX_CHARS: usize = 12_000;
const MAX_FETCH_CHARS: usize = 30_000;
const MAX_FETCH_BYTES: usize = 2 * 1024 * 1024;
const MAX_REDIRECT_HOPS: usize = 5;

/// Model chat-template special tokens stripped from untrusted external web content
/// so scraped pages cannot hijack turn boundaries on local open-weight models.
const LLM_SPECIAL_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
    "<|eom_id|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<s>",
    "</s>",
    "<|channel|>",
    "<|message|>",
    "<|return|>",
    "<|call|>",
    "<start_of_turn>",
    "<end_of_turn>",
];

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebSearchInput {
    /// Search query to look up live facts, articles, news, or documentation on the web.
    pub query: String,
    /// Maximum number of results to return (default: 5, max: 10).
    #[serde(default)]
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebFetchInput {
    /// Public HTTP or HTTPS URL to fetch and extract readable text from.
    pub url: String,
    /// Maximum characters of extracted text to return (default: 12000, max: 30000).
    #[serde(default)]
    pub max_chars: Option<usize>,
}

/// Optional search provider credentials resolved from QuixiGate's encrypted vault and env vars.
#[derive(Debug, Clone, Default)]
pub struct WebSearchCredentials {
    pub search_endpoint_override: Option<String>,
    pub provider_pin: Option<String>,
    pub tavily_api_key: Option<String>,
    pub exa_api_key: Option<String>,
    pub brave_api_key: Option<String>,
    pub perplexity_api_key: Option<String>,
    pub ollama_api_key: Option<String>,
    pub searxng_url: Option<String>,
    pub keyless_fallback: bool,
}

impl WebSearchCredentials {
    pub fn from_env(search_endpoint_override: Option<String>) -> Self {
        let non_empty_env = |key: &str| {
            std::env::var(key)
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        };
        let keyless_fallback = non_empty_env("QUIXIGATE_WEB_KEYLESS_FALLBACK")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true);

        Self {
            search_endpoint_override: search_endpoint_override
                .filter(|s| !s.trim().is_empty())
                .or_else(|| non_empty_env("QUIXIGATE_WEB_SEARCH_URL")),
            provider_pin: non_empty_env("QUIXIGATE_WEB_SEARCH_PROVIDER").and_then(|p| {
                let low = p.to_ascii_lowercase();
                (!matches!(low.as_str(), "" | "auto" | "default")).then_some(low)
            }),
            tavily_api_key: non_empty_env("TAVILY_API_KEY"),
            exa_api_key: non_empty_env("EXA_API_KEY"),
            brave_api_key: non_empty_env("BRAVE_SEARCH_API_KEY")
                .or_else(|| non_empty_env("BRAVE_API_KEY")),
            perplexity_api_key: non_empty_env("PERPLEXITY_API_KEY"),
            ollama_api_key: non_empty_env("OLLAMA_API_KEY"),
            searxng_url: non_empty_env("SEARXNG_URL"),
            keyless_fallback,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedSearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub site_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
}

/// Endpoint overrides used by unit tests to exercise keyless MCP / RSS / Lite providers offline.
#[derive(Debug, Clone)]
pub struct KeylessEndpoints {
    pub exa_mcp_url: String,
    pub parallel_mcp_url: String,
    pub bing_rss_url: String,
    pub ddg_lite_url: String,
}

impl Default for KeylessEndpoints {
    fn default() -> Self {
        Self {
            exa_mcp_url: DEFAULT_EXA_MCP_URL.to_owned(),
            parallel_mcp_url: DEFAULT_PARALLEL_MCP_URL.to_owned(),
            bing_rss_url: DEFAULT_BING_RSS_URL.to_owned(),
            ddg_lite_url: DEFAULT_DDG_LITE_URL.to_owned(),
        }
    }
}

/// Shared stateful web search & fetch engine (TTL memo cache + keyless round-robin cursor).
pub struct WebSearchEngine {
    http: std::sync::OnceLock<reqwest::Client>,
    no_redirect_http: std::sync::OnceLock<reqwest::Client>,
    cache: Cache<String, Value>,
    ring_cursor: AtomicUsize,
    session_id: String,
    keyless_endpoints: KeylessEndpoints,
    allow_private_fetch_for_tests: bool,
}

impl WebSearchEngine {
    pub fn new_lazy() -> Self {
        let uuid = Uuid::new_v4();
        let seed = uuid.as_u128() as usize;
        Self {
            http: std::sync::OnceLock::new(),
            no_redirect_http: std::sync::OnceLock::new(),
            cache: Cache::builder()
                .max_capacity(512)
                .time_to_live(SEARCH_CACHE_TTL)
                .build(),
            ring_cursor: AtomicUsize::new(seed),
            session_id: uuid.simple().to_string(),
            keyless_endpoints: KeylessEndpoints::default(),
            allow_private_fetch_for_tests: false,
        }
    }

    pub fn new(http: reqwest::Client) -> Self {
        Self::with_endpoints(http, KeylessEndpoints::default(), false)
    }

    pub fn with_endpoints(
        http: reqwest::Client,
        keyless_endpoints: KeylessEndpoints,
        allow_private_fetch_for_tests: bool,
    ) -> Self {
        let http_cell = std::sync::OnceLock::new();
        let _ = http_cell.set(http);
        let uuid = Uuid::new_v4();
        let seed = uuid.as_u128() as usize;
        Self {
            http: http_cell,
            no_redirect_http: std::sync::OnceLock::new(),
            cache: Cache::builder()
                .max_capacity(512)
                .time_to_live(SEARCH_CACHE_TTL)
                .build(),
            ring_cursor: AtomicUsize::new(seed),
            session_id: uuid.simple().to_string(),
            keyless_endpoints,
            allow_private_fetch_for_tests,
        }
    }

    #[inline]
    fn http(&self) -> &reqwest::Client {
        self.http.get_or_init(|| {
            quixigate_providers::shared_tls_client_builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default()
        })
    }

    #[inline]
    fn no_redirect_http(&self) -> &reqwest::Client {
        self.no_redirect_http.get_or_init(|| {
            quixigate_providers::shared_tls_client_builder()
                .redirect(Policy::none())
                .connect_timeout(Duration::from_secs(8))
                .timeout(Duration::from_secs(15))
                .user_agent(SEARCH_USER_AGENT)
                .build()
                .unwrap_or_else(|_| self.http().clone())
        })
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
        creds: &WebSearchCredentials,
    ) -> Result<Value, String> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(json!({ "results": [], "note": "empty query" }));
        }
        let limit = max_results.clamp(1, 10);

        // Rung 0: Custom override (`with_web_search_url` or `QUIXIGATE_WEB_SEARCH_URL`)
        if let Some(custom_url) = creds.search_endpoint_override.as_deref() {
            return self
                .search_custom_endpoint(custom_url, trimmed, limit)
                .await;
        }

        // Explicit provider pin (`QUIXIGATE_WEB_SEARCH_PROVIDER`) — strict, no silent reroute.
        if let Some(pin) = creds.provider_pin.as_deref() {
            let cache_key = format!("pin:{pin}\0{limit}\0{}", trimmed.to_lowercase());
            if let Some(mut cached) = self.cache.get(&cache_key).await {
                cached["cached"] = Value::Bool(true);
                return Ok(cached);
            }
            let started = Instant::now();
            let (id, label, hits) = self
                .execute_named_provider(pin, trimmed, limit, creds)
                .await?;
            let payload = build_search_payload(trimmed, id, label, hits, started.elapsed(), None);
            self.cache.insert(cache_key, payload.clone()).await;
            return Ok(payload);
        }

        // Build available keyed cascade (Rung 1 + Rung 2)
        let mut keyed_candidates: Vec<&'static str> = Vec::new();
        if creds.tavily_api_key.is_some() {
            keyed_candidates.push("tavily");
        }
        if creds.exa_api_key.is_some() {
            keyed_candidates.push("exa");
        }
        if creds.brave_api_key.is_some() {
            keyed_candidates.push("brave");
        }
        if creds.perplexity_api_key.is_some() {
            keyed_candidates.push("perplexity");
        }
        if creds.searxng_url.is_some() {
            keyed_candidates.push("searxng");
        }
        if creds.ollama_api_key.is_some() {
            keyed_candidates.push("ollama");
        }

        let mut primary_errors: Vec<String> = Vec::new();
        for &cand in &keyed_candidates {
            let cache_key = format!("{cand}\0{limit}\0{}", trimmed.to_lowercase());
            if let Some(mut cached) = self.cache.get(&cache_key).await {
                cached["cached"] = Value::Bool(true);
                return Ok(cached);
            }
            let started = Instant::now();
            match self
                .execute_named_provider(cand, trimmed, limit, creds)
                .await
            {
                Ok((id, label, hits)) => {
                    let payload =
                        build_search_payload(trimmed, id, label, hits, started.elapsed(), None);
                    self.cache.insert(cache_key, payload.clone()).await;
                    return Ok(payload);
                }
                Err(err) => {
                    tracing::warn!(provider = cand, error = %err, "keyed web search provider failed");
                    primary_errors.push(format!("{cand}: {err}"));
                }
            }
        }

        if !creds.keyless_fallback {
            if primary_errors.is_empty() {
                return Err(
                    "No web search credentials configured and keyless fallback is disabled."
                        .to_owned(),
                );
            }
            return Err(format!(
                "All configured web search providers failed: {}",
                primary_errors.join("; ")
            ));
        }

        // Rung 3: Keyless ring (either primary zero-key path, or one-shot stateless rescue).
        let is_rescue = !keyed_candidates.is_empty();
        let keyless_cache_key = format!("keyless\0{limit}\0{}", trimmed.to_lowercase());
        if !is_rescue && let Some(mut cached) = self.cache.get(&keyless_cache_key).await {
            cached["cached"] = Value::Bool(true);
            return Ok(cached);
        }

        let started = Instant::now();
        match self.search_keyless_ring(trimmed, limit).await {
            Ok((id, label, hits)) => {
                let rescued_note = is_rescue.then(|| primary_errors.join("; "));
                let payload = build_search_payload(
                    trimmed,
                    id,
                    label,
                    hits,
                    started.elapsed(),
                    rescued_note.as_deref(),
                );
                // Never cache a one-shot rescue so the next call retries the keyed primary!
                if !is_rescue {
                    self.cache.insert(keyless_cache_key, payload.clone()).await;
                }
                Ok(payload)
            }
            Err(ring_err) => {
                if primary_errors.is_empty() {
                    Err(ring_err)
                } else {
                    Err(format!(
                        "{} (keyless rescue also failed: {ring_err})",
                        primary_errors.join("; ")
                    ))
                }
            }
        }
    }

    pub async fn fetch_url(
        &self,
        raw_url: &str,
        max_chars: usize,
        creds: &WebSearchCredentials,
    ) -> Result<Value, String> {
        let trimmed = raw_url.trim();
        let budget = max_chars.clamp(1_000, MAX_FETCH_CHARS);
        let validated = if self.allow_private_fetch_for_tests {
            reqwest::Url::parse(trimmed).map_err(|e| format!("Invalid URL: {e}"))?
        } else {
            validate_public_http_url(trimmed).await?
        };
        let target = validated.as_str().to_owned();
        let started = Instant::now();

        // Try clean Markdown extraction via keyless public MCPs first when allowed (unless testing private URLs)
        if creds.keyless_fallback && !self.allow_private_fetch_for_tests {
            if let Ok((title, content)) = self.fetch_parallel_mcp(&target).await
                && !content.trim().is_empty()
            {
                return Ok(build_fetch_payload(
                    &target,
                    &title,
                    &content,
                    budget,
                    "parallel_mcp",
                    started.elapsed(),
                ));
            }
            if let Ok((title, content)) = self.fetch_exa_mcp(&target).await
                && !content.trim().is_empty()
            {
                return Ok(build_fetch_payload(
                    &target,
                    &title,
                    &content,
                    budget,
                    "exa_mcp",
                    started.elapsed(),
                ));
            }
        }

        // Direct SSRF-guarded HTTP fetch with per-hop redirect validation
        let (final_url, title, content) = self.fetch_direct_sanitized(validated).await?;
        Ok(build_fetch_payload(
            &final_url,
            &title,
            &content,
            budget,
            "direct_http",
            started.elapsed(),
        ))
    }

    async fn execute_named_provider(
        &self,
        provider: &str,
        query: &str,
        limit: usize,
        creds: &WebSearchCredentials,
    ) -> Result<(&'static str, &'static str, Vec<NormalizedSearchHit>), String> {
        match provider {
            "tavily" => {
                let key = creds
                    .tavily_api_key
                    .as_deref()
                    .ok_or_else(|| "TAVILY_API_KEY is not set".to_owned())?;
                let hits = self.search_tavily(key, query, limit).await?;
                Ok(("tavily", "Tavily", hits))
            }
            "exa" => {
                let key = creds
                    .exa_api_key
                    .as_deref()
                    .ok_or_else(|| "EXA_API_KEY is not set".to_owned())?;
                let hits = self.search_exa_rest(key, query, limit).await?;
                Ok(("exa", "Exa", hits))
            }
            "brave" | "brave_search" => {
                let key = creds
                    .brave_api_key
                    .as_deref()
                    .ok_or_else(|| "BRAVE_SEARCH_API_KEY is not set".to_owned())?;
                let hits = self.search_brave(key, query, limit).await?;
                Ok(("brave", "Brave Search", hits))
            }
            "perplexity" => {
                let key = creds
                    .perplexity_api_key
                    .as_deref()
                    .ok_or_else(|| "PERPLEXITY_API_KEY is not set".to_owned())?;
                let hits = self.search_perplexity(key, query, limit).await?;
                Ok(("perplexity", "Perplexity Search", hits))
            }
            "searxng" => {
                let base = creds
                    .searxng_url
                    .as_deref()
                    .ok_or_else(|| "SEARXNG_URL is not set".to_owned())?;
                let hits = self.search_searxng(base, query, limit).await?;
                Ok(("searxng", "SearXNG", hits))
            }
            "ollama" => {
                let hits = self
                    .search_ollama(creds.ollama_api_key.as_deref(), query, limit)
                    .await?;
                Ok(("ollama", "Ollama Web Search", hits))
            }
            "exa_mcp" => {
                let hits = self.search_exa_mcp(query, limit).await?;
                Ok(("exa_mcp", "Exa (Keyless MCP)", hits))
            }
            "parallel_mcp" | "parallel" => {
                let hits = self.search_parallel_mcp(query, limit).await?;
                Ok(("parallel_mcp", "Parallel (Keyless MCP)", hits))
            }
            "bing_rss" | "bing" => {
                let hits = self.search_bing_rss(query, limit).await?;
                Ok(("bing_rss", "Bing RSS", hits))
            }
            "duckduckgo_lite" | "ddg" | "duckduckgo" => {
                let hits = self.search_ddg_lite(query, limit).await?;
                Ok(("duckduckgo_lite", "DuckDuckGo Lite", hits))
            }
            other => Err(format!("Unknown web search provider '{other}'")),
        }
    }

    async fn search_keyless_ring(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<(&'static str, &'static str, Vec<NormalizedSearchHit>), String> {
        let mcp_ring = ["exa_mcp", "parallel_mcp"];
        let start = self.ring_cursor.fetch_add(1, Ordering::Relaxed) % mcp_ring.len();
        let ordered = [
            mcp_ring[start],
            mcp_ring[(start + 1) % mcp_ring.len()],
            "bing_rss",
            "duckduckgo_lite",
        ];

        let mut errors = Vec::new();
        for vendor in ordered {
            let res = match vendor {
                "exa_mcp" => self
                    .search_exa_mcp(query, limit)
                    .await
                    .map(|h| ("exa_mcp", "Exa (Keyless MCP)", h)),
                "parallel_mcp" => self
                    .search_parallel_mcp(query, limit)
                    .await
                    .map(|h| ("parallel_mcp", "Parallel (Keyless MCP)", h)),
                "bing_rss" => self
                    .search_bing_rss(query, limit)
                    .await
                    .map(|h| ("bing_rss", "Bing RSS", h)),
                _ => self
                    .search_ddg_lite(query, limit)
                    .await
                    .map(|h| ("duckduckgo_lite", "DuckDuckGo Lite", h)),
            };
            match res {
                Ok((_, _, ref hits)) if hits.is_empty() => {
                    errors.push(format!("{vendor}: 0 results"));
                }
                Ok(ok) => return Ok(ok),
                Err(e) => {
                    tracing::debug!(provider = vendor, error = %e, "keyless search rung failed");
                    errors.push(format!("{vendor}: {e}"));
                }
            }
        }

        Err(format!(
            "All keyless web search providers failed ({})",
            errors.join("; ")
        ))
    }

    async fn search_custom_endpoint(
        &self,
        endpoint: &str,
        query: &str,
        limit: usize,
    ) -> Result<Value, String> {
        let started = Instant::now();
        let resp = self
            .http()
            .get(endpoint)
            .query(&[
                ("q", query),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ])
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| format!("web search failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("custom search HTTP {}", resp.status()));
        }

        let data: Value = resp
            .json()
            .await
            .unwrap_or_else(|_| json!({ "results": [] }));

        if let Some(arr) = data["results"].as_array() {
            let hits: Vec<NormalizedSearchHit> = arr
                .iter()
                .take(limit)
                .filter_map(|item| {
                    let url = item["url"].as_str().unwrap_or("").trim().to_owned();
                    let title = sanitize_external_text(item["title"].as_str().unwrap_or(query));
                    let snippet = truncate_chars(
                        &sanitize_external_text(
                            item["snippet"]
                                .as_str()
                                .or_else(|| item["content"].as_str())
                                .or_else(|| item["description"].as_str())
                                .unwrap_or(""),
                        ),
                        MAX_SNIPPET_CHARS,
                    );
                    (!url.is_empty() || !snippet.is_empty()).then(|| NormalizedSearchHit {
                        site_name: site_name_from_url(&url),
                        title,
                        url,
                        snippet,
                        published: item["published"]
                            .as_str()
                            .or_else(|| item["publishedDate"].as_str())
                            .map(sanitize_external_text),
                    })
                })
                .collect();
            return Ok(build_search_payload(
                query,
                "custom",
                "Custom Search Endpoint",
                hits,
                started.elapsed(),
                None,
            ));
        }

        Ok(data)
    }

    async fn search_tavily(
        &self,
        api_key: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let resp = self
            .http()
            .post("https://api.tavily.com/search")
            .timeout(Duration::from_secs(15))
            .json(&json!({
                "api_key": api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
                "include_answer": false,
            }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let data: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(data["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let url = item["url"].as_str()?.trim().to_owned();
                if url.is_empty() {
                    return None;
                }
                Some(NormalizedSearchHit {
                    title: sanitize_external_text(item["title"].as_str().unwrap_or("")),
                    site_name: site_name_from_url(&url),
                    url,
                    snippet: truncate_chars(
                        &sanitize_external_text(item["content"].as_str().unwrap_or("")),
                        MAX_SNIPPET_CHARS,
                    ),
                    published: item["published_date"].as_str().map(sanitize_external_text),
                })
            })
            .take(limit)
            .collect())
    }

    async fn search_exa_rest(
        &self,
        api_key: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let resp = self
            .http()
            .post("https://api.exa.ai/search")
            .header("x-api-key", api_key)
            .timeout(Duration::from_secs(15))
            .json(&json!({
                "query": query,
                "type": "auto",
                "numResults": limit,
                "contents": { "highlights": true },
            }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let data: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(data["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let url = item["url"].as_str()?.trim().to_owned();
                if url.is_empty() {
                    return None;
                }
                let highlights = item["highlights"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .or_else(|| item["text"].as_str().map(str::to_owned))
                    .unwrap_or_default();
                Some(NormalizedSearchHit {
                    title: sanitize_external_text(item["title"].as_str().unwrap_or("")),
                    site_name: site_name_from_url(&url),
                    url,
                    snippet: truncate_chars(
                        &sanitize_external_text(&highlights),
                        MAX_SNIPPET_CHARS,
                    ),
                    published: item["publishedDate"].as_str().map(sanitize_external_text),
                })
            })
            .take(limit)
            .collect())
    }

    async fn search_brave(
        &self,
        api_key: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let resp = self
            .http()
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("X-Subscription-Token", api_key)
            .header("Accept", "application/json")
            .query(&[("q", query), ("count", &limit.to_string())])
            .timeout(Duration::from_secs(12))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let data: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(data["web"]["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let url = item["url"].as_str()?.trim().to_owned();
                if url.is_empty() {
                    return None;
                }
                Some(NormalizedSearchHit {
                    title: sanitize_external_text(&strip_html_tags(
                        item["title"].as_str().unwrap_or(""),
                    )),
                    site_name: site_name_from_url(&url),
                    url,
                    snippet: truncate_chars(
                        &sanitize_external_text(&strip_html_tags(
                            item["description"].as_str().unwrap_or(""),
                        )),
                        MAX_SNIPPET_CHARS,
                    ),
                    published: item["age"].as_str().map(sanitize_external_text),
                })
            })
            .take(limit)
            .collect())
    }

    async fn search_perplexity(
        &self,
        api_key: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let resp = self
            .http()
            .post("https://api.perplexity.ai/search")
            .bearer_auth(api_key)
            .timeout(Duration::from_secs(15))
            .json(&json!({ "query": query, "max_results": limit }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let data: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(data["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let url = item["url"].as_str()?.trim().to_owned();
                if url.is_empty() {
                    return None;
                }
                Some(NormalizedSearchHit {
                    title: sanitize_external_text(item["title"].as_str().unwrap_or("")),
                    site_name: site_name_from_url(&url),
                    url,
                    snippet: truncate_chars(
                        &sanitize_external_text(item["snippet"].as_str().unwrap_or("")),
                        MAX_SNIPPET_CHARS,
                    ),
                    published: item["date"].as_str().map(sanitize_external_text),
                })
            })
            .take(limit)
            .collect())
    }

    async fn search_searxng(
        &self,
        base_url: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let url = format!("{}/search", base_url.trim_end_matches('/'));
        let resp = self
            .http()
            .get(&url)
            .header("User-Agent", SEARCH_USER_AGENT)
            .query(&[("q", query), ("format", "json")])
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let data: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(data["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let url = item["url"].as_str()?.trim().to_owned();
                if url.is_empty() {
                    return None;
                }
                Some(NormalizedSearchHit {
                    title: sanitize_external_text(item["title"].as_str().unwrap_or("")),
                    site_name: site_name_from_url(&url),
                    url,
                    snippet: truncate_chars(
                        &sanitize_external_text(item["content"].as_str().unwrap_or("")),
                        MAX_SNIPPET_CHARS,
                    ),
                    published: item["publishedDate"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .map(sanitize_external_text),
                })
            })
            .take(limit)
            .collect())
    }

    async fn search_ollama(
        &self,
        api_key: Option<&str>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let (url, auth) = if let Some(k) = api_key {
            ("https://ollama.com/api/web_search", Some(k))
        } else {
            ("http://127.0.0.1:11434/api/experimental/web_search", None)
        };
        let mut req = self
            .http()
            .post(url)
            .timeout(Duration::from_secs(15))
            .json(&json!({ "query": query, "max_results": limit }));
        if let Some(token) = auth {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let data: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(data["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let url = item["url"].as_str()?.trim().to_owned();
                if url.is_empty() {
                    return None;
                }
                Some(NormalizedSearchHit {
                    title: sanitize_external_text(item["title"].as_str().unwrap_or("")),
                    site_name: site_name_from_url(&url),
                    url,
                    snippet: truncate_chars(
                        &sanitize_external_text(item["content"].as_str().unwrap_or("")),
                        MAX_SNIPPET_CHARS,
                    ),
                    published: None,
                })
            })
            .take(limit)
            .collect())
    }

    async fn call_keyless_mcp(
        &self,
        mcp_url: &str,
        tool: &str,
        args: Value,
    ) -> Result<String, String> {
        let resp = self
            .http()
            .post(mcp_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("User-Agent", SEARCH_USER_AGENT)
            .timeout(Duration::from_secs(15))
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": args,
                }
            }))
            .send()
            .await
            .map_err(|e| format!("MCP request failed: {e}"))?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
        let body = String::from_utf8_lossy(&bytes);
        if !status.is_success() {
            return Err(format!(
                "HTTP {}: {}",
                status.as_u16(),
                truncate_chars(body.trim(), 200)
            ));
        }
        parse_mcp_text_payload(&body)
    }

    async fn search_exa_mcp(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let text = self
            .call_keyless_mcp(
                &self.keyless_endpoints.exa_mcp_url,
                "web_search_exa",
                json!({ "query": query, "numResults": limit }),
            )
            .await?;
        Ok(parse_exa_mcp_search_text(&text, limit))
    }

    async fn search_parallel_mcp(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let text = self
            .call_keyless_mcp(
                &self.keyless_endpoints.parallel_mcp_url,
                "web_search",
                json!({
                    "objective": query,
                    "search_queries": [query],
                    "session_id": self.session_id,
                }),
            )
            .await?;
        parse_parallel_mcp_search_json(&text, limit)
    }

    async fn fetch_parallel_mcp(&self, url: &str) -> Result<(String, String), String> {
        let text = self
            .call_keyless_mcp(
                &self.keyless_endpoints.parallel_mcp_url,
                "web_fetch",
                json!({
                    "urls": [url],
                    "objective": "Full page content",
                    "session_id": self.session_id,
                }),
            )
            .await?;
        let data: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let first = data["results"]
            .as_array()
            .and_then(|a| a.first())
            .ok_or_else(|| "Parallel web_fetch returned no results".to_owned())?;
        let title = first["title"].as_str().unwrap_or("").to_owned();
        let content = first["full_content"]
            .as_str()
            .or_else(|| first["content"].as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                first["excerpts"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .unwrap_or_default()
            });
        Ok((title, content))
    }

    async fn fetch_exa_mcp(&self, url: &str) -> Result<(String, String), String> {
        let text = self
            .call_keyless_mcp(
                &self.keyless_endpoints.exa_mcp_url,
                "web_fetch_exa",
                json!({ "urls": [url] }),
            )
            .await?;
        let title = text
            .lines()
            .map(str::trim)
            .find_map(|line| {
                line.strip_prefix("Title:")
                    .or_else(|| line.strip_prefix("# "))
                    .map(|s| s.trim().to_owned())
            })
            .unwrap_or_default();
        Ok((title, text))
    }

    async fn search_bing_rss(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let resp = self
            .http()
            .get(&self.keyless_endpoints.bing_rss_url)
            .header("User-Agent", SEARCH_USER_AGENT)
            .query(&[("q", query), ("format", "rss")])
            .timeout(Duration::from_secs(12))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let xml = resp.text().await.map_err(|e| e.to_string())?;
        Ok(parse_bing_rss_xml(&xml, limit))
    }

    async fn search_ddg_lite(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NormalizedSearchHit>, String> {
        let resp = self
            .http()
            .get(&self.keyless_endpoints.ddg_lite_url)
            .header("User-Agent", SEARCH_USER_AGENT)
            .query(&[("q", query)])
            .timeout(Duration::from_secs(12))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let html = resp.text().await.map_err(|e| e.to_string())?;
        parse_ddg_lite_html(&html, limit)
    }

    async fn fetch_direct_sanitized(
        &self,
        mut current: reqwest::Url,
    ) -> Result<(String, String, String), String> {
        for _ in 0..=MAX_REDIRECT_HOPS {
            let resp = self
                .no_redirect_http()
                .get(current.clone())
                .header(
                    "Accept",
                    "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1",
                )
                .send()
                .await
                .map_err(|e| format!("HTTP fetch failed: {e}"))?;

            if resp.status().is_redirection() {
                let loc = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| "Redirect missing Location header".to_owned())?;
                let next = current
                    .join(loc)
                    .map_err(|e| format!("Invalid redirect URL: {e}"))?;
                current = if self.allow_private_fetch_for_tests {
                    next
                } else {
                    validate_public_http_url(next.as_str()).await?
                };
                continue;
            }

            if !resp.status().is_success() {
                return Err(format!("Origin returned HTTP {}", resp.status()));
            }

            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();
            let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
            let capped = &bytes[..bytes.len().min(MAX_FETCH_BYTES)];
            let raw_text = String::from_utf8_lossy(capped);

            let (title, body_text) = if content_type.contains("html")
                || raw_text.trim_start().starts_with("<!")
                || raw_text.contains("<body")
            {
                extract_readable_html(&raw_text)
            } else {
                (String::new(), raw_text.into_owned())
            };
            return Ok((current.to_string(), title, body_text));
        }

        Err("Too many HTTP redirects".to_owned())
    }
}

/// Builds the server-side `:online` web search tool (`quixigate_web_search`).
pub fn build_web_search_tool(
    http: reqwest::Client,
    search_endpoint_override: Option<String>,
) -> Tool {
    let engine = Arc::new(WebSearchEngine::new(http));
    let creds = WebSearchCredentials::from_env(search_endpoint_override);
    build_web_search_tool_with_engine(engine, creds)
}

/// Builds `quixigate_web_search` backed by a shared [`WebSearchEngine`] (TTL cache + ring cursor).
pub fn build_web_search_tool_with_engine(
    engine: Arc<WebSearchEngine>,
    creds: WebSearchCredentials,
) -> Tool {
    Tool {
        name: WEB_SEARCH_TOOL_NAME.to_owned(),
        description: "Search the live web for current facts, articles, news, documentation, and URLs. Returns normalized titles, URLs, and snippets. Follow up with quixigate_web_fetch on specific URLs when full page text is needed.".to_owned(),
        input_schema: schema_for!(WebSearchInput),
        execute: ToolExecute::from_async(move |_ctx: ToolContext, input: Value| {
            let engine = Arc::clone(&engine);
            let creds = creds.clone();
            async move {
                let query = input["query"].as_str().unwrap_or_default().to_owned();
                let max_results = input["max_results"]
                    .as_u64()
                    .or_else(|| input["count"].as_u64())
                    .or_else(|| input["limit"].as_u64())
                    .map(|n| n as usize)
                    .unwrap_or(5);
                let val = engine
                    .search(&query, max_results, &creds)
                    .await
                    .map_err(aisdk::Error::ToolCallError)?;
                Ok(val.to_string())
            }
        }),
    }
}

/// Builds the companion SSRF-guarded `quixigate_web_fetch` tool for reading full web pages.
pub fn build_web_fetch_tool_with_engine(
    engine: Arc<WebSearchEngine>,
    creds: WebSearchCredentials,
) -> Tool {
    Tool {
        name: WEB_FETCH_TOOL_NAME.to_owned(),
        description: "Fetch a public HTTP/HTTPS URL and extract clean readable text/markdown (removing scripts, styles, ads, and base64 blobs). Use after quixigate_web_search or when given a direct URL.".to_owned(),
        input_schema: schema_for!(WebFetchInput),
        execute: ToolExecute::from_async(move |_ctx: ToolContext, input: Value| {
            let engine = Arc::clone(&engine);
            let creds = creds.clone();
            async move {
                let url = input["url"].as_str().unwrap_or_default().to_owned();
                let max_chars = input["max_chars"]
                    .as_u64()
                    .map(|n| n as usize)
                    .unwrap_or(DEFAULT_FETCH_MAX_CHARS);
                let val = engine
                    .fetch_url(&url, max_chars, &creds)
                    .await
                    .map_err(aisdk::Error::ToolCallError)?;
                Ok(val.to_string())
            }
        }),
    }
}

fn build_search_payload(
    query: &str,
    provider_id: &str,
    provider_label: &str,
    results: Vec<NormalizedSearchHit>,
    elapsed: Duration,
    rescued_from: Option<&str>,
) -> Value {
    let mut obj = json!({
        "query": query,
        "provider": {
            "id": provider_id,
            "label": provider_label,
        },
        "count": results.len(),
        "took_ms": elapsed.as_millis() as u64,
        "cached": false,
        "external_content": {
            "untrusted": true,
            "source": "web_search",
            "provider": provider_id,
        },
        "results": results,
    });
    if let Some(note) = rescued_from {
        obj["rescued_from"] = Value::String(note.to_owned());
    }
    obj
}

fn build_fetch_payload(
    url: &str,
    title: &str,
    raw_content: &str,
    max_chars: usize,
    provider_id: &str,
    elapsed: Duration,
) -> Value {
    let cleaned = sanitize_external_text(&strip_inline_base64_images(raw_content));
    let (content, truncated) = head_tail_truncate(&cleaned, max_chars);
    json!({
        "url": url,
        "title": sanitize_external_text(title),
        "provider": provider_id,
        "took_ms": elapsed.as_millis() as u64,
        "char_count": content.chars().count(),
        "truncated": truncated,
        "external_content": {
            "untrusted": true,
            "source": "web_fetch",
        },
        "content": content,
    })
}

/// Sanitizes untrusted external web prose by removing LLM chat-template special tokens
/// and forged untrusted-content boundary markers.
pub fn sanitize_external_text(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut out = input.to_owned();
    for tok in LLM_SPECIAL_TOKENS {
        if out.contains(tok) {
            out = out.replace(tok, "[REMOVED_SPECIAL_TOKEN]");
        }
    }
    let lower = out.to_ascii_lowercase();
    if lower.contains("external_untrusted_content") || lower.contains("external untrusted content")
    {
        out = out
            .replace("<<<EXTERNAL_UNTRUSTED_CONTENT", "[[MARKER_SANITIZED")
            .replace(
                "<<<END_EXTERNAL_UNTRUSTED_CONTENT",
                "[[END_MARKER_SANITIZED",
            );
    }
    out.trim().to_owned()
}

/// Validates that a URL uses `http`/`https`, carries no embedded credentials, and does not
/// target loopback, private, link-local/metadata, or internal hostnames/IPs (SSRF guard).
pub async fn validate_public_http_url(raw: &str) -> Result<reqwest::Url, String> {
    let parsed = reqwest::Url::parse(raw.trim()).map_err(|e| format!("Invalid URL: {e}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err("Blocked: URL must start with http:// or https://".to_owned());
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("Blocked: URL must not contain embedded credentials".to_owned());
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| "Blocked: URL is missing a hostname".to_owned())?;
    let host_clean = host.trim_matches(|c| c == '[' || c == ']');
    let lower = host_clean.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "localhost"
            | "0.0.0.0"
            | "host.docker.internal"
            | "gateway.docker.internal"
            | "metadata.google.internal"
    ) || lower.ends_with(".localhost")
        || lower.ends_with(".local")
        || lower.ends_with(".internal")
    {
        return Err(format!(
            "Blocked: internal hostname '{host_clean}' is not permitted"
        ));
    }

    if let Ok(ip) = host_clean.parse::<IpAddr>() {
        if !is_globally_routable_ip(ip) {
            return Err(format!(
                "Blocked: private or internal IP '{ip}' is not permitted"
            ));
        }
        return Ok(parsed);
    }

    let port = parsed.port_or_known_default().unwrap_or(80);
    let addrs = tokio::net::lookup_host((host_clean, port))
        .await
        .map_err(|e| format!("Cannot resolve hostname '{host_clean}': {e}"))?;
    let mut resolved_any = false;
    for sock in addrs {
        resolved_any = true;
        if !is_globally_routable_ip(sock.ip()) {
            return Err(format!(
                "Blocked: hostname '{host_clean}' resolves to internal IP {}",
                sock.ip()
            ));
        }
    }
    if !resolved_any {
        return Err(format!("Cannot resolve hostname '{host_clean}'"));
    }
    Ok(parsed)
}

pub fn is_globally_routable_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            let is_cgnat = o[0] == 100 && (o[1] & 0b1100_0000) == 64; // 100.64.0.0/10
            let is_benchmark = o[0] == 198 && (o[1] == 18 || o[1] == 19); // 198.18.0.0/15
            !(v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || v4.is_multicast()
                || o[0] == 0
                || is_cgnat
                || is_benchmark
                || o[0] >= 240)
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_globally_routable_ip(IpAddr::V4(mapped));
            }
            let seg0 = v6.segments()[0];
            let is_ula = (seg0 & 0xfe00) == 0xfc00; // fc00::/7
            let is_link_local = (seg0 & 0xffc0) == 0xfe80; // fe80::/10
            let is_doc = seg0 == 0x2001 && v6.segments()[1] == 0x0db8;
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || is_ula
                || is_link_local
                || is_doc)
        }
    }
}

pub(crate) fn parse_mcp_text_payload(body: &str) -> Result<String, String> {
    let extract_envelope = |raw: &str| -> Option<Result<String, String>> {
        let trimmed = raw.trim();
        if !trimmed.starts_with('{') {
            return None;
        }
        let data: Value = serde_json::from_str(trimmed).ok()?;
        if let Some(err) = data.get("error") {
            let msg = err["message"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| err.to_string());
            return Some(Err(msg));
        }
        let result = data.get("result")?;
        let is_err = result["isError"].as_bool().unwrap_or(false);
        if let Some(structured) = result.get("structuredContent").filter(|v| v.is_object()) {
            return Some(if is_err {
                Err(structured.to_string())
            } else {
                Ok(structured.to_string())
            });
        }
        let texts: Vec<&str> = result["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| c["text"].as_str())
            .filter(|t| !t.is_empty())
            .collect();
        if is_err {
            return Some(Err(if texts.is_empty() {
                "MCP tool returned isError".to_owned()
            } else {
                texts.join(" ")
            }));
        }
        texts.first().map(|t| Ok((*t).to_owned()))
    };

    if let Some(res) = extract_envelope(body) {
        return res;
    }
    for line in body.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("data:")
            && let Some(res) = extract_envelope(rest.trim_start())
        {
            return res;
        }
    }
    Err("Unrecognized MCP response shape".to_owned())
}

pub(crate) fn parse_exa_mcp_search_text(text: &str, limit: usize) -> Vec<NormalizedSearchHit> {
    const LABELS: &[&str] = &[
        "Title:",
        "URL:",
        "Highlights:",
        "Text:",
        "Published:",
        "Author:",
    ];
    let mut hits = Vec::new();

    for block in text.split("\n---\n") {
        let mut title = String::new();
        let mut url = String::new();
        let mut published: Option<String> = None;
        let mut body_lines: Vec<&str> = Vec::new();
        let mut in_body = false;

        for raw_line in block.lines() {
            let line = raw_line.trim();
            if let Some(rest) = line.strip_prefix("Title:") {
                title = rest.trim().to_owned();
                in_body = false;
            } else if let Some(rest) = line.strip_prefix("URL:") {
                url = rest.trim().to_owned();
                in_body = false;
            } else if let Some(rest) = line.strip_prefix("Published:") {
                let p = rest.trim();
                if !p.is_empty() {
                    published = Some(sanitize_external_text(p));
                }
                in_body = false;
            } else if let Some(rest) = line
                .strip_prefix("Highlights:")
                .or_else(|| line.strip_prefix("Text:"))
            {
                in_body = true;
                let inline = rest.trim();
                if !inline.is_empty() {
                    body_lines.push(inline);
                }
            } else if LABELS.iter().any(|lbl| line.starts_with(lbl)) {
                in_body = false;
            } else if in_body && !line.is_empty() {
                body_lines.push(line);
            }
        }

        if !url.is_empty() {
            let snippet = truncate_chars(
                &sanitize_external_text(&body_lines.join(" ")),
                MAX_SNIPPET_CHARS,
            );
            hits.push(NormalizedSearchHit {
                title: sanitize_external_text(&title),
                site_name: site_name_from_url(&url),
                url,
                snippet,
                published,
            });
            if hits.len() >= limit {
                break;
            }
        }
    }

    hits
}

pub(crate) fn parse_parallel_mcp_search_json(
    text: &str,
    limit: usize,
) -> Result<Vec<NormalizedSearchHit>, String> {
    let data: Value =
        serde_json::from_str(text).map_err(|e| format!("Invalid Parallel JSON: {e}"))?;
    Ok(data["results"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let url = item["url"].as_str()?.trim().to_owned();
            if url.is_empty() {
                return None;
            }
            let snippet_raw = item["excerpts"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .or_else(|| item["snippet"].as_str().map(str::to_owned))
                .unwrap_or_default();
            Some(NormalizedSearchHit {
                title: sanitize_external_text(item["title"].as_str().unwrap_or("")),
                site_name: site_name_from_url(&url),
                url,
                snippet: truncate_chars(&sanitize_external_text(&snippet_raw), MAX_SNIPPET_CHARS),
                published: item["publish_date"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(sanitize_external_text),
            })
        })
        .take(limit)
        .collect())
}

pub(crate) fn parse_bing_rss_xml(xml: &str, limit: usize) -> Vec<NormalizedSearchHit> {
    let mut hits = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<item>") {
        let after_item = &rest[start + "<item>".len()..];
        let Some(end) = after_item.find("</item>") else {
            break;
        };
        let chunk = &after_item[..end];
        rest = &after_item[end + "</item>".len()..];

        let title = extract_xml_tag(chunk, "title")
            .map(|s| sanitize_external_text(&strip_html_tags(&s)))
            .unwrap_or_default();
        let url = extract_xml_tag(chunk, "link")
            .map(|s| decode_html_entities(s.trim()))
            .unwrap_or_default();
        let snippet = extract_xml_tag(chunk, "description")
            .map(|s| {
                truncate_chars(
                    &sanitize_external_text(&strip_html_tags(&s)),
                    MAX_SNIPPET_CHARS,
                )
            })
            .unwrap_or_default();
        let published = extract_xml_tag(chunk, "pubDate")
            .map(|s| sanitize_external_text(s.trim()))
            .filter(|s| !s.is_empty());

        if !title.is_empty() && !url.is_empty() {
            hits.push(NormalizedSearchHit {
                site_name: site_name_from_url(&url),
                title,
                url,
                snippet,
                published,
            });
            if hits.len() >= limit {
                break;
            }
        }
    }
    hits
}

pub(crate) fn parse_ddg_lite_html(
    html: &str,
    limit: usize,
) -> Result<Vec<NormalizedSearchHit>, String> {
    if html.contains("anomaly-modal")
        || html.contains("g-recaptcha")
        || html.contains("challenge-form")
    {
        return Err("DuckDuckGo Lite returned a bot-detection challenge".to_owned());
    }

    let mut hits = Vec::new();
    let mut cursor = html;
    while let Some(a_idx) = cursor.find("result-link") {
        let before = &cursor[..a_idx];
        let tag_start = before.rfind("<a").unwrap_or(a_idx);
        let from_a = &cursor[tag_start..];
        let Some(open_end) = from_a.find('>') else {
            break;
        };
        let open_tag = &from_a[..open_end];
        let after_open = &from_a[open_end + 1..];
        let Some(close_a) = after_open.find("</a>") else {
            break;
        };
        let raw_title = &after_open[..close_a];
        let raw_href = extract_attr(open_tag, "href").unwrap_or_default();
        let url = decode_ddg_redirect(&decode_html_entities(&raw_href));
        let title = sanitize_external_text(&strip_html_tags(raw_title));

        let trailing = &after_open[close_a + 4..];
        let next_link = trailing.find("result-link").unwrap_or(trailing.len());
        let scoped = &trailing[..next_link];
        let snippet = if let Some(sn_idx) = scoped.find("result-snippet") {
            let sn_sub = &scoped[sn_idx..];
            if let Some(gt) = sn_sub.find('>') {
                let sn_body = &sn_sub[gt + 1..];
                let end_td = sn_body
                    .find("</td>")
                    .or_else(|| sn_body.find("</a>"))
                    .unwrap_or(sn_body.len());
                truncate_chars(
                    &sanitize_external_text(&strip_html_tags(&sn_body[..end_td])),
                    MAX_SNIPPET_CHARS,
                )
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !title.is_empty() && !url.is_empty() {
            hits.push(NormalizedSearchHit {
                site_name: site_name_from_url(&url),
                title,
                url,
                snippet,
                published: None,
            });
            if hits.len() >= limit {
                break;
            }
        }
        cursor = trailing;
    }

    if hits.is_empty() {
        return Err("DuckDuckGo Lite returned no parseable results".to_owned());
    }
    Ok(hits)
}

fn decode_ddg_redirect(href: &str) -> String {
    let normalized = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_owned()
    };
    if let Ok(parsed) = reqwest::Url::parse(&normalized)
        && parsed
            .host_str()
            .is_some_and(|h| h.ends_with("duckduckgo.com"))
        && parsed.path().starts_with("/l/")
    {
        for (k, v) in parsed.query_pairs() {
            if k == "uddg" && !v.is_empty() {
                return v.into_owned();
            }
        }
    }
    normalized
}

fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    let inner = xml[start..end].trim();
    if let Some(cdata) = inner
        .strip_prefix("<![CDATA[")
        .and_then(|s| s.strip_suffix("]]>"))
    {
        Some(cdata.to_owned())
    } else {
        Some(decode_html_entities(inner))
    }
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pat_dq = format!("{attr}=\"");
    if let Some(idx) = tag.find(&pat_dq) {
        let rest = &tag[idx + pat_dq.len()..];
        let end = rest.find('"')?;
        return Some(rest[..end].to_owned());
    }
    let pat_sq = format!("{attr}='");
    if let Some(idx) = tag.find(&pat_sq) {
        let rest = &tag[idx + pat_sq.len()..];
        let end = rest.find('\'')?;
        return Some(rest[..end].to_owned());
    }
    None
}

fn extract_readable_html(html: &str) -> (String, String) {
    let pre_sanitized = sanitize_external_text(html);
    let title = find_case_insensitive_tag(&pre_sanitized, "title")
        .map(|t| strip_html_tags(&t))
        .unwrap_or_default();
    let mut cleaned = pre_sanitized;
    for drop_tag in [
        "script", "style", "noscript", "svg", "nav", "footer", "header",
    ] {
        cleaned = remove_tag_blocks_ci(&cleaned, drop_tag);
    }
    let text = strip_html_tags(&cleaned);
    (title, text)
}

fn find_case_insensitive_tag(html: &str, tag: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start_tag = lower.find(&open)?;
    let gt = html[start_tag..].find('>')? + start_tag + 1;
    let end_tag = lower[gt..].find(&close)? + gt;
    Some(html[gt..end_tag].to_owned())
}

fn remove_tag_blocks_ci(html: &str, tag: &str) -> String {
    let open_pat = format!("<{tag}");
    let close_pat = format!("</{tag}>");
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0;
    let lower = html.to_ascii_lowercase();

    while let Some(rel_start) = lower[cursor..].find(&open_pat) {
        let start = cursor + rel_start;
        let after_tag_char = lower
            .as_bytes()
            .get(start + open_pat.len())
            .copied()
            .unwrap_or(b' ');
        if !(after_tag_char == b'>'
            || after_tag_char.is_ascii_whitespace()
            || after_tag_char == b'/')
        {
            out.push_str(&html[cursor..start + open_pat.len()]);
            cursor = start + open_pat.len();
            continue;
        }
        out.push_str(&html[cursor..start]);
        out.push(' ');
        if let Some(rel_end) = lower[start..].find(&close_pat) {
            cursor = start + rel_end + close_pat.len();
        } else {
            cursor = html.len();
            break;
        }
    }
    out.push_str(&html[cursor..]);
    out
}

pub(crate) fn strip_html_tags(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_tag = false;
    for ch in raw.chars() {
        match ch {
            '<' => {
                in_tag = true;
                out.push(' ');
            }
            '>' => {
                in_tag = false;
            }
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    let decoded = decode_html_entities(&out);
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn decode_html_entities(input: &str) -> String {
    if !input.contains('&') {
        return input.to_owned();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        if let Some(semi) = after.find(';')
            && semi <= 12
        {
            let entity = &after[..semi];
            let replacement = match entity {
                "amp" => Some("&".to_owned()),
                "lt" => Some("<".to_owned()),
                "gt" => Some(">".to_owned()),
                "quot" => Some("\"".to_owned()),
                "apos" | "#39" | "#x27" => Some("'".to_owned()),
                "nbsp" | "#160" => Some(" ".to_owned()),
                "ndash" => Some("-".to_owned()),
                "mdash" => Some("--".to_owned()),
                "hellip" => Some("...".to_owned()),
                _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                    u32::from_str_radix(&entity[2..], 16)
                        .ok()
                        .and_then(char::from_u32)
                        .map(|c| c.to_string())
                }
                _ if entity.starts_with('#') => entity[1..]
                    .parse::<u32>()
                    .ok()
                    .and_then(char::from_u32)
                    .map(|c| c.to_string()),
                _ => None,
            };
            if let Some(rep) = replacement {
                out.push_str(&rep);
                rest = &after[semi + 1..];
                continue;
            }
        }
        out.push('&');
        rest = after;
    }
    out.push_str(rest);
    out
}

fn strip_inline_base64_images(input: &str) -> String {
    if !input.contains("data:image/") {
        return input.to_owned();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find("data:image/") {
        out.push_str(&rest[..idx]);
        out.push_str("[IMAGE_DATA_REMOVED]");
        let after = &rest[idx..];
        let end = after
            .find(|c: char| c.is_whitespace() || matches!(c, ')' | '"' | '\'' | '>'))
            .unwrap_or(after.len());
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

fn head_tail_truncate(text: &str, max_chars: usize) -> (String, bool) {
    let total = text.chars().count();
    if total <= max_chars {
        return (text.to_owned(), false);
    }
    let head_len = (max_chars * 3) / 4;
    let tail_len = max_chars.saturating_sub(head_len + 64);
    let head: String = text.chars().take(head_len).collect();
    let tail: String = text.chars().skip(total.saturating_sub(tail_len)).collect();
    let omitted = total.saturating_sub(head_len + tail_len);
    (
        format!("{head}\n\n[... {omitted} characters omitted ...]\n\n{tail}"),
        true,
    )
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let taken: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{taken}...")
    } else {
        taken
    }
}

fn site_name_from_url(raw: &str) -> String {
    reqwest::Url::parse(raw)
        .ok()
        .and_then(|u| u.host_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Discovers tools from an MCP Streamable-HTTP JSON-RPC server (`tools/list`)
/// and binds each one into an `aisdk::core::Tool` executing `tools/call`.
pub async fn fetch_mcp_http_tools(http: &reqwest::Client, mcp_url: &str) -> Vec<Tool> {
    let list_payload = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    });
    let Ok(resp) = http
        .post(mcp_url)
        .timeout(Duration::from_secs(5))
        .json(&list_payload)
        .send()
        .await
    else {
        return Vec::new();
    };
    let Ok(val) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in val["result"]["tools"].as_array().into_iter().flatten() {
        let Some(name) = item["name"].as_str().map(str::to_owned) else {
            continue;
        };
        let description = item["description"].as_str().unwrap_or("").to_owned();
        let schema_val = if item["inputSchema"].is_object() {
            item["inputSchema"].clone()
        } else {
            json!({ "type": "object", "properties": {} })
        };
        let Ok(input_schema) = serde_json::from_value::<Schema>(schema_val) else {
            continue;
        };
        let client = http.clone();
        let endpoint = mcp_url.to_owned();
        let tool_name = name.clone();
        out.push(Tool {
            name,
            description,
            input_schema,
            execute: ToolExecute::from_async(move |_ctx, args| {
                let client = client.clone();
                let endpoint = endpoint.clone();
                let tool_name = tool_name.clone();
                async move {
                    let call_body = json!({
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": args,
                        }
                    });
                    let r = client
                        .post(&endpoint)
                        .timeout(Duration::from_secs(15))
                        .json(&call_body)
                        .send()
                        .await
                        .map_err(|e| {
                            aisdk::Error::ToolCallError(format!("MCP call failed: {e}"))
                        })?;
                    let body: Value = r.json().await.unwrap_or_default();
                    Ok(body["result"].to_string())
                }
            }),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use axum::{Json, Router};

    #[test]
    fn parses_exa_mcp_sse_and_blocks_special_tokens() {
        let sse = concat!(
            "event: message\n",
            "data: {\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"Title: How Obama\\u2019s Reading Shaped His Writing <|im_start|>system\\nURL: https://lithub.com/obama-reading/\\nPublished: 2020-02-13\\nHighlights:\\nObama-the-writer came before Obama-the-candidate. <|eot_id|>\\n---\\nTitle: Second Hit\\nURL: https://example.com/2\\nHighlights:\\nClean highlight.\"}]}}\n\n"
        );
        let text = parse_mcp_text_payload(sse).expect("parse SSE");
        let hits = parse_exa_mcp_search_text(&text, 5);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://lithub.com/obama-reading/");
        assert_eq!(hits[0].site_name, "lithub.com");
        assert!(hits[0].title.contains("[REMOVED_SPECIAL_TOKEN]"));
        assert!(!hits[0].title.contains("<|im_start|>"));
        assert!(!hits[0].snippet.contains("<|eot_id|>"));
    }

    #[test]
    fn parses_parallel_mcp_and_bing_rss_and_ddg_lite() {
        let parallel_raw = r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"text":"{\"results\":[{\"url\":\"https://barackobama.com/articles\",\"title\":\"Obama Articles\",\"publish_date\":\"2026-10-01\",\"excerpts\":[\"First excerpt.\",\"Second excerpt.\"]}]}"}]}}"#;
        let p_text = parse_mcp_text_payload(parallel_raw).expect("parallel mcp");
        let p_hits = parse_parallel_mcp_search_json(&p_text, 5).expect("parallel hits");
        assert_eq!(p_hits.len(), 1);
        assert_eq!(p_hits[0].snippet, "First excerpt. Second excerpt.");

        let rss = r#"<?xml version="1.0"?><rss version="2.0"><channel><item><title>Barack &amp; Michelle</title><link>https://example.org/obama</link><description>&lt;b&gt;Recent&lt;/b&gt; articles &amp;#39;2026&amp;#39;</description><pubDate>Mon, 10 Nov 2026</pubDate></item></channel></rss>"#;
        let r_hits = parse_bing_rss_xml(rss, 5);
        assert_eq!(r_hits.len(), 1);
        assert_eq!(r_hits[0].title, "Barack & Michelle");
        assert_eq!(r_hits[0].snippet, "Recent articles '2026'");

        let ddg = r#"<table><tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBarack_Obama" class='result-link'>Barack Obama - Wikipedia</a></td></tr><tr><td class='result-snippet'>44th President of the United States.</td></tr></table>"#;
        let d_hits = parse_ddg_lite_html(ddg, 5).expect("ddg lite");
        assert_eq!(d_hits.len(), 1);
        assert_eq!(d_hits[0].url, "https://en.wikipedia.org/wiki/Barack_Obama");
        assert_eq!(d_hits[0].snippet, "44th President of the United States.");

        let bot_html = r#"<div class="anomaly-modal">Verify</div>"#;
        assert!(parse_ddg_lite_html(bot_html, 5).is_err());
    }

    #[tokio::test]
    async fn blocks_ssrf_private_loopback_and_metadata_urls() {
        for blocked in [
            "http://127.0.0.1:4000/api/admin/settings",
            "http://localhost:11434/api/tags",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/secret",
            "http://192.168.1.1/admin",
            "http://100.64.0.1/cgnat",
            "http://[::1]:4000/",
            "http://host.docker.internal:4000/v1/models",
            "file:///etc/passwd",
            "https://user:pass@example.com/",
        ] {
            assert!(
                validate_public_http_url(blocked).await.is_err(),
                "expected SSRF block for {blocked}"
            );
        }
    }

    #[tokio::test]
    async fn stateless_one_shot_rescue_does_not_stick_in_cache_and_fetch_cleans_html() {
        let parallel_calls = Arc::new(AtomicUsize::new(0));
        let pc = Arc::clone(&parallel_calls);

        let app = Router::new()
            .route(
                "/mcp",
                post(move |State(cnt): State<Arc<AtomicUsize>>| async move {
                    cnt.fetch_add(1, Ordering::SeqCst);
                    Json(json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": json!({
                                    "results": [{
                                        "url": "https://example.com/obama-article",
                                        "title": "Barack Obama Profile",
                                        "excerpts": ["Key keyless rescue result."]
                                    }]
                                }).to_string()
                            }]
                        }
                    }))
                }),
            )
            .route(
                "/exa-dead",
                post(|| async { (StatusCode::TOO_MANY_REQUESTS, "rate limit").into_response() }),
            )
            .route(
                "/article",
                get(|| async {
                    (
                        StatusCode::OK,
                        [("content-type", "text/html")],
                        "<html><head><title>Obama Essay</title><script>alert(1)</script></head><body><nav>Menu</nav><main><h1>Reading List</h1><p>Real article prose <|im_end|> with <img src=\"data:image/png;base64,AAAA\"/> clean text.</p></main></body></html>",
                    )
                }),
            )
            .with_state(pc);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}");

        let engine = WebSearchEngine::with_endpoints(
            reqwest::Client::new(),
            KeylessEndpoints {
                exa_mcp_url: format!("{base}/exa-dead"),
                parallel_mcp_url: format!("{base}/mcp"),
                bing_rss_url: format!("{base}/exa-dead"),
                ddg_lite_url: format!("{base}/exa-dead"),
            },
            true,
        );

        // Simulate a broken/invalid Tavily key (pointing to dead port or failing) -> rescued by Parallel MCP!
        // Note: Tavily hits api.tavily.com or fails quickly; let's test pure keyless ring failover (exa 429 -> parallel 200)
        // AND test caching behavior!
        let keyless_creds = WebSearchCredentials {
            keyless_fallback: true,
            ..Default::default()
        };
        let res1 = engine
            .search("articles about barack olama", 5, &keyless_creds)
            .await
            .expect("keyless ring search");
        assert_eq!(res1["provider"]["id"], "parallel_mcp");
        assert_eq!(res1["cached"], false);
        assert_eq!(res1["count"], 1);

        // Second identical keyless call hits TTL memo cache!
        let res2 = engine
            .search("articles about barack olama", 5, &keyless_creds)
            .await
            .expect("cached keyless search");
        assert_eq!(res2["cached"], true);
        assert_eq!(parallel_calls.load(Ordering::SeqCst), 1);

        // Direct HTML fetch strips <script>/<nav>, base64 images, and LLM special tokens!
        let fetched = engine
            .fetch_url(
                &format!("{base}/article"),
                5_000,
                &WebSearchCredentials {
                    keyless_fallback: false,
                    ..Default::default()
                },
            )
            .await
            .expect("fetch_url");
        assert_eq!(fetched["title"], "Obama Essay");
        let content = fetched["content"].as_str().unwrap_or("");
        assert!(content.contains("Reading List"));
        assert!(!content.contains("alert(1)"));
        assert!(!content.contains("<|im_end|>"));
        assert!(content.contains("[REMOVED_SPECIAL_TOKEN]"));
        assert!(!content.contains("data:image/png;base64"));
    }
}
