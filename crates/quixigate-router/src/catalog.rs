//! Offline-first + background-synced `models.dev` catalog, pricing calculator,
//! and unified model discovery across cloud seats and local runtimes.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::sync::Arc;

use arc_swap::ArcSwap;
use flate2::read::GzDecoder;
use quixigate_core::{AuthKind, LocalRuntimeStatus, ModelRoute, NanoUsd, OpenRouterPreset};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const EMBEDDED_MODELS_DEV_GZ: &[u8] = include_bytes!("../../../assets/models.dev.json.gz");
pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    /// Canonical `author/model` identifier (e.g. `anthropic/claude-sonnet-4-5` or `@preset/code-reviewer`).
    pub canonical_id: String,
    /// Bare model identifier without author prefix (e.g. `claude-sonnet-4-5`).
    pub bare_id: String,
    pub author: String,
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Map of `provider_id -> upstream_model_name`.
    pub providers: BTreeMap<String, String>,
    pub context_window: u64,
    pub max_output_tokens: u64,
    pub input_nanos_per_m: NanoUsd,
    pub output_nanos_per_m: NanoUsd,
    pub cache_read_nanos_per_m: NanoUsd,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub supports_structured_output: bool,
    /// Per-provider capability overrides from live catalog sync (e.g. `"openrouter"` -> `supported_parameters`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_supports_tools: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_supports_reasoning: BTreeMap<String, bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
    pub is_local: bool,
    #[serde(default)]
    pub is_preset: bool,
}

#[derive(Clone, Default)]
struct CatalogSnapshot {
    by_canonical: BTreeMap<String, CatalogModel>,
    bare_to_canonical: HashMap<String, String>,
}

#[derive(Clone)]
pub struct ModelCatalog {
    snapshot: Arc<ArcSwap<CatalogSnapshot>>,
    presets: Arc<ArcSwap<Vec<OpenRouterPreset>>>,
}

impl ModelCatalog {
    /// Loads and indexes the embedded `assets/models.dev.json.gz` snapshot in ~5ms.
    pub fn embedded() -> Self {
        let mut decoder = GzDecoder::new(EMBEDDED_MODELS_DEV_GZ);
        let mut json_bytes = Vec::new();
        let mut snap = if decoder.read_to_end(&mut json_bytes).is_ok()
            && let Ok(val) = serde_json::from_slice::<Value>(&json_bytes)
        {
            parse_models_dev_json(&val)
        } else {
            CatalogSnapshot::default()
        };
        insert_openrouter_meta_models(&mut snap.by_canonical, true);
        Self {
            snapshot: Arc::new(ArcSwap::from_pointee(snap)),
            presets: Arc::new(ArcSwap::from_pointee(Vec::new())),
        }
    }

    /// Refreshes the catalog from `https://models.dev/api.json` (or custom URL) atomically.
    pub async fn sync_from_url(&self, http: &reqwest::Client, url: &str) -> anyhow::Result<usize> {
        let val: Value = http
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let mut snap = parse_models_dev_json(&val);
        insert_openrouter_meta_models(&mut snap.by_canonical, true);
        let count = snap.by_canonical.len();
        if count > 0 {
            self.snapshot.store(Arc::new(snap));
        }
        Ok(count)
    }

    pub fn list_presets(&self) -> Vec<OpenRouterPreset> {
        self.presets.load().as_ref().clone()
    }

    pub fn set_presets(&self, presets: Vec<OpenRouterPreset>) {
        self.presets.store(Arc::new(presets));
    }

    pub fn clear_presets(&self) {
        self.presets.store(Arc::new(Vec::new()));
    }

    /// Syncs live OpenRouter `/models/user` (falling back to `/models`) and `/presets` (`@preset/<slug>`)
    /// into the catalog, filtering out `:batch`-only, non-text-output, expired, and guardrail-restricted models
    /// and pruning stale OpenRouter mappings left over from static seed data.
    pub async fn sync_openrouter(
        &self,
        http: &reqwest::Client,
        api_base_url: &str,
        bearer_token: Option<&str>,
    ) -> anyhow::Result<(usize, usize)> {
        let base = api_base_url.trim_end_matches('/');
        let user_models_url = format!("{base}/models/user");
        let models_url = format!("{base}/models");
        let presets_url = format!("{base}/presets");

        let clean_bearer = bearer_token.map(str::trim).filter(|t| !t.is_empty());

        // Prefer `/models/user` when authenticated so OpenRouter's server-side privacy, ZDR,
        // free-model-training opt-out, and workspace guardrail policies filter the catalog automatically.
        let mut used_user_filtered = false;
        let models_val: Value = if let Some(tok) = clean_bearer
            && let Ok(resp) = http.get(&user_models_url).bearer_auth(tok).send().await
            && resp.status().is_success()
            && let Ok(val) = resp.json::<Value>().await
            && val["data"].is_array()
        {
            used_user_filtered = true;
            val
        } else {
            let mut models_req = http.get(&models_url);
            if let Some(tok) = clean_bearer {
                models_req = models_req.bearer_auth(tok);
            }
            let models_resp = models_req.send().await?.error_for_status()?;
            models_resp.json().await?
        };

        let mut next_snap = (**self.snapshot.load()).clone();
        if models_val["data"].is_array() {
            for model in next_snap.by_canonical.values_mut() {
                model.providers.remove("openrouter");
                model.provider_supports_tools.remove("openrouter");
                model.provider_supports_reasoning.remove("openrouter");
            }
            next_snap
                .by_canonical
                .retain(|_, m| !m.providers.is_empty() || m.is_local || m.is_preset);
            next_snap
                .bare_to_canonical
                .retain(|_, canon| next_snap.by_canonical.contains_key(canon));
        }

        let today_ymd = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let mut synced_models = 0usize;
        let mut any_free_model = false;

        for item in models_val["data"].as_array().into_iter().flatten() {
            if !is_openrouter_model_usable_for_chat(item, &today_ymd) {
                continue;
            }
            let Some(raw_id) = item["id"].as_str().map(str::trim).filter(|s| !s.is_empty()) else {
                continue;
            };
            synced_models += 1;
            if raw_id.ends_with(":free") {
                any_free_model = true;
            }
            let (author, bare_id) = author_for_provider("openrouter", raw_id);
            let canonical_id = if raw_id.starts_with('~') {
                raw_id.to_owned()
            } else {
                format!("{author}/{}", bare_id.to_ascii_lowercase())
            };
            let in_nanos = usd_per_token_to_nanos_per_m(&item["pricing"]["prompt"]);
            let out_nanos = usd_per_token_to_nanos_per_m(&item["pricing"]["completion"]);
            let cache_nanos = usd_per_token_to_nanos_per_m(&item["pricing"]["input_cache_read"]);
            let ctx = item["context_length"]
                .as_u64()
                .or_else(|| item["top_provider"]["context_length"].as_u64())
                .unwrap_or(128_000);
            let max_out = item["top_provider"]["max_completion_tokens"]
                .as_u64()
                .unwrap_or(16_384);
            let params = item["supported_parameters"].as_array();
            let has_param =
                |p: &str| params.is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some(p)));
            // When `supported_parameters` is explicitly present, honor it strictly; otherwise default true
            let supports_tools =
                params.is_none_or(|_| has_param("tools") || has_param("tool_choice"));
            let supports_reasoning = has_param("reasoning")
                || has_param("include_reasoning")
                || has_param("reasoning_effort");
            let supports_structured =
                has_param("structured_outputs") || has_param("response_format");
            let supports_vision = item["architecture"]["input_modalities"]
                .as_array()
                .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some("image")));
            let created_date = item["created"]
                .as_i64()
                .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
                .map(|dt| dt.format("%Y-%m-%d").to_string());

            let entry = next_snap
                .by_canonical
                .entry(canonical_id.clone())
                .or_insert_with(|| CatalogModel {
                    canonical_id: canonical_id.clone(),
                    bare_id: bare_id.clone(),
                    author: author.clone(),
                    display_name: item["name"].as_str().unwrap_or(raw_id).to_owned(),
                    description: item["description"].as_str().map(str::to_owned),
                    family: None,
                    providers: BTreeMap::new(),
                    context_window: ctx,
                    max_output_tokens: max_out,
                    input_nanos_per_m: in_nanos,
                    output_nanos_per_m: out_nanos,
                    cache_read_nanos_per_m: cache_nanos,
                    supports_tools,
                    supports_reasoning,
                    supports_vision,
                    supports_structured_output: supports_structured,
                    provider_supports_tools: BTreeMap::new(),
                    provider_supports_reasoning: BTreeMap::new(),
                    release_date: created_date.clone(),
                    is_local: false,
                    is_preset: false,
                });
            let had_other_providers = entry.providers.keys().any(|k| k != "openrouter");
            entry
                .providers
                .insert("openrouter".to_owned(), raw_id.to_owned());
            entry
                .provider_supports_tools
                .insert("openrouter".to_owned(), supports_tools);
            entry
                .provider_supports_reasoning
                .insert("openrouter".to_owned(), supports_reasoning);
            if !had_other_providers {
                entry.supports_tools = supports_tools;
                entry.supports_reasoning = supports_reasoning;
                entry.supports_vision = supports_vision;
                entry.supports_structured_output = supports_structured;
                entry.context_window = ctx;
                entry.max_output_tokens = max_out;
            }
            if entry.release_date.is_none() {
                entry.release_date = created_date;
            }
            if entry.input_nanos_per_m.is_zero() && in_nanos.nanos() > 0 {
                entry.input_nanos_per_m = in_nanos;
                entry.output_nanos_per_m = out_nanos;
                if cache_nanos.nanos() > 0 {
                    entry.cache_read_nanos_per_m = cache_nanos;
                }
            }
            next_snap
                .bare_to_canonical
                .entry(bare_id.to_ascii_lowercase())
                .or_insert_with(|| canonical_id.clone());
        }

        let include_free_router = !used_user_filtered || any_free_model;
        insert_openrouter_meta_models(&mut next_snap.by_canonical, include_free_router);
        self.snapshot.store(Arc::new(next_snap));

        let mut synced_presets = Vec::new();
        if let Some(tok) = bearer_token.filter(|t| !t.trim().is_empty())
            && let Ok(pres_resp) = http.get(&presets_url).bearer_auth(tok.trim()).send().await
            && pres_resp.status().is_success()
            && let Ok(pres_val) = pres_resp.json::<Value>().await
        {
            let arr = pres_val["data"]
                .as_array()
                .or_else(|| pres_val["presets"].as_array());
            for p in arr.into_iter().flatten() {
                let Some(raw_slug) = p["slug"].as_str().filter(|s| !s.trim().is_empty()) else {
                    continue;
                };
                if p["status"]
                    .as_str()
                    .is_some_and(|st| st.eq_ignore_ascii_case("archived"))
                {
                    continue;
                }
                let slug = raw_slug.trim().trim_start_matches("@preset/").to_owned();
                if slug.is_empty() {
                    continue;
                }
                let id = p["id"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("preset_{slug}"));
                let name = p["name"]
                    .as_str()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or(&slug)
                    .to_owned();
                let description = p["description"]
                    .as_str()
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_owned);
                let underlying_model = p["designated_version"]["config"]["model"]
                    .as_str()
                    .or_else(|| p["config"]["model"].as_str())
                    .or_else(|| p["model"].as_str())
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_owned);
                let updated_at = p["updated_at"].as_str().map(str::to_owned);
                synced_presets.push(OpenRouterPreset {
                    id,
                    slug,
                    name,
                    description,
                    underlying_model,
                    updated_at,
                });
            }
            synced_presets.sort_by(|a, b| a.slug.cmp(&b.slug));
            self.presets.store(Arc::new(synced_presets.clone()));
        }

        let preset_count = self.presets.load().len();
        Ok((synced_models, preset_count))
    }

    /// Returns the total number of models currently routable through OpenRouter in the catalog.
    pub fn openrouter_model_count(&self) -> usize {
        self.snapshot
            .load()
            .by_canonical
            .values()
            .filter(|m| m.providers.contains_key("openrouter"))
            .count()
    }

    /// Looks up a catalog model by canonical `author/model`, `@preset/<slug>`, `~author/*-latest`,
    /// `provider/model`, or bare `model` name.
    pub fn lookup(&self, model: &str) -> Option<CatalogModel> {
        let snap = self.snapshot.load();
        let trimmed = model.trim();
        if let Some(slug) = trimmed.strip_prefix("@preset/") {
            let presets = self.presets.load();
            if let Some(p) = presets.iter().find(|p| p.slug.eq_ignore_ascii_case(slug)) {
                return Some(self.preset_to_catalog_model(p, &snap));
            }
        }
        if let Some(m) = snap.by_canonical.get(trimmed) {
            return Some(m.clone());
        }
        let lower = trimmed.to_ascii_lowercase();
        if let Some(m) = snap.by_canonical.get(&lower) {
            return Some(m.clone());
        }
        if let Some(canon) = snap.bare_to_canonical.get(&lower)
            && let Some(m) = snap.by_canonical.get(canon)
        {
            return Some(m.clone());
        }
        // Strip leading `~` and/or `-latest` suffix for dynamic OpenRouter aliases
        let untilded = lower.trim_start_matches('~');
        if let Some(m) = snap.by_canonical.get(untilded) {
            return Some(m.clone());
        }
        if let Some(prefix) = untilded.strip_suffix("-latest")
            && let Some(m) = snap
                .by_canonical
                .values()
                .filter(|m| m.canonical_id.starts_with(prefix))
                .max_by(|a, b| a.release_date.cmp(&b.release_date))
        {
            return Some(m.clone());
        }
        // Strip date suffix fallback (e.g. claude-sonnet-4-5-20250929 -> claude-sonnet-4-5)
        let bare = untilded.split('/').next_back().unwrap_or(untilded);
        if let Some(canon) = snap.bare_to_canonical.get(bare)
            && let Some(m) = snap.by_canonical.get(canon)
        {
            return Some(m.clone());
        }
        None
    }

    fn preset_to_catalog_model(
        &self,
        preset: &OpenRouterPreset,
        snap: &CatalogSnapshot,
    ) -> CatalogModel {
        let canonical_id = format!("@preset/{}", preset.slug);
        let mut providers = BTreeMap::new();
        providers.insert("openrouter".to_owned(), canonical_id.clone());

        let base = preset.underlying_model.as_deref().and_then(|um| {
            let low = um.trim().to_ascii_lowercase();
            snap.by_canonical.get(&low).or_else(|| {
                snap.bare_to_canonical
                    .get(&low)
                    .and_then(|c| snap.by_canonical.get(c))
            })
        });

        let desc = match (&preset.description, &preset.underlying_model) {
            (Some(d), Some(m)) => Some(format!("{d} (OpenRouter preset -> {m})")),
            (Some(d), None) => Some(d.clone()),
            (None, Some(m)) => Some(format!("OpenRouter preset targeting {m}")),
            (None, None) => Some("OpenRouter server-side model & prompt preset".to_owned()),
        };
        let display_name = if preset.name.trim().is_empty() || preset.name == preset.slug {
            canonical_id.clone()
        } else {
            format!("{} ({})", preset.name, canonical_id)
        };

        CatalogModel {
            canonical_id,
            bare_id: preset.slug.clone(),
            author: "preset".to_owned(),
            display_name,
            description: desc,
            family: Some("openrouter-preset".to_owned()),
            providers,
            context_window: base.map_or(200_000, |b| b.context_window),
            max_output_tokens: base.map_or(16_384, |b| b.max_output_tokens),
            input_nanos_per_m: base.map_or(NanoUsd::ZERO, |b| b.input_nanos_per_m),
            output_nanos_per_m: base.map_or(NanoUsd::ZERO, |b| b.output_nanos_per_m),
            cache_read_nanos_per_m: base.map_or(NanoUsd::ZERO, |b| b.cache_read_nanos_per_m),
            supports_tools: base.is_none_or(|b| b.supports_tools),
            supports_reasoning: base.is_some_and(|b| b.supports_reasoning),
            supports_vision: base.is_some_and(|b| b.supports_vision),
            supports_structured_output: base.is_none_or(|b| b.supports_structured_output),
            provider_supports_tools: base
                .map(|b| b.provider_supports_tools.clone())
                .unwrap_or_default(),
            provider_supports_reasoning: base
                .map(|b| b.provider_supports_reasoning.clone())
                .unwrap_or_default(),
            release_date: preset.updated_at.clone(),
            is_local: false,
            is_preset: true,
        }
    }

    /// Lists all catalog models merged with currently discovered local runtime models.
    pub fn list_models(&self, local_runtimes: &[LocalRuntimeStatus]) -> Vec<CatalogModel> {
        self.list_models_with_context(local_runtimes, false)
    }

    /// Lists catalog models merged with local runtimes and ( strictly when `openrouter_connected` is true )
    /// synced OpenRouter `@preset/<slug>` entries.
    pub fn list_models_with_context(
        &self,
        local_runtimes: &[LocalRuntimeStatus],
        openrouter_connected: bool,
    ) -> Vec<CatalogModel> {
        let snap = self.snapshot.load();
        let mut out: Vec<CatalogModel> = snap.by_canonical.values().cloned().collect();

        if openrouter_connected {
            let presets = self.presets.load();
            for p in presets.iter() {
                out.push(self.preset_to_catalog_model(p, &snap));
            }
        }

        for rt in local_runtimes.iter().filter(|r| r.healthy) {
            for model_name in &rt.models {
                let canonical_id = format!("{}/{}", rt.runtime_id, model_name);
                let mut providers = BTreeMap::new();
                providers.insert(rt.runtime_id.clone(), model_name.clone());
                let detail = rt.model_details.get(model_name);
                let context_window = detail
                    .and_then(|d| d.context_window)
                    .filter(|&c| c > 0)
                    .unwrap_or(131_072);
                let supports_vision = detail.is_some_and(|d| d.supports_vision);
                let mode_note = rt
                    .mode
                    .as_deref()
                    .map(|m| format!(" [{m} mode]"))
                    .unwrap_or_default();
                out.push(CatalogModel {
                    canonical_id,
                    bare_id: model_name.clone(),
                    author: rt.runtime_id.clone(),
                    display_name: format!("{} ({})", model_name, rt.display_name),
                    description: Some(format!(
                        "Local model served by {}{} at {}",
                        rt.display_name, mode_note, rt.base_url
                    )),
                    family: None,
                    providers,
                    context_window,
                    max_output_tokens: (context_window / 2).clamp(4_096, 32_768),
                    input_nanos_per_m: NanoUsd::ZERO,
                    output_nanos_per_m: NanoUsd::ZERO,
                    cache_read_nanos_per_m: NanoUsd::ZERO,
                    supports_tools: true,
                    supports_reasoning: true,
                    supports_vision,
                    supports_structured_output: true,
                    provider_supports_tools: BTreeMap::new(),
                    provider_supports_reasoning: BTreeMap::new(),
                    release_date: None,
                    is_local: true,
                    is_preset: false,
                });
            }
        }

        out.sort_by(|a, b| {
            b.is_local
                .cmp(&a.is_local)
                .then_with(|| b.is_preset.cmp(&a.is_preset))
                .then_with(|| b.release_date.cmp(&a.release_date))
                .then_with(|| a.canonical_id.cmp(&b.canonical_id))
        });
        out
    }

    /// Computes `(actual_cost_nanos, api_equiv_nanos)` using exact integer `NanoUsd` arithmetic.
    ///
    /// - Subscription OAuth seats (`openai-codex`, `anthropic-token`, `google-gemini-cli`,
    ///   `github-copilot`, `xai-oauth`, `qwen-portal`, `minimax-portal`, `chutes`), OpenRouter `:free`
    ///   models, and local runtimes (`ollama`, `lmstudio`, `llamacpp`, `vllm`, `mlx`) bill `$0` actual
    ///   spend while recording the equivalent public API cost in `api_equiv_nanos` for the Savings counter!
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cost(
        &self,
        provider_id: &str,
        auth_kind: AuthKind,
        model: &str,
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
        route_override: Option<&ModelRoute>,
    ) -> (NanoUsd, NanoUsd) {
        let stripped = model.strip_suffix(":free").unwrap_or(model);
        let cat = self.lookup(stripped);

        let default_local_in = NanoUsd(500_000_000); // $0.50 / 1M reference for local savings
        let default_local_out = NanoUsd(1_500_000_000); // $1.50 / 1M reference for local savings

        let is_free_openrouter_model =
            provider_id == "openrouter" && (model == "openrouter/free" || model.ends_with(":free"));

        let is_zero_marginal = is_free_openrouter_model
            || (auth_kind.is_zero_marginal_cost() && provider_id != "openrouter")
            || matches!(
                provider_id,
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

        let in_rate = route_override
            .and_then(|r| r.input_nanos_per_m.map(NanoUsd))
            .or_else(|| {
                cat.as_ref()
                    .map(|c| c.input_nanos_per_m)
                    .filter(|n| !n.is_zero())
            })
            .unwrap_or(if is_zero_marginal {
                default_local_in
            } else {
                NanoUsd(1_000_000_000)
            });

        let out_rate = route_override
            .and_then(|r| r.output_nanos_per_m.map(NanoUsd))
            .or_else(|| {
                cat.as_ref()
                    .map(|c| c.output_nanos_per_m)
                    .filter(|n| !n.is_zero())
            })
            .unwrap_or(if is_zero_marginal {
                default_local_out
            } else {
                NanoUsd(3_000_000_000)
            });

        let cache_rate = route_override
            .and_then(|r| r.cache_read_nanos_per_m.map(NanoUsd))
            .or_else(|| {
                cat.as_ref().map(|c| {
                    if c.cache_read_nanos_per_m.nanos() > 0 {
                        c.cache_read_nanos_per_m
                    } else {
                        NanoUsd(in_rate.nanos() / 10)
                    }
                })
            })
            .unwrap_or(NanoUsd(in_rate.nanos() / 10));

        let uncached_input = input_tokens.saturating_sub(cached_input_tokens);
        let in_cost = NanoUsd::from_tokens_and_rate(uncached_input, in_rate.nanos());
        let cache_cost = NanoUsd::from_tokens_and_rate(cached_input_tokens, cache_rate.nanos());
        let out_cost = NanoUsd::from_tokens_and_rate(output_tokens, out_rate.nanos());
        let equiv = in_cost.saturating_add(cache_cost).saturating_add(out_cost);

        if is_zero_marginal {
            (NanoUsd::ZERO, equiv)
        } else {
            (equiv, equiv)
        }
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::embedded()
    }
}

fn usd_per_token_to_nanos_per_m(v: &Value) -> NanoUsd {
    let parsed_f64 = v
        .as_str()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .or_else(|| v.as_f64());
    if let Some(per_tok) = parsed_f64
        && per_tok.is_finite()
        && per_tok > 0.0
    {
        let per_million = per_tok * 1_000_000.0;
        let formatted = format!("{per_million:.9}");
        return NanoUsd::parse_usd(&formatted).unwrap_or(NanoUsd::ZERO);
    }
    NanoUsd::ZERO
}

/// Filters raw OpenRouter `/api/v1/models(/user)` entries using OpenRouter's metadata so only
/// live, interactive text/chat-capable models (never `:batch`-only, image-only, or expired slugs)
/// enter the active catalog.
fn is_openrouter_model_usable_for_chat(item: &Value, today_ymd: &str) -> bool {
    let Some(raw_id) = item["id"].as_str().map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };
    if raw_id.ends_with(":batch") {
        return false;
    }

    // When `endpoints` is present (`["/api/v1/batches"]` vs `["/api/v1/chat/completions", ...]`),
    // require at least one synchronous chat/responses/messages endpoint.
    if let Some(endpoints) = item["endpoints"].as_array()
        && !endpoints.is_empty()
    {
        let has_interactive_endpoint = endpoints.iter().filter_map(Value::as_str).any(|ep| {
            ep.ends_with("/chat/completions")
                || ep.ends_with("/responses")
                || ep.ends_with("/messages")
        });
        if !has_interactive_endpoint {
            return false;
        }
    }

    // Require `"text"` output modality when `architecture.output_modalities` is specified.
    if let Some(out_mods) = item["architecture"]["output_modalities"].as_array()
        && !out_mods.is_empty()
        && !out_mods.iter().any(|m| m.as_str() == Some("text"))
    {
        return false;
    }
    if item["architecture"]["output_modalities"].is_null()
        && let Some(modality) = item["architecture"]["modality"].as_str()
        && let Some((_, rhs)) = modality.split_once("->")
        && !rhs.to_ascii_lowercase().contains("text")
    {
        return false;
    }

    // Skip sunset/expired endpoints (`"expiration_date": "YYYY-MM-DD"`).
    if let Some(exp) = item["expiration_date"].as_str().map(str::trim)
        && exp.len() >= 10
        && &exp[..10] < today_ymd
    {
        return false;
    }

    true
}

fn insert_openrouter_meta_models(
    by_canonical: &mut BTreeMap<String, CatalogModel>,
    include_free_router: bool,
) {
    let builtins = [
        (
            "openrouter/auto",
            "auto",
            "openrouter",
            "OpenRouter Auto Router",
            "Dynamically selects the optimal frontier model for your prompt via OpenRouter NotDiamond routing.",
            NanoUsd(1_000_000_000),
            NanoUsd(3_000_000_000),
            true,
        ),
        (
            "openrouter/free",
            "free",
            "openrouter",
            "OpenRouter Free Router",
            "Automatically routes across $0 free-tier models on OpenRouter with tool/vision matching.",
            NanoUsd::ZERO,
            NanoUsd::ZERO,
            true,
        ),
        (
            "~anthropic/claude-opus-latest",
            "claude-opus-latest",
            "anthropic",
            "Claude Opus (Latest via OpenRouter)",
            "OpenRouter dynamic latest alias resolving automatically to the newest Claude Opus release.",
            NanoUsd(5_000_000_000),
            NanoUsd(25_000_000_000),
            true,
        ),
        (
            "~anthropic/claude-sonnet-latest",
            "claude-sonnet-latest",
            "anthropic",
            "Claude Sonnet (Latest via OpenRouter)",
            "OpenRouter dynamic latest alias resolving automatically to the newest Claude Sonnet release.",
            NanoUsd(3_000_000_000),
            NanoUsd(15_000_000_000),
            true,
        ),
        (
            "~openai/gpt-latest",
            "gpt-latest",
            "openai",
            "OpenAI GPT (Latest via OpenRouter)",
            "OpenRouter dynamic latest alias resolving automatically to the newest flagship GPT release.",
            NanoUsd(2_500_000_000),
            NanoUsd(10_000_000_000),
            true,
        ),
        (
            "~google/gemini-pro-latest",
            "gemini-pro-latest",
            "google",
            "Gemini Pro (Latest via OpenRouter)",
            "OpenRouter dynamic latest alias resolving automatically to the newest Gemini Pro release.",
            NanoUsd(1_250_000_000),
            NanoUsd(10_000_000_000),
            true,
        ),
    ];

    for (canon, bare, author, name, desc, in_rate, out_rate, reasoning) in builtins {
        if canon == "openrouter/free" && !include_free_router {
            continue;
        }
        let entry = by_canonical
            .entry(canon.to_owned())
            .or_insert_with(|| CatalogModel {
                canonical_id: canon.to_owned(),
                bare_id: bare.to_owned(),
                author: author.to_owned(),
                display_name: name.to_owned(),
                description: Some(desc.to_owned()),
                family: Some("openrouter-dynamic".to_owned()),
                providers: BTreeMap::new(),
                context_window: 200_000,
                max_output_tokens: 32_768,
                input_nanos_per_m: in_rate,
                output_nanos_per_m: out_rate,
                cache_read_nanos_per_m: NanoUsd(in_rate.nanos() / 10),
                supports_tools: true,
                supports_reasoning: reasoning,
                supports_vision: true,
                supports_structured_output: true,
                provider_supports_tools: BTreeMap::new(),
                provider_supports_reasoning: BTreeMap::new(),
                release_date: None,
                is_local: false,
                is_preset: false,
            });
        entry
            .providers
            .insert("openrouter".to_owned(), canon.to_owned());
    }
}

fn usd_per_m_to_nanos(v: &Value) -> NanoUsd {
    if let Some(s) = v.as_str() {
        return NanoUsd::parse_usd(s).unwrap_or(NanoUsd::ZERO);
    }
    if let Some(n) = v.as_u64() {
        return NanoUsd((n as i64).saturating_mul(1_000_000_000));
    }
    if let Some(f) = v.as_f64()
        && f.is_finite()
        && f >= 0.0
    {
        // Format decimal representation from JSON float cleanly to integer nanos
        let formatted = format!("{f:.9}");
        return NanoUsd::parse_usd(&formatted).unwrap_or(NanoUsd::ZERO);
    }
    NanoUsd::ZERO
}

fn author_for_provider(provider_id: &str, model_id: &str) -> (String, String) {
    if let Some((author, rest)) = model_id.split_once('/') {
        return (author.to_ascii_lowercase(), rest.to_owned());
    }
    let author = match provider_id {
        "openai" => "openai",
        "anthropic" => "anthropic",
        "google" => "google",
        "xai" => "xai",
        "deepseek" => "deepseek",
        "mistral" => "mistral",
        "alibaba" | "alibaba-cn" => "qwen",
        "moonshotai" | "moonshotai-cn" => "moonshotai",
        "zhipuai" | "zhipuai-coding-plan" => "z-ai",
        other => other,
    };
    (author.to_owned(), model_id.to_owned())
}

fn parse_models_dev_json(root: &Value) -> CatalogSnapshot {
    let mut by_canonical: BTreeMap<String, CatalogModel> = BTreeMap::new();
    let mut bare_priority: HashMap<String, (i32, String)> = HashMap::new();

    let Some(providers_obj) = root.as_object() else {
        return CatalogSnapshot::default();
    };

    // Prioritize direct first-party provider entries when merging metadata & pricing.
    let provider_rank = |pid: &str| -> i32 {
        match pid {
            "openai" | "anthropic" | "google" | "xai" | "deepseek" | "mistral" => 100,
            "openrouter" => 90,
            "groq" | "togetherai" | "fireworks-ai" | "alibaba" => 80,
            _ => 50,
        }
    };

    for (provider_id, prov_val) in providers_obj {
        let Some(models_obj) = prov_val["models"].as_object() else {
            continue;
        };
        let rank = provider_rank(provider_id);

        for (raw_model_id, m) in models_obj {
            let (author, bare_id) = if let Some(canon) = m["canonical_model_id"].as_str()
                && let Some((a, b)) = canon.split_once('/')
            {
                (a.to_ascii_lowercase(), b.to_owned())
            } else {
                author_for_provider(provider_id, raw_model_id)
            };
            let canonical_id = format!("{author}/{}", bare_id.to_ascii_lowercase());
            let in_nanos = usd_per_m_to_nanos(&m["cost"]["input"]);
            let out_nanos = usd_per_m_to_nanos(&m["cost"]["output"]);
            let cache_nanos = usd_per_m_to_nanos(&m["cost"]["cache_read"]);
            let ctx = m["limit"]["context"].as_u64().unwrap_or(128_000);
            let max_out = m["limit"]["output"].as_u64().unwrap_or(8_192);
            let supports_tools = m["tool_call"].as_bool().unwrap_or(false);
            let supports_reasoning = m["reasoning"].as_bool().unwrap_or(false);
            let supports_structured = m["structured_output"].as_bool().unwrap_or(false);
            let supports_vision = m["modalities"]["input"]
                .as_array()
                .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some("image")));
            let release_date = m["release_date"].as_str().map(str::to_owned);

            let entry = by_canonical
                .entry(canonical_id.clone())
                .or_insert_with(|| CatalogModel {
                    canonical_id: canonical_id.clone(),
                    bare_id: bare_id.clone(),
                    author: author.clone(),
                    display_name: m["name"].as_str().unwrap_or(&bare_id).to_owned(),
                    description: m["description"].as_str().map(str::to_owned),
                    family: m["family"].as_str().map(str::to_owned),
                    providers: BTreeMap::new(),
                    context_window: ctx,
                    max_output_tokens: max_out,
                    input_nanos_per_m: in_nanos,
                    output_nanos_per_m: out_nanos,
                    cache_read_nanos_per_m: cache_nanos,
                    supports_tools,
                    supports_reasoning,
                    supports_vision,
                    supports_structured_output: supports_structured,
                    provider_supports_tools: BTreeMap::new(),
                    provider_supports_reasoning: BTreeMap::new(),
                    release_date: release_date.clone(),
                    is_local: false,
                    is_preset: false,
                });

            entry
                .providers
                .insert(provider_id.clone(), raw_model_id.clone());
            entry
                .provider_supports_tools
                .insert(provider_id.clone(), supports_tools);
            entry
                .provider_supports_reasoning
                .insert(provider_id.clone(), supports_reasoning);

            // Also register OAuth sibling provider mappings for first-party authors!
            match provider_id.as_str() {
                "openai" => {
                    entry
                        .providers
                        .entry("openai-codex".to_owned())
                        .or_insert_with(|| raw_model_id.clone());
                }
                "anthropic" => {
                    entry
                        .providers
                        .entry("anthropic-token".to_owned())
                        .or_insert_with(|| raw_model_id.clone());
                }
                "google" => {
                    entry
                        .providers
                        .entry("google-gemini-cli".to_owned())
                        .or_insert_with(|| raw_model_id.clone());
                }
                "xai" => {
                    entry
                        .providers
                        .entry("xai-oauth".to_owned())
                        .or_insert_with(|| raw_model_id.clone());
                }
                "alibaba" => {
                    entry
                        .providers
                        .entry("qwen-portal".to_owned())
                        .or_insert_with(|| raw_model_id.clone());
                }
                _ => {}
            }

            if rank >= 90 && in_nanos.nanos() > 0 {
                entry.input_nanos_per_m = in_nanos;
                entry.output_nanos_per_m = out_nanos;
                if cache_nanos.nanos() > 0 {
                    entry.cache_read_nanos_per_m = cache_nanos;
                }
                entry.context_window = ctx;
                entry.max_output_tokens = max_out;
            }

            let bare_lower = bare_id.to_ascii_lowercase();
            let update_bare = match bare_priority.get(&bare_lower) {
                Some((prev_rank, _)) => rank > *prev_rank,
                None => true,
            };
            if update_bare {
                bare_priority.insert(bare_lower, (rank, canonical_id));
            }
        }
    }

    let bare_to_canonical = bare_priority
        .into_iter()
        .map(|(k, (_, v))| (k, v))
        .collect();

    CatalogSnapshot {
        by_canonical,
        bare_to_canonical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::http::HeaderMap;
    use axum::routing::get;
    use serde_json::json;

    #[tokio::test]
    async fn sync_openrouter_prefers_user_models_filters_batch_and_prunes_stale_entries() {
        let app = Router::new()
            .route(
                "/api/v1/models",
                get(|| async {
                    axum::Json(json!({
                        "data": [
                            { "id": "liquid/lfm-3-1.7b:free", "name": "Blocked Free Model" }
                        ]
                    }))
                }),
            )
            .route(
                "/api/v1/models/user",
                get(|headers: HeaderMap| async move {
                    assert_eq!(
                        headers.get("authorization").and_then(|v| v.to_str().ok()),
                        Some("Bearer sk-or-user-test")
                    );
                    axum::Json(json!({
                        "data": [
                            {
                                "id": "google/gemini-4-argon",
                                "name": "Gemini 4 Argon",
                                "endpoints": ["/api/v1/chat/completions", "/api/v1/responses"],
                                "architecture": {
                                    "input_modalities": ["text", "image"],
                                    "output_modalities": ["text"]
                                },
                                "supported_parameters": ["tools", "reasoning"]
                            },
                            {
                                "id": "moondream/moondream3-preview",
                                "name": "Moondream 3",
                                "endpoints": ["/api/v1/chat/completions"],
                                "architecture": {
                                    "input_modalities": ["text", "image"],
                                    "output_modalities": ["text"]
                                },
                                "supported_parameters": ["temperature", "max_tokens"]
                            },
                            {
                                "id": "google/gemini-4-argon:batch",
                                "name": "Gemini 4 Argon Batch",
                                "endpoints": ["/api/v1/batches"],
                                "architecture": { "output_modalities": ["text"] }
                            },
                            {
                                "id": "black-forest-labs/flux-pro",
                                "name": "Flux Pro",
                                "endpoints": ["/api/v1/chat/completions"],
                                "architecture": { "output_modalities": ["image"] }
                            },
                            {
                                "id": "legacy/retired-model",
                                "name": "Retired",
                                "endpoints": ["/api/v1/chat/completions"],
                                "architecture": { "output_modalities": ["text"] },
                                "expiration_date": "2020-01-01"
                            }
                        ]
                    }))
                }),
            );

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let catalog = ModelCatalog::embedded();
        assert!(catalog.openrouter_model_count() > 50);

        let http = reqwest::Client::new();
        let (synced, _) = catalog
            .sync_openrouter(
                &http,
                &format!("http://{addr}/api/v1"),
                Some("sk-or-user-test"),
            )
            .await
            .unwrap();
        // Only the 2 interactive text chat models (`gemini-4-argon` and `moondream3-preview`) pass filtering!
        assert_eq!(synced, 2);

        // Guardrail-excluded `:free` model has `"openrouter"` stripped (even if another `models.dev` vendor
        // like `kilo` lists a slug with the same name), and `:batch`, image-only, and expired models are excluded:
        if let Some(lfm_free) = catalog.lookup("liquid/lfm-3-1.7b:free") {
            assert!(!lfm_free.providers.contains_key("openrouter"));
        }
        assert!(
            catalog
                .lookup("google/gemini-4-argon:batch")
                .is_none_or(|m| !m.providers.contains_key("openrouter"))
        );
        assert!(
            catalog
                .lookup("black-forest-labs/flux-pro")
                .is_none_or(|m| !m.providers.contains_key("openrouter"))
        );
        assert!(catalog.lookup("legacy/retired-model").is_none());
        // Because `/models/user` returned zero `:free` models (privacy settings block free training),
        // `openrouter/free` does NOT carry an `"openrouter"` provider mapping!
        assert!(
            catalog
                .lookup("openrouter/free")
                .is_none_or(|m| !m.providers.contains_key("openrouter"))
        );
        // `venice/*` still has its `venice` mapping, never `"openrouter"`:
        if let Some(venice) = catalog.lookup("venice/deepseek-v4-1-flash-uncensored") {
            assert!(!venice.providers.contains_key("openrouter"));
        }

        let argon = catalog.lookup("google/gemini-4-argon").unwrap();
        assert_eq!(
            argon.providers.get("openrouter").map(String::as_str),
            Some("google/gemini-4-argon")
        );
        assert_eq!(argon.provider_supports_tools.get("openrouter"), Some(&true));

        let moondream = catalog.lookup("moondream/moondream3-preview").unwrap();
        assert_eq!(
            moondream.provider_supports_tools.get("openrouter"),
            Some(&false)
        );
        assert!(!moondream.supports_tools);
    }
}
