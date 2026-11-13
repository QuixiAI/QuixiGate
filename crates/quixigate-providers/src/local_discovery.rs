//! Non-intrusive local runtime auto-discovery, `llama.cpp` (`llama-server`)
//! single-instance & multi-model router introspection, and rolling performance
//! tracker for Ollama, LM Studio, llama.cpp, Apple MLX, and vLLM.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use quixigate_core::{
    LocalEndpointConfig, LocalModelInfo, LocalRuntimeStatus, QuixigateError, Result,
};
use serde_json::{Value, json};

/// Normalizes common local runtime aliases (`"llama.cpp"`, `"llama-cpp"`, `"lm-studio"`)
/// into canonical QuixiGate runtime IDs.
pub fn normalize_local_runtime_id(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "llama.cpp" | "llama-cpp" | "llama_cpp" | "llama-server" | "llamacpp" => {
            "llamacpp".to_owned()
        }
        "lm-studio" | "lm_studio" | "lmstudio" => "lmstudio".to_owned(),
        other => other.to_owned(),
    }
}

fn ensure_v1_base(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") || trimmed.contains("/v1/") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/v1")
    }
}

#[derive(Debug, Clone)]
pub struct LocalEndpointSpec {
    pub runtime_id: String,
    pub display_name: String,
    pub base_url: String,
}

impl LocalEndpointSpec {
    pub fn defaults() -> Vec<Self> {
        let ollama_url = quixigate_core::brand::env_opt("QUIXIGATE_OLLAMA_URL")
            .or_else(|| {
                std::env::var("OLLAMA_HOST")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .map(|h| {
                        if h.starts_with("http://") || h.starts_with("https://") {
                            ensure_v1_base(&h)
                        } else {
                            ensure_v1_base(&format!("http://{h}"))
                        }
                    })
            })
            .unwrap_or_else(|| "http://127.0.0.1:11434/v1".to_owned());

        let lmstudio_url = quixigate_core::brand::env_opt("QUIXIGATE_LMSTUDIO_URL")
            .map(|u| ensure_v1_base(&u))
            .unwrap_or_else(|| "http://127.0.0.1:1234/v1".to_owned());

        let llamacpp_url = quixigate_core::brand::env_opt("QUIXIGATE_LLAMACPP_URL")
            .or_else(|| {
                std::env::var("LLAMA_SERVER_URL")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })
            .map(|u| ensure_v1_base(&u))
            .unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_owned());

        let mlx_url = quixigate_core::brand::env_opt("QUIXIGATE_MLX_URL")
            .map(|u| ensure_v1_base(&u))
            .unwrap_or_else(|| "http://127.0.0.1:8085/v1".to_owned());

        let vllm_url = quixigate_core::brand::env_opt("QUIXIGATE_VLLM_URL")
            .map(|u| ensure_v1_base(&u))
            .unwrap_or_else(|| "http://127.0.0.1:8000/v1".to_owned());

        vec![
            Self {
                runtime_id: "ollama".to_owned(),
                display_name: "Ollama".to_owned(),
                base_url: ollama_url,
            },
            Self {
                runtime_id: "lmstudio".to_owned(),
                display_name: "LM Studio".to_owned(),
                base_url: lmstudio_url,
            },
            Self {
                runtime_id: "llamacpp".to_owned(),
                display_name: "llama.cpp Server".to_owned(),
                base_url: llamacpp_url,
            },
            Self {
                runtime_id: "mlx".to_owned(),
                display_name: "Apple MLX".to_owned(),
                base_url: mlx_url,
            },
            Self {
                runtime_id: "vllm".to_owned(),
                display_name: "vLLM".to_owned(),
                base_url: vllm_url,
            },
        ]
    }
}

#[derive(Debug, Default, Clone)]
struct RollingPerf {
    samples: u64,
    avg_ttft_ms: f64,
    avg_tok_per_sec: f64,
}

#[derive(Clone)]
pub struct LocalRuntimeScanner {
    http: reqwest::Client,
    control_http: reqwest::Client,
    endpoints: Arc<RwLock<Vec<LocalEndpointSpec>>>,
    api_keys: Arc<DashMap<String, String>>,
    statuses: Arc<DashMap<String, LocalRuntimeStatus>>,
    perf: Arc<DashMap<String, RollingPerf>>,
    generation: Arc<std::sync::atomic::AtomicU64>,
}

impl LocalRuntimeScanner {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(850))
            .build()
            .unwrap_or_default();
        let control_http = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap_or_default();
        let endpoints = LocalEndpointSpec::defaults();
        let api_keys = DashMap::new();
        if let Ok(k) =
            std::env::var("QUIXIGATE_LLAMACPP_API_KEY").or_else(|_| std::env::var("LLAMA_API_KEY"))
            && !k.trim().is_empty()
        {
            api_keys.insert("llamacpp".to_owned(), k.trim().to_owned());
        }
        let statuses = DashMap::new();
        let now_ms = Utc::now().timestamp_millis();
        for ep in &endpoints {
            statuses.insert(
                ep.runtime_id.clone(),
                empty_status(ep, false, None, None, now_ms),
            );
        }
        Self {
            http,
            control_http,
            endpoints: Arc::new(RwLock::new(endpoints)),
            api_keys: Arc::new(api_keys),
            statuses: Arc::new(statuses),
            perf: Arc::new(DashMap::new()),
            generation: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::Acquire)
    }

    #[inline]
    fn bump_generation(&self) {
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    pub fn add_or_update_endpoint(&self, mut spec: LocalEndpointSpec) {
        spec.runtime_id = normalize_local_runtime_id(&spec.runtime_id);
        spec.base_url = spec.base_url.trim().to_owned();
        let now_ms = Utc::now().timestamp_millis();
        let mut eps = self.endpoints.write();
        if let Some(existing) = eps.iter_mut().find(|e| e.runtime_id == spec.runtime_id) {
            *existing = spec.clone();
        } else {
            eps.push(spec.clone());
        }
        drop(eps);
        self.statuses
            .entry(spec.runtime_id.clone())
            .or_insert_with(|| empty_status(&spec, false, None, None, now_ms));
        self.bump_generation();
    }

    pub fn remove_endpoint(&self, runtime_id: &str) -> bool {
        let norm = normalize_local_runtime_id(runtime_id);
        let defaults = LocalEndpointSpec::defaults();
        if let Some(def_spec) = defaults.into_iter().find(|d| d.runtime_id == norm) {
            self.add_or_update_endpoint(def_spec);
            return true;
        }
        let mut eps = self.endpoints.write();
        let before = eps.len();
        eps.retain(|e| e.runtime_id != norm);
        drop(eps);
        self.statuses.remove(&norm);
        self.api_keys.remove(&norm);
        self.bump_generation();
        before != self.endpoints.read().len()
    }

    /// Applies persisted custom/overridden local endpoints from `GatewaySettings`.
    pub fn apply_custom_endpoints(&self, custom: &[LocalEndpointConfig]) {
        for cfg in custom {
            if cfg.runtime_id.trim().is_empty() || cfg.base_url.trim().is_empty() {
                continue;
            }
            self.add_or_update_endpoint(LocalEndpointSpec {
                runtime_id: cfg.runtime_id.clone(),
                display_name: if cfg.display_name.trim().is_empty() {
                    cfg.runtime_id.clone()
                } else {
                    cfg.display_name.clone()
                },
                base_url: cfg.base_url.clone(),
            });
        }
    }

    /// Registers or clears an optional Bearer API key for a local runtime
    /// (e.g. `llama-server --api-key <KEY>` or authenticated vLLM/LM Studio).
    pub fn set_api_key(&self, runtime_id: &str, key: Option<String>) {
        let id = normalize_local_runtime_id(runtime_id);
        match key.map(|k| k.trim().to_owned()).filter(|k| !k.is_empty()) {
            Some(clean) => {
                self.api_keys.insert(id, clean);
            }
            None => {
                // Fall back to LLAMA_API_KEY env var if clearing a vault key for llamacpp
                if id == "llamacpp"
                    && let Ok(env_k) = std::env::var("QUIXIGATE_LLAMACPP_API_KEY")
                        .or_else(|_| std::env::var("LLAMA_API_KEY"))
                    && !env_k.trim().is_empty()
                {
                    self.api_keys.insert(id, env_k.trim().to_owned());
                } else {
                    self.api_keys.remove(&id);
                }
            }
        }
    }

    pub fn api_key_for(&self, runtime_id: &str) -> Option<String> {
        let id = normalize_local_runtime_id(runtime_id);
        self.api_keys.get(&id).map(|v| v.value().clone())
    }

    pub fn record_performance(
        &self,
        runtime_id: &str,
        ttft_ms: Option<u64>,
        tok_per_sec: Option<f64>,
    ) {
        let id = normalize_local_runtime_id(runtime_id);
        let mut entry = self.perf.entry(id.clone()).or_default();
        entry.samples += 1;
        let alpha = if entry.samples == 1 { 1.0 } else { 0.25 };
        if let Some(ttft) = ttft_ms {
            entry.avg_ttft_ms = entry.avg_ttft_ms * (1.0 - alpha) + (ttft as f64) * alpha;
        }
        if let Some(tps_val) = tok_per_sec.filter(|v| *v > 0.0) {
            entry.avg_tok_per_sec = entry.avg_tok_per_sec * (1.0 - alpha) + tps_val * alpha;
        }
        let avg_ttft = (entry.avg_ttft_ms > 0.0).then_some(entry.avg_ttft_ms.round() as u64);
        let avg_tps = (entry.avg_tok_per_sec > 0.0).then_some(entry.avg_tok_per_sec);
        drop(entry);

        if let Some(mut st) = self.statuses.get_mut(&id) {
            st.avg_ttft_ms = avg_ttft;
            st.avg_tok_per_sec = avg_tps;
        }
    }

    pub fn list_statuses(&self) -> Vec<LocalRuntimeStatus> {
        let eps = self.endpoints.read().clone();
        let mut out = Vec::with_capacity(eps.len());
        for ep in eps {
            if let Some(st) = self.statuses.get(&ep.runtime_id) {
                out.push(st.clone());
            }
        }
        out
    }

    pub fn get_status(&self, runtime_id: &str) -> Option<LocalRuntimeStatus> {
        let id = normalize_local_runtime_id(runtime_id);
        self.statuses.get(&id).map(|v| v.clone())
    }

    /// Probes all registered local endpoints concurrently with sub-second timeouts.
    pub async fn scan_all(&self) -> Vec<LocalRuntimeStatus> {
        let eps = self.endpoints.read().clone();
        let mut handles = Vec::with_capacity(eps.len());
        for ep in eps {
            let scanner = self.clone();
            handles.push(tokio::spawn(async move { scanner.probe_one(&ep).await }));
        }
        let mut results = Vec::new();
        for h in handles {
            if let Ok(st) = h.await {
                self.statuses.insert(st.runtime_id.clone(), st.clone());
                results.push(st);
            }
        }
        self.bump_generation();
        results
    }

    /// Triggers `POST /models/load` on a `llama-server` Router-mode runtime and refreshes status.
    pub async fn load_model(&self, runtime_id: &str, model: &str) -> Result<LocalRuntimeStatus> {
        self.control_llamacpp_model(runtime_id, model, true).await
    }

    /// Triggers `POST /models/unload` on a `llama-server` Router-mode runtime and refreshes status.
    pub async fn unload_model(&self, runtime_id: &str, model: &str) -> Result<LocalRuntimeStatus> {
        self.control_llamacpp_model(runtime_id, model, false).await
    }

    async fn control_llamacpp_model(
        &self,
        runtime_id: &str,
        model: &str,
        load: bool,
    ) -> Result<LocalRuntimeStatus> {
        let id = normalize_local_runtime_id(runtime_id);
        let ep = self
            .endpoints
            .read()
            .iter()
            .find(|e| e.runtime_id == id)
            .cloned()
            .ok_or_else(|| {
                QuixigateError::NotFound(format!("local runtime '{id}' not registered"))
            })?;

        let root = server_root(&ep.base_url);
        let action = if load { "load" } else { "unload" };
        let url = format!("{root}/models/{action}");
        let mut req = self
            .control_http
            .post(&url)
            .json(&json!({ "model": model }));
        if let Some(key) = self.api_key_for(&id) {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| QuixigateError::Upstream {
            status: 502,
            provider: id.clone(),
            message: format!("llama-server /models/{action} failed: {e}"),
        })?;
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or_default();
        if !status.is_success() {
            let msg = body["error"]["message"]
                .as_str()
                .unwrap_or("llama-server model control request failed")
                .to_owned();
            // Treat "model is already running" on load or "model is not running" on unload as idempotent
            let lower = msg.to_ascii_lowercase();
            let idempotent = (load && lower.contains("already running"))
                || (!load && lower.contains("not running"));
            if !idempotent {
                return Err(QuixigateError::Upstream {
                    status: status.as_u16(),
                    provider: id,
                    message: msg,
                });
            }
        }

        let updated = self.probe_one(&ep).await;
        self.statuses
            .insert(updated.runtime_id.clone(), updated.clone());
        self.bump_generation();
        Ok(updated)
    }

    async fn probe_one(&self, ep: &LocalEndpointSpec) -> LocalRuntimeStatus {
        let now_ms = Utc::now().timestamp_millis();
        let perf = self.perf.get(&ep.runtime_id).map(|p| p.clone());
        let avg_ttft_ms = perf
            .as_ref()
            .and_then(|p| (p.avg_ttft_ms > 0.0).then_some(p.avg_ttft_ms.round() as u64));
        let avg_tok_per_sec = perf
            .as_ref()
            .and_then(|p| (p.avg_tok_per_sec > 0.0).then_some(p.avg_tok_per_sec));
        let api_key = self.api_key_for(&ep.runtime_id);

        if ep.runtime_id == "ollama" {
            if let Some(st) = self
                .probe_ollama(ep, api_key.as_deref(), avg_ttft_ms, avg_tok_per_sec, now_ms)
                .await
            {
                return st;
            }
        } else if ep.runtime_id == "llamacpp" || looks_like_llamacpp_id(&ep.runtime_id) {
            return self
                .probe_llamacpp(ep, api_key.as_deref(), avg_ttft_ms, avg_tok_per_sec, now_ms)
                .await;
        } else if ep.runtime_id == "lmstudio"
            && let Some(st) = self
                .probe_lmstudio(ep, api_key.as_deref(), avg_ttft_ms, avg_tok_per_sec, now_ms)
                .await
        {
            return st;
        }

        self.probe_generic_openai(ep, api_key.as_deref(), avg_ttft_ms, avg_tok_per_sec, now_ms)
            .await
    }

    async fn probe_ollama(
        &self,
        ep: &LocalEndpointSpec,
        api_key: Option<&str>,
        avg_ttft_ms: Option<u64>,
        avg_tok_per_sec: Option<f64>,
        now_ms: i64,
    ) -> Option<LocalRuntimeStatus> {
        let root = server_root(&ep.base_url);
        let tags_url = format!("{root}/api/tags");
        let ps_url = format!("{root}/api/ps");
        let mut req = self.http.get(&tags_url);
        if let Some(k) = api_key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let val = resp.json::<Value>().await.ok()?;
        let models: Vec<String> = val["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m["name"].as_str().or(m["model"].as_str()))
            .map(str::to_owned)
            .collect();

        let mut loaded_models = Vec::new();
        let mut ps_req = self.http.get(&ps_url);
        if let Some(k) = api_key {
            ps_req = ps_req.bearer_auth(k);
        }
        if let Ok(ps_resp) = ps_req.send().await
            && ps_resp.status().is_success()
            && let Ok(ps_val) = ps_resp.json::<Value>().await
        {
            loaded_models = ps_val["models"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|m| m["name"].as_str().or(m["model"].as_str()))
                .map(str::to_owned)
                .collect();
        }

        let mut model_details = BTreeMap::new();
        for m in &models {
            let is_loaded = loaded_models.iter().any(|lm| lm == m);
            model_details.insert(
                m.clone(),
                LocalModelInfo {
                    context_window: None,
                    supports_vision: false,
                    status: Some(if is_loaded { "loaded" } else { "unloaded" }.to_owned()),
                    aliases: Vec::new(),
                },
            );
        }

        Some(LocalRuntimeStatus {
            runtime_id: ep.runtime_id.clone(),
            display_name: ep.display_name.clone(),
            base_url: ep.base_url.clone(),
            healthy: true,
            auth_required: false,
            mode: None,
            total_slots: None,
            supports_load_control: false,
            server_version: None,
            models,
            loaded_models,
            model_details,
            avg_ttft_ms,
            avg_tok_per_sec,
            last_checked_ms: now_ms,
        })
    }

    async fn probe_lmstudio(
        &self,
        ep: &LocalEndpointSpec,
        api_key: Option<&str>,
        avg_ttft_ms: Option<u64>,
        avg_tok_per_sec: Option<f64>,
        now_ms: i64,
    ) -> Option<LocalRuntimeStatus> {
        let root = server_root(&ep.base_url);
        let native_url = format!("{root}/api/v0/models");
        let mut req = self.http.get(&native_url);
        if let Some(k) = api_key {
            req = req.bearer_auth(k);
        }
        if let Ok(resp) = req.send().await {
            if matches!(resp.status().as_u16(), 401 | 403) {
                return Some(empty_status(ep, true, avg_ttft_ms, avg_tok_per_sec, now_ms));
            }
            if resp.status().is_success()
                && let Ok(val) = resp.json::<Value>().await
                && let Some(arr) = val["data"].as_array()
            {
                let mut models = Vec::new();
                let mut loaded_models = Vec::new();
                let mut model_details = BTreeMap::new();
                for item in arr {
                    let Some(id) = item["id"].as_str() else {
                        continue;
                    };
                    // Skip pure embedding models in chat model list? No, keep all so `/v1/embeddings` works too!
                    let state_str = item["state"].as_str().unwrap_or("loaded");
                    if state_str == "loaded" {
                        loaded_models.push(id.to_owned());
                    }
                    let ctx = item["loaded_context_length"]
                        .as_u64()
                        .or_else(|| item["max_context_length"].as_u64())
                        .filter(|&n| n > 0);
                    let supports_vision = item["type"].as_str() == Some("vlm");
                    models.push(id.to_owned());
                    model_details.insert(
                        id.to_owned(),
                        LocalModelInfo {
                            context_window: ctx,
                            supports_vision,
                            status: Some(state_str.to_owned()),
                            aliases: Vec::new(),
                        },
                    );
                }
                return Some(LocalRuntimeStatus {
                    runtime_id: ep.runtime_id.clone(),
                    display_name: ep.display_name.clone(),
                    base_url: ep.base_url.clone(),
                    healthy: true,
                    auth_required: false,
                    mode: None,
                    total_slots: None,
                    supports_load_control: false,
                    server_version: None,
                    models,
                    loaded_models,
                    model_details,
                    avg_ttft_ms,
                    avg_tok_per_sec,
                    last_checked_ms: now_ms,
                });
            }
        }
        None
    }

    /// Native `llama.cpp` (`llama-server`) probe supporting:
    /// 1. Public `/health` check (`200 ok` / `503 loading`) + `--api-key` `401` detection
    /// 2. Single-Model Instance Mode (`/props` `total_slots`, `n_ctx`, `modalities.vision`, `is_sleeping`, `data[0].meta`)
    /// 3. Multi-Model Router Mode (`props.role == "router"`, `max_instances`, `models_autoload`, per-model `status.value`, `aliases`, `/models/load` & `/models/unload`)
    async fn probe_llamacpp(
        &self,
        ep: &LocalEndpointSpec,
        api_key: Option<&str>,
        avg_ttft_ms: Option<u64>,
        avg_tok_per_sec: Option<f64>,
        now_ms: i64,
    ) -> LocalRuntimeStatus {
        let root = server_root(&ep.base_url);
        let health_url = format!("{root}/health");
        let props_url = format!("{root}/props");
        let models_url = models_endpoint(&ep.base_url);

        let health_status = self
            .http
            .get(&health_url)
            .send()
            .await
            .ok()
            .map(|r| r.status().as_u16());

        let mut models_req = self.http.get(&models_url);
        let mut props_req = self.http.get(&props_url);
        if let Some(k) = api_key {
            models_req = models_req.bearer_auth(k);
            props_req = props_req.bearer_auth(k);
        }

        let (models_res, props_res) = tokio::join!(models_req.send(), props_req.send());

        let Ok(models_resp) = models_res else {
            return empty_status(ep, false, avg_ttft_ms, avg_tok_per_sec, now_ms);
        };

        if matches!(models_resp.status().as_u16(), 401 | 403) {
            // Public /health succeeded (or server explicitly rejected /v1/models with 401):
            // surface `auth_required = true` so the operator knows `llama-server --api-key` is active!
            let reachable = health_status.is_some_and(|code| matches!(code, 200 | 503))
                || models_resp.status().as_u16() == 401;
            return empty_status(ep, reachable, avg_ttft_ms, avg_tok_per_sec, now_ms);
        }

        if !models_resp.status().is_success() {
            return empty_status(ep, false, avg_ttft_ms, avg_tok_per_sec, now_ms);
        }

        let Ok(models_json) = models_resp.json::<Value>().await else {
            return empty_status(ep, false, avg_ttft_ms, avg_tok_per_sec, now_ms);
        };

        let props_json: Value = match props_res {
            Ok(r) if r.status().is_success() => r.json::<Value>().await.unwrap_or_default(),
            _ => Value::Null,
        };

        let data_arr = models_json["data"]
            .as_array()
            .or_else(|| models_json["models"].as_array())
            .cloned()
            .unwrap_or_default();

        let is_router = props_json["role"].as_str() == Some("router")
            || data_arr
                .iter()
                .any(|item| item["status"]["value"].is_string());

        let server_version = props_json["build_info"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        if is_router {
            let models_autoload = props_json["models_autoload"].as_bool().unwrap_or(true);
            let total_slots = props_json["max_instances"]
                .as_u64()
                .filter(|&n| n > 0)
                .map(|n| n as u32);

            let mut models = Vec::new();
            let mut loaded_models = Vec::new();
            let mut model_details = BTreeMap::new();

            for item in &data_arr {
                let Some(id) = item["id"].as_str().or_else(|| item["name"].as_str()) else {
                    continue;
                };
                let status_val = item["status"]["value"].as_str().unwrap_or("loaded");
                let failed = item["status"]["failed"].as_bool().unwrap_or(false);
                if status_val == "incomplete" {
                    continue;
                }
                let aliases = parse_string_array(&item["aliases"]);
                let ctx = item["meta"]["n_ctx"]
                    .as_u64()
                    .filter(|&n| n > 0)
                    .or_else(|| item["context_length"].as_u64().filter(|&n| n > 0))
                    .or_else(|| item["meta"]["n_ctx_train"].as_u64().filter(|&n| n > 0));
                let supports_vision = has_image_modality(&item["architecture"]["input_modalities"]);

                if status_val == "loaded" {
                    loaded_models.push(id.to_owned());
                }
                let routable = !failed
                    && (models_autoload || matches!(status_val, "loaded" | "loading" | "sleeping"));
                if routable {
                    models.push(id.to_owned());
                }
                model_details.insert(
                    id.to_owned(),
                    LocalModelInfo {
                        context_window: ctx,
                        supports_vision,
                        status: Some(if failed {
                            "failed".to_owned()
                        } else {
                            status_val.to_owned()
                        }),
                        aliases,
                    },
                );
            }

            return LocalRuntimeStatus {
                runtime_id: ep.runtime_id.clone(),
                display_name: ep.display_name.clone(),
                base_url: ep.base_url.clone(),
                healthy: health_status != Some(503),
                auth_required: false,
                mode: Some("router".to_owned()),
                total_slots,
                supports_load_control: true,
                server_version,
                models,
                loaded_models,
                model_details,
                avg_ttft_ms,
                avg_tok_per_sec,
                last_checked_ms: now_ms,
            };
        }

        // Single-Model Instance Mode (`llama-server -m ...` or `-hf ...`, or compatible local server on :8080)
        let is_llamacpp_instance = props_json.get("total_slots").is_some()
            || data_arr.iter().any(|item| {
                item["owned_by"].as_str() == Some("llamacpp") || item.get("meta").is_some()
            });

        let total_slots = props_json["total_slots"]
            .as_u64()
            .filter(|&n| n > 0)
            .map(|n| n as u32);
        let props_ctx = props_json["default_generation_settings"]["n_ctx"]
            .as_u64()
            .filter(|&n| n > 0);
        let props_vision = props_json["modalities"]["vision"]
            .as_bool()
            .unwrap_or(false);
        let is_sleeping = props_json["is_sleeping"].as_bool().unwrap_or(false);
        let is_loading = health_status == Some(503);

        let mut models = Vec::new();
        let mut loaded_models = Vec::new();
        let mut model_details = BTreeMap::new();

        for item in &data_arr {
            let Some(id) = item["id"].as_str().or_else(|| item["name"].as_str()) else {
                continue;
            };
            let aliases = parse_string_array(&item["aliases"]);
            let ctx = item["meta"]["n_ctx"]
                .as_u64()
                .filter(|&n| n > 0)
                .or(props_ctx)
                .or_else(|| item["meta"]["n_ctx_train"].as_u64().filter(|&n| n > 0))
                .or_else(|| item["context_length"].as_u64().filter(|&n| n > 0));
            let supports_vision =
                props_vision || has_image_modality(&item["architecture"]["input_modalities"]);
            let state_label = if is_loading {
                "loading"
            } else if is_sleeping {
                "sleeping"
            } else {
                "loaded"
            };
            models.push(id.to_owned());
            if !is_loading && !is_sleeping {
                loaded_models.push(id.to_owned());
            }
            model_details.insert(
                id.to_owned(),
                LocalModelInfo {
                    context_window: ctx,
                    supports_vision,
                    status: Some(state_label.to_owned()),
                    aliases,
                },
            );
        }

        LocalRuntimeStatus {
            runtime_id: ep.runtime_id.clone(),
            display_name: ep.display_name.clone(),
            base_url: ep.base_url.clone(),
            healthy: !is_loading,
            auth_required: false,
            mode: is_llamacpp_instance.then(|| "instance".to_owned()),
            total_slots,
            supports_load_control: false,
            server_version,
            models,
            loaded_models,
            model_details,
            avg_ttft_ms,
            avg_tok_per_sec,
            last_checked_ms: now_ms,
        }
    }

    async fn probe_generic_openai(
        &self,
        ep: &LocalEndpointSpec,
        api_key: Option<&str>,
        avg_ttft_ms: Option<u64>,
        avg_tok_per_sec: Option<f64>,
        now_ms: i64,
    ) -> LocalRuntimeStatus {
        let models_url = models_endpoint(&ep.base_url);
        let mut req = self.http.get(&models_url);
        if let Some(k) = api_key {
            req = req.bearer_auth(k);
        }
        if let Ok(resp) = req.send().await {
            if matches!(resp.status().as_u16(), 401 | 403) {
                return empty_status(ep, true, avg_ttft_ms, avg_tok_per_sec, now_ms);
            }
            if resp.status().is_success()
                && let Ok(val) = resp.json::<Value>().await
            {
                let mut models = Vec::new();
                let mut model_details = BTreeMap::new();
                for item in val["data"]
                    .as_array()
                    .or_else(|| val["models"].as_array())
                    .into_iter()
                    .flatten()
                {
                    if let Some(id) = item["id"].as_str().or_else(|| item["name"].as_str()) {
                        let ctx = item["max_model_len"]
                            .as_u64()
                            .or_else(|| item["context_length"].as_u64())
                            .filter(|&n| n > 0);
                        models.push(id.to_owned());
                        model_details.insert(
                            id.to_owned(),
                            LocalModelInfo {
                                context_window: ctx,
                                supports_vision: has_image_modality(
                                    &item["architecture"]["input_modalities"],
                                ),
                                status: Some("loaded".to_owned()),
                                aliases: parse_string_array(&item["aliases"]),
                            },
                        );
                    }
                }
                return LocalRuntimeStatus {
                    runtime_id: ep.runtime_id.clone(),
                    display_name: ep.display_name.clone(),
                    base_url: ep.base_url.clone(),
                    healthy: true,
                    auth_required: false,
                    mode: None,
                    total_slots: None,
                    supports_load_control: false,
                    server_version: None,
                    loaded_models: models.clone(),
                    models,
                    model_details,
                    avg_ttft_ms,
                    avg_tok_per_sec,
                    last_checked_ms: now_ms,
                };
            }
        }
        empty_status(ep, false, avg_ttft_ms, avg_tok_per_sec, now_ms)
    }
}

impl Default for LocalRuntimeScanner {
    fn default() -> Self {
        Self::new()
    }
}

fn server_root(base_url: &str) -> &str {
    base_url
        .trim()
        .trim_end_matches('/')
        .trim_end_matches("/v1")
}

fn models_endpoint(base_url: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if base.ends_with("/models") {
        base.to_owned()
    } else if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    }
}

fn looks_like_llamacpp_id(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    lower.starts_with("llamacpp")
        || lower.starts_with("llama.cpp")
        || lower.starts_with("llama-cpp")
}

fn parse_string_array(v: &Value) -> Vec<String> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| x.as_str().filter(|s| !s.is_empty()).map(str::to_owned))
        .collect()
}

fn has_image_modality(v: &Value) -> bool {
    v.as_array().is_some_and(|arr| {
        arr.iter()
            .any(|m| matches!(m.as_str(), Some("image" | "vision")))
    })
}

fn empty_status(
    ep: &LocalEndpointSpec,
    auth_required: bool,
    avg_ttft_ms: Option<u64>,
    avg_tok_per_sec: Option<f64>,
    now_ms: i64,
) -> LocalRuntimeStatus {
    LocalRuntimeStatus {
        runtime_id: ep.runtime_id.clone(),
        display_name: ep.display_name.clone(),
        base_url: ep.base_url.clone(),
        healthy: false,
        auth_required,
        mode: None,
        total_slots: None,
        supports_load_control: false,
        server_version: None,
        models: Vec::new(),
        loaded_models: Vec::new(),
        model_details: BTreeMap::new(),
        avg_ttft_ms,
        avg_tok_per_sec,
        last_checked_ms: now_ms,
    }
}
