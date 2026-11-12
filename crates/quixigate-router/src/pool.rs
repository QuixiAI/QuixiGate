//! Multi-seat credential pool selector (`fill_first`, `least_used`, `round_robin`,
//! `random`), paid-API overflow gating, and model-scoped vs credential-wide error classifier.

use std::sync::atomic::{AtomicUsize, Ordering};

use quixigate_core::{
    AuthKind, CredentialRecord, GatewaySettings, LocalRuntimeStatus, ModelRoute, ModelSpec,
    PoolStrategy, now_ms,
};
use quixigate_db::Database;
use quixigate_providers::{LocalRuntimeScanner, normalize_local_runtime_id};
use rand::seq::SliceRandom;

use crate::catalog::ModelCatalog;

const MODEL_ENTITLEMENT_BENCH_MS: i64 = 365 * 24 * 60 * 60 * 1000;
const SOLE_SEAT_COOLDOWN_MS: i64 = 60 * 1000;
const POOLED_SEAT_COOLDOWN_MS: i64 = 5 * 60 * 1000;

#[derive(Debug, Clone)]
pub struct DispatchCandidate {
    pub provider_id: String,
    pub upstream_model: String,
    pub credential: Option<CredentialRecord>,
    pub base_url_override: Option<String>,
}

pub struct PoolSelector {
    rr_counter: AtomicUsize,
}

impl PoolSelector {
    pub fn new() -> Self {
        Self {
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Resolves a requested `ModelSpec` (plus optional `ModelRoute` alias) into an ordered
    /// failover list of healthy seats and/or discovered local runtimes.
    pub fn resolve_candidates(
        &self,
        spec: &ModelSpec,
        route: Option<&ModelRoute>,
        credentials: &[CredentialRecord],
        catalog: &ModelCatalog,
        local_scanner: &LocalRuntimeScanner,
        settings: &GatewaySettings,
    ) -> Vec<DispatchCandidate> {
        let now = now_ms();
        let strategy = route
            .map(|r| r.pool_strategy)
            .unwrap_or(settings.default_pool_strategy);
        let allow_paid = route.map(|r| r.allow_paid_overflow).unwrap_or(true) && !spec.free_only;
        let burst_enabled = settings.openrouter_burst_enabled && allow_paid;
        let burst_fallback_model = {
            let trimmed = settings.openrouter_burst_model.trim();
            if trimmed.is_empty() {
                quixigate_core::brand::DEFAULT_OPENROUTER_BURST_MODEL
            } else {
                trimmed
            }
        };

        let mut targets: Vec<(Option<String>, Option<String>, String)> = Vec::new();
        if let Some(r) = route
            && !r.targets.is_empty()
        {
            for t in &r.targets {
                let parsed = ModelSpec::parse(&t.model);
                let prov = t
                    .provider_id
                    .clone()
                    .or(parsed.pinned_target)
                    .or_else(|| spec.pinned_target.clone());
                targets.push((t.credential_id.clone(), prov, parsed.base_model));
            }
        } else {
            targets.push((
                None,
                spec.pinned_target.clone(),
                spec.base_model.clone(),
            ));
        }

        let local_statuses = local_scanner.list_statuses();
        let mut out = Vec::new();

        for (pinned_cred_id, pinned_provider, model_target) in targets {
            let mut oauth_or_local_tier = Vec::new();
            let mut paid_api_tier = Vec::new();

            let cat_entry = if spec.free_only {
                catalog
                    .lookup(&format!("{model_target}:free"))
                    .or_else(|| catalog.lookup(&model_target))
            } else {
                catalog.lookup(&model_target)
            };
            let (explicit_prefix, bare_model) = match model_target.split_once('/') {
                Some((p, m)) => (Some(p.to_ascii_lowercase()), m.to_owned()),
                None => (None, model_target.clone()),
            };

            // 1. Check stored credentials in SQLite
            for cred in credentials {
                if let Some(cid) = &pinned_cred_id
                    && &cred.id != cid
                {
                    continue;
                }
                if let Some(pin_prov) = &pinned_provider
                    && !provider_matches_pin(&cred.provider_id, pin_prov)
                {
                    continue;
                }
                if pinned_cred_id.is_none()
                    && !credential_serves_model(
                        cred,
                        explicit_prefix.as_deref(),
                        &model_target,
                        &bare_model,
                        cat_entry.as_ref(),
                        &local_statuses,
                    )
                {
                    continue;
                }

                let upstream_model = resolve_upstream_model_name(
                    &cred.provider_id,
                    &model_target,
                    explicit_prefix.as_deref(),
                    &bare_model,
                    cat_entry.as_ref(),
                    &local_statuses,
                    spec,
                    burst_enabled.then_some(burst_fallback_model),
                );
                if !cred.is_healthy_for(Some(&upstream_model), now) {
                    continue;
                }

                let cand = DispatchCandidate {
                    provider_id: normalize_if_local_provider(&cred.provider_id),
                    upstream_model,
                    base_url_override: cred.base_url.clone(),
                    credential: Some(cred.clone()),
                };

                let is_zero_marginal_seat = (cred.auth_kind.is_zero_marginal_cost()
                    && cred.provider_id != "openrouter")
                    || is_local_provider(&cred.provider_id);
                let is_openrouter_free_route = cred.provider_id == "openrouter"
                    && (spec.free_only || model_target == "openrouter/free");

                if is_zero_marginal_seat {
                    oauth_or_local_tier.push(cand);
                } else if allow_paid
                    || is_openrouter_free_route
                    || pinned_cred_id.is_some()
                    || pinned_provider.is_some()
                {
                    paid_api_tier.push(cand);
                }
            }

            // 2. Check auto-discovered local runtimes (zero-credential path)
            for rt in local_statuses.iter().filter(|r| r.healthy) {
                if let Some(pin_prov) = &pinned_provider
                    && !provider_matches_pin(&rt.runtime_id, pin_prov)
                    && pin_prov != "local"
                {
                    continue;
                }
                if let Some(upstream_model) = resolve_local_runtime_model(
                    rt,
                    &model_target,
                    explicit_prefix.as_deref(),
                    &bare_model,
                ) {
                    // Avoid duplicate if user already stored a credential for this local runtime
                    let already = oauth_or_local_tier
                        .iter()
                        .any(|c| c.provider_id == rt.runtime_id);
                    if !already {
                        oauth_or_local_tier.push(DispatchCandidate {
                            provider_id: rt.runtime_id.clone(),
                            upstream_model,
                            credential: None,
                            base_url_override: Some(rt.base_url.clone()),
                        });
                    }
                }
            }

            // Apply pool strategy within each tier (OAuth/Local seats ALWAYS come before paid API overflow,
            // and paid API overflow is gated if an OAuth seat exists with `allow_api_overflow == false`
            // unless route explicitly enables it).
            let any_oauth_for_target = credentials.iter().any(|c| {
                c.enabled
                    && c.provider_id != "openrouter"
                    && matches!(c.auth_kind, AuthKind::Oauth | AuthKind::SetupToken)
                    && credential_serves_model(
                        c,
                        explicit_prefix.as_deref(),
                        &model_target,
                        &bare_model,
                        cat_entry.as_ref(),
                        &local_statuses,
                    )
            });
            let seat_forbids_overflow = any_oauth_for_target
                && credentials
                    .iter()
                    .filter(|c| {
                        c.enabled
                            && c.provider_id != "openrouter"
                            && matches!(c.auth_kind, AuthKind::Oauth | AuthKind::SetupToken)
                            && credential_serves_model(
                                c,
                                explicit_prefix.as_deref(),
                                &model_target,
                                &bare_model,
                                cat_entry.as_ref(),
                                &local_statuses,
                            )
                    })
                    .all(|c| !c.allow_api_overflow)
                && route.is_none();

            self.order_tier(&mut oauth_or_local_tier, strategy);
            out.extend(oauth_or_local_tier);

            if !seat_forbids_overflow {
                self.order_tier(&mut paid_api_tier, strategy);
                out.extend(paid_api_tier);
            } else if burst_enabled && pinned_cred_id.is_none() && pinned_provider.is_none() {
                let mut openrouter_burst_only: Vec<DispatchCandidate> = paid_api_tier
                    .into_iter()
                    .filter(|c| c.provider_id == "openrouter")
                    .collect();
                self.order_tier(&mut openrouter_burst_only, strategy);
                out.extend(openrouter_burst_only);
            }

            // When Cloud Burst is enabled and a local-prefixed target (e.g. `ollama/qwen2.5-coder:7b` or `llamacpp/...`)
            // hasn't yet added an OpenRouter fallback candidate, append healthy OpenRouter seats using `burst_fallback_model`.
            if burst_enabled
                && pinned_cred_id.is_none()
                && pinned_provider.is_none()
                && !out.iter().any(|c| c.provider_id == "openrouter")
            {
                let mut extra_burst = Vec::new();
                for cred in credentials.iter().filter(|c| c.enabled && c.provider_id == "openrouter") {
                    let burst_upstream = apply_openrouter_modifiers(burst_fallback_model, spec);
                    if cred.is_healthy_for(Some(&burst_upstream), now) {
                        extra_burst.push(DispatchCandidate {
                            provider_id: "openrouter".to_owned(),
                            upstream_model: burst_upstream,
                            base_url_override: cred.base_url.clone(),
                            credential: Some(cred.clone()),
                        });
                    }
                }
                self.order_tier(&mut extra_burst, strategy);
                out.extend(extra_burst);
            }
        }

        out
    }

    fn order_tier(&self, tier: &mut [DispatchCandidate], strategy: PoolStrategy) {
        if tier.len() <= 1 {
            return;
        }
        match strategy {
            PoolStrategy::FillFirst => {
                tier.sort_by(|a, b| {
                    let pa = a.credential.as_ref().map(|c| c.priority).unwrap_or(0);
                    let pb = b.credential.as_ref().map(|c| c.priority).unwrap_or(0);
                    pa.cmp(&pb)
                });
            }
            PoolStrategy::LeastUsed => {
                tier.sort_by(|a, b| {
                    let ua = seat_usage_score(a.credential.as_ref());
                    let ub = seat_usage_score(b.credential.as_ref());
                    ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            PoolStrategy::RoundRobin => {
                let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) % tier.len();
                tier.rotate_left(idx);
            }
            PoolStrategy::Random => {
                let mut rng = rand::thread_rng();
                tier.shuffle(&mut rng);
            }
        }
    }
}

impl Default for PoolSelector {
    fn default() -> Self {
        Self::new()
    }
}

fn seat_usage_score(cred: Option<&CredentialRecord>) -> f64 {
    let Some(c) = cred else {
        return 0.0;
    };
    if let Some(snap) = &c.quota_snapshot
        && let Some(w) = snap.windows.first()
    {
        return w.used_percent;
    }
    (c.last_used_at_ms.unwrap_or(0) as f64) / 1e12
}

fn is_local_provider(provider_id: &str) -> bool {
    matches!(
        provider_id.trim().to_ascii_lowercase().as_str(),
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
    )
}

fn normalize_if_local_provider(provider_id: &str) -> String {
    if is_local_provider(provider_id) && provider_id != "local" {
        normalize_local_runtime_id(provider_id)
    } else {
        provider_id.to_owned()
    }
}

/// Resolves `model_target` against a discovered `LocalRuntimeStatus`, supporting:
/// - Exact model IDs including HuggingFace `org/repo:quant` slugs (`ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`)
/// - Runtime-prefixed slugs (`llamacpp/Qwen3-30B`, `llama.cpp/Qwen3-30B`, `ollama/qwen3:32b`, `lmstudio/...`)
/// - `llama-server` model `aliases` (`--alias` / router preset aliases)
fn resolve_local_runtime_model(
    rt: &LocalRuntimeStatus,
    model_target: &str,
    explicit_prefix: Option<&str>,
    bare_model: &str,
) -> Option<String> {
    if rt.models.iter().any(|m| m == model_target) {
        return Some(model_target.to_owned());
    }
    if let Some((id, _)) = rt
        .model_details
        .iter()
        .find(|(_, d)| d.aliases.iter().any(|a| a == model_target))
    {
        return Some(id.clone());
    }
    if let Some(prefix) = explicit_prefix {
        if provider_matches_pin(&rt.runtime_id, prefix) {
            if rt.models.iter().any(|m| m == bare_model) {
                return Some(bare_model.to_owned());
            }
            if let Some((id, _)) = rt
                .model_details
                .iter()
                .find(|(_, d)| d.aliases.iter().any(|a| a == bare_model))
            {
                return Some(id.clone());
            }
            return Some(bare_model.to_owned());
        }
        if prefix == "local" {
            if rt.models.iter().any(|m| m == bare_model) {
                return Some(bare_model.to_owned());
            }
            if let Some((id, _)) = rt
                .model_details
                .iter()
                .find(|(_, d)| d.aliases.iter().any(|a| a == bare_model))
            {
                return Some(id.clone());
            }
        }
    }
    None
}

fn provider_matches_pin(cred_provider: &str, pin: &str) -> bool {
    let p = pin.to_ascii_lowercase();
    let cp = cred_provider.to_ascii_lowercase();
    if cp == p {
        return true;
    }
    if is_local_provider(&cp) && is_local_provider(&p) && p != "local" && cp != "local" {
        return normalize_local_runtime_id(&cp) == normalize_local_runtime_id(&p);
    }
    matches!(
        (cp.as_str(), p.as_str()),
        ("openai-codex", "openai" | "codex")
            | ("anthropic-token", "anthropic" | "claude")
            | ("google-gemini-cli", "google" | "gemini")
            | ("xai-oauth", "xai" | "grok")
            | ("qwen-portal", "qwen" | "alibaba")
            | ("minimax-portal", "minimax")
            | ("github-copilot", "copilot")
            | ("fireworks-ai", "fireworks")
            | ("togetherai", "together")
    )
}

fn is_openrouter_native_slug(target: &str) -> bool {
    let t = target.trim();
    t.starts_with("@preset/")
        || t.starts_with('~')
        || t.eq_ignore_ascii_case("openrouter/auto")
        || t.eq_ignore_ascii_case("openrouter/free")
}

fn credential_serves_model(
    cred: &CredentialRecord,
    explicit_prefix: Option<&str>,
    full_target: &str,
    bare_model: &str,
    cat: Option<&crate::catalog::CatalogModel>,
    local_statuses: &[LocalRuntimeStatus],
) -> bool {
    let pid = cred.provider_id.as_str();
    if is_openrouter_native_slug(full_target) {
        return pid == "openrouter";
    }
    if is_local_provider(pid) {
        let norm_id = normalize_if_local_provider(pid);
        if let Some(prefix) = explicit_prefix
            && (provider_matches_pin(&norm_id, prefix) || prefix == "local")
        {
            return true;
        }
        if let Some(rt) = local_statuses.iter().find(|r| r.runtime_id == norm_id) {
            return resolve_local_runtime_model(rt, full_target, explicit_prefix, bare_model)
                .is_some();
        }
        return false;
    }
    // Aggregator & custom OpenAI-compatible gateways can serve any `author/model`
    // (unless explicitly prefixed with a local runtime like `ollama/...` or `llamacpp/...`).
    if matches!(
        pid,
        "openrouter"
            | "openaicompatible"
            | "openaichatcompletions"
            | "vercel"
            | "cloudflare-ai-gateway"
            | "helicone"
            | "litellm"
    ) {
        if let Some(prefix) = explicit_prefix
            && is_local_provider(prefix)
        {
            return false;
        }
        return true;
    }
    if pid == "github-copilot" {
        if let Some(author) = explicit_prefix {
            return matches!(author, "openai" | "anthropic" | "google" | "xai" | "github-copilot");
        }
        return true;
    }
    if let Some(prefix) = explicit_prefix {
        if provider_matches_pin(pid, prefix) || pid == prefix {
            return true;
        }
        if let Some(c) = cat
            && c.providers.contains_key(pid)
        {
            return true;
        }
        return false;
    }
    // Bare model name: check catalog or family heuristics
    if let Some(c) = cat
        && c.providers.contains_key(pid)
    {
        return true;
    }
    let lower = bare_model.to_ascii_lowercase();
    if lower.starts_with("claude") {
        return matches!(pid, "anthropic" | "anthropic-token" | "amazon-bedrock");
    }
    if lower.starts_with("gpt-")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("codex")
    {
        return matches!(pid, "openai" | "openai-codex");
    }
    if lower.starts_with("gemini") {
        return matches!(pid, "google" | "google-gemini-cli");
    }
    if lower.starts_with("grok") {
        return matches!(pid, "xai" | "xai-oauth");
    }
    if lower.starts_with("qwen") {
        return matches!(pid, "qwen-portal" | "alibaba" | "alibaba-cn" | "chutes");
    }
    if lower.starts_with("minimax") {
        return pid == "minimax-portal";
    }
    if lower.starts_with("deepseek") {
        return matches!(pid, "deepseek" | "chutes" | "fireworks-ai" | "togetherai");
    }
    true
}

fn apply_openrouter_modifiers(slug: &str, spec: &ModelSpec) -> String {
    if slug.eq_ignore_ascii_case("openrouter/auto") || slug.eq_ignore_ascii_case("openrouter/free")
    {
        return slug.to_owned();
    }
    if spec.nitro && !slug.ends_with(":nitro") {
        return format!("{slug}:nitro");
    }
    if spec.floor && !slug.ends_with(":floor") {
        return format!("{slug}:floor");
    }
    if spec.free_only && !slug.ends_with(":free") && !slug.starts_with("@preset/") {
        return format!("{slug}:free");
    }
    slug.to_owned()
}

#[allow(clippy::too_many_arguments)]
fn resolve_upstream_model_name(
    provider_id: &str,
    full_target: &str,
    explicit_prefix: Option<&str>,
    bare_model: &str,
    cat: Option<&crate::catalog::CatalogModel>,
    local_statuses: &[LocalRuntimeStatus],
    spec: &ModelSpec,
    burst_fallback_model: Option<&str>,
) -> String {
    if is_local_provider(provider_id) {
        let norm_id = normalize_if_local_provider(provider_id);
        if let Some(rt) = local_statuses.iter().find(|r| r.runtime_id == norm_id)
            && let Some(resolved) =
                resolve_local_runtime_model(rt, full_target, explicit_prefix, bare_model)
        {
            return resolved;
        }
        if let Some(prefix) = explicit_prefix
            && (provider_matches_pin(&norm_id, prefix) || prefix == "local")
        {
            return bare_model.to_owned();
        }
        return full_target.to_owned();
    }
    if provider_id == "openrouter" {
        let slug = if full_target.starts_with("@preset/")
            || full_target.starts_with('~')
            || full_target.eq_ignore_ascii_case("openrouter/auto")
            || full_target.eq_ignore_ascii_case("openrouter/free")
        {
            full_target.to_owned()
        } else if let Some(rest) = full_target.strip_prefix("openrouter/") {
            if rest.contains('/') {
                rest.to_owned()
            } else {
                full_target.to_owned()
            }
        } else if let Some(mapped) = cat.and_then(|c| c.providers.get("openrouter")) {
            mapped.clone()
        } else if full_target.contains('/') {
            full_target.to_owned()
        } else if let Some(c) = cat {
            c.canonical_id.clone()
        } else if let Some(fallback) = burst_fallback_model {
            fallback.to_owned()
        } else {
            bare_model.to_owned()
        };

        return apply_openrouter_modifiers(&slug, spec);
    }
    if let Some(c) = cat
        && let Some(mapped) = c.providers.get(provider_id)
    {
        return mapped.clone();
    }
    // First-party APIs expect bare model IDs (`gpt-5.4`, `claude-sonnet-4-5`, `gemini-2.5-pro`)
    if matches!(
        provider_id,
        "openai"
            | "openai-codex"
            | "anthropic"
            | "anthropic-token"
            | "google"
            | "google-gemini-cli"
            | "xai"
            | "xai-oauth"
            | "deepseek"
            | "mistral"
            | "groq"
            | "github-copilot"
            | "qwen-portal"
            | "minimax-portal"
    ) {
        return bare_model.to_owned();
    }
    if let Some(rest) = full_target.strip_prefix(&format!("{provider_id}/")) {
        return rest.to_owned();
    }
    full_target.to_owned()
}

/// Classifies an upstream `aisdk::Error` and persists model-scoped or credential-wide
/// health state into SQLite so subsequent turns/requests route cleanly around degraded seats.
pub fn record_upstream_failure(
    db: &Database,
    cred: &CredentialRecord,
    upstream_model: &str,
    err: &aisdk::Error,
    total_live_seats_for_provider: usize,
) -> (u16, String, Option<u64>) {
    let (status_opt, details) = match err {
        aisdk::Error::ApiError {
            status_code,
            details,
        } => (status_code.map(|s| s.as_u16()), details.clone()),
        other => (None, other.to_string()),
    };
    let status = status_opt.unwrap_or(502);
    let retry_after_secs = extract_retry_after_secs(&details);
    let now = now_ms();
    let lower = details.to_ascii_lowercase();

    // 1. Terminal auth revocation -> mark seat Dead
    if matches!(status, 401 | 403)
        && [
            "token_revoked",
            "token_invalidated",
            "invalid_grant",
            "account_deactivated",
        ]
        .iter()
        .any(|m| lower.contains(m))
    {
        let _ = db.mark_credential_dead(&cred.id, &details);
        return (status, details, retry_after_secs);
    }

    // 2. Model entitlement / data-policy rejection -> bench ONLY `(seat, model)` so sibling models stay live
    let is_entitlement = matches!(status, 400 | 403 | 404)
        && [
            "does not exist",
            "model_not_found",
            "not a valid model id",
            "not supported on this plan",
            "does not have access to model",
            "usage_not_included",
            "guardrail restrictions and data policy",
        ]
        .iter()
        .any(|m| lower.contains(m));
    if is_entitlement {
        let _ = db.record_model_cooldown(&cred.id, upstream_model, now + MODEL_ENTITLEMENT_BENCH_MS);
        return (status, details, retry_after_secs);
    }

    // 3. Rate-limit / quota window exhaustion (429 / 402 / 529)
    if matches!(status, 429 | 402 | 529) {
        let base_ttl_ms = retry_after_secs
            .map(|s| (s.clamp(1, 86_400) as i64) * 1000)
            .unwrap_or(if total_live_seats_for_provider <= 1 {
                SOLE_SEAT_COOLDOWN_MS
            } else {
                POOLED_SEAT_COOLDOWN_MS
            });
        let until_ms = now + base_ttl_ms;

        // Anthropic enforces rate limits per model; usage_window limits also bench per model
        let is_account_exhausted = [
            "usage_limit_reached",
            "insufficient_quota",
            "credit balance is too low",
        ]
        .iter()
        .any(|m| lower.contains(m));

        if !is_account_exhausted
            && (matches!(
                cred.provider_id.as_str(),
                "anthropic" | "anthropic-token" | "google-gemini-cli"
            ) || lower.contains("model rate limit")
                || lower.contains(&upstream_model.to_ascii_lowercase()))
        {
            let _ = db.record_model_cooldown(&cred.id, upstream_model, until_ms);
        } else {
            let _ = db.record_credential_cooldown(&cred.id, until_ms, &details);
        }
    }

    (status, details, retry_after_secs)
}

fn extract_retry_after_secs(details: &str) -> Option<u64> {
    let marker = "[retry_after=";
    let start = details.find(marker)? + marker.len();
    let rest = &details[start..];
    let end = rest.find(']')?;
    rest[..end].trim().parse::<u64>().ok()
}
