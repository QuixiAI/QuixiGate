//! Live subscription quota-window pollers (`wham/usage`, `api/oauth/usage`,
//! `retrieveUserQuota`, and OpenRouter `/credits`).

use quixigate_core::{
    CredentialSecret, QuixigateError, QuotaSnapshot, QuotaWindow, Result, now_ms,
};
use quixigate_db::Database;
use serde_json::{Value, json};

use crate::refresh::CredentialRefresher;

const WEEKLY_GAP_SECS: i64 = 3 * 24 * 3600;

#[derive(Clone)]
pub struct QuotaEndpoints {
    pub codex_wham_usage_url: String,
    pub anthropic_oauth_usage_url: String,
    pub gemini_quota_url: String,
    pub openrouter_credits_url: String,
}

impl Default for QuotaEndpoints {
    fn default() -> Self {
        Self {
            codex_wham_usage_url: "https://chatgpt.com/backend-api/wham/usage".to_owned(),
            anthropic_oauth_usage_url: "https://api.anthropic.com/api/oauth/usage".to_owned(),
            gemini_quota_url: "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
                .to_owned(),
            openrouter_credits_url: "https://openrouter.ai/api/v1/credits".to_owned(),
        }
    }
}

pub async fn poll_and_save_quota(
    db: &Database,
    refresher: &CredentialRefresher,
    credential_id: &str,
    endpoints: &QuotaEndpoints,
) -> Result<QuotaSnapshot> {
    let (record, secret) = refresher.ensure_fresh(credential_id).await?;
    let http = refresher.http();
    let now = now_ms();

    let snapshot = match (&secret, record.provider_id.as_str()) {
        (
            CredentialSecret::Codex {
                access, account_id, ..
            },
            _,
        ) => fetch_codex_quota(http, &endpoints.codex_wham_usage_url, access, account_id, now).await,
        (CredentialSecret::AnthropicSetupToken { token }, _) => {
            fetch_claude_quota(http, &endpoints.anthropic_oauth_usage_url, token, now).await
        }
        (CredentialSecret::GeminiCli { access, .. }, _) => {
            fetch_gemini_quota(http, &endpoints.gemini_quota_url, access, now).await
        }
        (CredentialSecret::ApiKey { key }, "openrouter") => {
            let credits_url = resolve_openrouter_credits_url(
                record.base_url.as_deref(),
                &endpoints.openrouter_credits_url,
            );
            fetch_openrouter_credits(http, &credits_url, key, now).await
        }
        (CredentialSecret::OpenRouterOidc { access, .. }, _) => {
            let credits_url = resolve_openrouter_credits_url(
                record.base_url.as_deref(),
                &endpoints.openrouter_credits_url,
            );
            fetch_openrouter_credits(http, &credits_url, access, now).await
        }
        _ => Ok(QuotaSnapshot {
            windows: Vec::new(),
            plan: None,
            balance_credits: None,
            fetched_at_ms: now,
            error: None,
        }),
    }?;

    db.save_quota_snapshot(credential_id, &snapshot)?;
    Ok(snapshot)
}

async fn fetch_codex_quota(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    account_id: &str,
    now: i64,
) -> Result<QuotaSnapshot> {
    let resp = http
        .get(url)
        .bearer_auth(token)
        .header("Accept", "application/json")
        .header("ChatGPT-Account-Id", account_id)
        .header("originator", "quixigate")
        .header("User-Agent", "quixigate/0.1.0")
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Codex wham/usage failed: {e}")))?;

    if !resp.status().is_success() {
        return Ok(QuotaSnapshot {
            windows: Vec::new(),
            plan: None,
            balance_credits: None,
            fetched_at_ms: now,
            error: Some(format!("HTTP {}", resp.status())),
        });
    }
    let data: Value = resp.json().await.unwrap_or_default();
    let mut windows = Vec::new();
    let primary_reset = data["rate_limit"]["primary_window"]["reset_at"].as_i64();

    for (key, primary) in [("primary_window", true), ("secondary_window", false)] {
        let win = &data["rate_limit"][key];
        if !win.is_object() {
            continue;
        }
        let secs = win["limit_window_seconds"]
            .as_i64()
            .unwrap_or(if primary { 18_000 } else { 86_400 });
        let hours = ((secs as f64) / 3600.0).round() as i64;
        let reset_at_sec = win["reset_at"].as_i64();
        let label = if primary || hours < 24 {
            format!("{hours}h")
        } else if hours >= 168
            || (reset_at_sec.is_some()
                && primary_reset.is_some()
                && reset_at_sec.unwrap_or(0) - primary_reset.unwrap_or(0) >= WEEKLY_GAP_SECS)
        {
            "Week".to_owned()
        } else {
            "Day".to_owned()
        };
        windows.push(QuotaWindow {
            label,
            used_percent: clamp_pct(win["used_percent"].as_f64().unwrap_or(0.0)),
            reset_at_ms: reset_at_sec.map(|s| s * 1000),
        });
    }

    let balance_credits = data["credits"]["balance"]
        .as_f64()
        .or_else(|| data["credits"]["balance"].as_str()?.parse().ok());

    Ok(QuotaSnapshot {
        windows,
        plan: data["plan_type"].as_str().map(str::to_owned),
        balance_credits,
        fetched_at_ms: now,
        error: None,
    })
}

async fn fetch_claude_quota(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    now: i64,
) -> Result<QuotaSnapshot> {
    let resp = http
        .get(url)
        .bearer_auth(token)
        .header("Accept", "application/json")
        .header("User-Agent", "quixigate/0.1.0")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "oauth-2025-04-20")
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Anthropic oauth/usage failed: {e}")))?;

    if !resp.status().is_success() {
        return Ok(QuotaSnapshot {
            windows: Vec::new(),
            plan: Some("setup-token".to_owned()),
            balance_credits: None,
            fetched_at_ms: now,
            error: Some(format!("HTTP {}", resp.status())),
        });
    }
    let data: Value = resp.json().await.unwrap_or_default();
    let mut windows = Vec::new();
    for (field, label) in [
        ("five_hour", "5h"),
        ("seven_day", "Week"),
        ("seven_day_sonnet", "Sonnet"),
        ("seven_day_opus", "Opus"),
    ] {
        if let Some(util) = data[field]["utilization"].as_f64() {
            let reset_at_ms = data[field]["resets_at"]
                .as_str()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.timestamp_millis())
                .or_else(|| data[field]["resets_at"].as_i64().map(|s| s * 1000));
            windows.push(QuotaWindow {
                label: label.to_owned(),
                used_percent: clamp_pct(util),
                reset_at_ms,
            });
        }
    }
    if data["extra_usage"]["is_enabled"] == true
        && let Some(util) = data["extra_usage"]["utilization"].as_f64()
    {
        windows.push(QuotaWindow {
            label: "Extra usage".to_owned(),
            used_percent: clamp_pct(util),
            reset_at_ms: None,
        });
    }
    Ok(QuotaSnapshot {
        windows,
        plan: Some("claude-subscription".to_owned()),
        balance_credits: None,
        fetched_at_ms: now,
        error: None,
    })
}

async fn fetch_gemini_quota(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    now: i64,
) -> Result<QuotaSnapshot> {
    let resp = http
        .post(url)
        .bearer_auth(token)
        .json(&json!({}))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Gemini retrieveUserQuota failed: {e}")))?;

    if !resp.status().is_success() {
        return Ok(QuotaSnapshot {
            windows: Vec::new(),
            plan: None,
            balance_credits: None,
            fetched_at_ms: now,
            error: Some(format!("HTTP {}", resp.status())),
        });
    }
    let data: Value = resp.json().await.unwrap_or_default();
    let mut pro_rem = 1.0f64;
    let mut pro_found = false;
    let mut flash_rem = 1.0f64;
    let mut flash_found = false;

    for bucket in data["buckets"].as_array().into_iter().flatten() {
        let model_id = bucket["modelId"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let frac = bucket["remainingFraction"].as_f64().unwrap_or(1.0);
        if model_id.contains("pro") {
            pro_found = true;
            pro_rem = pro_rem.min(frac);
        }
        if model_id.contains("flash") {
            flash_found = true;
            flash_rem = flash_rem.min(frac);
        }
    }
    let mut windows = Vec::new();
    if pro_found {
        windows.push(QuotaWindow {
            label: "Pro".to_owned(),
            used_percent: clamp_pct((1.0 - pro_rem) * 100.0),
            reset_at_ms: None,
        });
    }
    if flash_found {
        windows.push(QuotaWindow {
            label: "Flash".to_owned(),
            used_percent: clamp_pct((1.0 - flash_rem) * 100.0),
            reset_at_ms: None,
        });
    }
    Ok(QuotaSnapshot {
        windows,
        plan: Some("code-assist".to_owned()),
        balance_credits: None,
        fetched_at_ms: now,
        error: None,
    })
}

fn resolve_openrouter_credits_url(base_url: Option<&str>, configured_url: &str) -> String {
    if configured_url != "https://openrouter.ai/api/v1/credits" {
        return configured_url.to_owned();
    }
    if let Some(base) = base_url.filter(|b| !b.trim().is_empty()) {
        return format!("{}/credits", base.trim_end_matches('/'));
    }
    configured_url.to_owned()
}

async fn fetch_openrouter_credits(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    now: i64,
) -> Result<QuotaSnapshot> {
    let resp = http
        .get(url)
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("OpenRouter credits failed: {e}")))?;
    if !resp.status().is_success() {
        return Ok(QuotaSnapshot {
            windows: Vec::new(),
            plan: None,
            balance_credits: None,
            fetched_at_ms: now,
            error: Some(format!("HTTP {}", resp.status())),
        });
    }
    let data: Value = resp.json().await.unwrap_or_default();
    let total = data["data"]["total_credits"].as_f64().unwrap_or(0.0);
    let used = data["data"]["total_usage"].as_f64().unwrap_or(0.0);
    let used_pct = if total > 0.0 {
        clamp_pct((used / total) * 100.0)
    } else {
        0.0
    };
    let mut windows = vec![QuotaWindow {
        label: "Credits".to_owned(),
        used_percent: used_pct,
        reset_at_ms: None,
    }];
    let mut plan = "openrouter".to_owned();

    if let Some(base) = url.strip_suffix("/credits")
        && let Ok(key_resp) = http
            .get(format!("{base}/key"))
            .bearer_auth(api_key)
            .send()
            .await
        && key_resp.status().is_success()
        && let Ok(kbody) = key_resp.json::<Value>().await
    {
        let kd = &kbody["data"];
        if kd["is_free_tier"].as_bool() == Some(true) {
            plan = "openrouter-free".to_owned();
        } else {
            plan = "openrouter-paid".to_owned();
        }
        if let Some(limit) = kd["limit"].as_f64().filter(|&l| l > 0.0) {
            let key_used = kd["usage"]
                .as_f64()
                .or_else(|| kd["limit_remaining"].as_f64().map(|rem| (limit - rem).max(0.0)))
                .unwrap_or(0.0);
            windows.push(QuotaWindow {
                label: "Key Limit".to_owned(),
                used_percent: clamp_pct((key_used / limit) * 100.0),
                reset_at_ms: None,
            });
        }
    }

    Ok(QuotaSnapshot {
        windows,
        plan: Some(plan),
        balance_credits: Some((total - used).max(0.0)),
        fetched_at_ms: now,
        error: None,
    })
}

fn clamp_pct(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 100.0)
    } else {
        0.0
    }
}
