//! Single-flight token refresh coalescing with 30-second post-refresh replay
//! cache and write-before-return SQLite persistence.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use quixigate_core::{CredentialRecord, CredentialSecret, QuixigateError, Result, now_ms};
use quixigate_db::Database;
use tokio::sync::Mutex as AsyncMutex;

use crate::providers;

const DEFAULT_REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;
const OPENROUTER_OIDC_REFRESH_SKEW_MS: i64 = 2 * 60 * 1000;
const XAI_REFRESH_SKEW_MS: i64 = 60 * 60 * 1000;
const REPLAY_CACHE_TTL: Duration = Duration::from_secs(30);

const TERMINAL_AUTH_MARKERS: &[&str] = &[
    "token_invalidated",
    "token_revoked",
    "invalid_token",
    "invalid_grant",
    "unauthorized_client",
    "refresh_token_reused",
];

#[derive(Clone)]
pub struct RefreshEndpoints {
    pub codex_token_url: String,
    pub google_token_url: String,
    pub qwen_token_url: String,
    pub chutes_token_url: String,
    pub copilot_domain_override: Option<String>,
    pub allow_loopback_xai: bool,
}

impl Default for RefreshEndpoints {
    fn default() -> Self {
        Self {
            codex_token_url: providers::CODEX_TOKEN_URL.to_owned(),
            google_token_url: providers::GOOGLE_TOKEN_URL.to_owned(),
            qwen_token_url: providers::QWEN_TOKEN_URL.to_owned(),
            chutes_token_url: providers::CHUTES_TOKEN_URL.to_owned(),
            copilot_domain_override: None,
            allow_loopback_xai: false,
        }
    }
}

#[derive(Clone)]
pub struct CredentialRefresher {
    db: Database,
    http: reqwest::Client,
    flights: Arc<DashMap<String, Arc<AsyncMutex<()>>>>,
    replay_cache: Arc<DashMap<String, (Instant, CredentialRecord, CredentialSecret)>>,
    endpoints: RefreshEndpoints,
}

impl CredentialRefresher {
    pub fn new(db: Database) -> Self {
        Self::with_endpoints(db, RefreshEndpoints::default())
    }

    pub fn with_endpoints(db: Database, endpoints: RefreshEndpoints) -> Self {
        Self {
            db,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            flights: Arc::new(DashMap::new()),
            replay_cache: Arc::new(DashMap::new()),
            endpoints,
        }
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Returns a guaranteed-fresh `(CredentialRecord, CredentialSecret)` for `credential_id`.
    /// Concurrent callers for the same expiring OAuth credential coalesce onto a single upstream
    /// refresh POST and receive the persisted rotated credential.
    pub async fn ensure_fresh(
        &self,
        credential_id: &str,
    ) -> Result<(CredentialRecord, CredentialSecret)> {
        let (record, secret) = self.db.get_credential_with_secret(credential_id)?;
        if !needs_refresh(&secret) {
            return Ok((record, secret));
        }

        let lock = self
            .flights
            .entry(credential_id.to_owned())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone();

        let _guard = lock.lock().await;

        // Check post-refresh replay cache inside single-flight lock!
        if let Some(cached) = self.replay_cache.get(credential_id)
            && cached.0 > Instant::now()
        {
            return Ok((cached.1.clone(), cached.2.clone()));
        }

        // Re-read from SQLite in case a sibling updated before cache prune.
        let (record, secret) = self.db.get_credential_with_secret(credential_id)?;
        if !needs_refresh(&secret) {
            return Ok((record, secret));
        }

        let refreshed_res = self.perform_refresh(&secret).await;
        match refreshed_res {
            Ok((renewed, new_base_url)) => {
                // Write-before-return: never return a rotated single-use token unless committed to SQLite!
                if let Err(persist_err) = self.db.update_credential_secret(
                    credential_id,
                    &renewed,
                    new_base_url.as_deref(),
                ) {
                    let _ = self
                        .db
                        .mark_credential_dead(credential_id, "credential_persist_failed");
                    return Err(persist_err);
                }
                let updated_record = self.db.get_credential(credential_id)?;
                self.replay_cache.insert(
                    credential_id.to_owned(),
                    (
                        Instant::now() + REPLAY_CACHE_TTL,
                        updated_record.clone(),
                        renewed.clone(),
                    ),
                );
                Ok((updated_record, renewed))
            }
            Err(err) => {
                let msg = err.to_string();
                let lower = msg.to_ascii_lowercase();
                if TERMINAL_AUTH_MARKERS.iter().any(|m| lower.contains(m)) {
                    let _ = self.db.mark_credential_dead(credential_id, &msg);
                }
                Err(err)
            }
        }
    }

    async fn perform_refresh(
        &self,
        secret: &CredentialSecret,
    ) -> Result<(CredentialSecret, Option<String>)> {
        match secret {
            CredentialSecret::Codex {
                refresh,
                account_id,
                ..
            } => {
                let renewed = providers::refresh_codex_token(
                    &self.http,
                    &self.endpoints.codex_token_url,
                    refresh,
                    account_id,
                )
                .await?;
                Ok((renewed, None))
            }
            CredentialSecret::Xai {
                refresh,
                token_endpoint,
                ..
            } => {
                let renewed = providers::refresh_xai_token(
                    &self.http,
                    token_endpoint,
                    refresh,
                    self.endpoints.allow_loopback_xai,
                )
                .await?;
                Ok((renewed, None))
            }
            CredentialSecret::GeminiCli {
                refresh,
                project_id,
                email,
                client_id,
                client_secret,
                ..
            } => {
                let renewed = providers::refresh_gemini_token(
                    &self.http,
                    &self.endpoints.google_token_url,
                    refresh,
                    project_id,
                    email.as_deref(),
                    client_id,
                    client_secret,
                )
                .await?;
                Ok((renewed, None))
            }
            CredentialSecret::GitHubCopilot {
                github_token,
                enterprise_domain,
                ..
            } => {
                let exchange_url =
                    if let Some(override_url) = &self.endpoints.copilot_domain_override {
                        format!(
                            "{}/copilot_internal/v2/token",
                            override_url.trim_end_matches('/')
                        )
                    } else {
                        let dom = enterprise_domain.as_deref().unwrap_or("github.com");
                        format!("https://api.{dom}/copilot_internal/v2/token")
                    };
                let renewed = providers::exchange_github_for_copilot(
                    &self.http,
                    &exchange_url,
                    github_token,
                    enterprise_domain.as_deref(),
                )
                .await?;
                let next_base = match &renewed {
                    CredentialSecret::GitHubCopilot { base_url, .. } => Some(base_url.clone()),
                    _ => None,
                };
                Ok((renewed, next_base))
            }
            CredentialSecret::QwenPortal {
                refresh,
                resource_url,
                ..
            } => {
                let renewed = providers::refresh_qwen_token(
                    &self.http,
                    &self.endpoints.qwen_token_url,
                    refresh,
                    resource_url.as_deref(),
                )
                .await?;
                let next_base = match &renewed {
                    CredentialSecret::QwenPortal { resource_url, .. } => resource_url.clone(),
                    _ => None,
                };
                Ok((renewed, next_base))
            }
            CredentialSecret::Chutes {
                refresh,
                email,
                account_id,
                client_id,
                client_secret,
                ..
            } => {
                let renewed = providers::refresh_chutes_token(
                    &self.http,
                    &self.endpoints.chutes_token_url,
                    refresh,
                    email.as_deref(),
                    account_id.as_deref(),
                    client_id,
                    client_secret.as_deref(),
                )
                .await?;
                Ok((renewed, None))
            }
            CredentialSecret::OpenRouterOidc {
                federation_policy_id,
                subject_token,
                subject_token_env,
                token_endpoint,
                ..
            } => {
                let fresh_subject = subject_token_env
                    .as_deref()
                    .and_then(|var| std::env::var(var).ok())
                    .filter(|v| !v.trim().is_empty())
                    .or_else(|| subject_token.clone().filter(|v| !v.trim().is_empty()))
                    .ok_or_else(|| {
                        QuixigateError::Unauthorized(
                            "OpenRouter OIDC token expired and no fresh subject_token or env var is available; sign in again".to_owned(),
                        )
                    })?;
                let renewed = providers::exchange_openrouter_oidc_token(
                    &self.http,
                    token_endpoint,
                    &fresh_subject,
                    federation_policy_id.as_deref(),
                    subject_token_env.as_deref(),
                )
                .await?;
                Ok((renewed, None))
            }
            CredentialSecret::MiniMaxPortal { .. } => Err(QuixigateError::Unauthorized(
                "MiniMax Portal token has expired; sign in again in QuixiGate".to_owned(),
            )),
            CredentialSecret::ApiKey { .. }
            | CredentialSecret::AnthropicSetupToken { .. }
            | CredentialSecret::LocalNoauth => Ok((secret.clone(), None)),
        }
    }
}

fn needs_refresh(secret: &CredentialSecret) -> bool {
    let now = now_ms();
    match secret {
        CredentialSecret::Xai { expires_ms, .. } => *expires_ms <= now + XAI_REFRESH_SKEW_MS,
        CredentialSecret::OpenRouterOidc { expires_ms, .. } => {
            *expires_ms <= now + OPENROUTER_OIDC_REFRESH_SKEW_MS
        }
        CredentialSecret::Codex { expires_ms, .. }
        | CredentialSecret::GeminiCli { expires_ms, .. }
        | CredentialSecret::GitHubCopilot { expires_ms, .. }
        | CredentialSecret::QwenPortal { expires_ms, .. }
        | CredentialSecret::MiniMaxPortal { expires_ms, .. }
        | CredentialSecret::Chutes { expires_ms, .. } => {
            *expires_ms <= now + DEFAULT_REFRESH_SKEW_MS
        }
        CredentialSecret::ApiKey { .. }
        | CredentialSecret::AnthropicSetupToken { .. }
        | CredentialSecret::LocalNoauth => false,
    }
}
