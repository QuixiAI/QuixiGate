//! OAuth wizard session manager (for both Web UI modals and CLI login),
//! single-flight token refresher, and live subscription quota pollers.

pub mod flow;
pub mod providers;
pub mod quota;
pub mod refresh;

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use quixigate_core::{
    AuthKind, CredentialRecord, QuixigateError, Result, now_ms,
};
use quixigate_db::Database;
use providers::{OAUTH_PROVIDER_CATALOG, OAuthProviderDescriptor, PollOutcome};
use quota::QuotaEndpoints;
use refresh::CredentialRefresher;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WizardStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthWizardSession {
    pub session_id: String,
    pub provider_id: String,
    pub label: String,
    pub status: WizardStatus,
    pub authorize_url: Option<String>,
    pub verification_uri: Option<String>,
    pub user_code: Option<String>,
    pub accepts_manual_code: bool,
    pub credential: Option<CredentialRecord>,
    pub error: Option<String>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StartOAuthRequest {
    pub provider_id: String,
    pub label: Option<String>,
    pub priority: Option<i32>,
    pub prefer_device_code: Option<bool>,
    pub setup_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub region: Option<String>,
    pub enterprise_domain: Option<String>,
}

struct PendingPkce {
    provider_id: String,
    label: String,
    priority: i32,
    verifier: String,
    state: Option<String>,
    redirect_uri: String,
    client_id: Option<String>,
    client_secret: Option<String>,
    region: Option<String>,
}

#[derive(Clone)]
pub struct OAuthWizardManager {
    db: Database,
    refresher: CredentialRefresher,
    quota_endpoints: QuotaEndpoints,
    sessions: Arc<DashMap<String, OAuthWizardSession>>,
    pkce_states: Arc<DashMap<String, PendingPkce>>,
}

impl OAuthWizardManager {
    pub fn new(db: Database, refresher: CredentialRefresher) -> Self {
        Self {
            db,
            refresher,
            quota_endpoints: QuotaEndpoints::default(),
            sessions: Arc::new(DashMap::new()),
            pkce_states: Arc::new(DashMap::new()),
        }
    }

    pub fn with_quota_endpoints(mut self, quota_endpoints: QuotaEndpoints) -> Self {
        self.quota_endpoints = quota_endpoints;
        self
    }

    pub fn catalog(&self) -> &'static [OAuthProviderDescriptor] {
        OAUTH_PROVIDER_CATALOG
    }

    pub fn get_session(&self, session_id: &str) -> Result<OAuthWizardSession> {
        self.sessions
            .get(session_id)
            .map(|s| s.clone())
            .ok_or_else(|| QuixigateError::NotFound(format!("OAuth session '{session_id}' not found")))
    }

    /// Starts any enabled OAuth / PKCE flow for the Web UI or CLI.
    pub async fn start_flow(&self, req: StartOAuthRequest) -> Result<OAuthWizardSession> {
        let provider_id = req.provider_id.trim().to_owned();
        #[cfg(not(feature = "subscription-oauth"))]
        if provider_id != "openrouter" {
            return Err(QuixigateError::Validation(format!(
                "Consumer subscription OAuth ('{provider_id}') is disabled in this build (compiled without `subscription-oauth` feature)"
            )));
        }
        let default_prio = if provider_id == "openrouter" { 200 } else { 50 };
        let priority = req.priority.unwrap_or(default_prio);
        let label = req
            .label
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| format!("{provider_id} seat"));
        let session_id = format!("oas_{}", Uuid::new_v4().simple());
        let http = self.refresher.http().clone();

        // Instant synchronous flow: Anthropic setup token.
        if provider_id == "anthropic-token" {
            let raw = req.setup_token.as_deref().unwrap_or_default();
            let secret = providers::validate_anthropic_setup_token(raw)?;
            let cred = self.db.upsert_credential(
                None,
                "anthropic-token",
                &label,
                AuthKind::SetupToken,
                Some(providers::ANTHROPIC_DEFAULT_BASE_URL),
                priority,
                true,
                true,
                &secret,
            )?;
            let session = OAuthWizardSession {
                session_id: session_id.clone(),
                provider_id,
                label,
                status: WizardStatus::Completed,
                authorize_url: None,
                verification_uri: None,
                user_code: None,
                accepts_manual_code: false,
                credential: Some(cred),
                error: None,
                created_at_ms: now_ms(),
            };
            self.sessions.insert(session_id, session.clone());
            return Ok(session);
        }

        match provider_id.as_str() {
            "openai-codex" => {
                let use_device = req.prefer_device_code.unwrap_or(false);
                if !use_device
                    && let Ok(listener) =
                        TcpListener::bind(("127.0.0.1", 1455)).await
                {
                    let verifier = flow::random_token();
                    let state = flow::random_token();
                    let challenge = flow::pkce_challenge(&verifier);
                    let auth_url = providers::build_codex_authorize_url(
                        providers::CODEX_AUTHORIZE_URL,
                        providers::CODEX_REDIRECT_URI,
                        &challenge,
                        &state,
                    )?;
                    let session = OAuthWizardSession {
                        session_id: session_id.clone(),
                        provider_id: provider_id.clone(),
                        label: label.clone(),
                        status: WizardStatus::Pending,
                        authorize_url: Some(auth_url),
                        verification_uri: None,
                        user_code: None,
                        accepts_manual_code: true,
                        credential: None,
                        error: None,
                        created_at_ms: now_ms(),
                    };
                    self.sessions.insert(session_id.clone(), session.clone());
                    self.pkce_states.insert(
                        session_id.clone(),
                        PendingPkce {
                            provider_id: provider_id.clone(),
                            label: label.clone(),
                            priority,
                            verifier: verifier.clone(),
                            state: Some(state.clone()),
                            redirect_uri: providers::CODEX_REDIRECT_URI.to_owned(),
                            client_id: None,
                            client_secret: None,
                            region: None,
                        },
                    );
                    let mgr = self.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        if let Ok(code) = flow::wait_for_callback(
                            listener,
                            "/auth/callback",
                            Some(&state),
                            Duration::from_secs(300),
                        )
                        .await
                        {
                            let _ = mgr.complete_manual_code(&sid, &code).await;
                        }
                    });
                    return Ok(session);
                }

                // Device code path (default fallback if port 1455 busy or explicitly requested).
                let (device_auth_id, user_code, interval) =
                    providers::start_codex_device_flow(&http, providers::CODEX_DEVICE_USERCODE_URL)
                        .await?;
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id: provider_id.clone(),
                    label: label.clone(),
                    status: WizardStatus::Pending,
                    authorize_url: Some(providers::CODEX_DEVICE_VERIFY_URL.to_owned()),
                    verification_uri: Some(providers::CODEX_DEVICE_VERIFY_URL.to_owned()),
                    user_code: Some(user_code.clone()),
                    accepts_manual_code: false,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id.clone(), session.clone());
                let mgr = self.clone();
                let sid = session_id.clone();
                tokio::spawn(async move {
                    mgr.poll_device_loop(
                        &sid,
                        "openai-codex",
                        &label,
                        AuthKind::Oauth,
                        priority,
                        interval,
                        900,
                        move |client| {
                            let auth_id = device_auth_id.clone();
                            let ucode = user_code.clone();
                            async move {
                                providers::poll_codex_device_once(
                                    &client,
                                    providers::CODEX_DEVICE_TOKEN_URL,
                                    providers::CODEX_TOKEN_URL,
                                    &auth_id,
                                    &ucode,
                                )
                                .await
                            }
                        },
                    )
                    .await;
                });
                Ok(session)
            }

            "xai-oauth" => {
                let start = providers::start_xai_device_flow(
                    &http,
                    providers::XAI_DISCOVERY_URL,
                    false,
                )
                .await?;
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id: provider_id.clone(),
                    label: label.clone(),
                    status: WizardStatus::Pending,
                    authorize_url: Some(start.verification_url.clone()),
                    verification_uri: Some(start.verification_url.clone()),
                    user_code: Some(start.user_code.clone()),
                    accepts_manual_code: false,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id.clone(), session.clone());
                let mgr = self.clone();
                let sid = session_id.clone();
                tokio::spawn(async move {
                    let tok_ep = start.token_endpoint;
                    let dcode = start.device_code;
                    mgr.poll_device_loop(
                        &sid,
                        "xai-oauth",
                        &label,
                        AuthKind::Oauth,
                        priority,
                        start.interval_secs,
                        start.expires_in_secs,
                        move |client| {
                            let ep = tok_ep.clone();
                            let dc = dcode.clone();
                            async move { providers::poll_xai_device_once(&client, &ep, &dc).await }
                        },
                    )
                    .await;
                });
                Ok(session)
            }

            "github-copilot" => {
                let domain = req
                    .enterprise_domain
                    .clone()
                    .filter(|d| !d.trim().is_empty())
                    .unwrap_or_else(|| "github.com".to_owned());
                let start = providers::start_copilot_device_flow(
                    &http,
                    &format!("https://{domain}/login/device/code"),
                    &domain,
                )
                .await?;
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id: provider_id.clone(),
                    label: label.clone(),
                    status: WizardStatus::Pending,
                    authorize_url: Some(start.verification_uri.clone()),
                    verification_uri: Some(start.verification_uri.clone()),
                    user_code: Some(start.user_code.clone()),
                    accepts_manual_code: false,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id.clone(), session.clone());
                let mgr = self.clone();
                let sid = session_id.clone();
                let ent = req.enterprise_domain.clone();
                tokio::spawn(async move {
                    let gh_tok_url = format!("https://{}/login/oauth/access_token", start.domain);
                    let cop_url = format!("https://api.{}/copilot_internal/v2/token", start.domain);
                    let dc = start.device_code;
                    mgr.poll_device_loop(
                        &sid,
                        "github-copilot",
                        &label,
                        AuthKind::Oauth,
                        priority,
                        start.interval_secs,
                        start.expires_in_secs,
                        move |client| {
                            let g = gh_tok_url.clone();
                            let c = cop_url.clone();
                            let d = dc.clone();
                            let e = ent.clone();
                            async move {
                                providers::poll_copilot_device_once(&client, &g, &c, &d, e.as_deref())
                                    .await
                            }
                        },
                    )
                    .await;
                });
                Ok(session)
            }

            "google-gemini-cli" => {
                let (cid, csec) = providers::resolve_gemini_client_credentials(
                    req.client_id.as_deref(),
                    req.client_secret.as_deref(),
                )?;
                let verifier = flow::random_token();
                let state = flow::random_token();
                let challenge = flow::pkce_challenge(&verifier);
                let auth_url = providers::build_gemini_authorize_url(
                    providers::GOOGLE_AUTH_URL,
                    providers::GEMINI_CLI_REDIRECT_URI,
                    &cid,
                    &challenge,
                    &state,
                )?;
                self.pkce_states.insert(
                    session_id.clone(),
                    PendingPkce {
                        provider_id: provider_id.clone(),
                        label: label.clone(),
                        priority,
                        verifier,
                        state: Some(state.clone()),
                        redirect_uri: providers::GEMINI_CLI_REDIRECT_URI.to_owned(),
                        client_id: Some(cid),
                        client_secret: Some(csec),
                        region: None,
                    },
                );
                if let Ok(listener) = TcpListener::bind(("127.0.0.1", 8085)).await {
                    let mgr = self.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        if let Ok(code) = flow::wait_for_callback(
                            listener,
                            "/oauth2callback",
                            Some(&state),
                            Duration::from_secs(300),
                        )
                        .await
                        {
                            let _ = mgr.complete_manual_code(&sid, &code).await;
                        }
                    });
                }
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id,
                    label,
                    status: WizardStatus::Pending,
                    authorize_url: Some(auth_url),
                    verification_uri: None,
                    user_code: None,
                    accepts_manual_code: true,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id, session.clone());
                Ok(session)
            }

            "openrouter" => {
                let verifier = flow::random_token();
                let challenge = flow::pkce_challenge(&verifier);
                let nonce = flow::random_token();
                let path = format!("/callback/{nonce}");
                let listener = TcpListener::bind(("127.0.0.1", 0)).await.ok();
                let redirect_uri = listener
                    .as_ref()
                    .and_then(|l| l.local_addr().ok())
                    .map(|addr| format!("http://127.0.0.1:{}{path}", addr.port()));
                let auth_url = providers::build_openrouter_authorize_url(
                    providers::OPENROUTER_AUTH_URL,
                    redirect_uri.as_deref(),
                    &challenge,
                )?;
                self.pkce_states.insert(
                    session_id.clone(),
                    PendingPkce {
                        provider_id: provider_id.clone(),
                        label: label.clone(),
                        priority,
                        verifier,
                        state: None,
                        redirect_uri: redirect_uri.unwrap_or_default(),
                        client_id: None,
                        client_secret: None,
                        region: req.region.clone(),
                    },
                );
                if let Some(listener) = listener {
                    let mgr = self.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        if let Ok(code) =
                            flow::wait_for_callback(listener, &path, None, Duration::from_secs(300))
                                .await
                        {
                            let _ = mgr.complete_manual_code(&sid, &code).await;
                        }
                    });
                }
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id,
                    label,
                    status: WizardStatus::Pending,
                    authorize_url: Some(auth_url),
                    verification_uri: None,
                    user_code: None,
                    accepts_manual_code: true,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id, session.clone());
                Ok(session)
            }

            "qwen-portal" => {
                let start =
                    providers::start_qwen_device_flow(&http, providers::QWEN_DEVICE_CODE_URL)
                        .await?;
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id: provider_id.clone(),
                    label: label.clone(),
                    status: WizardStatus::Pending,
                    authorize_url: Some(start.verification_url.clone()),
                    verification_uri: Some(start.verification_url.clone()),
                    user_code: Some(start.user_code.clone()),
                    accepts_manual_code: false,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id.clone(), session.clone());
                let mgr = self.clone();
                let sid = session_id.clone();
                tokio::spawn(async move {
                    let dc = start.device_code;
                    let ver = start.verifier;
                    mgr.poll_device_loop(
                        &sid,
                        "qwen-portal",
                        &label,
                        AuthKind::Oauth,
                        priority,
                        start.interval_secs,
                        start.expires_in_secs,
                        move |client| {
                            let d = dc.clone();
                            let v = ver.clone();
                            async move {
                                providers::poll_qwen_device_once(
                                    &client,
                                    providers::QWEN_TOKEN_URL,
                                    &d,
                                    &v,
                                )
                                .await
                            }
                        },
                    )
                    .await;
                });
                Ok(session)
            }

            "minimax-portal" => {
                let region = req.region.unwrap_or_else(|| "global".to_owned());
                let base = providers::minimax_auth_base(&region);
                let start = providers::start_minimax_flow(&http, base, &region).await?;
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id: provider_id.clone(),
                    label: label.clone(),
                    status: WizardStatus::Pending,
                    authorize_url: Some(start.verification_uri.clone()),
                    verification_uri: Some(start.verification_uri.clone()),
                    user_code: Some(start.user_code.clone()),
                    accepts_manual_code: false,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id.clone(), session.clone());
                let mgr = self.clone();
                let sid = session_id.clone();
                tokio::spawn(async move {
                    let uc = start.user_code;
                    let ver = start.verifier;
                    let reg = start.region;
                    let ab = start.auth_base;
                    mgr.poll_device_loop(
                        &sid,
                        "minimax-portal",
                        &label,
                        AuthKind::Oauth,
                        priority,
                        start.interval_secs,
                        start.expires_in_secs,
                        move |client| {
                            let u = uc.clone();
                            let v = ver.clone();
                            let r = reg.clone();
                            let b = ab.clone();
                            async move { providers::poll_minimax_once(&client, &b, &r, &u, &v).await }
                        },
                    )
                    .await;
                });
                Ok(session)
            }

            "chutes" => {
                let cid = req
                    .client_id
                    .clone()
                    .or_else(|| std::env::var("CHUTES_CLIENT_ID").ok())
                    .unwrap_or_else(|| "quixigate".to_owned());
                let verifier = flow::random_token();
                let state = flow::random_token();
                let challenge = flow::pkce_challenge(&verifier);
                let listener = TcpListener::bind(("127.0.0.1", 0)).await.ok();
                let redirect_uri = listener
                    .as_ref()
                    .and_then(|l| l.local_addr().ok())
                    .map(|a| format!("http://127.0.0.1:{}/callback", a.port()))
                    .unwrap_or_else(|| "http://127.0.0.1:1456/callback".to_owned());
                let auth_url = providers::build_chutes_authorize_url(
                    providers::CHUTES_AUTHORIZE_URL,
                    &redirect_uri,
                    &cid,
                    &challenge,
                    &state,
                )?;
                self.pkce_states.insert(
                    session_id.clone(),
                    PendingPkce {
                        provider_id: provider_id.clone(),
                        label: label.clone(),
                        priority,
                        verifier,
                        state: Some(state.clone()),
                        redirect_uri,
                        client_id: Some(cid),
                        client_secret: req.client_secret,
                        region: None,
                    },
                );
                if let Some(listener) = listener {
                    let mgr = self.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        if let Ok(code) = flow::wait_for_callback(
                            listener,
                            "/callback",
                            Some(&state),
                            Duration::from_secs(300),
                        )
                        .await
                        {
                            let _ = mgr.complete_manual_code(&sid, &code).await;
                        }
                    });
                }
                let session = OAuthWizardSession {
                    session_id: session_id.clone(),
                    provider_id,
                    label,
                    status: WizardStatus::Pending,
                    authorize_url: Some(auth_url),
                    verification_uri: None,
                    user_code: None,
                    accepts_manual_code: true,
                    credential: None,
                    error: None,
                    created_at_ms: now_ms(),
                };
                self.sessions.insert(session_id, session.clone());
                Ok(session)
            }

            other => Err(QuixigateError::Validation(format!(
                "unsupported OAuth provider '{other}'"
            ))),
        }
    }

    /// Completes a PKCE flow using either the loopback callback `code` or a redirect URL pasted by the user.
    pub async fn complete_manual_code(
        &self,
        session_id: &str,
        raw_code_or_url: &str,
    ) -> Result<OAuthWizardSession> {
        let Some((_, pending)) = self.pkce_states.remove(session_id) else {
            return self.get_session(session_id);
        };
        let http = self.refresher.http();
        let code = match flow::extract_code_from_input(raw_code_or_url, pending.state.as_deref()) {
            Ok(c) => c,
            Err(err) => {
                self.mark_failed(session_id, &err.to_string());
                return Err(err);
            }
        };

        let exchanged = match pending.provider_id.as_str() {
            "openai-codex" => providers::exchange_codex_code(
                http,
                providers::CODEX_TOKEN_URL,
                &code,
                &pending.verifier,
                &pending.redirect_uri,
            )
            .await
            .map(|s| (s, AuthKind::Oauth, Some(providers::CODEX_DEFAULT_BASE_URL.to_owned()))),

            "google-gemini-cli" => providers::complete_gemini_login(
                http,
                providers::GOOGLE_TOKEN_URL,
                providers::GOOGLE_USERINFO_URL,
                providers::GOOGLE_CODE_ASSIST_ENDPOINT,
                &pending.redirect_uri,
                &code,
                &pending.verifier,
                pending.client_id.as_deref().unwrap_or_default(),
                pending.client_secret.as_deref().unwrap_or_default(),
            )
            .await
            .map(|s| {
                (
                    s,
                    AuthKind::Oauth,
                    Some(providers::GOOGLE_CODE_ASSIST_ENDPOINT.to_owned()),
                )
            }),

            "openrouter" => providers::exchange_openrouter_code(
                http,
                providers::OPENROUTER_KEYS_URL,
                &code,
                &pending.verifier,
            )
            .await
            .map(|s| {
                let base = providers::openrouter_base_url_for_region(pending.region.as_deref());
                (s, AuthKind::ApiKey, Some(base.to_owned()))
            }),

            "chutes" => providers::exchange_chutes_code(
                http,
                providers::CHUTES_TOKEN_URL,
                providers::CHUTES_USERINFO_URL,
                &code,
                &pending.verifier,
                pending.client_id.as_deref().unwrap_or("quixigate"),
                pending.client_secret.as_deref(),
                &pending.redirect_uri,
            )
            .await
            .map(|s| {
                (
                    s,
                    AuthKind::Oauth,
                    Some(providers::CHUTES_DEFAULT_BASE_URL.to_owned()),
                )
            }),

            other => Err(QuixigateError::Validation(format!("unknown PKCE provider '{other}'"))),
        };

        match exchanged {
            Ok((secret, kind, base_url)) => {
                let cred = self.db.upsert_credential(
                    None,
                    &pending.provider_id,
                    &pending.label,
                    kind,
                    base_url.as_deref(),
                    pending.priority,
                    true,
                    true,
                    &secret,
                )?;
                let _ = quota::poll_and_save_quota(
                    &self.db,
                    &self.refresher,
                    &cred.id,
                    &self.quota_endpoints,
                )
                .await;
                let fresh_cred = self.db.get_credential(&cred.id).unwrap_or(cred);
                if let Some(mut sess) = self.sessions.get_mut(session_id) {
                    sess.status = WizardStatus::Completed;
                    sess.credential = Some(fresh_cred);
                }
                self.get_session(session_id)
            }
            Err(err) => {
                self.mark_failed(session_id, &err.to_string());
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn poll_device_loop<F, Fut>(
        &self,
        session_id: &str,
        provider_id: &str,
        label: &str,
        auth_kind: AuthKind,
        priority: i32,
        mut interval_secs: u64,
        expires_in_secs: u64,
        mut poll_fn: F,
    ) where
        F: FnMut(reqwest::Client) -> Fut,
        Fut: std::future::Future<Output = Result<PollOutcome>>,
    {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(expires_in_secs);
        let http = self.refresher.http().clone();

        while tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_secs(interval_secs.max(1))).await;
            match poll_fn(http.clone()).await {
                Ok(PollOutcome::Pending) => continue,
                Ok(PollOutcome::SlowDown) => {
                    interval_secs = (interval_secs + 3).min(15);
                }
                Ok(PollOutcome::Authorized { secret, base_url }) => {
                    match self.db.upsert_credential(
                        None,
                        provider_id,
                        label,
                        auth_kind,
                        base_url.as_deref(),
                        priority,
                        true,
                        true,
                        &secret,
                    ) {
                        Ok(cred) => {
                            let _ = quota::poll_and_save_quota(
                                &self.db,
                                &self.refresher,
                                &cred.id,
                                &self.quota_endpoints,
                            )
                            .await;
                            let fresh_cred = self.db.get_credential(&cred.id).unwrap_or(cred);
                            if let Some(mut sess) = self.sessions.get_mut(session_id) {
                                sess.status = WizardStatus::Completed;
                                sess.credential = Some(fresh_cred);
                            }
                        }
                        Err(e) => self.mark_failed(session_id, &e.to_string()),
                    }
                    return;
                }
                Err(err) => {
                    self.mark_failed(session_id, &err.to_string());
                    return;
                }
            }
        }
        self.mark_failed(session_id, "device authorization timed out");
    }

    fn mark_failed(&self, session_id: &str, reason: &str) {
        if let Some(mut sess) = self.sessions.get_mut(session_id) {
            sess.status = WizardStatus::Failed;
            sess.error = Some(reason.to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::extract::State;
    use axum::routing::{get, post};
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use quixigate_core::CredentialSecret;
    use serde_json::json;

    use super::refresh::{CredentialRefresher, RefreshEndpoints};
    use super::*;

    fn fake_codex_jwt(account_id: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        let payload = URL_SAFE_NO_PAD.encode(
            json!({
                "https://api.openai.com/auth": { "chatgpt_account_id": account_id }
            })
            .to_string()
            .as_bytes(),
        );
        format!("{header}.{payload}.sig")
    }

    #[tokio::test]
    async fn single_flight_coalesces_20_concurrent_rotating_token_refreshes() {
        let hits = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/oauth/token",
                post(move |State(counter): State<Arc<AtomicUsize>>| async move {
                        let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        tokio::time::sleep(Duration::from_millis(40)).await;
                        axum::Json(json!({
                            "access_token": fake_codex_jwt(&format!("acct_{n}")),
                            "refresh_token": format!("rotated_rt_{n}"),
                            "expires_in": 3600
                        }))
                }),
            )
            .route(
                "/wham/usage",
                get(|| async {
                    axum::Json(json!({
                        "plan_type": "pro",
                        "rate_limit": {
                            "primary_window": { "limit_window_seconds": 18000, "used_percent": 42.5, "reset_at": 1999999999i64 }
                        },
                        "credits": { "balance": 120.0 }
                    }))
                }),
            )
            .with_state(Arc::clone(&hits));

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let db = Database::open_in_memory().unwrap();
        // Store an already-expiring Codex credential.
        let expiring = CredentialSecret::Codex {
            access: fake_codex_jwt("acct_0"),
            refresh: "initial_single_use_rt".to_owned(),
            expires_ms: now_ms() + 1_000, // inside 5-min refresh skew!
            account_id: "acct_0".to_owned(),
        };
        let cred = db
            .upsert_credential(
                None,
                "openai-codex",
                "Seat #1",
                AuthKind::Oauth,
                None,
                10,
                true,
                true,
                &expiring,
            )
            .unwrap();

        let refresher = CredentialRefresher::with_endpoints(
            db.clone(),
            RefreshEndpoints {
                codex_token_url: format!("http://{addr}/oauth/token"),
                ..RefreshEndpoints::default()
            },
        );

        // Fire 20 concurrent refresh callers!
        let mut tasks = Vec::new();
        for _ in 0..20 {
            let r = refresher.clone();
            let cid = cred.id.clone();
            tasks.push(tokio::spawn(async move { r.ensure_fresh(&cid).await.unwrap() }));
        }
        for t in tasks {
            let (_, sec) = t.await.unwrap();
            match sec {
                CredentialSecret::Codex { refresh, account_id, .. } => {
                    assert_eq!(refresh, "rotated_rt_1");
                    assert_eq!(account_id, "acct_1");
                }
                _ => panic!("expected Codex secret"),
            }
        }
        // Exactly ONE upstream refresh HTTP POST occurred!
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // Also verify live quota polling against mock wham/usage!
        let q = quota::poll_and_save_quota(
            &db,
            &refresher,
            &cred.id,
            &QuotaEndpoints {
                codex_wham_usage_url: format!("http://{addr}/wham/usage"),
                ..QuotaEndpoints::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(q.plan.as_deref(), Some("pro"));
        assert_eq!(q.windows[0].label, "5h");
        assert_eq!(q.windows[0].used_percent, 42.5);
    }
}
