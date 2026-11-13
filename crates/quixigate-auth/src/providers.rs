//! Protocol implementations for all 9 first-class OAuth & subscription portals:
//! 1. `openai-codex` (ChatGPT Plus/Pro/Team PKCE + Device Code)
//! 2. `xai-oauth` (SuperGrok / X OIDC Device Code)
//! 3. `anthropic-token` (Claude Code `sk-ant-oat01-...` Setup Token)
//! 4. `google-gemini-cli` (Google Cloud Code Assist OAuth PKCE + project onboarding)
//! 5. `github-copilot` (GitHub Device Code + Copilot token exchange + `proxy-ep`)
//! 6. `openrouter` (OpenRouter PKCE 1-click API key minting)
//! 7. `qwen-portal` (Qwen Device Code + PKCE)
//! 8. `minimax-portal` (MiniMax User Code + PKCE, global & CN)
//! 9. `chutes` (Chutes IDP OAuth PKCE)

use quixigate_core::{AuthKind, CredentialSecret, QuixigateError, Result, now_ms};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::flow;

pub const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const CODEX_DEVICE_USERCODE_URL: &str =
    "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const CODEX_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub const CODEX_DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";
pub const CODEX_DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
pub const CODEX_DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

pub const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
pub const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
pub const XAI_DISCOVERY_URL: &str = "https://auth.x.ai/.well-known/openid-configuration";
pub const XAI_DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

pub const ANTHROPIC_SETUP_TOKEN_PREFIX: &str = "sk-ant-oat01-";
pub const ANTHROPIC_SETUP_TOKEN_MIN_LEN: usize = 80;
pub const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

pub const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
pub const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
pub const GOOGLE_USERINFO_URL: &str = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json";
pub const GOOGLE_CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
pub const GEMINI_CLI_REDIRECT_URI: &str = "http://localhost:8085/oauth2callback";

pub const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
pub const COPILOT_DEFAULT_BASE_URL: &str = "https://api.individual.githubcopilot.com";

pub const OPENROUTER_AUTH_URL: &str = "https://openrouter.ai/auth";
pub const OPENROUTER_KEYS_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
pub const OPENROUTER_OIDC_TOKEN_URL: &str = "https://openrouter.ai/api/v1/oauth/token";
pub const OPENROUTER_KEY_INFO_URL: &str = "https://openrouter.ai/api/v1/key";
pub const OPENROUTER_CREDITS_URL: &str = "https://openrouter.ai/api/v1/credits";
pub const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";
pub const OPENROUTER_PRESETS_URL: &str = "https://openrouter.ai/api/v1/presets";
pub const OPENROUTER_DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

pub const QWEN_CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";
pub const QWEN_DEVICE_CODE_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/device/code";
pub const QWEN_TOKEN_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/token";
pub const QWEN_DEFAULT_BASE_URL: &str = "https://portal.qwen.ai/v1";

pub const MINIMAX_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";
pub const MINIMAX_GLOBAL_AUTH_BASE: &str = "https://api.minimax.io";
pub const MINIMAX_CN_AUTH_BASE: &str = "https://api.minimaxi.com";

pub const CHUTES_AUTHORIZE_URL: &str = "https://api.chutes.ai/idp/authorize";
pub const CHUTES_TOKEN_URL: &str = "https://api.chutes.ai/idp/token";
pub const CHUTES_USERINFO_URL: &str = "https://api.chutes.ai/idp/userinfo";
pub const CHUTES_DEFAULT_BASE_URL: &str = "https://api.chutes.ai/v1";

const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const MINIMAX_USER_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:user_code";

/// Descriptor of an OAuth/subscription provider surfaced in CLI & Web UI wizards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthProviderDescriptor {
    pub provider_id: &'static str,
    pub vendor_id: &'static str,
    pub display_name: &'static str,
    pub auth_kind: AuthKind,
    pub flow_style: &'static str, // "pkce_or_device" | "device_code" | "setup_token" | "pkce"
    pub default_base_url: &'static str,
    pub description: &'static str,
}

pub const OAUTH_PROVIDER_CATALOG: &[OAuthProviderDescriptor] = &[
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "openai-codex",
        vendor_id: "openai",
        display_name: "ChatGPT / Codex (Plus, Pro, Team)",
        auth_kind: AuthKind::Oauth,
        flow_style: "pkce_or_device",
        default_base_url: CODEX_DEFAULT_BASE_URL,
        description: "Sign in with your ChatGPT subscription via browser PKCE or device code.",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "xai-oauth",
        vendor_id: "xai",
        display_name: "Grok / SuperGrok (X Subscription)",
        auth_kind: AuthKind::Oauth,
        flow_style: "device_code",
        default_base_url: XAI_DEFAULT_BASE_URL,
        description: "Device-code sign-in with an X Premium / SuperGrok account.",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "anthropic-token",
        vendor_id: "anthropic",
        display_name: "Claude Subscription (Setup Token)",
        auth_kind: AuthKind::SetupToken,
        flow_style: "setup_token",
        default_base_url: ANTHROPIC_DEFAULT_BASE_URL,
        description: "Paste a claude setup-token (sk-ant-oat01-...) stored exclusively in QuixiGate's vault.",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "google-gemini-cli",
        vendor_id: "google",
        display_name: "Google Gemini (Cloud Code Assist)",
        auth_kind: AuthKind::Oauth,
        flow_style: "pkce",
        default_base_url: GOOGLE_CODE_ASSIST_ENDPOINT,
        description: "OAuth PKCE sign-in for Google Cloud Code Assist (Pro & Flash quotas).",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "github-copilot",
        vendor_id: "github-copilot",
        display_name: "GitHub Copilot",
        auth_kind: AuthKind::Oauth,
        flow_style: "device_code",
        default_base_url: COPILOT_DEFAULT_BASE_URL,
        description: "Device-code sign-in using your Individual, Business, or Enterprise Copilot seat.",
    },
    OAuthProviderDescriptor {
        provider_id: "openrouter",
        vendor_id: "openrouter",
        display_name: "OpenRouter (1-Click PKCE)",
        auth_kind: AuthKind::ApiKey,
        flow_style: "pkce",
        default_base_url: OPENROUTER_DEFAULT_BASE_URL,
        description: "Mints a dedicated OpenRouter API key via browser PKCE in one click.",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "qwen-portal",
        vendor_id: "alibaba",
        display_name: "Qwen Portal",
        auth_kind: AuthKind::Oauth,
        flow_style: "device_code",
        default_base_url: QWEN_DEFAULT_BASE_URL,
        description: "Free-tier & subscription Qwen models via device code + PKCE.",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "minimax-portal",
        vendor_id: "minimax",
        display_name: "MiniMax Portal",
        auth_kind: AuthKind::Oauth,
        flow_style: "device_code",
        default_base_url: "https://api.minimax.io/anthropic",
        description: "MiniMax global & CN portal sign-in via user code + PKCE.",
    },
    #[cfg(feature = "subscription-oauth")]
    OAuthProviderDescriptor {
        provider_id: "chutes",
        vendor_id: "chutes",
        display_name: "Chutes AI",
        auth_kind: AuthKind::Oauth,
        flow_style: "pkce",
        default_base_url: CHUTES_DEFAULT_BASE_URL,
        description: "Open-weights inference via Chutes OAuth PKCE.",
    },
];

#[derive(Debug, Clone)]
pub enum PollOutcome {
    Pending,
    SlowDown,
    Authorized {
        secret: CredentialSecret,
        base_url: Option<String>,
    },
}

// ── 1. Anthropic Setup Token ─────────────────────────────────────────────────

pub fn validate_anthropic_setup_token(raw: &str) -> Result<CredentialSecret> {
    let cleaned: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.is_empty() {
        return Err(QuixigateError::Validation(
            "Anthropic setup token is empty".to_owned(),
        ));
    }
    if !cleaned.starts_with(ANTHROPIC_SETUP_TOKEN_PREFIX) {
        return Err(QuixigateError::Validation(format!(
            "Anthropic setup token must start with '{ANTHROPIC_SETUP_TOKEN_PREFIX}'"
        )));
    }
    if cleaned.len() < ANTHROPIC_SETUP_TOKEN_MIN_LEN {
        return Err(QuixigateError::Validation(format!(
            "Anthropic setup token is too short (expected at least {ANTHROPIC_SETUP_TOKEN_MIN_LEN} characters)"
        )));
    }
    Ok(CredentialSecret::AnthropicSetupToken { token: cleaned })
}

// ── 2. OpenAI Codex (PKCE + Device Code + Refresh) ──────────────────────────

pub fn build_codex_authorize_url(
    authorize_endpoint: &str,
    redirect_uri: &str,
    challenge: &str,
    state: &str,
) -> Result<String> {
    let url = reqwest::Url::parse_with_params(
        authorize_endpoint,
        [
            ("response_type", "code"),
            ("client_id", CODEX_CLIENT_ID),
            ("redirect_uri", redirect_uri),
            ("scope", "openid profile email offline_access"),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "pi"),
        ],
    )
    .map_err(|e| QuixigateError::Validation(format!("invalid Codex authorize URL: {e}")))?;
    Ok(url.into())
}

pub async fn exchange_codex_code(
    http: &reqwest::Client,
    token_url: &str,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<CredentialSecret> {
    codex_token_request(
        http,
        token_url,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", CODEX_CLIENT_ID),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", redirect_uri),
        ],
        None,
    )
    .await
}

pub async fn refresh_codex_token(
    http: &reqwest::Client,
    token_url: &str,
    refresh: &str,
    fallback_account_id: &str,
) -> Result<CredentialSecret> {
    codex_token_request(
        http,
        token_url,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
            ("client_id", CODEX_CLIENT_ID),
        ],
        Some(fallback_account_id),
    )
    .await
}

async fn codex_token_request(
    http: &reqwest::Client,
    token_url: &str,
    form: &[(&str, &str)],
    fallback_account_id: Option<&str>,
) -> Result<CredentialSecret> {
    let resp = http
        .post(token_url)
        .form(form)
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Codex token request failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        let code = body["error"]
            .as_str()
            .or_else(|| body["error"]["code"].as_str())
            .unwrap_or("oauth_error");
        let desc = body["error_description"]
            .as_str()
            .or_else(|| body["error"]["message"].as_str())
            .unwrap_or("Codex OAuth rejected");
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "openai-codex".to_owned(),
            message: format!("{code}: {desc}"),
        });
    }
    let access = body["access_token"].as_str().ok_or_else(|| {
        QuixigateError::Validation("Codex response missing access_token".to_owned())
    })?;
    let refresh = body["refresh_token"].as_str().ok_or_else(|| {
        QuixigateError::Validation("Codex response missing refresh_token".to_owned())
    })?;
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);
    let account_id = flow::jwt_payload(access)
        .and_then(|claims| {
            claims["https://api.openai.com/auth"]["chatgpt_account_id"]
                .as_str()
                .map(str::to_owned)
        })
        .or_else(|| fallback_account_id.map(str::to_owned))
        .ok_or_else(|| {
            QuixigateError::Validation("Codex JWT did not contain chatgpt_account_id".to_owned())
        })?;

    Ok(CredentialSecret::Codex {
        access: access.to_owned(),
        refresh: refresh.to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        account_id,
    })
}

pub async fn start_codex_device_flow(
    http: &reqwest::Client,
    usercode_url: &str,
) -> Result<(String, String, u64)> {
    let body: Value = http
        .post(usercode_url)
        .json(&json!({ "client_id": CODEX_CLIENT_ID }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Codex device code request failed: {e}")))?
        .error_for_status()
        .map_err(|e| QuixigateError::Validation(format!("Codex device code rejected: {e}")))?
        .json()
        .await
        .map_err(|e| QuixigateError::Validation(format!("invalid Codex device JSON: {e}")))?;
    let auth_id = body["device_auth_id"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("missing device_auth_id".to_owned()))?
        .to_owned();
    let user_code = body["user_code"]
        .as_str()
        .or_else(|| body["usercode"].as_str())
        .ok_or_else(|| QuixigateError::Validation("missing user_code".to_owned()))?
        .to_owned();
    let interval = interval_secs(&body["interval"]).max(2);
    Ok((auth_id, user_code, interval))
}

pub async fn poll_codex_device_once(
    http: &reqwest::Client,
    device_token_url: &str,
    oauth_token_url: &str,
    device_auth_id: &str,
    user_code: &str,
) -> Result<PollOutcome> {
    let resp = http
        .post(device_token_url)
        .json(&json!({
            "device_auth_id": device_auth_id,
            "user_code": user_code,
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Codex device poll failed: {e}")))?;
    if matches!(resp.status().as_u16(), 403 | 404) {
        return Ok(PollOutcome::Pending);
    }
    if !resp.status().is_success() {
        return Err(QuixigateError::Validation(format!(
            "Codex device flow failed with HTTP {}",
            resp.status()
        )));
    }
    let done: Value = resp.json().await.unwrap_or_default();
    let code = done["authorization_code"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("missing authorization_code".to_owned()))?;
    let verifier = done["code_verifier"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("missing code_verifier".to_owned()))?;
    let secret = exchange_codex_code(
        http,
        oauth_token_url,
        code,
        verifier,
        CODEX_DEVICE_REDIRECT_URI,
    )
    .await?;
    Ok(PollOutcome::Authorized {
        secret,
        base_url: Some(CODEX_DEFAULT_BASE_URL.to_owned()),
    })
}

// ── 3. xAI Grok (OIDC Discovery + Device Code + Refresh) ────────────────────

#[derive(Debug, Clone)]
pub struct XaiDeviceStart {
    pub device_code: String,
    pub user_code: String,
    pub verification_url: String,
    pub token_endpoint: String,
    pub interval_secs: u64,
    pub expires_in_secs: u64,
}

pub async fn start_xai_device_flow(
    http: &reqwest::Client,
    discovery_url: &str,
    allow_loopback_for_test: bool,
) -> Result<XaiDeviceStart> {
    let disc: Value = http
        .get(discovery_url)
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("xAI discovery request failed: {e}")))?
        .error_for_status()
        .map_err(|e| QuixigateError::Validation(format!("xAI discovery failed: {e}")))?
        .json()
        .await
        .map_err(|e| QuixigateError::Validation(format!("invalid xAI discovery JSON: {e}")))?;

    let device_endpoint = trusted_xai_url(
        &disc["device_authorization_endpoint"],
        allow_loopback_for_test,
    )?;
    let token_endpoint =
        trusted_xai_url(&disc["token_endpoint"], allow_loopback_for_test)?.to_owned();

    let start: Value = http
        .post(device_endpoint)
        .form(&[("client_id", XAI_CLIENT_ID), ("scope", XAI_SCOPE)])
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("xAI device code request failed: {e}")))?
        .error_for_status()
        .map_err(|e| QuixigateError::Validation(format!("xAI rejected device code start: {e}")))?
        .json()
        .await
        .map_err(|e| QuixigateError::Validation(format!("invalid xAI device JSON: {e}")))?;

    let device_code = start["device_code"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("xAI missing device_code".to_owned()))?
        .to_owned();
    let user_code = start["user_code"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("xAI missing user_code".to_owned()))?
        .to_owned();
    let verify_raw = if start["verification_uri_complete"].is_string() {
        &start["verification_uri_complete"]
    } else {
        &start["verification_uri"]
    };
    let verification_url = trusted_xai_url(verify_raw, allow_loopback_for_test)?.to_owned();

    Ok(XaiDeviceStart {
        device_code,
        user_code,
        verification_url,
        token_endpoint,
        interval_secs: interval_secs(&start["interval"]).max(1),
        expires_in_secs: start["expires_in"].as_u64().unwrap_or(300).max(60),
    })
}

pub async fn poll_xai_device_once(
    http: &reqwest::Client,
    token_endpoint: &str,
    device_code: &str,
) -> Result<PollOutcome> {
    xai_token_request(
        http,
        token_endpoint,
        &[
            ("grant_type", DEVICE_GRANT),
            ("client_id", XAI_CLIENT_ID),
            ("device_code", device_code),
        ],
        None,
    )
    .await
}

pub async fn refresh_xai_token(
    http: &reqwest::Client,
    token_endpoint: &str,
    refresh: &str,
    allow_loopback_for_test: bool,
) -> Result<CredentialSecret> {
    let endpoint_val = Value::String(token_endpoint.to_owned());
    let verified_endpoint = trusted_xai_url(&endpoint_val, allow_loopback_for_test)?;
    match xai_token_request(
        http,
        verified_endpoint,
        &[
            ("grant_type", "refresh_token"),
            ("client_id", XAI_CLIENT_ID),
            ("refresh_token", refresh),
        ],
        Some(refresh),
    )
    .await?
    {
        PollOutcome::Authorized { secret, .. } => Ok(secret),
        _ => Err(QuixigateError::Unauthorized(
            "xAI refresh returned pending state".to_owned(),
        )),
    }
}

async fn xai_token_request(
    http: &reqwest::Client,
    token_endpoint: &str,
    form: &[(&str, &str)],
    prev_refresh: Option<&str>,
) -> Result<PollOutcome> {
    let resp = http
        .post(token_endpoint)
        .header("Accept", "application/json")
        .form(form)
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("xAI token request failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return match body["error"].as_str() {
            Some("authorization_pending") => Ok(PollOutcome::Pending),
            Some("slow_down") => Ok(PollOutcome::SlowDown),
            err => Err(QuixigateError::Upstream {
                status: status.as_u16(),
                provider: "xai-oauth".to_owned(),
                message: err.unwrap_or("xAI token request rejected").to_owned(),
            }),
        };
    }
    let access = body["access_token"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("xAI missing access_token".to_owned()))?;
    let refresh = body["refresh_token"]
        .as_str()
        .or(prev_refresh)
        .ok_or_else(|| QuixigateError::Validation("xAI missing refresh_token".to_owned()))?;
    let expires_ms = match body["expires_in"].as_u64() {
        Some(secs) => now_ms() + (secs as i64) * 1000,
        None => flow::jwt_payload(access)
            .and_then(|c| c["exp"].as_u64())
            .map_or(now_ms() + 3_600_000, |exp| (exp as i64) * 1000),
    };
    Ok(PollOutcome::Authorized {
        secret: CredentialSecret::Xai {
            access: access.to_owned(),
            refresh: refresh.to_owned(),
            expires_ms,
            token_endpoint: token_endpoint.to_owned(),
        },
        base_url: Some(XAI_DEFAULT_BASE_URL.to_owned()),
    })
}

pub fn trusted_xai_url(val: &Value, allow_loopback: bool) -> Result<&str> {
    let raw = val
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("xAI OIDC discovery URL missing".to_owned()))?;
    let parsed = reqwest::Url::parse(raw)
        .map_err(|e| QuixigateError::Validation(format!("invalid xAI URL: {e}")))?;
    let host = parsed.host_str().unwrap_or("");
    if parsed.scheme() == "https" && (host == "x.ai" || host.ends_with(".x.ai")) {
        return Ok(raw);
    }
    if allow_loopback && parsed.scheme() == "http" && matches!(host, "127.0.0.1" | "localhost") {
        return Ok(raw);
    }
    Err(QuixigateError::Validation(format!(
        "xAI discovery URL points outside trusted *.x.ai domain: {raw}"
    )))
}

// ── 4. GitHub Copilot (Device Code + Copilot Internal Token Exchange) ───────

#[derive(Debug, Clone)]
pub struct CopilotDeviceStart {
    pub domain: String,
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub interval_secs: u64,
    pub expires_in_secs: u64,
}

pub fn derive_copilot_base_url(copilot_token: &str, enterprise_domain: Option<&str>) -> String {
    if let Ok(re) = Regex::new(r"proxy-ep=([^;]+)")
        && let Some(caps) = re.captures(copilot_token)
    {
        let proxy_host = &caps[1];
        let api_host = proxy_host
            .strip_prefix("proxy.")
            .map(|rest| format!("api.{rest}"))
            .unwrap_or_else(|| proxy_host.to_owned());
        return format!("https://{api_host}");
    }
    if let Some(ent) = enterprise_domain.filter(|d| !d.is_empty() && *d != "github.com") {
        return format!("https://copilot-api.{ent}");
    }
    COPILOT_DEFAULT_BASE_URL.to_owned()
}

pub async fn start_copilot_device_flow(
    http: &reqwest::Client,
    device_code_url: &str,
    domain: &str,
) -> Result<CopilotDeviceStart> {
    let data: Value = http
        .post(device_code_url)
        .header("Accept", "application/json")
        .header("User-Agent", "GitHubCopilotChat/0.35.0")
        .json(&json!({
            "client_id": COPILOT_CLIENT_ID,
            "scope": "read:user",
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("GitHub device code failed: {e}")))?
        .error_for_status()
        .map_err(|e| QuixigateError::Validation(format!("GitHub device code rejected: {e}")))?
        .json()
        .await
        .map_err(|e| QuixigateError::Validation(format!("invalid GitHub device JSON: {e}")))?;

    Ok(CopilotDeviceStart {
        domain: domain.to_owned(),
        device_code: data["device_code"]
            .as_str()
            .ok_or_else(|| QuixigateError::Validation("missing device_code".to_owned()))?
            .to_owned(),
        user_code: data["user_code"]
            .as_str()
            .ok_or_else(|| QuixigateError::Validation("missing user_code".to_owned()))?
            .to_owned(),
        verification_uri: data["verification_uri"]
            .as_str()
            .ok_or_else(|| QuixigateError::Validation("missing verification_uri".to_owned()))?
            .to_owned(),
        interval_secs: interval_secs(&data["interval"]).max(2),
        expires_in_secs: data["expires_in"].as_u64().unwrap_or(900),
    })
}

pub async fn poll_copilot_device_once(
    http: &reqwest::Client,
    github_token_url: &str,
    copilot_exchange_url: &str,
    device_code: &str,
    enterprise_domain: Option<&str>,
) -> Result<PollOutcome> {
    let data: Value = http
        .post(github_token_url)
        .header("Accept", "application/json")
        .header("User-Agent", "GitHubCopilotChat/0.35.0")
        .json(&json!({
            "client_id": COPILOT_CLIENT_ID,
            "device_code": device_code,
            "grant_type": DEVICE_GRANT,
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("GitHub token poll failed: {e}")))?
        .json()
        .await
        .unwrap_or_default();

    if let Some(gh_token) = data["access_token"].as_str() {
        let secret =
            exchange_github_for_copilot(http, copilot_exchange_url, gh_token, enterprise_domain)
                .await?;
        let base = match &secret {
            CredentialSecret::GitHubCopilot { base_url, .. } => Some(base_url.clone()),
            _ => None,
        };
        return Ok(PollOutcome::Authorized {
            secret,
            base_url: base,
        });
    }
    match data["error"].as_str().unwrap_or("") {
        "authorization_pending" => Ok(PollOutcome::Pending),
        "slow_down" => Ok(PollOutcome::SlowDown),
        other => Err(QuixigateError::Validation(format!(
            "GitHub Copilot device flow failed: {other}"
        ))),
    }
}

pub async fn exchange_github_for_copilot(
    http: &reqwest::Client,
    copilot_exchange_url: &str,
    github_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<CredentialSecret> {
    let resp = http
        .get(copilot_exchange_url)
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {github_token}"))
        .header("User-Agent", "GitHubCopilotChat/0.35.0")
        .header("Editor-Version", "vscode/1.107.0")
        .header("Editor-Plugin-Version", "copilot-chat/0.35.0")
        .header("Copilot-Integration-Id", "vscode-chat")
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Copilot token exchange failed: {e}")))?;
    let status = resp.status();
    let data: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "github-copilot".to_owned(),
            message: data["message"]
                .as_str()
                .unwrap_or("Copilot token exchange rejected")
                .to_owned(),
        });
    }
    let token = data["token"].as_str().ok_or_else(|| {
        QuixigateError::Validation("Copilot token response missing 'token'".to_owned())
    })?;
    let expires_at = data["expires_at"]
        .as_i64()
        .unwrap_or_else(|| (now_ms() / 1000) + 1800);
    let base_url = derive_copilot_base_url(token, enterprise_domain);

    Ok(CredentialSecret::GitHubCopilot {
        github_token: github_token.to_owned(),
        access: token.to_owned(),
        expires_ms: expires_at * 1000,
        base_url,
        enterprise_domain: enterprise_domain.map(str::to_owned),
    })
}

// ── 5. Google Gemini CLI / Cloud Code Assist ────────────────────────────────

pub fn resolve_gemini_client_credentials(
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> Result<(String, String)> {
    let cid = client_id
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| std::env::var("GEMINI_CLI_OAUTH_CLIENT_ID").ok())
        .or_else(|| quixigate_core::brand::env_opt("QUIXIGATE_GEMINI_OAUTH_CLIENT_ID"))
        .ok_or_else(|| {
            QuixigateError::Validation(
                "Google Gemini OAuth requires client_id (pass in UI/CLI or set GEMINI_CLI_OAUTH_CLIENT_ID)".to_owned(),
            )
        })?;
    let csec = client_secret
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| std::env::var("GEMINI_CLI_OAUTH_CLIENT_SECRET").ok())
        .or_else(|| quixigate_core::brand::env_opt("QUIXIGATE_GEMINI_OAUTH_CLIENT_SECRET"))
        .ok_or_else(|| {
            QuixigateError::Validation(
                "Google Gemini OAuth requires client_secret (pass in UI/CLI or set GEMINI_CLI_OAUTH_CLIENT_SECRET)".to_owned(),
            )
        })?;
    Ok((cid, csec))
}

pub fn build_gemini_authorize_url(
    auth_url: &str,
    redirect_uri: &str,
    client_id: &str,
    challenge: &str,
    state: &str,
) -> Result<String> {
    let scopes = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile";
    let url = reqwest::Url::parse_with_params(
        auth_url,
        [
            ("client_id", client_id),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri),
            ("scope", scopes),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("access_type", "offline"),
            ("prompt", "consent"),
        ],
    )
    .map_err(|e| QuixigateError::Validation(format!("invalid Google authorize URL: {e}")))?;
    Ok(url.into())
}

#[allow(clippy::too_many_arguments)]
pub async fn complete_gemini_login(
    http: &reqwest::Client,
    token_url: &str,
    userinfo_url: &str,
    code_assist_endpoint: &str,
    redirect_uri: &str,
    code: &str,
    verifier: &str,
    client_id: &str,
    client_secret: &str,
) -> Result<CredentialSecret> {
    let resp = http
        .post(token_url)
        .form(&[
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("code", code),
            ("grant_type", "authorization_code"),
            ("redirect_uri", redirect_uri),
            ("code_verifier", verifier),
        ])
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Google token exchange failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "google-gemini-cli".to_owned(),
            message: body["error_description"]
                .as_str()
                .or_else(|| body["error"].as_str())
                .unwrap_or("Google token exchange rejected")
                .to_owned(),
        });
    }
    let access = body["access_token"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("Google missing access_token".to_owned()))?;
    let refresh = body["refresh_token"].as_str().unwrap_or_default();
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    let email = match http.get(userinfo_url).bearer_auth(access).send().await {
        Ok(r) if r.status().is_success() => r
            .json::<Value>()
            .await
            .ok()
            .and_then(|v| v["email"].as_str().map(str::to_owned)),
        _ => None,
    };

    let project_id = discover_gemini_project(http, code_assist_endpoint, access).await?;
    Ok(CredentialSecret::GeminiCli {
        access: access.to_owned(),
        refresh: refresh.to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        project_id,
        email,
        client_id: client_id.to_owned(),
        client_secret: client_secret.to_owned(),
    })
}

async fn discover_gemini_project(
    http: &reqwest::Client,
    code_assist_endpoint: &str,
    access_token: &str,
) -> Result<String> {
    let meta = json!({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    });
    if let Ok(resp) = http
        .post(format!(
            "{}/v1internal:loadCodeAssist",
            code_assist_endpoint.trim_end_matches('/')
        ))
        .bearer_auth(access_token)
        .header("User-Agent", "google-api-nodejs-client/9.15.1")
        .header("X-Goog-Api-Client", "google-cloud-sdk")
        .header("Client-Metadata", meta.to_string())
        .json(&json!({ "metadata": meta }))
        .send()
        .await
        && resp.status().is_success()
        && let Ok(data) = resp.json::<Value>().await
    {
        if let Some(proj) = data["cloudaicompanionProject"].as_str() {
            return Ok(proj.to_owned());
        }
        if let Some(proj) = data["cloudaicompanionProject"]["id"].as_str() {
            return Ok(proj.to_owned());
        }
    }
    if let Ok(env_proj) = std::env::var("GOOGLE_CLOUD_PROJECT")
        && !env_proj.trim().is_empty()
    {
        return Ok(env_proj);
    }
    Err(QuixigateError::Validation(
        "could not discover Google Cloud Code Assist project (set GOOGLE_CLOUD_PROJECT if needed)"
            .to_owned(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub async fn refresh_gemini_token(
    http: &reqwest::Client,
    token_url: &str,
    refresh: &str,
    project_id: &str,
    email: Option<&str>,
    client_id: &str,
    client_secret: &str,
) -> Result<CredentialSecret> {
    let resp = http
        .post(token_url)
        .form(&[
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
        ])
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Google token refresh failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "google-gemini-cli".to_owned(),
            message: body["error"]
                .as_str()
                .unwrap_or("Google refresh rejected")
                .to_owned(),
        });
    }
    let access = body["access_token"].as_str().ok_or_else(|| {
        QuixigateError::Validation("Google refresh missing access_token".to_owned())
    })?;
    let next_refresh = body["refresh_token"].as_str().unwrap_or(refresh);
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    Ok(CredentialSecret::GeminiCli {
        access: access.to_owned(),
        refresh: next_refresh.to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        project_id: project_id.to_owned(),
        email: email.map(str::to_owned),
        client_id: client_id.to_owned(),
        client_secret: client_secret.to_owned(),
    })
}

// ── 6. OpenRouter (PKCE 1-Click Key Minting, OIDC Token Exchange & Key Inspection) ──

pub fn openrouter_base_url_for_region(region: Option<&str>) -> &'static str {
    match region.map(|r| r.trim().to_ascii_lowercase()).as_deref() {
        Some("us") => "https://us.openrouter.ai/api/v1",
        Some("eu") => "https://eu.openrouter.ai/api/v1",
        _ => OPENROUTER_DEFAULT_BASE_URL,
    }
}

pub fn build_openrouter_authorize_url(
    auth_url: &str,
    callback_url: Option<&str>,
    challenge: &str,
) -> Result<String> {
    let mut params = vec![
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
    ];
    if let Some(cb) = callback_url {
        params.push(("callback_url", cb));
    } else {
        params.push(("key_label", "quixigate"));
    }
    let url = reqwest::Url::parse_with_params(auth_url, params)
        .map_err(|e| QuixigateError::Validation(format!("invalid OpenRouter auth URL: {e}")))?;
    Ok(url.into())
}

pub async fn exchange_openrouter_code(
    http: &reqwest::Client,
    keys_url: &str,
    code: &str,
    verifier: &str,
) -> Result<CredentialSecret> {
    let resp = http
        .post(keys_url)
        .header(
            "HTTP-Referer",
            quixigate_core::brand::DEFAULT_OPENROUTER_APP_URL,
        )
        .header(
            "X-Title",
            quixigate_core::brand::DEFAULT_OPENROUTER_APP_TITLE,
        )
        .json(&json!({
            "code": code,
            "code_verifier": verifier,
            "code_challenge_method": "S256",
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("OpenRouter key exchange failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        let msg = body["error"]["message"]
            .as_str()
            .or_else(|| body["message"].as_str())
            .unwrap_or("OpenRouter rejected the authorization code");
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "openrouter".to_owned(),
            message: msg.to_owned(),
        });
    }
    let key = body["key"]
        .as_str()
        .or_else(|| body["data"]["key"].as_str())
        .filter(|k| !k.trim().is_empty())
        .ok_or_else(|| {
            QuixigateError::Validation("OpenRouter response missing 'key'".to_owned())
        })?;
    Ok(CredentialSecret::ApiKey {
        key: key.trim().to_owned(),
    })
}

/// Exchanges an external OIDC / Workload Identity JWT for a short-lived OpenRouter access token
/// via RFC 8693 (`grant_type=urn:ietf:params:oauth:grant-type:token-exchange`).
pub async fn exchange_openrouter_oidc_token(
    http: &reqwest::Client,
    token_url: &str,
    subject_token: &str,
    federation_policy_id: Option<&str>,
    subject_token_env: Option<&str>,
) -> Result<CredentialSecret> {
    let trimmed_subject = subject_token.trim();
    if trimmed_subject.is_empty() {
        return Err(QuixigateError::Validation(
            "OpenRouter OIDC subject_token is empty".to_owned(),
        ));
    }
    let mut form = vec![
        (
            "grant_type",
            "urn:ietf:params:oauth:grant-type:token-exchange",
        ),
        ("subject_token", trimmed_subject),
        ("subject_token_type", "urn:ietf:params:oauth:token-type:jwt"),
    ];
    let clean_policy = federation_policy_id
        .map(str::trim)
        .filter(|p| !p.is_empty());
    if let Some(policy) = clean_policy {
        form.push(("federation_policy_id", policy));
    }

    let resp = http
        .post(token_url)
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .await
        .map_err(|e| {
            QuixigateError::Internal(format!("OpenRouter OIDC token exchange failed: {e}"))
        })?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        let msg = body["error_description"]
            .as_str()
            .or_else(|| body["error"]["message"].as_str())
            .or_else(|| body["error"].as_str())
            .unwrap_or("OpenRouter OIDC token exchange rejected");
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "openrouter".to_owned(),
            message: msg.to_owned(),
        });
    }
    let access = body["access_token"]
        .as_str()
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| {
            QuixigateError::Validation("OpenRouter OIDC response missing access_token".to_owned())
        })?;
    let expires_in = body["expires_in"].as_u64().unwrap_or(900);
    Ok(CredentialSecret::OpenRouterOidc {
        access: access.trim().to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        federation_policy_id: clean_policy.map(str::to_owned),
        subject_token: Some(trimmed_subject.to_owned()),
        subject_token_env: subject_token_env
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        token_endpoint: token_url.to_owned(),
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenRouterAccountInfo {
    pub key_label: Option<String>,
    pub key_limit: Option<f64>,
    pub key_limit_remaining: Option<f64>,
    pub key_usage: Option<f64>,
    pub is_free_tier: bool,
    pub is_management_key: bool,
    pub total_credits: Option<f64>,
    pub total_usage: Option<f64>,
    pub remaining_credits: Option<f64>,
}

/// Validates and introspects an OpenRouter API key or OIDC Bearer token against
/// `GET {api_base_url}/key` and `GET {api_base_url}/credits`.
pub async fn inspect_openrouter_account(
    http: &reqwest::Client,
    api_base_url: &str,
    bearer_token: &str,
) -> Result<OpenRouterAccountInfo> {
    let base = api_base_url.trim_end_matches('/');
    let key_url = format!("{base}/key");
    let credits_url = format!("{base}/credits");

    let key_resp = http
        .get(&key_url)
        .bearer_auth(bearer_token.trim())
        .header(
            "HTTP-Referer",
            quixigate_core::brand::DEFAULT_OPENROUTER_APP_URL,
        )
        .header(
            "X-Title",
            quixigate_core::brand::DEFAULT_OPENROUTER_APP_TITLE,
        )
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("OpenRouter key inspection failed: {e}")))?;

    let mut info = OpenRouterAccountInfo::default();
    let mut validated = false;

    if key_resp.status().is_success() {
        validated = true;
        let body: Value = key_resp.json().await.unwrap_or_default();
        let d = &body["data"];
        info.key_label = d["label"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        info.key_limit = d["limit"].as_f64();
        info.key_limit_remaining = d["limit_remaining"].as_f64();
        info.key_usage = d["usage"].as_f64();
        info.is_free_tier = d["is_free_tier"].as_bool().unwrap_or(false);
        info.is_management_key = d["is_management_key"].as_bool().unwrap_or(false);
    } else if matches!(key_resp.status().as_u16(), 401 | 403) {
        // Double-check `/credits` in case an OIDC inference-scoped token isn't accepted on `/key`.
        let cred_check = http
            .get(&credits_url)
            .bearer_auth(bearer_token.trim())
            .send()
            .await;
        if let Ok(cr) = cred_check
            && cr.status().is_success()
        {
            let body: Value = cr.json().await.unwrap_or_default();
            let total = body["data"]["total_credits"].as_f64().unwrap_or(0.0);
            let used = body["data"]["total_usage"].as_f64().unwrap_or(0.0);
            info.total_credits = Some(total);
            info.total_usage = Some(used);
            info.remaining_credits = Some((total - used).max(0.0));
            return Ok(info);
        }
        return Err(QuixigateError::Unauthorized(
            "OpenRouter rejected the credential (HTTP 401/403)".to_owned(),
        ));
    }

    if let Ok(cr) = http
        .get(&credits_url)
        .bearer_auth(bearer_token.trim())
        .send()
        .await
        && cr.status().is_success()
    {
        validated = true;
        let body: Value = cr.json().await.unwrap_or_default();
        let total = body["data"]["total_credits"].as_f64().unwrap_or(0.0);
        let used = body["data"]["total_usage"].as_f64().unwrap_or(0.0);
        info.total_credits = Some(total);
        info.total_usage = Some(used);
        info.remaining_credits = Some((total - used).max(0.0));
    }

    if !validated {
        return Err(QuixigateError::Validation(
            "Could not validate OpenRouter credential against /key or /credits".to_owned(),
        ));
    }
    Ok(info)
}

// ── 7. Qwen Portal (Device Code + PKCE) ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QwenDeviceStart {
    pub device_code: String,
    pub user_code: String,
    pub verification_url: String,
    pub verifier: String,
    pub interval_secs: u64,
    pub expires_in_secs: u64,
}

pub async fn start_qwen_device_flow(
    http: &reqwest::Client,
    device_code_url: &str,
) -> Result<QwenDeviceStart> {
    let verifier = flow::random_token();
    let challenge = flow::pkce_challenge(&verifier);
    let data: Value = http
        .post(device_code_url)
        .json(&json!({
            "client_id": QWEN_CLIENT_ID,
            "scope": "openid profile email model.completion",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Qwen device flow start failed: {e}")))?
        .error_for_status()
        .map_err(|e| QuixigateError::Validation(format!("Qwen device code rejected: {e}")))?
        .json()
        .await
        .map_err(|e| QuixigateError::Validation(format!("invalid Qwen JSON: {e}")))?;

    let verify = data["verification_uri_complete"]
        .as_str()
        .or_else(|| data["verification_uri"].as_str())
        .ok_or_else(|| QuixigateError::Validation("Qwen missing verification_uri".to_owned()))?
        .to_owned();

    Ok(QwenDeviceStart {
        device_code: data["device_code"]
            .as_str()
            .ok_or_else(|| QuixigateError::Validation("Qwen missing device_code".to_owned()))?
            .to_owned(),
        user_code: data["user_code"]
            .as_str()
            .ok_or_else(|| QuixigateError::Validation("Qwen missing user_code".to_owned()))?
            .to_owned(),
        verification_url: verify,
        verifier,
        interval_secs: interval_secs(&data["interval"]).max(2),
        expires_in_secs: data["expires_in"].as_u64().unwrap_or(900),
    })
}

pub async fn poll_qwen_device_once(
    http: &reqwest::Client,
    token_url: &str,
    device_code: &str,
    verifier: &str,
) -> Result<PollOutcome> {
    let data: Value = http
        .post(token_url)
        .json(&json!({
            "client_id": QWEN_CLIENT_ID,
            "device_code": device_code,
            "grant_type": DEVICE_GRANT,
            "code_verifier": verifier,
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Qwen poll failed: {e}")))?
        .json()
        .await
        .unwrap_or_default();

    if let Some(access) = data["access_token"].as_str() {
        let refresh = data["refresh_token"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let expires_in = data["expires_in"].as_u64().unwrap_or(3600);
        let resource_url = data["resource_url"].as_str().map(str::to_owned);
        return Ok(PollOutcome::Authorized {
            base_url: Some(
                resource_url
                    .clone()
                    .unwrap_or_else(|| QWEN_DEFAULT_BASE_URL.to_owned()),
            ),
            secret: CredentialSecret::QwenPortal {
                access: access.to_owned(),
                refresh,
                expires_ms: now_ms() + (expires_in as i64) * 1000,
                resource_url,
            },
        });
    }
    match data["error"].as_str().unwrap_or("") {
        "authorization_pending" => Ok(PollOutcome::Pending),
        "slow_down" => Ok(PollOutcome::SlowDown),
        other => Err(QuixigateError::Validation(format!(
            "Qwen device flow failed: {other}"
        ))),
    }
}

pub async fn refresh_qwen_token(
    http: &reqwest::Client,
    token_url: &str,
    refresh: &str,
    prev_resource_url: Option<&str>,
) -> Result<CredentialSecret> {
    let resp = http
        .post(token_url)
        .json(&json!({
            "client_id": QWEN_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh,
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Qwen refresh failed: {e}")))?;
    let status = resp.status();
    let data: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "qwen-portal".to_owned(),
            message: data["error"]
                .as_str()
                .unwrap_or("Qwen refresh failed")
                .to_owned(),
        });
    }
    let access = data["access_token"].as_str().ok_or_else(|| {
        QuixigateError::Validation("Qwen refresh missing access_token".to_owned())
    })?;
    let next_refresh = data["refresh_token"].as_str().unwrap_or(refresh);
    let expires_in = data["expires_in"].as_u64().unwrap_or(3600);
    Ok(CredentialSecret::QwenPortal {
        access: access.to_owned(),
        refresh: next_refresh.to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        resource_url: data["resource_url"]
            .as_str()
            .or(prev_resource_url)
            .map(str::to_owned),
    })
}

// ── 8. MiniMax Portal (User Code + PKCE) ────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MiniMaxDeviceStart {
    pub region: String,
    pub auth_base: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verifier: String,
    pub interval_secs: u64,
    pub expires_in_secs: u64,
}

pub fn minimax_auth_base(region: &str) -> &'static str {
    if region.eq_ignore_ascii_case("cn") {
        MINIMAX_CN_AUTH_BASE
    } else {
        MINIMAX_GLOBAL_AUTH_BASE
    }
}

pub async fn start_minimax_flow(
    http: &reqwest::Client,
    auth_base: &str,
    region: &str,
) -> Result<MiniMaxDeviceStart> {
    let verifier = flow::random_token();
    let challenge = flow::pkce_challenge(&verifier);
    let state = flow::random_token();
    let data: Value = http
        .post(format!("{}/oauth/code", auth_base.trim_end_matches('/')))
        .json(&json!({
            "client_id": MINIMAX_CLIENT_ID,
            "scope": "group_id profile model.completion",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("MiniMax code request failed: {e}")))?
        .error_for_status()
        .map_err(|e| QuixigateError::Validation(format!("MiniMax code rejected: {e}")))?
        .json()
        .await
        .map_err(|e| QuixigateError::Validation(format!("invalid MiniMax JSON: {e}")))?;

    if data["state"].as_str() != Some(&state) {
        return Err(QuixigateError::Validation(
            "MiniMax state mismatch (CSRF check failed)".to_owned(),
        ));
    }

    Ok(MiniMaxDeviceStart {
        region: region.to_owned(),
        auth_base: auth_base.to_owned(),
        user_code: data["user_code"]
            .as_str()
            .ok_or_else(|| QuixigateError::Validation("MiniMax missing user_code".to_owned()))?
            .to_owned(),
        verification_uri: data["verification_uri"]
            .as_str()
            .ok_or_else(|| {
                QuixigateError::Validation("MiniMax missing verification_uri".to_owned())
            })?
            .to_owned(),
        verifier,
        interval_secs: interval_secs(&data["interval"]).max(2),
        expires_in_secs: data["expired_in"]
            .as_u64()
            .or_else(|| data["expires_in"].as_u64())
            .unwrap_or(300),
    })
}

pub async fn poll_minimax_once(
    http: &reqwest::Client,
    auth_base: &str,
    region: &str,
    user_code: &str,
    verifier: &str,
) -> Result<PollOutcome> {
    let data: Value = http
        .post(format!("{}/oauth/token", auth_base.trim_end_matches('/')))
        .json(&json!({
            "client_id": MINIMAX_CLIENT_ID,
            "grant_type": MINIMAX_USER_CODE_GRANT,
            "user_code": user_code,
            "code_verifier": verifier,
        }))
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("MiniMax poll failed: {e}")))?
        .json()
        .await
        .unwrap_or_default();

    if data["status"] == "success" || data["access_token"].is_string() {
        let access = data["access_token"]
            .as_str()
            .or_else(|| data["access"].as_str())
            .ok_or_else(|| QuixigateError::Validation("MiniMax missing access_token".to_owned()))?;
        let refresh = data["refresh_token"]
            .as_str()
            .or_else(|| data["refresh"].as_str())
            .unwrap_or_default();
        let raw_exp = data["expired_in"]
            .as_i64()
            .or_else(|| data["expires_in"].as_i64())
            .unwrap_or(3600);
        let expires_ms = if raw_exp < 10_000_000_000 {
            now_ms() + raw_exp * 1000
        } else {
            raw_exp * 1000
        };
        let resource_url = data["resource_url"].as_str().map(str::to_owned);
        let base_url = format!("{}/anthropic", minimax_auth_base(region));
        return Ok(PollOutcome::Authorized {
            base_url: Some(base_url),
            secret: CredentialSecret::MiniMaxPortal {
                access: access.to_owned(),
                refresh: refresh.to_owned(),
                expires_ms,
                region: region.to_owned(),
                resource_url,
            },
        });
    }
    if data["status"] == "pending" || data["error"] == "authorization_pending" {
        return Ok(PollOutcome::Pending);
    }
    if data["error"] == "slow_down" {
        return Ok(PollOutcome::SlowDown);
    }
    Err(QuixigateError::Validation(format!(
        "MiniMax OAuth failed: {}",
        data["error"]
            .as_str()
            .or_else(|| data["status"].as_str())
            .unwrap_or("unknown")
    )))
}

// ── 9. Chutes OAuth PKCE ────────────────────────────────────────────────────

pub fn build_chutes_authorize_url(
    authorize_url: &str,
    redirect_uri: &str,
    client_id: &str,
    challenge: &str,
    state: &str,
) -> Result<String> {
    let url = reqwest::Url::parse_with_params(
        authorize_url,
        [
            ("response_type", "code"),
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("scope", "openid profile email offline_access"),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
        ],
    )
    .map_err(|e| QuixigateError::Validation(format!("invalid Chutes authorize URL: {e}")))?;
    Ok(url.into())
}

#[allow(clippy::too_many_arguments)]
pub async fn exchange_chutes_code(
    http: &reqwest::Client,
    token_url: &str,
    userinfo_url: &str,
    code: &str,
    verifier: &str,
    client_id: &str,
    client_secret: Option<&str>,
    redirect_uri: &str,
) -> Result<CredentialSecret> {
    let mut form = vec![
        ("grant_type", "authorization_code"),
        ("client_id", client_id),
        ("code", code),
        ("code_verifier", verifier),
        ("redirect_uri", redirect_uri),
    ];
    if let Some(sec) = client_secret.filter(|s| !s.is_empty()) {
        form.push(("client_secret", sec));
    }
    let resp = http
        .post(token_url)
        .form(&form)
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Chutes token exchange failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "chutes".to_owned(),
            message: body["error"]
                .as_str()
                .unwrap_or("Chutes exchange rejected")
                .to_owned(),
        });
    }
    let access = body["access_token"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("Chutes missing access_token".to_owned()))?;
    let refresh = body["refresh_token"]
        .as_str()
        .ok_or_else(|| QuixigateError::Validation("Chutes missing refresh_token".to_owned()))?;
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    let (email, account_id) = match http.get(userinfo_url).bearer_auth(access).send().await {
        Ok(r) if r.status().is_success() => match r.json::<Value>().await {
            Ok(v) => (
                v["username"]
                    .as_str()
                    .or_else(|| v["email"].as_str())
                    .map(str::to_owned),
                v["sub"].as_str().map(str::to_owned),
            ),
            Err(_) => (None, None),
        },
        _ => (None, None),
    };

    Ok(CredentialSecret::Chutes {
        access: access.to_owned(),
        refresh: refresh.to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        email,
        account_id,
        client_id: client_id.to_owned(),
        client_secret: client_secret.map(str::to_owned),
    })
}

#[allow(clippy::too_many_arguments)]
pub async fn refresh_chutes_token(
    http: &reqwest::Client,
    token_url: &str,
    refresh: &str,
    email: Option<&str>,
    account_id: Option<&str>,
    client_id: &str,
    client_secret: Option<&str>,
) -> Result<CredentialSecret> {
    let mut form = vec![
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh),
        ("client_id", client_id),
    ];
    if let Some(sec) = client_secret.filter(|s| !s.is_empty()) {
        form.push(("client_secret", sec));
    }
    let resp = http
        .post(token_url)
        .form(&form)
        .send()
        .await
        .map_err(|e| QuixigateError::Internal(format!("Chutes refresh failed: {e}")))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: "chutes".to_owned(),
            message: body["error"]
                .as_str()
                .unwrap_or("Chutes refresh rejected")
                .to_owned(),
        });
    }
    let access = body["access_token"].as_str().ok_or_else(|| {
        QuixigateError::Validation("Chutes refresh missing access_token".to_owned())
    })?;
    let next_refresh = body["refresh_token"].as_str().unwrap_or(refresh);
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);
    Ok(CredentialSecret::Chutes {
        access: access.to_owned(),
        refresh: next_refresh.to_owned(),
        expires_ms: now_ms() + (expires_in as i64) * 1000,
        email: email.map(str::to_owned),
        account_id: account_id.map(str::to_owned),
        client_id: client_id.to_owned(),
        client_secret: client_secret.map(str::to_owned),
    })
}

pub(crate) fn interval_secs(val: &Value) -> u64 {
    val.as_u64()
        .or_else(|| val.as_str()?.parse().ok())
        .unwrap_or(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_anthropic_setup_token_and_copilot_proxy_ep() {
        let valid_tok = format!("sk-ant-oat01-{}", "a".repeat(80));
        assert!(validate_anthropic_setup_token(&valid_tok).is_ok());
        assert!(validate_anthropic_setup_token("sk-ant-api03-short").is_err());

        let copilot_tok = "tid=abc;exp=1900000;proxy-ep=proxy.enterprise.githubcopilot.com;sku=ent";
        assert_eq!(
            derive_copilot_base_url(copilot_tok, None),
            "https://api.enterprise.githubcopilot.com"
        );
        assert_eq!(
            derive_copilot_base_url("tid=abc", None),
            "https://api.individual.githubcopilot.com"
        );
    }

    #[test]
    fn pins_xai_discovery_domain() {
        assert!(trusted_xai_url(&json!("https://auth.x.ai/oauth2/token"), false).is_ok());
        assert!(trusted_xai_url(&json!("https://x.ai.attacker.com/token"), false).is_err());
        assert!(trusted_xai_url(&json!("http://auth.x.ai/token"), false).is_err());
        assert!(trusted_xai_url(&json!("http://127.0.0.1:9999/token"), true).is_ok());
    }
}
