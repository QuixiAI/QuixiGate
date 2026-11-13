//! Shared `axum` application state, Master Key verifier, UI gate, and structured HTTP error envelope.

use std::sync::Arc;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use parking_lot::Mutex;
use quixigate_auth::OAuthWizardManager;
use quixigate_auth::quota::QuotaEndpoints;
use quixigate_core::{QuixigateError, now_ms};
use quixigate_db::Database;
use quixigate_router::GatewayRouter;
use quixigate_telemetry::TelemetryHub;
use rand::RngCore;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Generates a cryptographically random 256-bit gateway Master Key (`sk-qx-master-...`).
pub fn generate_master_key() -> String {
    let mut entropy = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut entropy);
    format!(
        "{}{}",
        quixigate_core::brand::MASTER_KEY_PREFIX,
        URL_SAFE_NO_PAD.encode(entropy)
    )
}

fn sha256_digest(raw: &[u8]) -> [u8; 32] {
    Sha256::digest(raw).into()
}

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub router: GatewayRouter,
    pub telemetry: TelemetryHub,
    pub oauth_wizard: OAuthWizardManager,
    pub quota_endpoints: QuotaEndpoints,
    /// Fallback loopback indicator for in-process `Router::oneshot` tests without TCP `ConnectInfo`.
    pub bound_loopback: bool,
    pub started_at_ms: i64,
    /// Constant-time SHA-256 digest of `QUIXIGATE_MASTER_KEY` / `LITELLM_MASTER_KEY` (or startup-generated key).
    master_key_sha256: Option<[u8; 32]>,
    /// Optional HMAC-SHA256 secret (`QUIXIGATE_JWT_SECRET` / `JWT_SECRET`) for inbound `HS256` JWT bearer auth.
    jwt_secret: Option<Arc<Vec<u8>>>,
    /// Inbound OIDC Discovery & JWKS (`RS*`/`PS*`/`ES*`/`EdDSA`) cache and verifier.
    pub oidc: crate::oidc::OidcVerifier,
    /// Optional startup/runtime UI override (`--ui`, `--no-ui`, `QUIXIGATE_UI_ENABLED`, `DISABLE_ADMIN_UI`,
    /// or ephemeral Management API toggle). Cleared or updated dynamically whenever `/ui` is toggled via API.
    ui_override: Arc<Mutex<Option<bool>>>,
}

impl AppState {
    pub fn new(
        db: Database,
        router: GatewayRouter,
        telemetry: TelemetryHub,
        oauth_wizard: OAuthWizardManager,
        bound_loopback: bool,
    ) -> Self {
        let jwt_secret = std::env::var("QUIXIGATE_JWT_SECRET")
            .ok()
            .or_else(|| std::env::var("JWT_SECRET").ok())
            .map(|s| s.trim().as_bytes().to_vec())
            .filter(|b| !b.is_empty())
            .map(Arc::new);
        Self {
            db,
            router,
            telemetry,
            oauth_wizard,
            quota_endpoints: QuotaEndpoints::default(),
            bound_loopback,
            started_at_ms: now_ms(),
            master_key_sha256: None,
            jwt_secret,
            oidc: crate::oidc::OidcVerifier::new(),
            ui_override: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_quota_endpoints(mut self, endpoints: QuotaEndpoints) -> Self {
        self.quota_endpoints = endpoints;
        self
    }

    pub fn with_master_key(mut self, raw_key: impl AsRef<str>) -> Self {
        let trimmed = raw_key.as_ref().trim();
        if !trimmed.is_empty() {
            self.master_key_sha256 = Some(sha256_digest(trimmed.as_bytes()));
        }
        self
    }

    pub fn with_jwt_secret(mut self, raw_secret: impl AsRef<[u8]>) -> Self {
        let bytes = raw_secret.as_ref();
        if !bytes.is_empty() {
            self.jwt_secret = Some(Arc::new(bytes.to_vec()));
        } else {
            self.jwt_secret = None;
        }
        self
    }

    pub fn jwt_secret(&self) -> Option<&[u8]> {
        self.jwt_secret.as_deref().map(Vec::as_slice)
    }

    pub fn with_ui_override(self, ui_enabled: Option<bool>) -> Self {
        *self.ui_override.lock() = ui_enabled;
        self
    }

    /// Creates a core-local `AppState` clone with a dedicated outbound `reqwest::Client` connection
    /// pool pinned to the calling worker thread's Tokio reactor, while sharing `Database`,
    /// `ModelCatalog`, `PoolSelector`, and `TelemetryHub`.
    pub fn fork_for_worker_core(&self) -> Self {
        let mut forked = self.clone();
        forked.router = self
            .router
            .fork_with_http(quixigate_providers::build_inference_http_client());
        forked
    }

    pub const fn is_ui_compiled_in() -> bool {
        cfg!(feature = "ui")
    }

    pub fn ui_runtime_override(&self) -> Option<bool> {
        *self.ui_override.lock()
    }

    pub fn clear_ui_override(&self) {
        *self.ui_override.lock() = None;
    }

    /// Dynamically enables or disables the `/ui` convenience dashboard at runtime via the Management API,
    /// without requiring any CLI flag change, config file edit, or server restart.
    pub fn set_ui_enabled(&self, enabled: bool, persist: bool) -> quixigate_core::Result<Value> {
        if enabled && !Self::is_ui_compiled_in() {
            return Err(QuixigateError::Validation(
                "Web UI assets were excluded at compile time (--no-default-features); rebuild with default 'ui' feature to enable /ui".to_owned(),
            ));
        }
        if persist {
            let mut settings = self.db.get_settings()?;
            settings.ui_enabled = enabled;
            self.db.update_settings(&settings)?;
            *self.ui_override.lock() = None;
        } else {
            *self.ui_override.lock() = Some(enabled);
        }
        Ok(self.ui_status_json())
    }

    pub fn ui_status_json(&self) -> Value {
        let compiled_in = Self::is_ui_compiled_in();
        let persisted = self.db.settings_arc().ui_enabled;
        let runtime_override = self.ui_runtime_override();
        let enabled = self.is_ui_enabled();
        json!({
            "compiled_in": compiled_in,
            "enabled": enabled,
            "persisted_ui_enabled": persisted,
            "runtime_override": runtime_override,
            "ui_url": enabled.then_some("/ui"),
        })
    }

    pub fn has_master_key(&self) -> bool {
        self.master_key_sha256.is_some()
    }

    /// Constant-time verification of a candidate bearer / `x-litellm-api-key` / `x-quixigate-admin-key` against the Master Key.
    pub fn verify_master_key(&self, candidate: &str) -> bool {
        let Some(expected) = &self.master_key_sha256 else {
            return false;
        };
        let trimmed = candidate.trim();
        if trimmed.is_empty() {
            return false;
        }
        let actual = sha256_digest(trimmed.as_bytes());
        actual.ct_eq(expected).unwrap_u8() == 1
    }

    /// Returns whether the optional `/ui` convenience dashboard is compiled in and enabled.
    pub fn is_ui_enabled(&self) -> bool {
        if !Self::is_ui_compiled_in() {
            return false;
        }
        if let Some(forced) = *self.ui_override.lock() {
            return forced;
        }
        self.db.settings_arc().ui_enabled
    }
}

#[derive(Debug)]
pub struct ApiError(pub QuixigateError);

impl From<QuixigateError> for ApiError {
    fn from(value: QuixigateError) -> Self {
        Self(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = self.0.http_status();
        let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let (error_type, error_code) = match &self.0 {
            QuixigateError::Validation(_) => ("invalid_request_error", "invalid_request"),
            QuixigateError::Unauthorized(_) => ("authentication_error", "unauthorized"),
            QuixigateError::Forbidden(_) => ("permission_error", "forbidden"),
            QuixigateError::NotFound(_) => ("not_found_error", "not_found"),
            QuixigateError::BudgetExceeded(_) => ("insufficient_quota", "budget_exceeded"),
            QuixigateError::RateLimited { .. } => ("rate_limit_error", "rate_limit_exceeded"),
            QuixigateError::Upstream { status, .. } => (
                "upstream_error",
                if *status == 504 {
                    "upstream_timeout"
                } else {
                    "upstream_failure"
                },
            ),
            QuixigateError::NoAvailableRoute(_) => ("service_unavailable", "no_available_provider"),
            QuixigateError::Vault(_) => ("server_error", "vault_error"),
            QuixigateError::Storage(_) => ("storage_error", "storage_error"),
            QuixigateError::Internal(_) => ("server_error", "internal_error"),
        };

        let mut err_obj = serde_json::Map::new();
        err_obj.insert("message".into(), json!(self.0.to_string()));
        err_obj.insert("type".into(), json!(error_type));
        err_obj.insert("code".into(), json!(error_code));
        if let QuixigateError::Upstream { provider, .. } = &self.0 {
            err_obj.insert("provider".into(), json!(provider));
        }

        let mut resp = (
            status,
            Json(json!({
                "type": "error",
                "error": Value::Object(err_obj),
            })),
        )
            .into_response();

        if let QuixigateError::RateLimited {
            retry_after_secs: Some(secs),
            ..
        } = &self.0
            && let Ok(hv) = secs.to_string().parse()
        {
            resp.headers_mut()
                .insert(axum::http::header::RETRY_AFTER, hv);
        }

        resp
    }
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;
