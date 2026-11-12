//! Structured error type with protocol-accurate HTTP status codes.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, QuixigateError>;

#[derive(Debug, Error)]
pub enum QuixigateError {
    #[error("authentication required: {0}")]
    Unauthorized(String),

    #[error("forbidden: {0}")]
    Forbidden(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid request: {0}")]
    Validation(String),

    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),

    #[error("rate limit exceeded: {message}")]
    RateLimited {
        message: String,
        retry_after_secs: Option<u64>,
    },

    #[error("upstream provider '{provider}' error ({status}): {message}")]
    Upstream {
        status: u16,
        provider: String,
        message: String,
    },

    #[error("no healthy credentials or providers available for model '{0}'")]
    NoAvailableRoute(String),

    #[error("vault encryption error: {0}")]
    Vault(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("{0}")]
    Internal(String),
}

impl QuixigateError {
    pub fn http_status(&self) -> u16 {
        self.status_code()
    }

    pub fn status_code(&self) -> u16 {
        match self {
            Self::Unauthorized(_) => 401,
            Self::Forbidden(_) => 403,
            Self::NotFound(_) => 404,
            Self::Validation(_) => 400,
            Self::BudgetExceeded(_) => 402,
            Self::RateLimited { .. } => 429,
            Self::Upstream { status, .. } => {
                if (400..=599).contains(status) {
                    *status
                } else {
                    502
                }
            }
            Self::NoAvailableRoute(_) => 503,
            Self::Vault(_) | Self::Storage(_) | Self::Internal(_) => 500,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Unauthorized(_) => "invalid_api_key",
            Self::Forbidden(_) => "permission_denied",
            Self::NotFound(_) => "model_not_found",
            Self::Validation(_) => "invalid_request_error",
            Self::BudgetExceeded(_) => "insufficient_quota",
            Self::RateLimited { .. } => "rate_limit_exceeded",
            Self::Upstream { .. } => "upstream_provider_error",
            Self::NoAvailableRoute(_) => "no_available_provider",
            Self::Vault(_) => "vault_error",
            Self::Storage(_) => "storage_error",
            Self::Internal(_) => "internal_error",
        }
    }
}
