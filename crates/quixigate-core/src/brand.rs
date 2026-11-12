//! Centralized brand and attribution identity constants for QuixiGate (`QuixiAI/QuixiGate`).

pub const BRAND_NAME: &str = "QuixiGate";
pub const BRAND_SLUG: &str = "quixigate";
pub const BIN_NAME: &str = "quixigate";
pub const DATA_DIR_NAME: &str = ".quixigate";
pub const DB_FILE_NAME: &str = "quixigate.db";

pub const VIRTUAL_KEY_PREFIX: &str = "sk-qx-";
pub const MASTER_KEY_PREFIX: &str = "sk-qx-master-";
pub const SESSION_TOKEN_PREFIX: &str = "qxs_";
pub const SESSION_COOKIE_NAME: &str = "quixigate_session";

/// Default `HTTP-Referer` sent on OpenRouter upstream calls for app leaderboard & signup attribution.
pub const DEFAULT_OPENROUTER_APP_URL: &str = "https://quixi.ai";
/// Default `X-Title` sent on OpenRouter upstream calls for app leaderboard & signup attribution.
pub const DEFAULT_OPENROUTER_APP_TITLE: &str = "QuixiGate";
/// Default OpenRouter fallback model used when `openrouter_burst_enabled` is active and a local-only tag overflows.
pub const DEFAULT_OPENROUTER_BURST_MODEL: &str = "openrouter/auto";

#[inline]
pub fn is_virtual_key_token(token: &str) -> bool {
    token.starts_with(VIRTUAL_KEY_PREFIX)
}

#[inline]
pub fn is_session_token(token: &str) -> bool {
    token.starts_with(SESSION_TOKEN_PREFIX)
}

/// Returns a trimmed non-empty environment variable value if present.
pub fn env_opt(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}
