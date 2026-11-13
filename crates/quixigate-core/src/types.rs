//! Core domain records shared across QuixiGate crates.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Datelike, Duration as ChronoDuration, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::money::NanoUsd;

/// Current Unix epoch timestamp in milliseconds.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserRole {
    Admin,
    Member,
    Viewer,
}

impl UserRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admin => "admin",
            Self::Member => "member",
            Self::Viewer => "viewer",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "admin" => Self::Admin,
            "viewer" => Self::Viewer,
            _ => Self::Member,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetWindow {
    Hourly,
    Daily,
    Weekly,
    #[default]
    Monthly,
    Total,
}

impl BudgetWindow {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hourly => "hourly",
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
            Self::Total => "total",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "hourly" => Self::Hourly,
            "daily" => Self::Daily,
            "weekly" => Self::Weekly,
            "total" => Self::Total,
            _ => Self::Monthly,
        }
    }

    /// Computes the next UTC boundary reset timestamp in milliseconds, or `None` for `Total`.
    pub fn next_reset_ms(self, from_ms: i64) -> Option<i64> {
        let dt = Utc
            .timestamp_millis_opt(from_ms)
            .single()
            .unwrap_or_else(Utc::now);
        match self {
            Self::Hourly => Some((dt + ChronoDuration::hours(1)).timestamp_millis()),
            Self::Daily => Some((dt + ChronoDuration::days(1)).timestamp_millis()),
            Self::Weekly => Some((dt + ChronoDuration::days(7)).timestamp_millis()),
            Self::Monthly => {
                let (year, month) = if dt.month() == 12 {
                    (dt.year() + 1, 1)
                } else {
                    (dt.year(), dt.month() + 1)
                };
                Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0)
                    .single()
                    .map(|next| next.timestamp_millis())
                    .or_else(|| Some((dt + ChronoDuration::days(30)).timestamp_millis()))
            }
            Self::Total => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub display_name: String,
    pub role: UserRole,
    pub has_password: bool,
    #[serde(default)]
    pub blocked: bool,
    pub budget_limit_nanos: Option<NanoUsd>,
    pub budget_window: BudgetWindow,
    pub spend_nanos: NanoUsd,
    pub window_reset_at_ms: Option<i64>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamMember {
    pub user_id: String,
    pub role: UserRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub blocked: bool,
    #[serde(default)]
    pub disable_logging: bool,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    pub budget_limit_nanos: Option<NanoUsd>,
    pub budget_window: BudgetWindow,
    pub spend_nanos: NanoUsd,
    pub window_reset_at_ms: Option<i64>,
    pub member_ids: Vec<String>,
    #[serde(default)]
    pub members: Vec<TeamMember>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub model_budgets_nanos: BTreeMap<String, NanoUsd>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndUser {
    pub user_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(default)]
    pub blocked: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_id: Option<String>,
    pub budget_limit_nanos: Option<NanoUsd>,
    pub budget_window: BudgetWindow,
    pub spend_nanos: NanoUsd,
    pub window_reset_at_ms: Option<i64>,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetRecord {
    pub budget_id: String,
    pub max_budget_nanos: Option<NanoUsd>,
    pub soft_budget_nanos: Option<NanoUsd>,
    pub max_parallel_requests: Option<u32>,
    pub tpm_limit: Option<u64>,
    pub rpm_limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub model_max_budget_nanos: BTreeMap<String, NanoUsd>,
    pub budget_reset_at_ms: Option<i64>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    pub tag_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_id: Option<String>,
    pub max_budget_nanos: Option<NanoUsd>,
    pub spend_nanos: NanoUsd,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessGroupRecord {
    pub access_group: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub model_names: Vec<String>,
    #[serde(default)]
    pub mcp_server_ids: Vec<String>,
    pub max_budget_nanos: Option<NanoUsd>,
    pub tpm_limit: Option<u64>,
    pub rpm_limit: Option<u32>,
    pub spend_nanos: NanoUsd,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualKey {
    pub id: String,
    pub key_prefix: String,
    pub name: String,
    pub user_id: String,
    pub team_id: Option<String>,
    pub budget_limit_nanos: Option<NanoUsd>,
    pub budget_window: BudgetWindow,
    pub spend_nanos: NanoUsd,
    pub window_reset_at_ms: Option<i64>,
    pub rpm_limit: Option<u32>,
    pub tpm_limit: Option<u64>,
    pub max_concurrent: Option<u32>,
    pub allowed_models: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub model_budgets_nanos: BTreeMap<String, NanoUsd>,
    #[serde(default)]
    pub allowed_cidrs: Vec<String>,
    #[serde(default = "default_true")]
    pub allow_spend_read: bool,
    pub log_payloads: Option<bool>,
    pub expires_at_ms: Option<i64>,
    pub revoked_at_ms: Option<i64>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthKind {
    Oauth,
    SetupToken,
    ApiKey,
    LocalNoauth,
}

impl AuthKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oauth => "oauth",
            Self::SetupToken => "setup_token",
            Self::ApiKey => "api_key",
            Self::LocalNoauth => "local_noauth",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "oauth" => Self::Oauth,
            "setup_token" => Self::SetupToken,
            "local_noauth" => Self::LocalNoauth,
            _ => Self::ApiKey,
        }
    }

    /// True when usage does not incur per-token API charges (subscription OAuth or local server).
    pub fn is_zero_marginal_cost(self) -> bool {
        matches!(self, Self::Oauth | Self::SetupToken | Self::LocalNoauth)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialStatus {
    #[default]
    Ok,
    Exhausted,
    Dead,
}

impl CredentialStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Exhausted => "exhausted",
            Self::Dead => "dead",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "exhausted" => Self::Exhausted,
            "dead" => Self::Dead,
            _ => Self::Ok,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolStrategy {
    #[default]
    FillFirst,
    LeastUsed,
    RoundRobin,
    Random,
}

/// Encrypted secret material stored in `credentials.encrypted_payload`.
/// Never serialized into public API responses or logs.
#[derive(Clone, Serialize, Deserialize, Zeroize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CredentialSecret {
    ApiKey {
        key: String,
    },
    Codex {
        access: String,
        refresh: String,
        expires_ms: i64,
        account_id: String,
    },
    Xai {
        access: String,
        refresh: String,
        expires_ms: i64,
        token_endpoint: String,
    },
    AnthropicSetupToken {
        token: String,
    },
    GeminiCli {
        access: String,
        refresh: String,
        expires_ms: i64,
        project_id: String,
        email: Option<String>,
        client_id: String,
        client_secret: String,
    },
    GitHubCopilot {
        github_token: String,
        access: String,
        expires_ms: i64,
        base_url: String,
        enterprise_domain: Option<String>,
    },
    QwenPortal {
        access: String,
        refresh: String,
        expires_ms: i64,
        resource_url: Option<String>,
    },
    MiniMaxPortal {
        access: String,
        refresh: String,
        expires_ms: i64,
        region: String,
        resource_url: Option<String>,
    },
    Chutes {
        access: String,
        refresh: String,
        expires_ms: i64,
        email: Option<String>,
        account_id: Option<String>,
        client_id: String,
        client_secret: Option<String>,
    },
    OpenRouterOidc {
        access: String,
        expires_ms: i64,
        federation_policy_id: Option<String>,
        subject_token: Option<String>,
        subject_token_env: Option<String>,
        token_endpoint: String,
    },
    LocalNoauth,
}

impl std::fmt::Debug for CredentialSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CredentialSecret(<redacted>)")
    }
}

impl CredentialSecret {
    pub fn expires_at_ms(&self) -> Option<i64> {
        match self {
            Self::Codex { expires_ms, .. }
            | Self::Xai { expires_ms, .. }
            | Self::GeminiCli { expires_ms, .. }
            | Self::GitHubCopilot { expires_ms, .. }
            | Self::QwenPortal { expires_ms, .. }
            | Self::MiniMaxPortal { expires_ms, .. }
            | Self::Chutes { expires_ms, .. }
            | Self::OpenRouterOidc { expires_ms, .. } => Some(*expires_ms),
            Self::ApiKey { .. } | Self::AnthropicSetupToken { .. } | Self::LocalNoauth => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaWindow {
    pub label: String,
    pub used_percent: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reset_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuotaSnapshot {
    pub windows: Vec<QuotaWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance_credits: Option<f64>,
    pub fetched_at_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Safe public metadata view of a stored credential/seat (never includes raw secrets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRecord {
    pub id: String,
    pub provider_id: String,
    pub label: String,
    pub auth_kind: AuthKind,
    pub base_url: Option<String>,
    pub priority: i32,
    pub enabled: bool,
    pub allow_api_overflow: bool,
    pub status: CredentialStatus,
    pub status_reason: Option<String>,
    pub cooldown_until_ms: Option<i64>,
    pub model_cooldowns: BTreeMap<String, i64>,
    pub quota_snapshot: Option<QuotaSnapshot>,
    pub expires_at_ms: Option<i64>,
    pub last_used_at_ms: Option<i64>,
    pub created_at_ms: i64,
}

impl CredentialRecord {
    /// Checks whether this seat is currently usable for `model` at `now_ms`.
    pub fn is_healthy_for(&self, model: Option<&str>, now_ms: i64) -> bool {
        if !self.enabled || self.status == CredentialStatus::Dead {
            return false;
        }
        if self.status == CredentialStatus::Exhausted
            && self.cooldown_until_ms.is_some_and(|until| until > now_ms)
        {
            return false;
        }
        if let Some(until) = self.cooldown_until_ms
            && until > now_ms
        {
            return false;
        }
        if let Some(model) = model {
            if self
                .model_cooldowns
                .get(model)
                .is_some_and(|&until| until > now_ms)
            {
                return false;
            }
        } else if self.model_cooldowns.values().any(|&until| until > now_ms) {
            return false;
        }
        // Proactive quota check: skip if a primary window is at 100% with future reset.
        if let Some(snapshot) = &self.quota_snapshot
            && snapshot.windows.iter().any(|w| {
                w.used_percent >= 100.0 && w.reset_at_ms.is_some_and(|reset| reset > now_ms)
            })
        {
            return false;
        }
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTarget {
    /// Canonical model or `provider/model` target.
    pub model: String,
    /// Optional explicit provider or credential ID pin.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRoute {
    pub alias: String,
    pub targets: Vec<RouteTarget>,
    #[serde(default)]
    pub pool_strategy: PoolStrategy,
    #[serde(default = "default_true")]
    pub allow_paid_overflow: bool,
    #[serde(default)]
    pub blocked: bool,
    pub input_nanos_per_m: Option<i64>,
    pub output_nanos_per_m: Option<i64>,
    pub cache_read_nanos_per_m: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at_ms: Option<i64>,
    pub updated_at_ms: i64,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngressProtocol {
    ChatCompletions,
    TextCompletions,
    Responses,
    ResponsesWs,
    AnthropicMessages,
    Gemini,
    VercelUi,
    Embeddings,
    Images,
    Audio,
    Moderations,
    Rerank,
    Search,
    Mcp,
}

impl IngressProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat_completions",
            Self::TextCompletions => "text_completions",
            Self::Responses => "responses",
            Self::ResponsesWs => "responses_ws",
            Self::AnthropicMessages => "anthropic_messages",
            Self::Gemini => "gemini",
            Self::VercelUi => "vercel_ui",
            Self::Embeddings => "embeddings",
            Self::Images => "images",
            Self::Audio => "audio",
            Self::Moderations => "moderations",
            Self::Rerank => "rerank",
            Self::Search => "search",
            Self::Mcp => "mcp",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "text_completions" => Self::TextCompletions,
            "responses" => Self::Responses,
            "responses_ws" => Self::ResponsesWs,
            "anthropic_messages" => Self::AnthropicMessages,
            "gemini" => Self::Gemini,
            "vercel_ui" => Self::VercelUi,
            "embeddings" => Self::Embeddings,
            "images" => Self::Images,
            "audio" => Self::Audio,
            "moderations" => Self::Moderations,
            "rerank" => Self::Rerank,
            "search" => Self::Search,
            "mcp" => Self::Mcp,
            _ => Self::ChatCompletions,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLog {
    pub id: String,
    pub timestamp_ms: i64,
    pub ingress_protocol: IngressProtocol,
    pub requested_model: String,
    pub resolved_model: String,
    pub provider_id: String,
    pub credential_id: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub virtual_key_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub status_code: u16,
    pub stream: bool,
    pub fallback_attempts: u32,
    pub cache_hit: bool,
    pub ttft_ms: Option<u64>,
    pub duration_ms: u64,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub output_tokens: u64,
    pub cost_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
    pub otel_trace_id: Option<String>,
    pub error_message: Option<String>,
    pub request_body_json: Option<String>,
    pub response_body_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: String,
    pub timestamp_ms: i64,
    pub actor_id: String,
    pub actor_role: String,
    pub action: String,
    pub table_name: String,
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_value: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_values: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendReportRow {
    pub group_key: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub total_tokens: u64,
    pub spend_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
    pub savings_nanos: NanoUsd,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretScannerMode {
    #[default]
    Off,
    Redact,
    Block,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OtlpProtocol {
    #[default]
    HttpProtobuf,
    HttpJson,
    Grpc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewaySettings {
    /// Enforce auth even on `127.0.0.1` loopback when true.
    #[serde(default)]
    pub require_auth: bool,
    /// Permit passwordless local access from local container runtimes (`host.docker.internal`,
    /// `host.containers.internal`, `host.orb.internal`, `host.lima.internal`, and local Docker/Podman/OrbStack
    /// bridge subnets) when `require_auth` is false. Enabled by default so Open WebUI / LibreChat in Docker work seamlessly.
    #[serde(default = "default_trust_local_containers")]
    pub trust_local_containers: bool,
    /// Optional additional trusted IP/CIDR ranges (e.g. `100.64.0.0/10` for Tailscale or `192.168.1.0/24` for home LAN)
    /// allowed without an API key when `require_auth` is false.
    #[serde(default = "default_trusted_cidrs")]
    pub trusted_cidrs: Vec<String>,
    /// Optional global IP/CIDR allowlist (`allowed_ips`). Empty allows all network peers; non-empty rejects
    /// requests from TCP peers outside these CIDRs with `403 ip_not_allowed`.
    #[serde(default = "default_allowed_cidrs")]
    pub allowed_cidrs: Vec<String>,
    /// Split-plane flag (`DISABLE_ADMIN_ENDPOINTS`): rejects `/api/admin/*` and management routes on data-plane nodes.
    #[serde(default = "default_disable_admin_endpoints")]
    pub disable_admin_endpoints: bool,
    /// Split-plane flag (`DISABLE_LLM_API_ENDPOINTS`): rejects inference routes on control-plane-only nodes.
    #[serde(default = "default_disable_inference_endpoints")]
    pub disable_inference_endpoints: bool,
    /// Serve `Disallow: /` on `GET /robots.txt` to block web crawlers (default `true`).
    #[serde(default = "default_block_robots")]
    pub block_robots: bool,
    /// Maximum allowed HTTP request body size in megabytes (default `32` MB).
    #[serde(default = "default_max_request_size_mb")]
    pub max_request_size_mb: usize,
    /// Required JSON paths or headers on inference requests (e.g. `["user", "metadata.tags"]`).
    #[serde(default)]
    pub enforced_params: Vec<String>,
    /// Optional global tag-level max budgets (`tag -> NanoUsd`).
    #[serde(default)]
    pub tag_budgets_nanos: BTreeMap<String, NanoUsd>,
    /// In-process secret leak guardrail (`off` | `redact` | `block`).
    #[serde(default = "default_secret_scanner_mode")]
    pub secret_scanner_mode: SecretScannerMode,
    /// Case-insensitive banned phrases/keywords rejected pre-flight before upstream dispatch.
    #[serde(default)]
    pub banned_keywords: Vec<String>,
    /// Honor `x-quixigate-no-log: true` / `x-litellm-disable-callbacks: true` per-request privacy header (default `true`).
    #[serde(default = "default_true")]
    pub allow_no_log_header: bool,
    /// Expose native Streamable-HTTP Gateway MCP server at `POST /mcp` (default `true`).
    #[serde(default = "default_true")]
    pub mcp_server_enabled: bool,
    /// Optional custom OpenAPI `/openapi.json` & `/docs` title (`DOCS_TITLE` / `QUIXIGATE_DOCS_TITLE`).
    #[serde(
        default = "default_docs_title",
        skip_serializing_if = "Option::is_none"
    )]
    pub docs_title: Option<String>,
    /// Optional custom OpenAPI `/openapi.json` description (`DOCS_DESCRIPTION` / `QUIXIGATE_DOCS_DESCRIPTION`).
    #[serde(
        default = "default_docs_description",
        skip_serializing_if = "Option::is_none"
    )]
    pub docs_description: Option<String>,
    /// Filter management & admin endpoints out of `/openapi.json` for unauthenticated/non-admin callers (`DOCS_FILTERED`).
    #[serde(default = "default_docs_filtered")]
    pub docs_filtered: bool,
    /// Mount the optional convenience Web UI under `/ui` (default true; disable for pure headless API operation).
    #[serde(default = "default_true")]
    pub ui_enabled: bool,
    /// Capture prompt/response JSON payloads in SQLite & OTLP spans by default.
    #[serde(default)]
    pub capture_payloads_default: bool,
    /// Maximum trace rows retained in SQLite ring buffer (default 10,000).
    #[serde(default = "default_max_trace_rows")]
    pub max_trace_rows: usize,
    /// Enable exact-match response caching globally unless overridden per request.
    #[serde(default)]
    pub cache_enabled_default: bool,
    /// Cache TTL in seconds (default 3600).
    #[serde(default = "default_cache_ttl")]
    pub cache_ttl_secs: u64,
    /// Enable OpenTelemetry / Arize Phoenix OTLP export.
    #[serde(default)]
    pub otlp_enabled: bool,
    /// OTLP HTTP `/v1/traces` or gRPC `:4317` endpoint (e.g. `http://localhost:6006/v1/traces`).
    #[serde(default = "default_phoenix_endpoint")]
    pub otlp_endpoint: String,
    #[serde(default)]
    pub otlp_protocol: OtlpProtocol,
    /// Optional Bearer API key for Arize Phoenix Cloud / OTLP collectors.
    #[serde(default)]
    pub otlp_api_key: Option<String>,
    /// `openinference.project.name` attribute (default `"quixigate"`).
    #[serde(default = "default_project_name")]
    pub openinference_project_name: String,
    /// Arize Phoenix UI base URL for deep-linking traces (`http://localhost:6006`).
    #[serde(default = "default_phoenix_ui_url")]
    pub phoenix_ui_url: String,
    /// Default multi-seat pool selection strategy.
    #[serde(default)]
    pub default_pool_strategy: PoolStrategy,
    /// Auto-discover local inference servers (`ollama`, `lmstudio`, `llamacpp`, `vllm`, `mlx`).
    #[serde(default = "default_true")]
    pub local_discovery_enabled: bool,
    /// Custom or overridden local runtime endpoints persisted across restarts.
    #[serde(default)]
    pub custom_local_endpoints: Vec<LocalEndpointConfig>,
    /// Optional MCP Streamable-HTTP JSON-RPC server URLs for server-side tool execution.
    #[serde(default)]
    pub mcp_server_urls: Vec<String>,
    /// OpenRouter attribution `HTTP-Referer` URL sent when routing via OpenRouter.
    #[serde(default = "default_openrouter_app_url")]
    pub openrouter_app_url: String,
    /// OpenRouter attribution `X-Title` header sent when routing via OpenRouter.
    #[serde(default = "default_openrouter_app_title")]
    pub openrouter_app_title: String,
    /// When true, automatically bursts onto OpenRouter whenever local runtimes or OAuth subscription seats
    /// saturate (`429`/`5xx`), are offline, or do not host the requested model.
    #[serde(default)]
    pub openrouter_burst_enabled: bool,
    /// Fallback OpenRouter model/router used when bursting a local-only tag (defaults to `"openrouter/auto"`).
    #[serde(default = "default_openrouter_burst_model")]
    pub openrouter_burst_model: String,
    /// Optional OIDC Issuer URL (e.g. Dex, Keycloak, Okta, Cloudflare Access) for `/.well-known/openid-configuration` discovery & `iss` verification.
    #[serde(default = "default_jwt_issuer_url", alias = "oidc_issuer_url")]
    pub jwt_issuer_url: Option<String>,
    /// Optional explicit JWKS URL(s) (`QUIXIGATE_JWKS_URL` / `JWT_PUBLIC_KEY_URL`, comma-separated supported).
    #[serde(default = "default_jwt_jwks_url", alias = "jwt_public_key_url")]
    pub jwt_jwks_url: Option<String>,
    /// Optional required JWT `aud` claim (single value or comma-separated list of accepted audiences).
    #[serde(default = "default_jwt_audience")]
    pub jwt_audience: Option<String>,
    /// Claim path used to resolve the QuixiGate username/ID (default `"sub"`, e.g. `"email"` or `"preferred_username"`).
    #[serde(default = "default_jwt_user_claim")]
    pub jwt_user_claim: String,
    /// Claim path used to inspect group/role strings (default `"groups"`, dot-paths like `"realm_access.roles"` supported).
    #[serde(default = "default_jwt_roles_claim")]
    pub jwt_roles_claim: String,
    /// Role/group names in `jwt_roles_claim` that grant QuixiGate `UserRole::Admin`.
    #[serde(default = "default_jwt_admin_roles")]
    pub jwt_admin_roles: Vec<String>,
    /// Optional claim path used to map OIDC callers to an QuixiGate `Team` (e.g. `"team_id"`, `"groups"`).
    #[serde(default = "default_jwt_team_claim")]
    pub jwt_team_claim: Option<String>,
    /// Automatically create/sync QuixiGate `User` records for valid OIDC/JWKS tokens (default `true`).
    #[serde(default = "default_jwt_auto_provision")]
    pub jwt_auto_provision_users: bool,
}

/// First-class OpenRouter Preset (`@preset/<slug>`) synced from `GET /api/v1/presets`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenRouterPreset {
    pub id: String,
    pub slug: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underlying_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalEndpointConfig {
    pub runtime_id: String,
    pub display_name: String,
    pub base_url: String,
}

const fn default_max_trace_rows() -> usize {
    10_000
}

const fn default_cache_ttl() -> u64 {
    3_600
}

fn parse_env_bool(names: &[&str], fallback: bool) -> bool {
    for name in names {
        if let Ok(v) = std::env::var(name) {
            match v.trim().to_ascii_lowercase().as_str() {
                "0" | "false" | "no" | "off" => return false,
                "1" | "true" | "yes" | "on" => return true,
                _ => {}
            }
        }
    }
    fallback
}

fn default_allowed_cidrs() -> Vec<String> {
    std::env::var("QUIXIGATE_ALLOWED_IPS")
        .or_else(|_| std::env::var("ALLOWED_IPS"))
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn default_disable_admin_endpoints() -> bool {
    parse_env_bool(
        &[
            "DISABLE_ADMIN_ENDPOINTS",
            "QUIXIGATE_DISABLE_ADMIN_ENDPOINTS",
        ],
        false,
    )
}

fn default_disable_inference_endpoints() -> bool {
    parse_env_bool(
        &[
            "DISABLE_LLM_API_ENDPOINTS",
            "QUIXIGATE_DISABLE_INFERENCE_ENDPOINTS",
        ],
        false,
    )
}

fn default_block_robots() -> bool {
    parse_env_bool(&["BLOCK_ROBOTS", "QUIXIGATE_BLOCK_ROBOTS"], true)
}

fn default_max_request_size_mb() -> usize {
    std::env::var("MAX_REQUEST_SIZE_MB")
        .or_else(|_| std::env::var("QUIXIGATE_MAX_REQUEST_SIZE_MB"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(32)
}

fn default_secret_scanner_mode() -> SecretScannerMode {
    match crate::brand::env_opt("QUIXIGATE_SECRET_SCANNER")
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("redact") => SecretScannerMode::Redact,
        Some("block") => SecretScannerMode::Block,
        _ => SecretScannerMode::Off,
    }
}

fn default_docs_title() -> Option<String> {
    crate::brand::env_opt("QUIXIGATE_DOCS_TITLE").or_else(|| crate::brand::env_opt("DOCS_TITLE"))
}

fn default_docs_description() -> Option<String> {
    crate::brand::env_opt("QUIXIGATE_DOCS_DESCRIPTION")
        .or_else(|| crate::brand::env_opt("DOCS_DESCRIPTION"))
}

fn default_docs_filtered() -> bool {
    parse_env_bool(&["DOCS_FILTERED", "QUIXIGATE_DOCS_FILTERED"], false)
}

fn default_phoenix_endpoint() -> String {
    "http://localhost:6006/v1/traces".to_owned()
}

fn default_phoenix_ui_url() -> String {
    "http://localhost:6006".to_owned()
}

fn default_project_name() -> String {
    crate::brand::BRAND_SLUG.to_owned()
}

fn default_openrouter_app_url() -> String {
    crate::brand::env_opt("QUIXIGATE_OPENROUTER_APP_URL")
        .unwrap_or_else(|| crate::brand::DEFAULT_OPENROUTER_APP_URL.to_owned())
}

fn default_openrouter_app_title() -> String {
    crate::brand::env_opt("QUIXIGATE_OPENROUTER_APP_TITLE")
        .unwrap_or_else(|| crate::brand::DEFAULT_OPENROUTER_APP_TITLE.to_owned())
}

fn default_openrouter_burst_model() -> String {
    crate::brand::DEFAULT_OPENROUTER_BURST_MODEL.to_owned()
}

fn default_jwt_issuer_url() -> Option<String> {
    crate::brand::env_opt("QUIXIGATE_OIDC_ISSUER_URL")
        .or_else(|| crate::brand::env_opt("OIDC_ISSUER_URL"))
}

fn default_jwt_jwks_url() -> Option<String> {
    crate::brand::env_opt("QUIXIGATE_JWKS_URL")
        .or_else(|| crate::brand::env_opt("JWT_PUBLIC_KEY_URL"))
}

fn default_jwt_audience() -> Option<String> {
    crate::brand::env_opt("QUIXIGATE_JWT_AUDIENCE")
        .or_else(|| crate::brand::env_opt("JWT_AUDIENCE"))
}

fn default_jwt_user_claim() -> String {
    crate::brand::env_opt("QUIXIGATE_JWT_USER_CLAIM").unwrap_or_else(|| "sub".to_owned())
}

fn default_jwt_roles_claim() -> String {
    crate::brand::env_opt("QUIXIGATE_JWT_ROLES_CLAIM").unwrap_or_else(|| "groups".to_owned())
}

fn default_jwt_admin_roles() -> Vec<String> {
    if let Ok(raw) = std::env::var("QUIXIGATE_JWT_ADMIN_ROLES") {
        let parsed: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    vec!["quixigate-admin".to_owned(), "proxy_admin".to_owned()]
}

fn default_jwt_team_claim() -> Option<String> {
    crate::brand::env_opt("QUIXIGATE_JWT_TEAM_CLAIM")
}

fn default_jwt_auto_provision() -> bool {
    parse_env_bool(&["QUIXIGATE_JWT_AUTO_PROVISION"], true)
}

fn default_trust_local_containers() -> bool {
    parse_env_bool(&["QUIXIGATE_TRUST_CONTAINERS"], true)
}

fn default_trusted_cidrs() -> Vec<String> {
    std::env::var("QUIXIGATE_TRUSTED_CIDRS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

impl Default for GatewaySettings {
    fn default() -> Self {
        Self {
            require_auth: false,
            trust_local_containers: default_trust_local_containers(),
            trusted_cidrs: default_trusted_cidrs(),
            allowed_cidrs: default_allowed_cidrs(),
            disable_admin_endpoints: default_disable_admin_endpoints(),
            disable_inference_endpoints: default_disable_inference_endpoints(),
            block_robots: default_block_robots(),
            max_request_size_mb: default_max_request_size_mb(),
            enforced_params: Vec::new(),
            tag_budgets_nanos: BTreeMap::new(),
            secret_scanner_mode: default_secret_scanner_mode(),
            banned_keywords: Vec::new(),
            allow_no_log_header: true,
            mcp_server_enabled: true,
            docs_title: default_docs_title(),
            docs_description: default_docs_description(),
            docs_filtered: default_docs_filtered(),
            ui_enabled: true,
            capture_payloads_default: false,
            max_trace_rows: default_max_trace_rows(),
            cache_enabled_default: false,
            cache_ttl_secs: default_cache_ttl(),
            otlp_enabled: false,
            otlp_endpoint: default_phoenix_endpoint(),
            otlp_protocol: OtlpProtocol::HttpProtobuf,
            otlp_api_key: None,
            openinference_project_name: default_project_name(),
            phoenix_ui_url: default_phoenix_ui_url(),
            default_pool_strategy: PoolStrategy::FillFirst,
            local_discovery_enabled: true,
            custom_local_endpoints: Vec::new(),
            mcp_server_urls: Vec::new(),
            openrouter_app_url: default_openrouter_app_url(),
            openrouter_app_title: default_openrouter_app_title(),
            openrouter_burst_enabled: false,
            openrouter_burst_model: default_openrouter_burst_model(),
            jwt_issuer_url: default_jwt_issuer_url(),
            jwt_jwks_url: default_jwt_jwks_url(),
            jwt_audience: default_jwt_audience(),
            jwt_user_claim: default_jwt_user_claim(),
            jwt_roles_claim: default_jwt_roles_claim(),
            jwt_admin_roles: default_jwt_admin_roles(),
            jwt_team_claim: default_jwt_team_claim(),
            jwt_auto_provision_users: default_jwt_auto_provision(),
        }
    }
}

/// Per-model capability & residency metadata discovered from local runtimes
/// (`llama-server` single-instance & router modes, Ollama `/api/ps`, LM Studio `/api/v0/models`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalModelInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

/// Discovered local model runtime status surfaced in the dashboard & `/v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalRuntimeStatus {
    pub runtime_id: String,
    pub display_name: String,
    pub base_url: String,
    pub healthy: bool,
    #[serde(default)]
    pub auth_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_slots: Option<u32>,
    #[serde(default)]
    pub supports_load_control: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_version: Option<String>,
    pub models: Vec<String>,
    pub loaded_models: Vec<String>,
    #[serde(default)]
    pub model_details: BTreeMap<String, LocalModelInfo>,
    pub avg_ttft_ms: Option<u64>,
    pub avg_tok_per_sec: Option<f64>,
    pub last_checked_ms: i64,
}
