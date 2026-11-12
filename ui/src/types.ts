// Shared TypeScript interfaces matching QuixiGate's Rust `/api/admin/*` contracts (`NanoUsd` exact integer nanos).

export type UserRole = 'admin' | 'member' | 'viewer';
export type BudgetWindow = 'hourly' | 'daily' | 'weekly' | 'monthly' | 'total';
export type AuthKind = 'oauth' | 'setup_token' | 'api_key' | 'local_noauth';
export type CredentialStatus = 'ok' | 'exhausted' | 'dead';
export type PoolStrategy = 'fill_first' | 'least_used' | 'round_robin' | 'random';
export type OtlpProtocol = 'http_protobuf' | 'http_json' | 'grpc';

export interface User {
  id: string;
  username: string;
  display_name: string;
  role: UserRole;
  has_password: boolean;
  budget_limit_nanos: number | null;
  budget_window: BudgetWindow;
  spend_nanos: number;
  window_reset_at_ms: number | null;
  created_at_ms: number;
}

export interface Team {
  id: string;
  name: string;
  budget_limit_nanos: number | null;
  budget_window: BudgetWindow;
  spend_nanos: number;
  window_reset_at_ms: number | null;
  member_ids: string[];
  created_at_ms: number;
}

export interface VirtualKey {
  id: string;
  key_prefix: string;
  name: string;
  user_id: string;
  team_id: string | null;
  budget_limit_nanos: number | null;
  budget_window: BudgetWindow;
  spend_nanos: number;
  window_reset_at_ms: number | null;
  rpm_limit: number | null;
  tpm_limit: number | null;
  max_concurrent: number | null;
  allowed_models: string[];
  log_payloads: boolean | null;
  expires_at_ms: number | null;
  revoked_at_ms: number | null;
  created_at_ms: number;
}

export interface QuotaWindow {
  label: string;
  used_percent: number;
  reset_at_ms?: number | null;
}

export interface QuotaSnapshot {
  windows: QuotaWindow[];
  plan?: string | null;
  balance_credits?: number | null;
  fetched_at_ms: number;
  error?: string | null;
}

export interface CredentialRecord {
  id: string;
  provider_id: string;
  label: string;
  auth_kind: AuthKind;
  base_url: string | null;
  priority: number;
  enabled: boolean;
  allow_api_overflow: boolean;
  status: CredentialStatus;
  status_reason: string | null;
  cooldown_until_ms: number | null;
  model_cooldowns: Record<string, number>;
  quota_snapshot: QuotaSnapshot | null;
  expires_at_ms: number | null;
  last_used_at_ms: number | null;
  created_at_ms: number;
}

export interface OAuthProviderDescriptor {
  provider_id: string;
  vendor_id: string;
  display_name: string;
  auth_kind: AuthKind;
  flow_style: 'pkce_or_device' | 'device_code' | 'setup_token' | 'pkce';
  default_base_url: string;
  description: string;
}

export interface ApiProviderDescriptor {
  provider_id: string;
  display_name: string;
}

export interface OAuthWizardSession {
  session_id: string;
  provider_id: string;
  label: string;
  status: 'pending' | 'completed' | 'failed';
  authorize_url: string | null;
  verification_uri: string | null;
  user_code: string | null;
  accepts_manual_code: boolean;
  credential: CredentialRecord | null;
  error: string | null;
  created_at_ms: number;
}

export interface RouteTarget {
  model: string;
  provider_id?: string | null;
  credential_id?: string | null;
}

export interface ModelRoute {
  alias: string;
  targets: RouteTarget[];
  pool_strategy: PoolStrategy;
  allow_paid_overflow: boolean;
  input_nanos_per_m: number | null;
  output_nanos_per_m: number | null;
  cache_read_nanos_per_m: number | null;
  updated_at_ms: number;
}

export interface CatalogModel {
  canonical_id: string;
  bare_id: string;
  author: string;
  display_name: string;
  description?: string | null;
  family?: string | null;
  providers: Record<string, string>;
  context_window: number;
  max_output_tokens: number;
  input_nanos_per_m: number;
  output_nanos_per_m: number;
  cache_read_nanos_per_m: number;
  supports_tools: boolean;
  supports_reasoning: boolean;
  supports_vision: boolean;
  supports_structured_output: boolean;
  release_date?: string | null;
  is_local: boolean;
  is_preset?: boolean;
}

export interface OpenRouterPreset {
  id: string;
  slug: string;
  name: string;
  description?: string | null;
  underlying_model?: string | null;
  updated_at?: string | null;
}

export interface OpenRouterStatusResponse {
  connected: boolean;
  seats: CredentialRecord[];
  env_key_detected: boolean;
  env_key_masked?: string | null;
  oidc_env_detected?: string | null;
  catalog_model_count: number;
  preset_count: number;
  presets: OpenRouterPreset[];
  app_url: string;
  app_title: string;
  burst_enabled?: boolean;
  burst_model?: string;
}

export interface LocalModelInfo {
  context_window?: number | null;
  supports_vision: boolean;
  status?: string | null;
  aliases?: string[];
}

export interface LocalEndpointConfig {
  runtime_id: string;
  display_name: string;
  base_url: string;
}

export interface LocalRuntimeStatus {
  runtime_id: string;
  display_name: string;
  base_url: string;
  healthy: boolean;
  auth_required?: boolean;
  mode?: string | null;
  total_slots?: number | null;
  supports_load_control?: boolean;
  server_version?: string | null;
  models: string[];
  loaded_models: string[];
  model_details?: Record<string, LocalModelInfo>;
  avg_ttft_ms: number | null;
  avg_tok_per_sec: number | null;
  last_checked_ms: number;
}

export interface ProviderSpendSummary {
  provider_id: string;
  requests: number;
  tokens: number;
  spend_nanos: number;
  api_equiv_nanos: number;
}

export interface ModelSpendSummary {
  model: string;
  requests: number;
  tokens: number;
  avg_ttft_ms: number | null;
  spend_nanos: number;
  api_equiv_nanos: number;
}

export interface DashboardMetrics {
  active_requests: number;
  total_requests_24h: number;
  error_requests_24h: number;
  cache_hits_24h: number;
  input_tokens_24h: number;
  output_tokens_24h: number;
  cached_tokens_24h: number;
  avg_tok_per_sec_24h: number;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  spend_24h_nanos: number;
  spend_30d_nanos: number;
  savings_30d_nanos: number;
  by_provider: ProviderSpendSummary[];
  by_model: ModelSpendSummary[];
}

export interface RequestLog {
  id: string;
  timestamp_ms: number;
  ingress_protocol: string;
  requested_model: string;
  resolved_model: string;
  provider_id: string;
  credential_id: string | null;
  user_id: string | null;
  team_id: string | null;
  virtual_key_id: string | null;
  status_code: number;
  stream: boolean;
  fallback_attempts: number;
  cache_hit: boolean;
  ttft_ms: number | null;
  duration_ms: number;
  input_tokens: number;
  cached_input_tokens: number;
  reasoning_tokens: number;
  output_tokens: number;
  cost_nanos: number;
  api_equiv_nanos: number;
  otel_trace_id: string | null;
  error_message: string | null;
  request_body_json: string | null;
  response_body_json: string | null;
}

export interface GatewaySettings {
  require_auth: boolean;
  trust_local_containers?: boolean;
  trusted_cidrs?: string[];
  ui_enabled?: boolean;
  capture_payloads_default: boolean;
  max_trace_rows: number;
  cache_enabled_default: boolean;
  cache_ttl_secs: number;
  otlp_enabled: boolean;
  otlp_endpoint: string;
  otlp_protocol: OtlpProtocol;
  otlp_api_key: string | null;
  openinference_project_name: string;
  phoenix_ui_url: string;
  default_pool_strategy: PoolStrategy;
  local_discovery_enabled: boolean;
  custom_local_endpoints?: LocalEndpointConfig[];
  mcp_server_urls?: string[];
  openrouter_app_url?: string;
  openrouter_app_title?: string;
  openrouter_burst_enabled?: boolean;
  openrouter_burst_model?: string;
}

export interface OverviewResponse {
  uptime_ms: number;
  bound_loopback: boolean;
  ui_enabled?: boolean;
  openrouter_connected?: boolean;
  openrouter_presets_count?: number;
  metrics: DashboardMetrics;
  credentials: CredentialRecord[];
  local_runtimes: LocalRuntimeStatus[];
  routes: ModelRoute[];
  settings: GatewaySettings;
  recent_traces: RequestLog[];
}

export interface AuthStatusResponse {
  require_auth: boolean;
  bound_loopback: boolean;
  direct_loopback?: boolean;
  has_master_key?: boolean;
  has_password_user: boolean;
  authenticated: boolean;
  via_loopback_trust: boolean;
  via_master_key?: boolean;
  user: User | null;
}

export interface TestSpanResult {
  ok: boolean;
  trace_id: string;
  endpoint: string;
  status_code: number;
  phoenix_url: string;
  message: string;
}

export interface ChatMessageMetadata {
  model?: string;
  provider?: string;
  ttftMs?: number | null;
  durationMs?: number;
  inputTokens?: number;
  cachedTokens?: number;
  reasoningTokens?: number;
  outputTokens?: number;
  costNanos?: number;
  costFormatted?: string;
  apiEquivNanos?: number;
  apiEquivFormatted?: string;
  cacheHit?: boolean;
  fallbackAttempts?: number;
}
