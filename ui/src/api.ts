import type {
  ApiProviderDescriptor,
  AuthStatusResponse,
  CatalogModel,
  CredentialRecord,
  GatewaySettings,
  LocalRuntimeStatus,
  ModelRoute,
  OAuthProviderDescriptor,
  OAuthWizardSession,
  OpenRouterPreset,
  OpenRouterStatusResponse,
  OverviewResponse,
  RequestLog,
  Team,
  TestSpanResult,
  User,
  VirtualKey,
} from './types';

export class UnauthorizedError extends Error {
  constructor(message = 'Authentication required') {
    super(message);
    this.name = 'UnauthorizedError';
  }
}

let onUnauthorizedCallback: (() => void) | null = null;

export function setOnUnauthorized(cb: (() => void) | null) {
  onUnauthorizedCallback = cb;
}

/**
 * Formats integer NanoUsd (`1e9 = $1.00`) using exact BigInt arithmetic (never floating-point ledger math).
 */
export function formatNanos(nanos: number | null | undefined): string {
  if (nanos === null || nanos === undefined) return '—';
  const big = BigInt(Math.trunc(nanos));
  const neg = big < 0n;
  const abs = neg ? -big : big;
  const whole = abs / 1_000_000_000n;
  const rem = abs % 1_000_000_000n;
  const sign = neg ? '-' : '';
  if (rem === 0n) {
    return `${sign}$${whole.toString()}.00`;
  }
  const fracRaw = rem.toString().padStart(9, '0').replace(/0+$/, '');
  const frac = fracRaw.length < 2 ? fracRaw.padEnd(2, '0') : fracRaw;
  // Trim visual precision for dashboard readability while keeping sub-cent detail
  const displayFrac = whole > 0n && frac.length > 4 ? frac.slice(0, 4) : frac.slice(0, 6);
  return `${sign}$${whole.toString()}.${displayFrac}`;
}

export function formatTokens(count: number | null | undefined): string {
  if (!count) return '0';
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(2)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return count.toLocaleString();
}

export function formatRelativeTime(timestampMs: number | null | undefined): string {
  if (!timestampMs) return '—';
  const diffSec = Math.round((timestampMs - Date.now()) / 1000);
  const abs = Math.abs(diffSec);
  if (abs < 60) return diffSec >= 0 ? `in ${abs}s` : `${abs}s ago`;
  const mins = Math.round(abs / 60);
  if (mins < 60) return diffSec >= 0 ? `in ${mins}m` : `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return diffSec >= 0 ? `in ${hours}h` : `${hours}h ago`;
  const days = Math.round(hours / 24);
  return diffSec >= 0 ? `in ${days}d` : `${days}d ago`;
}

/**
 * Returns true ONLY if `model` can actually be served right now by a healthy
 * local runtime (`m.is_local`) or an enabled credential in `enabledProviders`.
 * Never surfaces OpenRouter or unconfigured vendor models unless that provider is added.
 */
export function isModelRoutable(
  m: CatalogModel,
  enabledProviders: ReadonlySet<string>,
): boolean {
  if (m.is_local) return true;
  if (enabledProviders.size === 0) return false;
  const isOpenRouterExclusive =
    Boolean(m.is_preset) ||
    m.canonical_id.startsWith('@preset/') ||
    m.canonical_id.startsWith('~') ||
    m.canonical_id === 'openrouter/auto' ||
    m.canonical_id === 'openrouter/free';
  if (isOpenRouterExclusive) {
    return enabledProviders.has('openrouter') && 'openrouter' in m.providers;
  }
  if (
    enabledProviders.has('openaicompatible') ||
    enabledProviders.has('vercel') ||
    enabledProviders.has('cloudflare-ai-gateway') ||
    enabledProviders.has('helicone')
  ) {
    return true;
  }

  for (const p of Object.keys(m.providers)) {
    if (enabledProviders.has(p)) return true;
    if (p === 'openai' && enabledProviders.has('openai-codex')) return true;
    if (p === 'anthropic' && enabledProviders.has('anthropic-token')) return true;
    if (p === 'google' && enabledProviders.has('google-gemini-cli')) return true;
    if (p === 'xai' && enabledProviders.has('xai-oauth')) return true;
    if (p === 'alibaba' && enabledProviders.has('qwen-portal')) return true;
    if (p === 'minimax' && enabledProviders.has('minimax-portal')) return true;
    if (
      (p === 'openai' || p === 'anthropic' || p === 'google') &&
      enabledProviders.has('github-copilot')
    ) {
      return true;
    }
  }

  if (enabledProviders.has('qwen-portal') && m.author === 'qwen') return true;
  if (enabledProviders.has('minimax-portal') && m.author === 'minimax') return true;
  return false;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (res.status === 401 && !path.includes('/api/admin/auth/')) {
    onUnauthorizedCallback?.();
    throw new UnauthorizedError();
  }

  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) {
    const msg = data?.error?.message || data?.message || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data as T;
}

export const api = {
  // Auth
  getAuthStatus: () => requestJson<AuthStatusResponse>('/api/admin/auth/status'),
  bootstrapPassword: (password: string, username?: string) =>
    requestJson<{ ok: boolean; user: User }>('/api/admin/auth/bootstrap', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  login: (username: string, password: string) =>
    requestJson<{ ok: boolean; user: User }>('/api/admin/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  logout: () =>
    requestJson<{ ok: boolean }>('/api/admin/auth/logout', { method: 'POST' }),
  setPassword: (password: string) =>
    requestJson<User>('/api/admin/auth/password', {
      method: 'POST',
      body: JSON.stringify({ password }),
    }),

  // Overview
  getOverview: () => requestJson<OverviewResponse>('/api/admin/overview'),

  // Providers & OAuth Seats
  getProvidersCatalog: () =>
    requestJson<{
      oauth_providers: OAuthProviderDescriptor[];
      api_providers: ApiProviderDescriptor[];
    }>('/api/admin/providers'),
  getCredentials: () => requestJson<CredentialRecord[]>('/api/admin/credentials'),
  createCredential: (payload: {
    id?: string;
    provider_id: string;
    label: string;
    auth_kind: 'api_key' | 'setup_token' | 'local_noauth';
    api_key?: string;
    setup_token?: string;
    base_url?: string;
    priority?: number;
    enabled?: boolean;
    allow_api_overflow?: boolean;
  }) =>
    requestJson<CredentialRecord>('/api/admin/credentials', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  patchCredential: (
    id: string,
    payload: {
      label?: string;
      base_url?: string | null;
      priority?: number;
      enabled?: boolean;
      allow_api_overflow?: boolean;
    },
  ) =>
    requestJson<CredentialRecord>(`/api/admin/credentials/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteCredential: (id: string) =>
    requestJson<{ deleted: boolean }>(`/api/admin/credentials/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  resetCredentialHealth: (id: string) =>
    requestJson<CredentialRecord>(
      `/api/admin/credentials/${encodeURIComponent(id)}/reset-health`,
      { method: 'POST' },
    ),
  refreshCredentialQuota: (id: string) =>
    requestJson<CredentialRecord>(
      `/api/admin/credentials/${encodeURIComponent(id)}/refresh-quota`,
      { method: 'POST' },
    ),
  testCredential: (id: string, model?: string) =>
    requestJson<{
      ok: boolean;
      latency_ms: number;
      preview?: string;
      credential: CredentialRecord;
    }>(`/api/admin/credentials/${encodeURIComponent(id)}/test`, {
      method: 'POST',
      body: JSON.stringify({ model }),
    }),

  // OAuth Wizard
  startOAuth: (payload: {
    provider_id: string;
    label?: string;
    priority?: number;
    prefer_device_code?: boolean;
    setup_token?: string;
    client_id?: string;
    client_secret?: string;
    region?: string;
    enterprise_domain?: string;
  }) =>
    requestJson<OAuthWizardSession>('/api/admin/oauth/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getOAuthSession: (sessionId: string) =>
    requestJson<OAuthWizardSession>(
      `/api/admin/oauth/sessions/${encodeURIComponent(sessionId)}`,
    ),
  completeOAuthSession: (sessionId: string, code: string) =>
    requestJson<OAuthWizardSession>(
      `/api/admin/oauth/sessions/${encodeURIComponent(sessionId)}/complete`,
      { method: 'POST', body: JSON.stringify({ code }) },
    ),

  // First-Class OpenRouter Hub
  getOpenRouterStatus: () =>
    requestJson<OpenRouterStatusResponse>('/api/admin/openrouter/status'),
  connectOpenRouter: (payload: {
    method: 'pkce' | 'env' | 'api_key' | 'oidc';
    label?: string;
    api_key?: string;
    env_var?: string;
    subject_token?: string;
    subject_token_env?: string;
    federation_policy_id?: string;
    token_endpoint?: string;
    base_url?: string;
    region?: string;
    priority?: number;
    allow_api_overflow?: boolean;
  }) =>
    requestJson<{
      mode: 'pkce' | 'connected';
      session?: OAuthWizardSession;
      credential?: CredentialRecord;
      synced_models?: number;
      synced_presets?: number;
      presets?: OpenRouterPreset[];
    }>('/api/admin/openrouter/connect', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  syncOpenRouter: () =>
    requestJson<{
      ok: boolean;
      synced_models: number;
      synced_presets: number;
      presets: OpenRouterPreset[];
      seats: CredentialRecord[];
    }>('/api/admin/openrouter/sync', { method: 'POST' }),
  setOpenRouterBurst: (payload: { enabled?: boolean; burst_model?: string }) =>
    requestJson<{
      enabled: boolean;
      burst_model: string;
      openrouter_connected: boolean;
      openrouter_seats: number;
      app_url: string;
      app_title: string;
    }>('/api/admin/burst', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Models, Routes & Local Runtimes
  getModelsBundle: () =>
    requestJson<{
      models: CatalogModel[];
      routes: ModelRoute[];
      credentials: CredentialRecord[];
      presets?: OpenRouterPreset[];
    }>('/api/admin/models'),
  syncModelsDev: (url?: string) =>
    requestJson<{ ok: boolean; synced_models: number }>('/api/admin/models/sync', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),
  upsertRoute: (payload: {
    alias: string;
    targets: { model: string; provider_id?: string | null; credential_id?: string | null }[];
    pool_strategy: string;
    allow_paid_overflow: boolean;
  }) =>
    requestJson<ModelRoute>('/api/admin/routes', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteRoute: (alias: string) =>
    requestJson<{ deleted: boolean }>(`/api/admin/routes/${encodeURIComponent(alias)}`, {
      method: 'DELETE',
    }),
  scanLocalRuntimes: () =>
    requestJson<LocalRuntimeStatus[]>('/api/admin/local-runtimes/scan', {
      method: 'POST',
    }),
  registerLocalRuntime: (payload: {
    runtime_id: string;
    display_name: string;
    base_url: string;
    api_key?: string;
  }) =>
    requestJson<LocalRuntimeStatus[]>('/api/admin/local-runtimes', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  loadLocalModel: (runtimeId: string, model: string) =>
    requestJson<LocalRuntimeStatus[]>(
      `/api/admin/local-runtimes/${encodeURIComponent(runtimeId)}/load`,
      {
        method: 'POST',
        body: JSON.stringify({ model }),
      },
    ),
  unloadLocalModel: (runtimeId: string, model: string) =>
    requestJson<LocalRuntimeStatus[]>(
      `/api/admin/local-runtimes/${encodeURIComponent(runtimeId)}/unload`,
      {
        method: 'POST',
        body: JSON.stringify({ model }),
      },
    ),

  // Users, Teams & Keys
  getUsers: () => requestJson<User[]>('/api/admin/users'),
  createUser: (payload: {
    username: string;
    display_name?: string;
    role?: string;
    password?: string;
    budget_usd?: string;
    budget_window?: string;
  }) =>
    requestJson<User>('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  patchUser: (
    id: string,
    payload: {
      display_name?: string;
      role?: string;
      password?: string;
      budget_usd?: string | null;
      budget_window?: string;
    },
  ) =>
    requestJson<User>(`/api/admin/users/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteUser: (id: string) =>
    requestJson<{ deleted: boolean }>(`/api/admin/users/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  getTeams: () => requestJson<Team[]>('/api/admin/teams'),
  createTeam: (payload: {
    name: string;
    budget_usd?: string;
    budget_window?: string;
    member_ids: string[];
  }) =>
    requestJson<Team>('/api/admin/teams', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteTeam: (id: string) =>
    requestJson<{ deleted: boolean }>(`/api/admin/teams/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  getKeys: () => requestJson<VirtualKey[]>('/api/admin/keys'),
  createKey: (payload: {
    name: string;
    user_id?: string;
    team_id?: string;
    budget_usd?: string;
    budget_window?: string;
    rpm_limit?: number | null;
    tpm_limit?: number | null;
    max_concurrent?: number | null;
    allowed_models: string[];
    log_payloads?: boolean | null;
  }) =>
    requestJson<{ key: VirtualKey; raw_key: string }>('/api/admin/keys', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  rotateKey: (id: string, grace_seconds = 300) =>
    requestJson<{ key: VirtualKey; raw_key: string; grace_seconds: number }>(
      `/api/admin/keys/${encodeURIComponent(id)}/rotate`,
      { method: 'POST', body: JSON.stringify({ grace_seconds }) },
    ),
  revokeKey: (id: string) =>
    requestJson<VirtualKey>(`/api/admin/keys/${encodeURIComponent(id)}/revoke`, {
      method: 'POST',
    }),
  deleteKey: (id: string) =>
    requestJson<{ deleted: boolean }>(`/api/admin/keys/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  // Traces, Settings & Config
  getTraces: (params?: {
    model?: string;
    provider_id?: string;
    status_min?: number;
    limit?: number;
  }) => {
    const q = new URLSearchParams();
    if (params?.model) q.set('model', params.model);
    if (params?.provider_id) q.set('provider_id', params.provider_id);
    if (params?.status_min) q.set('status_min', String(params.status_min));
    if (params?.limit) q.set('limit', String(params.limit));
    const qs = q.toString();
    return requestJson<RequestLog[]>(`/api/admin/traces${qs ? `?${qs}` : ''}`);
  },
  getSettings: () => requestJson<GatewaySettings>('/api/admin/settings'),
  saveSettings: (settings: GatewaySettings) =>
    requestJson<GatewaySettings>('/api/admin/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
  getUiStatus: () =>
    requestJson<{
      compiled_in: boolean;
      enabled: boolean;
      persisted_ui_enabled: boolean;
      runtime_override: boolean | null;
      ui_url: string | null;
    }>('/api/admin/ui'),
  setUiStatus: (enabled: boolean, persist = true) =>
    requestJson<{
      compiled_in: boolean;
      enabled: boolean;
      persisted_ui_enabled: boolean;
      runtime_override: boolean | null;
      ui_url: string | null;
    }>('/api/admin/ui', {
      method: 'POST',
      body: JSON.stringify({ enabled, persist }),
    }),
  applyPhoenixPreset: (payload?: {
    phoenix_ui_url?: string;
    otlp_endpoint?: string;
    project_name?: string;
    api_key?: string;
  }) =>
    requestJson<GatewaySettings>('/api/admin/telemetry/phoenix-preset', {
      method: 'POST',
      body: JSON.stringify(payload ?? {}),
    }),
  sendTestSpan: () =>
    requestJson<TestSpanResult>('/api/admin/telemetry/test-span', {
      method: 'POST',
    }),
  clearCache: () =>
    requestJson<{ ok: boolean; cleared_entries: number }>('/api/admin/cache/clear', {
      method: 'POST',
    }),
  exportConfig: (format: 'toml' | 'yaml') =>
    requestJson<{ format: string; content: string }>(
      `/api/admin/config/export?format=${format}`,
    ),
  importConfig: (content: string, format?: 'toml' | 'yaml') =>
    requestJson<{ ok: boolean; applied_items: number }>('/api/admin/config/import', {
      method: 'POST',
      body: JSON.stringify({ content, format }),
    }),
};
