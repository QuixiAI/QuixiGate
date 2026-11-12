import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { formatNanos, formatTokens, isModelRoutable } from './api';
import { QuotaRing } from './components/QuotaRing';

describe('isModelRoutable provider gating', () => {
  const openrouterOnlyModel = {
    canonical_id: 'meta-llama/llama-4-maverick',
    bare_id: 'llama-4-maverick',
    author: 'meta-llama',
    display_name: 'Llama 4 Maverick',
    providers: { openrouter: 'meta-llama/llama-4-maverick' },
    context_window: 128000,
    max_output_tokens: 8192,
    input_nanos_per_m: 200_000_000,
    output_nanos_per_m: 600_000_000,
    cache_read_nanos_per_m: 0,
    supports_tools: true,
    supports_reasoning: false,
    supports_vision: true,
    supports_structured_output: true,
    is_local: false,
  };

  const claudeModel = {
    ...openrouterOnlyModel,
    canonical_id: 'anthropic/claude-sonnet-4-5',
    bare_id: 'claude-sonnet-4-5',
    author: 'anthropic',
    display_name: 'Claude Sonnet 4.5',
    providers: {
      anthropic: 'claude-sonnet-4-5',
      openrouter: 'anthropic/claude-sonnet-4-5',
    },
  };

  it('hides OpenRouter and cloud models when no providers are configured', () => {
    const empty = new Set<string>();
    expect(isModelRoutable(openrouterOnlyModel, empty)).toBe(false);
    expect(isModelRoutable(claudeModel, empty)).toBe(false);
    expect(
      isModelRoutable({ ...openrouterOnlyModel, is_local: true }, empty),
    ).toBe(true);
  });

  it('surfaces only Anthropic models when anthropic-token is added, and OpenRouter models/@presets only when openrouter is added', () => {
    const presetModel = {
      ...openrouterOnlyModel,
      canonical_id: '@preset/code-reviewer',
      bare_id: 'code-reviewer',
      author: 'preset',
      display_name: '@preset/code-reviewer',
      is_preset: true,
    };
    const tildeLatestModel = {
      ...openrouterOnlyModel,
      canonical_id: '~anthropic/claude-opus-latest',
      bare_id: 'claude-opus-latest',
      author: '~anthropic',
      display_name: '~anthropic/claude-opus-latest',
    };
    const autoRouterModel = {
      ...openrouterOnlyModel,
      canonical_id: 'openrouter/auto',
      bare_id: 'auto',
      author: 'openrouter',
      display_name: 'OpenRouter Auto Router',
    };

    const claudeAndOpenAiCompat = new Set(['anthropic-token', 'openaicompatible']);
    expect(isModelRoutable(claudeModel, claudeAndOpenAiCompat)).toBe(true);
    expect(isModelRoutable(presetModel, claudeAndOpenAiCompat)).toBe(false);
    expect(isModelRoutable(tildeLatestModel, claudeAndOpenAiCompat)).toBe(false);
    expect(isModelRoutable(autoRouterModel, claudeAndOpenAiCompat)).toBe(false);

    const withOpenRouter = new Set(['openrouter']);
    const veniceOnlyModel = {
      ...openrouterOnlyModel,
      canonical_id: 'venice/deepseek-v4-1-flash-uncensored',
      bare_id: 'deepseek-v4-1-flash-uncensored',
      author: 'venice',
      display_name: 'DeepSeek Uncensored (Venice)',
      providers: { venice: 'deepseek-v4-1-flash-uncensored' },
    };
    expect(isModelRoutable(openrouterOnlyModel, withOpenRouter)).toBe(true);
    expect(isModelRoutable(presetModel, withOpenRouter)).toBe(true);
    expect(isModelRoutable(tildeLatestModel, withOpenRouter)).toBe(true);
    expect(isModelRoutable(autoRouterModel, withOpenRouter)).toBe(true);
    expect(isModelRoutable(veniceOnlyModel, withOpenRouter)).toBe(false);
  });
});

describe('NanoUsd BigInt formatter', () => {
  it('formats zero, whole dollars, and sub-cent nano precision accurately without float drift', () => {
    expect(formatNanos(0)).toBe('$0.00');
    expect(formatNanos(1_000_000_000)).toBe('$1.00');
    expect(formatNanos(25_500_000_000)).toBe('$25.50');
    expect(formatNanos(1_250_000)).toBe('$0.00125');
    expect(formatNanos(-3_000_000_000)).toBe('-$3.00');
    expect(formatNanos(null)).toBe('—');
  });

  it('formats token counts cleanly', () => {
    expect(formatTokens(0)).toBe('0');
    expect(formatTokens(842)).toBe('842');
    expect(formatTokens(14_500)).toBe('14.5k');
    expect(formatTokens(2_400_000)).toBe('2.40M');
  });
});

describe('QuotaRing component', () => {
  afterEach(() => cleanup());

  it('renders quota window label and percentage', () => {
    render(
      <QuotaRing
        window={{
          label: '5h Window',
          used_percent: 42.4,
          reset_at_ms: Date.now() + 3_600_000,
        }}
      />,
    );
    expect(screen.getByText('5h Window')).toBeInTheDocument();
    expect(screen.getByText('42%')).toBeInTheDocument();
  });
});

describe('App shell & AuthGate integration', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
    window.history.pushState({}, '', '/');
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders AuthGate when require_auth is active and session is unauthenticated', async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url.includes('/api/admin/auth/status')) {
        return new Response(
          JSON.stringify({
            require_auth: true,
            bound_loopback: true,
            has_password_user: true,
            authenticated: false,
            via_loopback_trust: false,
            user: null,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response('{}', { status: 200 });
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('Sign in to QuixiGate')).toBeInTheDocument();
    });
    expect(
      screen.getByText('Strict HttpOnly session authentication is active.'),
    ).toBeInTheDocument();
  });

  it('renders Command Center KPIs and navigates between tabs on loopback trust', async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url.includes('/api/admin/auth/status')) {
        return new Response(
          JSON.stringify({
            require_auth: false,
            bound_loopback: true,
            has_password_user: false,
            authenticated: true,
            via_loopback_trust: true,
            user: {
              id: 'local',
              username: 'local',
              display_name: 'Local Operator',
              role: 'admin',
              has_password: false,
              budget_limit_nanos: null,
              budget_window: 'monthly',
              spend_nanos: 0,
              window_reset_at_ms: null,
              created_at_ms: 1700000000000,
            },
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.includes('/api/admin/overview')) {
        return new Response(
          JSON.stringify({
            uptime_ms: 12000,
            bound_loopback: true,
            metrics: {
              active_requests: 0,
              total_requests_24h: 18,
              error_requests_24h: 0,
              cache_hits_24h: 4,
              input_tokens_24h: 12000,
              output_tokens_24h: 6400,
              cached_tokens_24h: 2000,
              avg_tok_per_sec_24h: 94.6,
              ttft_p50_ms: 185,
              ttft_p95_ms: 310,
              spend_24h_nanos: 250_000_000,
              spend_30d_nanos: 1_500_000_000,
              savings_30d_nanos: 42_000_000_000,
              by_provider: [],
              by_model: [],
            },
            credentials: [
              {
                id: 'seat-1',
                provider_id: 'openai-codex',
                label: 'ChatGPT Plus Seat #1',
                auth_kind: 'oauth',
                base_url: null,
                priority: 10,
                enabled: true,
                allow_api_overflow: false,
                status: 'ok',
                status_reason: null,
                cooldown_until_ms: null,
                model_cooldowns: {},
                quota_snapshot: {
                  windows: [{ label: '5h', used_percent: 28 }],
                  fetched_at_ms: Date.now(),
                },
                expires_at_ms: null,
                last_used_at_ms: null,
                created_at_ms: Date.now(),
              },
            ],
            local_runtimes: [
              {
                runtime_id: 'ollama',
                display_name: 'Ollama',
                base_url: 'http://127.0.0.1:11434/v1',
                healthy: true,
                models: ['qwen2.5-coder:14b'],
                loaded_models: ['qwen2.5-coder:14b'],
                avg_ttft_ms: 95,
                avg_tok_per_sec: 72.4,
                last_checked_ms: Date.now(),
              },
              {
                runtime_id: 'llamacpp',
                display_name: 'llama.cpp Server',
                base_url: 'http://127.0.0.1:8080/v1',
                healthy: true,
                mode: 'router',
                total_slots: 2,
                supports_load_control: true,
                models: ['Qwen3-Coder-30B-Q4_K_M.gguf'],
                loaded_models: ['Qwen3-Coder-30B-Q4_K_M.gguf'],
                avg_ttft_ms: 62,
                avg_tok_per_sec: 88.5,
                last_checked_ms: Date.now(),
              },
            ],
            routes: [],
            settings: {
              require_auth: false,
              capture_payloads_default: true,
              max_trace_rows: 10000,
              cache_enabled_default: false,
              cache_ttl_secs: 3600,
              otlp_enabled: false,
              otlp_endpoint: 'http://127.0.0.1:6006/v1/traces',
              otlp_protocol: 'http_protobuf',
              otlp_api_key: null,
              openinference_project_name: 'quixigate',
              phoenix_ui_url: 'http://127.0.0.1:6006',
              default_pool_strategy: 'fill_first',
              local_discovery_enabled: true,
            },
            recent_traces: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.includes('/api/admin/keys')) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (url.includes('/api/admin/users')) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (url.includes('/api/admin/teams')) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      return new Response(JSON.stringify({}), { status: 200 });
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('94.6')).toBeInTheDocument();
    });
    expect(screen.getByText('$42.00')).toBeInTheDocument();
    expect(screen.getByText('ChatGPT Plus Seat #1')).toBeInTheDocument();
    expect(screen.getByText('llama.cpp Server')).toBeInTheDocument();
    expect(screen.getByText('router')).toBeInTheDocument();
    expect(screen.getByText('2 slots')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Users, Teams & Keys'));
    expect(window.location.pathname).toBe('/ui/keys');
  });
});
