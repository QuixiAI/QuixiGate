import { useEffect, useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  Cpu,
  DollarSign,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react';
import { api, formatNanos, formatRelativeTime, formatTokens } from '../api';
import { QuotaRing } from '../components/QuotaRing';
import type { OverviewResponse } from '../types';

const QUOTA_POLLABLE_PROVIDERS = new Set([
  'openai-codex',
  'anthropic-token',
  'google-gemini-cli',
  'openrouter',
]);

export function CommandCenter({
  onNavigate,
  onOpenPlaygroundModel,
}: {
  onNavigate: (tab: string) => void;
  onOpenPlaygroundModel: (modelId: string) => void;
}) {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanningLocal, setScanningLocal] = useState(false);
  const [refreshingId, setRefreshingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const res = await api.getOverview();
      setData(res);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load gateway overview');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 4000);
    return () => clearInterval(timer);
  }, []);

  async function handleScanLocal() {
    setScanningLocal(true);
    try {
      await api.scanLocalRuntimes();
      await load();
    } finally {
      setScanningLocal(false);
    }
  }

  async function handleRefreshQuota(credId: string) {
    setRefreshingId(credId);
    try {
      await api.refreshCredentialQuota(credId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Quota refresh failed');
    } finally {
      setRefreshingId(null);
    }
  }

  if (loading && !data) {
    return (
      <div className="p-8 text-sm text-slate-400">Loading QuixiGate telemetry…</div>
    );
  }

  const m = data?.metrics;
  const healthyLocals = (data?.local_runtimes ?? []).filter((r) => r.healthy);
  const localModelCount = healthyLocals.reduce((acc, r) => acc + r.models.length, 0);
  const oauthSeats = (data?.credentials ?? []).filter(
    (c) =>
      c.enabled &&
      c.provider_id !== 'openrouter' &&
      (c.auth_kind === 'oauth' || c.auth_kind === 'setup_token'),
  );
  const openrouterSeats = (data?.credentials ?? []).filter(
    (c) => c.enabled && c.provider_id === 'openrouter',
  );
  const openrouterCredits = openrouterSeats[0]?.quota_snapshot?.balance_credits;
  const openrouterPresetCount = data?.openrouter_presets_count ?? 0;

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-[1440px] mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Command Center
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Real-time throughput, subscription seat quotas, local inference servers, and exact{' '}
            <code className="text-indigo-300 text-xs">NanoUsd</code> accounting.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() => void load()}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 px-3.5 py-2 text-xs font-medium text-slate-200 transition cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => onNavigate('chat')}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-3.5 py-2 text-xs font-medium text-white transition cursor-pointer"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Open Playground
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* 4 Flagship KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Streaming Velocity (24h)</span>
            <Zap className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-2xl font-semibold font-mono text-white tabular-nums">
              {m ? m.avg_tok_per_sec_24h.toFixed(1) : '0.0'}
            </span>
            <span className="text-xs text-slate-400">tok/s</span>
          </div>
          <div className="mt-2 flex items-center gap-3 text-xs text-slate-400 font-mono tabular-nums">
            <span>p50 TTFT: {m?.ttft_p50_ms != null ? `${m.ttft_p50_ms}ms` : '—'}</span>
            <span>p95: {m?.ttft_p95_ms != null ? `${m.ttft_p95_ms}ms` : '—'}</span>
          </div>
        </div>

        <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Requests & Cache (24h)</span>
            <Activity className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-2xl font-semibold font-mono text-white tabular-nums">
              {m ? m.total_requests_24h.toLocaleString() : '0'}
            </span>
            <span className="text-xs text-slate-400">
              reqs ({m?.cache_hits_24h ?? 0} cache hits)
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono tabular-nums">
            In: {formatTokens(m?.input_tokens_24h)} · Out:{' '}
            {formatTokens(m?.output_tokens_24h)} · Cached:{' '}
            {formatTokens(m?.cached_tokens_24h)}
          </div>
        </div>

        <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Paid API Spend (NanoUsd)</span>
            <DollarSign className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-2xl font-semibold font-mono text-white tabular-nums">
              {formatNanos(m?.spend_24h_nanos ?? 0)}
            </span>
            <span className="text-xs text-slate-400">24h</span>
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono tabular-nums">
            30d billed: {formatNanos(m?.spend_30d_nanos ?? 0)}
          </div>
        </div>

        <div className="rounded-xl bg-gradient-to-br from-emerald-500/12 via-[#0d111a] to-[#0d111a] border border-emerald-500/25 p-5">
          <div className="flex items-center justify-between text-xs text-emerald-300">
            <span>OAuth & Local Savings (30d)</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-2xl font-semibold font-mono text-emerald-300 tabular-nums">
              {formatNanos(m?.savings_30d_nanos ?? 0)}
            </span>
            <span className="text-xs text-emerald-300/80">API-equiv saved</span>
          </div>
          <div className="mt-2 text-xs text-slate-400">
            $0 marginal cost via subscription seats & local VRAM
          </div>
        </div>
      </div>

      {/* Three Co-Equal First-Class Pillars Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl bg-[#0d111a] border border-emerald-500/20 p-4 flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <HardDrive className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-xs font-semibold text-white">
                1. Local Runtimes
              </span>
              <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 font-mono text-[10px] text-emerald-300">
                $0 VRAM
              </span>
            </div>
            <p className="mt-1.5 text-xs text-slate-300 font-mono">
              {healthyLocals.length > 0
                ? `${healthyLocals.length} online · ${localModelCount} local models`
                : 'Auto-discovery ready (llama.cpp, Ollama, LM Studio, MLX, vLLM)'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => void handleScanLocal()}
            disabled={scanningLocal}
            className="shrink-0 rounded-lg bg-white/6 hover:bg-white/10 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 cursor-pointer"
          >
            {scanningLocal ? 'Scanning…' : 'Scan'}
          </button>
        </div>

        <div className="rounded-xl bg-[#0d111a] border border-indigo-500/20 p-4 flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Cpu className="h-3.5 w-3.5 text-indigo-400" />
              <span className="text-xs font-semibold text-white">
                2. OAuth Subscriptions
              </span>
              <span className="rounded-full bg-indigo-500/15 px-2 py-0.5 font-mono text-[10px] text-indigo-300">
                $0 Marginal
              </span>
            </div>
            <p className="mt-1.5 text-xs text-slate-300 font-mono">
              {oauthSeats.length > 0
                ? `${oauthSeats.length} active subscription ${oauthSeats.length === 1 ? 'seat' : 'seats'}`
                : 'ChatGPT/Codex, Claude, Gemini CLI, Copilot, Grok, Qwen, MiniMax'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('providers')}
            className="shrink-0 rounded-lg bg-white/6 hover:bg-white/10 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 cursor-pointer"
          >
            Seats
          </button>
        </div>

        <div className="rounded-xl bg-[#0d111a] border border-violet-500/20 p-4 flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-3.5 w-3.5 text-violet-400" />
              <span className="text-xs font-semibold text-white">
                3. OpenRouter
              </span>
              {openrouterSeats.length > 0 ? (
                <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 font-mono text-[10px] text-emerald-300">
                  Connected
                </span>
              ) : (
                <span className="rounded-full bg-white/6 px-2 py-0.5 font-mono text-[10px] text-slate-400">
                  Optional
                </span>
              )}
            </div>
            <p className="mt-1.5 text-xs text-slate-300 font-mono">
              {openrouterSeats.length > 0
                ? `${openrouterCredits != null ? `$${openrouterCredits.toFixed(2)} credits · ` : ''}${openrouterPresetCount} @presets`
                : 'Not connected · PKCE, OIDC, API key, or env'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('providers')}
            className="shrink-0 rounded-lg bg-white/6 hover:bg-white/10 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 cursor-pointer"
          >
            {openrouterSeats.length > 0 ? 'Manage' : 'Connect'}
          </button>
        </div>
      </div>

      {/* Seat Pool & Local Runtimes */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Multi-Seat Subscription & API Pool */}
        <div className="xl:col-span-2 rounded-xl bg-[#0d111a] border border-white/8 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-white">
                Multi-Seat Credential Pool
              </h2>
              <p className="text-xs text-slate-400">
                Sticky <code className="text-indigo-300">fill_first</code> routing drains $0 OAuth seats before touching paid API keys.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('providers')}
              className="inline-flex items-center gap-1 text-xs font-medium text-indigo-400 hover:text-indigo-300 cursor-pointer"
            >
              Manage Seats
              <ArrowUpRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {!data?.credentials.length ? (
            <div className="rounded-xl border border-dashed border-white/10 p-6 text-center">
              <p className="text-sm text-slate-300">No cloud seats configured yet</p>
              <p className="text-xs text-slate-400 mt-1">
                Sign in with ChatGPT/Codex, Claude Setup Token, Gemini CLI, Copilot, Grok, OpenRouter, Qwen, MiniMax, Chutes, or any API key.
              </p>
              <button
                type="button"
                onClick={() => onNavigate('providers')}
                className="mt-4 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
              >
                Connect Provider Seat
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {data.credentials.map((cred) => {
                const statusBadge =
                  cred.status === 'ok'
                    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                    : cred.status === 'exhausted'
                      ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                      : 'bg-rose-500/15 text-rose-300 border-rose-500/30';
                const modelCooldownEntries = Object.entries(cred.model_cooldowns ?? {});

                return (
                  <div
                    key={cred.id}
                    className="rounded-xl bg-slate-950/70 border border-white/6 p-4 flex flex-col gap-3"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="rounded bg-white/6 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                          #{cred.priority}
                        </span>
                        <span className="font-medium text-sm text-white">
                          {cred.label}
                        </span>
                        <span className="rounded-md bg-indigo-500/15 border border-indigo-500/25 px-2 py-0.5 text-[11px] font-mono text-indigo-300">
                          {cred.provider_id}
                        </span>
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${statusBadge}`}
                        >
                          {cred.status}
                        </span>
                        {!cred.enabled && (
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-400">
                            disabled
                          </span>
                        )}
                      </div>
                      {QUOTA_POLLABLE_PROVIDERS.has(cred.provider_id) ? (
                        <button
                          type="button"
                          onClick={() => void handleRefreshQuota(cred.id)}
                          disabled={refreshingId === cred.id}
                          className="inline-flex items-center gap-1.5 rounded-md bg-white/5 hover:bg-white/10 px-2.5 py-1 text-xs text-slate-300 cursor-pointer"
                        >
                          <RefreshCw
                            className={`h-3 w-3 ${refreshingId === cred.id ? 'animate-spin' : ''}`}
                          />
                          Sync Quota
                        </button>
                      ) : (
                        <span className="rounded-md bg-sky-500/10 border border-sky-500/20 px-2 py-0.5 text-[11px] font-mono text-sky-300">
                          {cred.allow_api_overflow ? 'Overflow Seat' : 'Pinned Only'}
                        </span>
                      )}
                    </div>

                    {cred.quota_snapshot?.windows?.length ? (
                      <div className="flex flex-wrap gap-2.5">
                        {cred.quota_snapshot.windows.map((win) => (
                          <QuotaRing key={win.label} window={win} />
                        ))}
                      </div>
                    ) : null}

                    {modelCooldownEntries.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {modelCooldownEntries.map(([mod, until]) => (
                          <span
                            key={mod}
                            className="rounded bg-amber-500/10 border border-amber-500/25 px-2 py-0.5 text-[11px] font-mono text-amber-300"
                          >
                            {mod} cooldown {formatRelativeTime(until)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Local Runtimes Discovery */}
        <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-white">
                  Local Runtimes
                </h2>
              </div>
              <button
                type="button"
                onClick={() => void handleScanLocal()}
                disabled={scanningLocal}
                className="inline-flex items-center gap-1.5 rounded-md bg-white/6 hover:bg-white/10 px-2.5 py-1 text-xs text-slate-200 cursor-pointer"
              >
                <RefreshCw className={`h-3 w-3 ${scanningLocal ? 'animate-spin' : ''}`} />
                Scan Ports
              </button>
            </div>

            <div className="space-y-2.5">
              {(data?.local_runtimes ?? []).map((rt) => (
                <div
                  key={rt.runtime_id}
                  className="rounded-lg bg-slate-950/70 border border-white/6 p-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className={`h-2 w-2 shrink-0 rounded-full ${
                          rt.healthy
                            ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]'
                            : rt.auth_required
                              ? 'bg-amber-400 shadow-[0_0_8px_#fbbf24]'
                              : 'bg-slate-600'
                        }`}
                      />
                      <span className="text-xs font-medium text-white truncate">
                        {rt.display_name}
                      </span>
                      {rt.mode && (
                        <span className="rounded bg-indigo-500/15 border border-indigo-500/25 px-1.5 py-0.2 font-mono text-[10px] text-indigo-300">
                          {rt.mode}
                        </span>
                      )}
                      {rt.total_slots != null && rt.total_slots > 0 && (
                        <span className="rounded bg-white/6 px-1.5 py-0.2 font-mono text-[10px] text-slate-300">
                          {rt.total_slots} slots
                        </span>
                      )}
                    </div>
                    {rt.healthy ? (
                      <span className="font-mono text-[11px] text-emerald-300 shrink-0">
                        {rt.models.length} {rt.models.length === 1 ? 'model' : 'models'}
                      </span>
                    ) : rt.auth_required ? (
                      <button
                        type="button"
                        onClick={() => onNavigate('providers')}
                        className="rounded bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 font-mono text-[10px] text-amber-300 hover:bg-amber-500/25 cursor-pointer shrink-0"
                      >
                        API Key Required
                      </button>
                    ) : (
                      <span className="font-mono text-[11px] text-slate-400 shrink-0">
                        offline
                      </span>
                    )}
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-slate-500 truncate">
                    {rt.base_url}
                  </div>
                  {rt.healthy && rt.models.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {rt.models.slice(0, 4).map((mod) => {
                        const isLoaded = rt.loaded_models.includes(mod);
                        const detail = rt.model_details?.[mod];
                        return (
                          <button
                            key={mod}
                            type="button"
                            onClick={() => onOpenPlaygroundModel(`${rt.runtime_id}/${mod}`)}
                            title={
                              detail?.context_window
                                ? `${Math.round(detail.context_window / 1024)}k ctx${detail.supports_vision ? ' · vision' : ''}${isLoaded ? ' · loaded in memory' : ''}`
                                : undefined
                            }
                            className="inline-flex items-center gap-1.5 rounded bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/25 px-2 py-0.5 text-[11px] font-mono text-emerald-300 cursor-pointer"
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${
                                isLoaded ? 'bg-emerald-400' : 'bg-slate-500'
                              }`}
                            />
                            <span className="truncate max-w-[180px]">{mod}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                  {(rt.avg_tok_per_sec || rt.avg_ttft_ms) && (
                    <div className="mt-2 flex items-center gap-3 text-[11px] font-mono text-slate-400">
                      {rt.avg_tok_per_sec && <span>{rt.avg_tok_per_sec.toFixed(1)} tok/s</span>}
                      {rt.avg_ttft_ms && <span>TTFT {rt.avg_ttft_ms}ms</span>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 rounded-lg bg-white/[0.03] border border-white/6 p-3 flex items-center gap-2.5 text-xs text-slate-400">
            <HardDrive className="h-4 w-4 text-slate-400 shrink-0" />
            <span>
              Local completions run at <strong className="text-slate-200">$0.00</strong> while tracking cloud API-equivalent savings.
            </span>
          </div>
        </div>
      </div>

      {/* Recent Live Requests Table */}
      <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-white">Recent Gateway Traffic</h2>
          <button
            type="button"
            onClick={() => onNavigate('traces')}
            className="inline-flex items-center gap-1 text-xs font-medium text-indigo-400 hover:text-indigo-300 cursor-pointer"
          >
            Live Trace Inspector
            <ArrowUpRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {!data?.recent_traces.length ? (
          <div className="py-6 text-center text-xs text-slate-400">
            No requests recorded yet — fire a prompt in the Playground or run{' '}
            <code className="text-indigo-300">quixigate run claude</code>.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-400 border-b border-white/8">
                <tr>
                  <th className="py-2 pr-4 font-medium">Time</th>
                  <th className="py-2 pr-4 font-medium">Ingress</th>
                  <th className="py-2 pr-4 font-medium">Model</th>
                  <th className="py-2 pr-4 font-medium">Provider</th>
                  <th className="py-2 pr-4 font-medium">Status</th>
                  <th className="py-2 pr-4 font-medium">Latency</th>
                  <th className="py-2 pr-4 font-medium">Tokens</th>
                  <th className="py-2 font-medium text-right">Billed / Saved</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 font-mono">
                {data.recent_traces.map((t) => {
                  const saved = Math.max(0, t.api_equiv_nanos - t.cost_nanos);
                  return (
                    <tr key={t.id} className="hover:bg-white/[0.02]">
                      <td className="py-2.5 pr-4 text-slate-400">
                        {formatRelativeTime(t.timestamp_ms)}
                      </td>
                      <td className="py-2.5 pr-4 text-slate-300">
                        {t.ingress_protocol}
                      </td>
                      <td className="py-2.5 pr-4 text-white">
                        {t.resolved_model}
                        {t.cache_hit && (
                          <span className="ml-2 rounded bg-sky-500/15 px-1.5 py-0.5 text-[10px] text-sky-300">
                            CACHE
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 pr-4 text-indigo-300">{t.provider_id}</td>
                      <td className="py-2.5 pr-4">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[11px] ${
                            t.status_code < 400
                              ? 'bg-emerald-500/15 text-emerald-300'
                              : 'bg-rose-500/15 text-rose-300'
                          }`}
                        >
                          {t.status_code}
                        </span>
                      </td>
                      <td className="py-2.5 pr-4 text-slate-300 tabular-nums">
                        {t.ttft_ms != null ? `${t.ttft_ms}ms / ` : ''}
                        {t.duration_ms}ms
                      </td>
                      <td className="py-2.5 pr-4 text-slate-300 tabular-nums">
                        {t.input_tokens + t.output_tokens}
                      </td>
                      <td className="py-2.5 text-right tabular-nums">
                        <span className="text-slate-200">{formatNanos(t.cost_nanos)}</span>
                        {saved > 0 && (
                          <span className="ml-2 text-emerald-400">
                            (saved {formatNanos(saved)})
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
