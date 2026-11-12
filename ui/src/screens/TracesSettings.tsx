import { useEffect, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  Download,
  ExternalLink,
  Flame,
  Lock,
  Pause,
  Play,
  RefreshCw,
  Save,
  Send,
  Settings2,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { api, formatNanos, formatRelativeTime } from '../api';
import type {
  GatewaySettings,
  OtlpProtocol,
  PoolStrategy,
  RequestLog,
  TestSpanResult,
} from '../types';

export function TracesSettings({
  initialSubTab = 'traces',
}: {
  initialSubTab?: 'traces' | 'settings';
}) {
  const [subTab, setSubTab] = useState<'traces' | 'settings'>(initialSubTab);

  // Traces State
  const [traces, setTraces] = useState<RequestLog[]>([]);
  const [liveStream, setLiveStream] = useState(true);
  const [modelFilter, setModelFilter] = useState('');
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [selectedTrace, setSelectedTrace] = useState<RequestLog | null>(null);

  // Settings State
  const [settings, setSettings] = useState<GatewaySettings | null>(null);
  const [testSpanResult, setTestSpanResult] = useState<TestSpanResult | null>(null);
  const [sendingSpan, setSendingSpan] = useState(false);
  const [newOperatorPassword, setNewOperatorPassword] = useState('');
  const [configFormat, setConfigFormat] = useState<'toml' | 'yaml'>('toml');
  const [configText, setConfigText] = useState('');
  const [banner, setBanner] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  useEffect(() => {
    setSubTab(initialSubTab);
  }, [initialSubTab]);

  async function loadTraces() {
    const list = await api.getTraces({
      model: modelFilter || undefined,
      status_min: errorsOnly ? 400 : undefined,
      limit: 120,
    });
    setTraces(list);
  }

  async function loadSettings() {
    const s = await api.getSettings();
    setSettings(s);
    const exported = await api.exportConfig(configFormat);
    setConfigText(exported.content);
  }

  useEffect(() => {
    void loadTraces();
    void loadSettings();
  }, []);

  useEffect(() => {
    void loadTraces();
  }, [modelFilter, errorsOnly]);

  // Live SSE Trace Bus
  useEffect(() => {
    if (!liveStream) return;
    const es = new EventSource('/api/admin/traces/stream');
    const handler = (ev: MessageEvent) => {
      try {
        const incoming = JSON.parse(ev.data) as RequestLog;
        if (errorsOnly && incoming.status_code < 400) return;
        if (
          modelFilter &&
          !incoming.resolved_model.toLowerCase().includes(modelFilter.toLowerCase()) &&
          !incoming.requested_model.toLowerCase().includes(modelFilter.toLowerCase())
        ) {
          return;
        }
        setTraces((prev) => [
          incoming,
          ...prev.filter((t) => t.id !== incoming.id).slice(0, 149),
        ]);
      } catch {
        // ignore malformed keepalive
      }
    };
    es.addEventListener('trace', handler);
    return () => {
      es.removeEventListener('trace', handler);
      es.close();
    };
  }, [liveStream, modelFilter, errorsOnly]);

  async function handleSaveSettings(e: React.FormEvent) {
    e.preventDefault();
    if (!settings) return;
    try {
      if (newOperatorPassword.trim()) {
        await api.setPassword(newOperatorPassword.trim());
        setNewOperatorPassword('');
      }
      const updated = await api.saveSettings(settings);
      setSettings(updated);
      setBanner({ kind: 'ok', text: 'Gateway settings saved to SQLite.' });
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Failed to save settings',
      });
    }
  }

  async function handlePhoenixPreset() {
    try {
      const updated = await api.applyPhoenixPreset({
        phoenix_ui_url: settings?.phoenix_ui_url || 'http://localhost:6006',
        otlp_endpoint: settings?.otlp_endpoint || 'http://localhost:6006/v1/traces',
        project_name: settings?.openinference_project_name || 'quixigate',
      });
      setSettings(updated);
      setBanner({
        kind: 'ok',
        text: `Arize Phoenix OpenInference preset enabled (${updated.otlp_endpoint})!`,
      });
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Failed to apply Arize Phoenix preset',
      });
    }
  }

  async function handleSendTestSpan() {
    setSendingSpan(true);
    setTestSpanResult(null);
    try {
      if (settings) {
        await api.saveSettings(settings);
      }
      const res = await api.sendTestSpan();
      setTestSpanResult(res);
      setBanner({ kind: res.ok ? 'ok' : 'err', text: res.message });
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'OTLP test span failed',
      });
    } finally {
      setSendingSpan(false);
    }
  }

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-[1440px] mx-auto">
      {/* Header & Section Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Observability, Arize Phoenix & Settings
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Live SSE request waterfall, OpenInference OTLP protobuf exporter, response cache, and declarative config sync.
          </p>
        </div>

        <div className="inline-flex rounded-xl bg-[#0d111a] border border-white/10 p-1">
          <button
            type="button"
            onClick={() => setSubTab('traces')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-medium cursor-pointer ${
              subTab === 'traces'
                ? 'bg-indigo-500 text-white'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            Live Trace Inspector
          </button>
          <button
            type="button"
            onClick={() => setSubTab('settings')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-medium cursor-pointer ${
              subTab === 'settings'
                ? 'bg-indigo-500 text-white'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Settings2 className="h-3.5 w-3.5" />
            Phoenix & Gateway Settings
          </button>
        </div>
      </div>

      {banner && (
        <div
          className={`rounded-xl border px-4 py-3 text-xs flex items-center justify-between ${
            banner.kind === 'ok'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-200'
          }`}
        >
          <span>{banner.text}</span>
          <button
            type="button"
            onClick={() => setBanner(null)}
            className="text-slate-400 hover:text-white cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {subTab === 'traces' ? (
        <div className="space-y-4">
          {/* Filter Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2.5 flex-1">
              <input
                type="text"
                value={modelFilter}
                onChange={(e) => setModelFilter(e.target.value)}
                placeholder="Filter traces by model slug..."
                className="w-64 rounded-xl bg-[#0d111a] border border-white/10 px-3.5 py-2 text-xs text-white"
              />
              <label className="flex items-center gap-2 rounded-xl bg-[#0d111a] border border-white/10 px-3.5 py-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={errorsOnly}
                  onChange={(e) => setErrorsOnly(e.target.checked)}
                />
                Errors only (&ge; 400)
              </label>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setLiveStream((v) => !v)}
                className={`inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-2 text-xs font-medium cursor-pointer ${
                  liveStream
                    ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                    : 'bg-[#0d111a] border-white/10 text-slate-300'
                }`}
              >
                {liveStream ? (
                  <Pause className="h-3.5 w-3.5" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
                {liveStream ? 'Live SSE Active' : 'Paused'}
              </button>
              <a
                href="/api/admin/spend/export?format=csv"
                download="quixigate-spend-export.csv"
                className="inline-flex items-center gap-1.5 rounded-xl bg-[#0d111a] border border-white/10 hover:border-white/20 px-3.5 py-2 text-xs text-slate-200"
              >
                <Download className="h-3.5 w-3.5" />
                Export CSV
              </a>
              <button
                type="button"
                onClick={() => void loadTraces()}
                className="inline-flex items-center gap-1.5 rounded-xl bg-[#0d111a] border border-white/10 px-3.5 py-2 text-xs text-slate-200 cursor-pointer"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Reload
              </button>
            </div>
          </div>

          {/* Trace Table */}
          <div className="rounded-xl bg-[#0d111a] border border-white/8 overflow-hidden">
            {traces.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-400">
                Waiting for gateway traffic… Live SSE traces appear here instantaneously.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950/70 text-slate-400 border-b border-white/8">
                    <tr>
                      <th className="py-3 px-4 font-medium">When</th>
                      <th className="py-3 px-4 font-medium">Protocol</th>
                      <th className="py-3 px-4 font-medium">Requested &rarr; Resolved</th>
                      <th className="py-3 px-4 font-medium">Provider</th>
                      <th className="py-3 px-4 font-medium">Status</th>
                      <th className="py-3 px-4 font-medium">TTFT / Total</th>
                      <th className="py-3 px-4 font-medium">Tokens</th>
                      <th className="py-3 px-4 font-medium text-right">NanoUsd</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono">
                    {traces.map((t) => (
                      <tr
                        key={t.id}
                        onClick={() => setSelectedTrace(t)}
                        className="hover:bg-white/[0.03] cursor-pointer"
                      >
                        <td className="py-2.5 px-4 text-slate-400">
                          {formatRelativeTime(t.timestamp_ms)}
                        </td>
                        <td className="py-2.5 px-4 text-slate-300">
                          {t.ingress_protocol}
                        </td>
                        <td className="py-2.5 px-4 text-white">
                          <span>{t.resolved_model}</span>
                          {t.cache_hit && (
                            <span className="ml-2 rounded bg-sky-500/15 px-1.5 py-0.5 text-[10px] text-sky-300">
                              HIT
                            </span>
                          )}
                          {t.fallback_attempts > 0 && (
                            <span className="ml-2 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">
                              +{t.fallback_attempts} failover
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-4 text-indigo-300">{t.provider_id}</td>
                        <td className="py-2.5 px-4">
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
                        <td className="py-2.5 px-4 text-slate-300 tabular-nums">
                          {t.ttft_ms != null ? `${t.ttft_ms}ms / ` : ''}
                          {t.duration_ms}ms
                        </td>
                        <td className="py-2.5 px-4 text-slate-300 tabular-nums">
                          {t.input_tokens} &rarr; {t.output_tokens}
                        </td>
                        <td className="py-2.5 px-4 text-right text-slate-200 tabular-nums">
                          {formatNanos(t.cost_nanos)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      ) : (
        settings && (
          <div className="space-y-6">
            {/* 1-Click Arize Phoenix OpenInference Card */}
            <div className="rounded-2xl bg-gradient-to-br from-orange-500/12 via-[#0d111a] to-[#0d111a] border border-orange-500/30 p-6 space-y-4">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  <div className="h-10 w-10 rounded-xl bg-orange-500/15 border border-orange-500/30 flex items-center justify-center text-orange-400 shrink-0">
                    <Flame className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-white">
                      Arize Phoenix OpenInference Integration
                    </h2>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Native Rust OTLP <code className="text-orange-300">ExportTraceServiceRequest</code> protobuf exporter emitting{' '}
                      <code className="text-orange-300">openinference.span.kind="LLM"</code>, token counts, costs, tool calls, and privacy-masked prompts.
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  <button
                    type="button"
                    onClick={() => void handlePhoenixPreset()}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-orange-500 hover:bg-orange-400 px-3.5 py-2 text-xs font-semibold text-slate-950 cursor-pointer"
                  >
                    <Flame className="h-3.5 w-3.5" />
                    Apply 1-Click Phoenix Preset (:6006)
                  </button>
                  <button
                    type="button"
                    disabled={sendingSpan}
                    onClick={() => void handleSendTestSpan()}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/15 px-3.5 py-2 text-xs font-medium text-white cursor-pointer"
                  >
                    <Send className="h-3.5 w-3.5" />
                    {sendingSpan ? 'Sending…' : 'Send Test Span'}
                  </button>
                  <a
                    href={settings.phoenix_ui_url || 'http://localhost:6006'}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 rounded-lg bg-white/5 hover:bg-white/10 px-3 py-2 text-xs text-slate-200"
                  >
                    Open Phoenix UI
                    <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-2">
                <div className="md:col-span-2">
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    OTLP Traces Endpoint
                  </label>
                  <input
                    type="text"
                    value={settings.otlp_endpoint}
                    onChange={(e) =>
                      setSettings({ ...settings, otlp_endpoint: e.target.value })
                    }
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    OpenInference Project
                  </label>
                  <input
                    type="text"
                    value={settings.openinference_project_name}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        openinference_project_name: e.target.value,
                      })
                    }
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Wire Protocol
                  </label>
                  <select
                    value={settings.otlp_protocol}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        otlp_protocol: e.target.value as OtlpProtocol,
                      })
                    }
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="http_protobuf">HTTP/Protobuf (Phoenix Default :6006)</option>
                    <option value="grpc">gRPC (:4317)</option>
                    <option value="http_json">HTTP/JSON</option>
                  </select>
                </div>
              </div>

              <label className="flex items-center gap-2 text-xs text-slate-200 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.otlp_enabled}
                  onChange={(e) =>
                    setSettings({ ...settings, otlp_enabled: e.target.checked })
                  }
                />
                Enable live OpenInference OTLP export on every gateway completion
              </label>

              {testSpanResult && (
                <div className="rounded-xl bg-slate-950/90 border border-emerald-500/30 p-3.5 text-xs flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-emerald-300">
                    <CheckCircle2 className="h-4 w-4" />
                    <span>{testSpanResult.message}</span>
                  </div>
                  <code className="text-[11px] text-slate-400">
                    trace_id={testSpanResult.trace_id}
                  </code>
                </div>
              )}
            </div>

            {/* Core Gateway Settings Form */}
            <form
              onSubmit={handleSaveSettings}
              className="rounded-2xl bg-[#0d111a] border border-white/8 p-6 space-y-5"
            >
              <h3 className="text-sm font-semibold text-white">
                Security, Caching & Routing Defaults
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Default Pool Strategy
                  </label>
                  <select
                    value={settings.default_pool_strategy}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        default_pool_strategy: e.target.value as PoolStrategy,
                      })
                    }
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="fill_first">fill_first (Sticky Prompt Cache)</option>
                    <option value="least_used">least_used (Even Quota Drain)</option>
                    <option value="round_robin">round_robin</option>
                    <option value="random">random</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Response Cache TTL (seconds)
                  </label>
                  <input
                    type="number"
                    value={settings.cache_ttl_secs}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        cache_ttl_secs: Number(e.target.value),
                      })
                    }
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Set / Rotate Operator Password
                  </label>
                  <div className="relative">
                    <Lock className="h-3.5 w-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      type="password"
                      value={newOperatorPassword}
                      onChange={(e) => setNewOperatorPassword(e.target.value)}
                      placeholder="Leave blank to keep current"
                      className="w-full rounded-lg bg-slate-950 border border-white/10 pl-8 pr-3 py-2 text-xs text-white"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  MCP Streamable-HTTP Tool Servers (comma-separated JSON-RPC URLs)
                </label>
                <input
                  type="text"
                  value={(settings.mcp_server_urls ?? []).join(', ')}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      mcp_server_urls: e.target.value
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean),
                    })
                  }
                  placeholder="http://127.0.0.1:8787/mcp, http://127.0.0.1:8788/mcp"
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.cache_enabled_default}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        cache_enabled_default: e.target.checked,
                      })
                    }
                  />
                  Enable BLAKE3 exact-match response cache by default
                </label>

                <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.capture_payloads_default}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        capture_payloads_default: e.target.checked,
                      })
                    }
                  />
                  Capture prompt/completion payloads in traces & OpenInference spans
                </label>

                <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.local_discovery_enabled}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        local_discovery_enabled: e.target.checked,
                      })
                    }
                  />
                  Auto-discover local runtimes (Ollama, LM Studio, llama.cpp, vLLM)
                </label>

                <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.require_auth}
                    onChange={(e) =>
                      setSettings({ ...settings, require_auth: e.target.checked })
                    }
                  />
                  Require strict authentication even on 127.0.0.1 loopback
                </label>

                <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.trust_local_containers ?? true}
                    disabled={settings.require_auth}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        trust_local_containers: e.target.checked,
                      })
                    }
                  />
                  Trust local Docker / Podman / OrbStack containers (host.docker.internal)
                </label>

                <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.ui_enabled ?? true}
                    onChange={(e) =>
                      setSettings({ ...settings, ui_enabled: e.target.checked })
                    }
                  />
                  Enable optional Web UI at /ui (live toggle via POST /api/admin/ui/enable | /disable)
                </label>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Additional Trusted Subnets / CIDRs (optional comma-separated, e.g. 100.64.0.0/10 for Tailscale)
                </label>
                <input
                  type="text"
                  value={(settings.trusted_cidrs ?? []).join(', ')}
                  disabled={settings.require_auth}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      trusted_cidrs: e.target.value
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean),
                    })
                  }
                  placeholder="100.64.0.0/10, 192.168.1.0/24"
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white disabled:opacity-50"
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-white/8">
                <button
                  type="button"
                  onClick={async () => {
                    const res = await api.clearCache();
                    setBanner({
                      kind: 'ok',
                      text: `Purged response cache (${res.cleared_entries} SQLite entries cleared).`,
                    });
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500/12 hover:bg-rose-500/20 px-3.5 py-2 text-xs text-rose-300 cursor-pointer"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Purge Response Cache
                </button>

                <button
                  type="submit"
                  className="inline-flex items-center gap-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-semibold text-white cursor-pointer"
                >
                  <Save className="h-3.5 w-3.5" />
                  Save Gateway Settings
                </button>
              </div>
            </form>

            {/* Declarative TOML / YAML Config Import & Export */}
            <div className="rounded-2xl bg-[#0d111a] border border-white/8 p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    Declarative Config Sync (TOML / YAML with <code className="text-indigo-300">{'${ENV}'}</code>)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Export redacted config for GitOps or paste TOML/YAML to import settings, API credentials, and Smart Routes.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <select
                    value={configFormat}
                    onChange={async (e) => {
                      const fmt = e.target.value as 'toml' | 'yaml';
                      setConfigFormat(fmt);
                      const res = await api.exportConfig(fmt);
                      setConfigText(res.content);
                    }}
                    className="rounded-lg bg-slate-950 border border-white/10 px-3 py-1.5 text-xs text-white"
                  >
                    <option value="toml">TOML</option>
                    <option value="yaml">YAML</option>
                  </select>

                  <button
                    type="button"
                    onClick={async () => {
                      const res = await api.exportConfig(configFormat);
                      setConfigText(res.content);
                    }}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-white/8 hover:bg-white/12 px-3 py-1.5 text-xs text-slate-200 cursor-pointer"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Refresh Export
                  </button>

                  <button
                    type="button"
                    onClick={async () => {
                      try {
                        const res = await api.importConfig(configText, configFormat);
                        setBanner({
                          kind: 'ok',
                          text: `Imported configuration (${res.applied_items} items synced).`,
                        });
                        await loadSettings();
                      } catch (err) {
                        setBanner({
                          kind: 'err',
                          text: err instanceof Error ? err.message : 'Config import failed',
                        });
                      }
                    }}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 px-3.5 py-1.5 text-xs font-semibold text-slate-950 cursor-pointer"
                  >
                    <Upload className="h-3.5 w-3.5" />
                    Apply Config
                  </button>
                </div>
              </div>

              <textarea
                rows={10}
                value={configText}
                onChange={(e) => setConfigText(e.target.value)}
                className="w-full rounded-xl bg-slate-950 border border-white/10 p-4 text-xs font-mono text-slate-200 leading-relaxed"
              />
            </div>
          </div>
        )
      )}

      {/* Trace Detail Drawer */}
      {selectedTrace && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-xs flex justify-end">
          <div className="w-full max-w-2xl bg-[#0d111a] border-l border-white/12 h-full overflow-y-auto p-6 space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-base font-semibold text-white font-mono">
                  {selectedTrace.id}
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  {selectedTrace.ingress_protocol} &middot; {selectedTrace.provider_id} &middot;{' '}
                  {selectedTrace.resolved_model}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedTrace(null)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-slate-400 text-[10px]">HTTP Status</div>
                <div className="text-white font-semibold mt-0.5">
                  {selectedTrace.status_code}
                </div>
              </div>
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-slate-400 text-[10px]">TTFT / Duration</div>
                <div className="text-white font-semibold mt-0.5">
                  {selectedTrace.ttft_ms ?? '—'}ms / {selectedTrace.duration_ms}ms
                </div>
              </div>
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-slate-400 text-[10px]">Billed Cost</div>
                <div className="text-white font-semibold mt-0.5">
                  {formatNanos(selectedTrace.cost_nanos)}
                </div>
              </div>
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-slate-400 text-[10px]">API-Equiv Value</div>
                <div className="text-emerald-300 font-semibold mt-0.5">
                  {formatNanos(selectedTrace.api_equiv_nanos)}
                </div>
              </div>
            </div>

            {selectedTrace.otel_trace_id && (
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3 flex items-center justify-between text-xs">
                <span className="font-mono text-slate-300">
                  otel.trace_id: {selectedTrace.otel_trace_id}
                </span>
                <a
                  href={`${settings?.phoenix_ui_url ?? 'http://localhost:6006'}/projects`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-orange-300 hover:text-orange-200"
                >
                  View in Phoenix
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
            )}

            {selectedTrace.error_message && (
              <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 p-3 text-xs font-mono text-rose-200">
                {selectedTrace.error_message}
              </div>
            )}

            <div className="space-y-2">
              <div className="text-xs font-semibold text-slate-300">
                Captured Request Payload
              </div>
              <pre className="rounded-xl bg-slate-950 border border-white/10 p-4 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-64">
                {selectedTrace.request_body_json
                  ? JSON.stringify(JSON.parse(selectedTrace.request_body_json), null, 2)
                  : 'Payload capture disabled (privacy-first default — enable in Settings to record prompts).'}
              </pre>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-semibold text-slate-300">
                Captured Response Payload
              </div>
              <pre className="rounded-xl bg-slate-950 border border-white/10 p-4 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-64">
                {selectedTrace.response_body_json
                  ? JSON.stringify(JSON.parse(selectedTrace.response_body_json), null, 2)
                  : 'Payload capture disabled.'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
