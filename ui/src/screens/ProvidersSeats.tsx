import { useEffect, useState } from 'react';
import {
  ArrowDown,
  ArrowUp,
  Check,
  Copy,
  ExternalLink,
  KeyRound,
  Plus,
  RefreshCw,
  RotateCcw,
  Server,
  Shield,
  Trash2,
  X,
  Zap,
} from 'lucide-react';
import { api, formatRelativeTime } from '../api';
import { QuotaRing } from '../components/QuotaRing';
import type {
  ApiProviderDescriptor,
  CredentialRecord,
  LocalRuntimeStatus,
  OAuthProviderDescriptor,
  OAuthWizardSession,
  OpenRouterStatusResponse,
} from '../types';

const QUOTA_POLLABLE_PROVIDERS = new Set([
  'openai-codex',
  'anthropic-token',
  'google-gemini-cli',
  'openrouter',
]);

const LOCAL_RUNTIME_PRESETS = [
  {
    runtime_id: 'llamacpp',
    display_name: 'llama.cpp Server',
    base_url: 'http://127.0.0.1:8080/v1',
  },
  {
    runtime_id: 'ollama',
    display_name: 'Ollama',
    base_url: 'http://127.0.0.1:11434/v1',
  },
  {
    runtime_id: 'lmstudio',
    display_name: 'LM Studio',
    base_url: 'http://127.0.0.1:1234/v1',
  },
  {
    runtime_id: 'mlx',
    display_name: 'Apple MLX',
    base_url: 'http://127.0.0.1:8085/v1',
  },
  {
    runtime_id: 'vllm',
    display_name: 'vLLM',
    base_url: 'http://127.0.0.1:8000/v1',
  },
] as const;

export function ProvidersSeats() {
  const [oauthCatalog, setOauthCatalog] = useState<OAuthProviderDescriptor[]>([]);
  const [apiCatalog, setApiCatalog] = useState<ApiProviderDescriptor[]>([]);
  const [credentials, setCredentials] = useState<CredentialRecord[]>([]);
  const [localRuntimes, setLocalRuntimes] = useState<LocalRuntimeStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [banner, setBanner] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  // OAuth Wizard Modal State
  const [activeOAuth, setActiveOAuth] = useState<OAuthProviderDescriptor | null>(null);
  const [seatLabel, setSeatLabel] = useState('');
  const [seatPriority, setSeatPriority] = useState(50);
  const [preferDeviceCode, setPreferDeviceCode] = useState(false);
  const [setupTokenInput, setSetupTokenInput] = useState('');
  const [regionInput, setRegionInput] = useState('global');
  const [enterpriseDomain, setEnterpriseDomain] = useState('');
  const [customClientId, setCustomClientId] = useState('');
  const [wizardSession, setWizardSession] = useState<OAuthWizardSession | null>(null);
  const [manualCallbackCode, setManualCallbackCode] = useState('');
  const [wizardBusy, setWizardBusy] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  // API-Key Modal State
  const [showApiModal, setShowApiModal] = useState(false);
  const [apiSearch, setApiSearch] = useState('');
  const [selectedApiProvider, setSelectedApiProvider] = useState('openai');
  const [apiLabel, setApiLabel] = useState('');
  const [apiKeyVal, setApiKeyVal] = useState('');
  const [apiBaseUrl, setApiBaseUrl] = useState('');
  const [apiPriority, setApiPriority] = useState(200);
  const [apiAllowOverflow, setApiAllowOverflow] = useState(true);
  const [apiBusy, setApiBusy] = useState(false);

  // Custom Local Runtime Modal State
  const [showLocalModal, setShowLocalModal] = useState(false);
  const [localId, setLocalId] = useState('llamacpp');
  const [localName, setLocalName] = useState('llama.cpp Server');
  const [localUrl, setLocalUrl] = useState('http://127.0.0.1:8080/v1');
  const [localApiKey, setLocalApiKey] = useState('');
  const [localControlBusy, setLocalControlBusy] = useState<string | null>(null);

  // First-Class OpenRouter Hub State
  const [orStatus, setOrStatus] = useState<OpenRouterStatusResponse | null>(null);
  const [orModeModal, setOrModeModal] = useState<'api_key' | 'oidc' | null>(null);
  const [orKeyInput, setOrKeyInput] = useState('');
  const [orLabelInput, setOrLabelInput] = useState('');
  const [orRegion, setOrRegion] = useState('global');
  const [orOidcToken, setOrOidcToken] = useState('');
  const [orOidcEnvVar, setOrOidcEnvVar] = useState('');
  const [orFederationPolicyId, setOrFederationPolicyId] = useState('');
  const [orBusy, setOrBusy] = useState(false);

  async function loadAll() {
    try {
      const [cat, creds, ov, openrouter] = await Promise.all([
        api.getProvidersCatalog(),
        api.getCredentials(),
        api.getOverview(),
        api.getOpenRouterStatus().catch(() => null),
      ]);
      setOauthCatalog(cat.oauth_providers);
      setApiCatalog(cat.api_providers);
      setCredentials(creds);
      setLocalRuntimes(ov.local_runtimes);
      if (openrouter) setOrStatus(openrouter);
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Failed to load providers',
      });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadAll();
  }, []);

  // Poll pending OAuth wizard session automatically
  useEffect(() => {
    if (!wizardSession || wizardSession.status !== 'pending') return;
    const interval = setInterval(async () => {
      try {
        const next = await api.getOAuthSession(wizardSession.session_id);
        setWizardSession(next);
        if (next.status === 'completed') {
          setBanner({
            kind: 'ok',
            text: `Connected seat "${next.label}" (${next.provider_id})!`,
          });
          await loadAll();
        }
      } catch {
        // Keep polling gently
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [wizardSession]);

  function openOAuthModal(desc: OAuthProviderDescriptor) {
    setActiveOAuth(desc);
    const existingCount = credentials.filter((c) => c.provider_id === desc.provider_id).length;
    setSeatLabel(`${desc.display_name.split('(')[0].trim()} #${existingCount + 1}`);
    setSeatPriority(10 + existingCount * 10);
    setPreferDeviceCode(desc.flow_style === 'device_code');
    setSetupTokenInput('');
    setManualCallbackCode('');
    setWizardSession(null);
  }

  async function handleStartOAuth(e: React.FormEvent) {
    e.preventDefault();
    if (!activeOAuth) return;
    setWizardBusy(true);
    try {
      const sess = await api.startOAuth({
        provider_id: activeOAuth.provider_id,
        label: seatLabel,
        priority: seatPriority,
        prefer_device_code: preferDeviceCode,
        setup_token: setupTokenInput || undefined,
        region: activeOAuth.provider_id === 'minimax-portal' ? regionInput : undefined,
        enterprise_domain: enterpriseDomain || undefined,
        client_id: customClientId || undefined,
      });
      setWizardSession(sess);
      if (sess.status === 'completed') {
        setBanner({
          kind: 'ok',
          text: `Saved seat "${sess.label}" in AES-256-GCM vault!`,
        });
        setActiveOAuth(null);
        await loadAll();
      }
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Could not start OAuth flow',
      });
    } finally {
      setWizardBusy(false);
    }
  }

  async function handleCompleteManualCode(e: React.FormEvent) {
    e.preventDefault();
    if (!wizardSession || !manualCallbackCode.trim()) return;
    setWizardBusy(true);
    try {
      const sess = await api.completeOAuthSession(
        wizardSession.session_id,
        manualCallbackCode.trim(),
      );
      setWizardSession(sess);
      if (sess.status === 'completed') {
        setBanner({ kind: 'ok', text: `Seat "${sess.label}" authorized!` });
        setActiveOAuth(null);
        await loadAll();
      }
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Callback code exchange failed',
      });
    } finally {
      setWizardBusy(false);
    }
  }

  async function handleSaveApiCredential(e: React.FormEvent) {
    e.preventDefault();
    setApiBusy(true);
    try {
      const created = await api.createCredential({
        provider_id: selectedApiProvider,
        label: apiLabel.trim() || `${selectedApiProvider} API key`,
        auth_kind: apiKeyVal.trim() ? 'api_key' : 'local_noauth',
        api_key: apiKeyVal.trim() || undefined,
        base_url: apiBaseUrl.trim() || undefined,
        priority: apiPriority,
        enabled: true,
        allow_api_overflow: apiAllowOverflow,
      });
      setBanner({
        kind: 'ok',
        text: `Added credential "${created.label}" (${created.provider_id})`,
      });
      setShowApiModal(false);
      setApiKeyVal('');
      setApiLabel('');
      setApiBaseUrl('');
      await loadAll();
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Failed to save API credential',
      });
    } finally {
      setApiBusy(false);
    }
  }

  async function movePriority(cred: CredentialRecord, delta: number) {
    const nextPrio = Math.max(1, cred.priority + delta);
    await api.patchCredential(cred.id, { priority: nextPrio });
    await loadAll();
  }

  async function toggleEnabled(cred: CredentialRecord) {
    await api.patchCredential(cred.id, { enabled: !cred.enabled });
    await loadAll();
  }

  async function toggleOverflow(cred: CredentialRecord) {
    await api.patchCredential(cred.id, {
      allow_api_overflow: !cred.allow_api_overflow,
    });
    await loadAll();
  }

  async function handleTestSeat(cred: CredentialRecord) {
    try {
      const res = await api.testCredential(cred.id);
      setBanner({
        kind: 'ok',
        text: `Seat "${cred.label}" verified fresh in ${res.latency_ms}ms.`,
      });
      await loadAll();
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Seat check failed',
      });
    }
  }

  async function handleRegisterLocal(e: React.FormEvent) {
    e.preventDefault();
    try {
      const updated = await api.registerLocalRuntime({
        runtime_id: localId,
        display_name: localName,
        base_url: localUrl,
        api_key: localApiKey.trim() || undefined,
      });
      setLocalRuntimes(updated);
      setShowLocalModal(false);
      setLocalApiKey('');
      await loadAll();
      setBanner({ kind: 'ok', text: `Configured local runtime "${localName}"` });
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Failed to register local runtime',
      });
    }
  }

  async function handleLocalModelControl(
    runtimeId: string,
    model: string,
    action: 'load' | 'unload',
  ) {
    const key = `${runtimeId}:${model}`;
    setLocalControlBusy(key);
    try {
      const updated =
        action === 'load'
          ? await api.loadLocalModel(runtimeId, model)
          : await api.unloadLocalModel(runtimeId, model);
      setLocalRuntimes(updated);
      setBanner({
        kind: 'ok',
        text: `${action === 'load' ? 'Loading' : 'Unloaded'} "${model}" on ${runtimeId}`,
      });
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : `Failed to ${action} model`,
      });
    } finally {
      setLocalControlBusy(null);
    }
  }

  const filteredApiProviders = apiCatalog.filter(
    (p) =>
      p.provider_id.toLowerCase().includes(apiSearch.toLowerCase()) ||
      p.display_name.toLowerCase().includes(apiSearch.toLowerCase()),
  );
  const subscriptionOAuthCatalog = oauthCatalog.filter(
    (d) => d.provider_id !== 'openrouter',
  );
  const openRouterDescriptor = oauthCatalog.find(
    (d) => d.provider_id === 'openrouter',
  );

  async function handleConnectOpenRouterEnv() {
    setOrBusy(true);
    try {
      const res = await api.connectOpenRouter({
        method: 'env',
        region: orRegion,
      });
      setBanner({
        kind: 'ok',
        text: `Imported OPENROUTER_API_KEY · ${res.synced_models ?? 0} models · ${res.synced_presets ?? 0} @presets synced!`,
      });
      await loadAll();
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Failed to import OPENROUTER_API_KEY',
      });
    } finally {
      setOrBusy(false);
    }
  }

  async function handleSubmitOpenRouterModal(e: React.FormEvent) {
    e.preventDefault();
    if (!orModeModal) return;
    setOrBusy(true);
    try {
      const res =
        orModeModal === 'api_key'
          ? await api.connectOpenRouter({
              method: 'api_key',
              api_key: orKeyInput.trim(),
              label: orLabelInput.trim() || undefined,
              region: orRegion,
            })
          : await api.connectOpenRouter({
              method: 'oidc',
              subject_token: orOidcToken.trim() || undefined,
              subject_token_env: orOidcEnvVar.trim() || undefined,
              federation_policy_id: orFederationPolicyId.trim() || undefined,
              label: orLabelInput.trim() || undefined,
              region: orRegion,
            });
      setBanner({
        kind: 'ok',
        text: `Connected OpenRouter (${orModeModal.toUpperCase()}) · ${res.synced_models ?? 0} models · ${res.synced_presets ?? 0} @presets synced!`,
      });
      setOrModeModal(null);
      setOrKeyInput('');
      setOrOidcToken('');
      await loadAll();
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'OpenRouter connection failed',
      });
    } finally {
      setOrBusy(false);
    }
  }

  async function handleSyncOpenRouter() {
    setOrBusy(true);
    try {
      const res = await api.syncOpenRouter();
      setBanner({
        kind: 'ok',
        text: `Synced ${res.synced_models} OpenRouter models & ${res.synced_presets} @preset/* presets!`,
      });
      await loadAll();
    } catch (err) {
      setBanner({
        kind: 'err',
        text: err instanceof Error ? err.message : 'OpenRouter sync failed',
      });
    } finally {
      setOrBusy(false);
    }
  }

  if (loading) {
    return <div className="p-8 text-sm text-slate-400">Loading provider vault…</div>;
  }

  return (
    <div className="p-6 lg:p-8 space-y-7 max-w-[1440px] mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Providers: Local · OAuth Subscriptions · OpenRouter
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Three co-equal first-class pillars (<code className="text-emerald-300 text-xs">$0</code> local runtimes,{' '}
            <code className="text-indigo-300 text-xs">$0</code> OAuth subscription seats, and optional{' '}
            <code className="text-violet-300 text-xs">OpenRouter</code>) + {apiCatalog.length} direct{' '}
            <code className="text-indigo-300 text-xs">aisdk</code> API providers.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() => setShowLocalModal(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 px-3.5 py-2 text-xs font-medium text-slate-200 cursor-pointer"
          >
            <Server className="h-3.5 w-3.5 text-emerald-400" />
            Add Local Server
          </button>
          <button
            type="button"
            onClick={() => setShowApiModal(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-3.5 py-2 text-xs font-medium text-white cursor-pointer"
          >
            <KeyRound className="h-3.5 w-3.5" />
            Add API Key ({apiCatalog.length} Providers)
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

      {/* Pillar 2: Subscription OAuth Portals ($0 Marginal Cost) */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-semibold text-white">
              OAuth Subscriptions ($0 Marginal Cost)
            </h2>
            <p className="text-xs text-slate-400">
              Pool multiple personal or team subscriptions — drained first via sticky <code className="text-indigo-300">fill_first</code> routing.
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {subscriptionOAuthCatalog.map((desc) => {
            const seatCount = credentials.filter(
              (c) => c.provider_id === desc.provider_id,
            ).length;
            return (
              <div
                key={desc.provider_id}
                className="rounded-xl bg-[#0d111a] border border-white/8 p-5 flex flex-col justify-between hover:border-indigo-500/35 transition"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-semibold text-white">
                        {desc.display_name}
                      </h3>
                      <span className="font-mono text-[11px] text-indigo-300">
                        {desc.provider_id}
                      </span>
                    </div>
                    {seatCount > 0 ? (
                      <span className="shrink-0 whitespace-nowrap rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300">
                        {seatCount} {seatCount === 1 ? 'seat' : 'seats'}
                      </span>
                    ) : (
                      <span className="shrink-0 whitespace-nowrap rounded-full bg-white/5 px-2.5 py-0.5 text-[11px] text-slate-400">
                        {desc.flow_style.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                  <p className="mt-2.5 text-xs text-slate-400 leading-relaxed">
                    {desc.description}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-white/6 flex items-center justify-between">
                  <span className="text-[11px] text-slate-500 font-mono truncate max-w-[180px]">
                    {desc.auth_kind === 'api_key' ? 'PKCE Key Mint' : '$0 Subscription Seat'}
                  </span>
                  <button
                    type="button"
                    onClick={() => openOAuthModal(desc)}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-white/8 hover:bg-indigo-500 hover:text-white px-3 py-1.5 text-xs font-medium text-slate-200 transition cursor-pointer"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    {seatCount > 0 ? 'Add Another Seat' : 'Connect Seat'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Pillar 3: First-Class OpenRouter Hub (Optional) */}
      <div className="rounded-xl bg-gradient-to-br from-violet-500/8 via-[#0d111a] to-[#0d111a] border border-violet-500/25 p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-sm font-semibold text-white">
                OpenRouter Hub
              </h2>
              {orStatus?.connected ? (
                <span className="rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300">
                  Connected ({orStatus.seats.filter((s) => s.enabled).length}{' '}
                  {orStatus.seats.filter((s) => s.enabled).length === 1 ? 'seat' : 'seats'})
                </span>
              ) : (
                <span className="rounded-full bg-white/6 px-2.5 py-0.5 text-[11px] text-slate-300">
                  Optional · Not Connected
                </span>
              )}
              {orStatus?.connected && (
                <span className="rounded-md bg-violet-500/15 border border-violet-500/30 px-2 py-0.5 font-mono text-[11px] text-violet-200">
                  {orStatus.catalog_model_count} models · {orStatus.preset_count} @presets
                </span>
              )}
            </div>
            <p className="mt-1 text-xs text-slate-400">
              Universal 300+ model catalog, server-side <code className="text-violet-300">@preset/&lt;slug&gt;</code> presets,{' '}
              <code className="text-violet-300">~author/*-latest</code> aliases, and <code className="text-violet-300">:nitro</code> /{' '}
              <code className="text-violet-300">:floor</code> modifiers. Strictly optional — zero OpenRouter models appear until connected.
            </p>
          </div>

          {orStatus?.connected && (
            <button
              type="button"
              disabled={orBusy}
              onClick={() => void handleSyncOpenRouter()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/35 px-3 py-1.5 text-xs font-medium text-violet-200 cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${orBusy ? 'animate-spin' : ''}`} />
              Sync Models &amp; @Presets
            </button>
          )}
        </div>

        {/* 4 Frictionless Connection Methods (matching `ori login`) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
          <button
            type="button"
            disabled={orBusy || !openRouterDescriptor}
            onClick={() => {
              if (openRouterDescriptor) {
                openOAuthModal(openRouterDescriptor);
                setSeatPriority(200);
              }
            }}
            className="text-left rounded-xl bg-slate-950/80 hover:bg-slate-900 border border-violet-500/30 p-3.5 transition cursor-pointer"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-white">
                Sign in with Browser
              </span>
              <span className="rounded bg-violet-500/20 px-1.5 py-0.5 font-mono text-[10px] text-violet-200">
                PKCE
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Recommended · 1-click key minting via <code className="text-slate-300">openrouter.ai/auth</code>.
            </p>
          </button>

          <button
            type="button"
            disabled={orBusy}
            onClick={() => {
              setOrLabelInput('');
              setOrModeModal('oidc');
            }}
            className="text-left rounded-xl bg-slate-950/80 hover:bg-slate-900 border border-white/10 hover:border-violet-500/35 p-3.5 transition cursor-pointer"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-white">
                OpenRouter OIDC
              </span>
              <span className="rounded bg-indigo-500/15 px-1.5 py-0.5 font-mono text-[10px] text-indigo-300">
                RFC 8693
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Exchange Workload Identity JWT with single-flight auto-refresh.
            </p>
          </button>

          <button
            type="button"
            disabled={orBusy}
            onClick={() => {
              setOrLabelInput('');
              setOrKeyInput('');
              setOrModeModal('api_key');
            }}
            className="text-left rounded-xl bg-slate-950/80 hover:bg-slate-900 border border-white/10 hover:border-violet-500/35 p-3.5 transition cursor-pointer"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-white">
                Enter API Key
              </span>
              <span className="rounded bg-white/8 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                sk-or-v1
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Validate against <code className="text-slate-300">/key</code> &amp; <code className="text-slate-300">/credits</code> and store in Vault.
            </p>
          </button>

          <button
            type="button"
            disabled={orBusy || !orStatus?.env_key_detected}
            onClick={() => void handleConnectOpenRouterEnv()}
            className={`text-left rounded-xl border p-3.5 transition ${
              orStatus?.env_key_detected
                ? 'bg-emerald-500/8 hover:bg-emerald-500/15 border-emerald-500/35 cursor-pointer'
                : 'bg-slate-950/40 border-white/6 opacity-60 cursor-not-allowed'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-white">
                Import from Env
              </span>
              <span
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                  orStatus?.env_key_detected
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-white/6 text-slate-400'
                }`}
              >
                {orStatus?.env_key_detected ? 'Detected' : 'Unset'}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400 font-mono truncate">
              {orStatus?.env_key_detected
                ? `OPENROUTER_API_KEY (${orStatus.env_key_masked ?? 'set'})`
                : 'OPENROUTER_API_KEY not set in daemon env'}
            </p>
          </button>
        </div>

        {/* 1-Click Cloud Burst Overflow Bar */}
        <div className="pt-3 border-t border-white/8 flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2 text-xs font-medium text-white">
              <span>Cloud Burst Overflow (Local / Sub → OpenRouter)</span>
              <span className="rounded bg-white/6 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                X-Title: {orStatus?.app_title ?? 'QuixiGate'}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Try $0 local GPUs &amp; OAuth subscriptions first; automatically burst to{' '}
              <code className="text-violet-300">{orStatus?.burst_model ?? 'openrouter/auto'}</code>{' '}
              on <code className="text-slate-300">429</code> / saturation.
            </p>
          </div>
          <button
            type="button"
            disabled={orBusy}
            onClick={async () => {
              setOrBusy(true);
              try {
                const next = !(orStatus?.burst_enabled ?? false);
                const res = await api.setOpenRouterBurst({ enabled: next });
                setBanner({
                  kind: 'ok',
                  text: res.enabled
                    ? `Cloud Burst ENABLED → fallback ${res.burst_model}`
                    : 'Cloud Burst DISABLED (strict local/subscription isolation)',
                });
                await loadAll();
              } catch (err) {
                setBanner({
                  kind: 'err',
                  text: err instanceof Error ? err.message : 'Failed to toggle Cloud Burst',
                });
              } finally {
                setOrBusy(false);
              }
            }}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition cursor-pointer ${
              orStatus?.burst_enabled
                ? 'bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-200'
                : 'bg-white/7 hover:bg-white/12 border border-white/12 text-slate-200'
            }`}
          >
            {orStatus?.burst_enabled ? 'Burst: Enabled' : 'Enable Cloud Burst'}
          </button>
        </div>

        {orStatus?.connected && orStatus.presets.length > 0 && (
          <div className="pt-2 border-t border-white/8">
            <div className="text-[11px] text-slate-400 mb-2">
              Synced OpenRouter Presets (<code className="text-violet-300">@preset/&lt;slug&gt;</code>):
            </div>
            <div className="flex flex-wrap gap-1.5">
              {orStatus.presets.slice(0, 12).map((p) => (
                <span
                  key={p.id}
                  title={p.description ?? p.underlying_model ?? p.name}
                  className="rounded-md bg-violet-500/15 border border-violet-500/30 px-2 py-0.5 font-mono text-[11px] text-violet-200"
                >
                  @preset/{p.slug}
                </span>
              ))}
              {orStatus.presets.length > 12 && (
                <span className="text-[11px] text-slate-400 px-1.5 py-0.5">
                  +{orStatus.presets.length - 12} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Configured Seats Pool Table */}
      <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-white">
              Active Vault Seats ({credentials.length})
            </h2>
            <p className="text-xs text-slate-400">
              Lower priority numbers execute first (<code className="text-indigo-300">fill_first</code>). Paid API seats only receive overflow when <strong className="text-slate-200">API Overflow</strong> is enabled.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void loadAll()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-white/6 hover:bg-white/10 px-3 py-1.5 text-xs text-slate-200 cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>

        {credentials.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-400">
            No credentials stored yet. Connect an OAuth subscription seat above or add an API key.
          </div>
        ) : (
          <div className="space-y-3">
            {credentials.map((cred) => (
              <div
                key={cred.id}
                className="rounded-xl bg-slate-950/75 border border-white/7 p-4 space-y-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <div className="inline-flex items-center rounded-md bg-slate-900 border border-white/10">
                      <button
                        type="button"
                        title="Increase priority (lower number)"
                        onClick={() => void movePriority(cred, -10)}
                        className="p-1 text-slate-400 hover:text-white cursor-pointer"
                      >
                        <ArrowUp className="h-3 w-3" />
                      </button>
                      <span className="px-2 font-mono text-xs text-white tabular-nums">
                        #{cred.priority}
                      </span>
                      <button
                        type="button"
                        title="Decrease priority (higher number)"
                        onClick={() => void movePriority(cred, 10)}
                        className="p-1 text-slate-400 hover:text-white cursor-pointer"
                      >
                        <ArrowDown className="h-3 w-3" />
                      </button>
                    </div>

                    <span className="text-sm font-semibold text-white">
                      {cred.label}
                    </span>
                    <span className="rounded bg-indigo-500/15 border border-indigo-500/25 px-2 py-0.5 font-mono text-[11px] text-indigo-300">
                      {cred.provider_id}
                    </span>
                    <span className="rounded bg-white/6 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                      {cred.auth_kind}
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                        cred.status === 'ok'
                          ? 'bg-emerald-500/15 text-emerald-300'
                          : cred.status === 'exhausted'
                            ? 'bg-amber-500/15 text-amber-300'
                            : 'bg-rose-500/15 text-rose-300'
                      }`}
                    >
                      {cred.status}
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    {cred.auth_kind === 'api_key' && (
                      <button
                        type="button"
                        onClick={() => void toggleOverflow(cred)}
                        className={`rounded-md border px-2.5 py-1 text-[11px] font-medium cursor-pointer ${
                          cred.allow_api_overflow
                            ? 'bg-sky-500/15 border-sky-500/30 text-sky-300'
                            : 'bg-slate-900 border-white/10 text-slate-400'
                        }`}
                      >
                        API Overflow: {cred.allow_api_overflow ? 'ON' : 'OFF'}
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={() => void toggleEnabled(cred)}
                      className={`rounded-md border px-2.5 py-1 text-[11px] font-medium cursor-pointer ${
                        cred.enabled
                          ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                          : 'bg-slate-900 border-white/10 text-slate-400'
                      }`}
                    >
                      {cred.enabled ? 'Enabled' : 'Disabled'}
                    </button>

                    <button
                      type="button"
                      onClick={() => void handleTestSeat(cred)}
                      className="inline-flex items-center gap-1 rounded-md bg-white/6 hover:bg-white/12 px-2.5 py-1 text-[11px] text-slate-200 cursor-pointer"
                    >
                      <Zap className="h-3 w-3 text-amber-400" />
                      Verify
                    </button>

                    {QUOTA_POLLABLE_PROVIDERS.has(cred.provider_id) && (
                      <button
                        type="button"
                        onClick={async () => {
                          await api.refreshCredentialQuota(cred.id);
                          await loadAll();
                        }}
                        className="inline-flex items-center gap-1 rounded-md bg-white/6 hover:bg-white/12 px-2.5 py-1 text-[11px] text-slate-200 cursor-pointer"
                      >
                        <RefreshCw className="h-3 w-3" />
                        Quota
                      </button>
                    )}

                    {cred.status !== 'ok' && (
                      <button
                        type="button"
                        onClick={async () => {
                          await api.resetCredentialHealth(cred.id);
                          await loadAll();
                        }}
                        className="inline-flex items-center gap-1 rounded-md bg-amber-500/15 hover:bg-amber-500/25 px-2.5 py-1 text-[11px] text-amber-300 cursor-pointer"
                      >
                        <RotateCcw className="h-3 w-3" />
                        Reset Health
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={async () => {
                        await api.deleteCredential(cred.id);
                        await loadAll();
                      }}
                      className="inline-flex items-center gap-1 rounded-md bg-rose-500/10 hover:bg-rose-500/20 px-2 py-1 text-[11px] text-rose-300 cursor-pointer"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>

                {cred.quota_snapshot?.windows?.length ? (
                  <div className="flex flex-wrap gap-2.5">
                    {cred.quota_snapshot.windows.map((w) => (
                      <QuotaRing key={w.label} window={w} />
                    ))}
                  </div>
                ) : null}

                <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-500 font-mono">
                  <span>ID: {cred.id}</span>
                  {cred.base_url && <span>Endpoint: {cred.base_url}</span>}
                  {cred.expires_at_ms && (
                    <span>Token refreshes {formatRelativeTime(cred.expires_at_ms)}</span>
                  )}
                  {cred.status_reason && (
                    <span className="text-amber-300">Note: {cred.status_reason}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Local Runtimes Strip */}
      <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h2 className="text-sm font-semibold text-white">
              Local Inference Servers ({localRuntimes.length})
            </h2>
            <p className="text-xs text-slate-400">
              Native auto-discovery for <code className="text-emerald-300">llama-server</code> (single-instance &amp; multi-model router), Ollama, LM Studio, Apple MLX, and vLLM.
            </p>
          </div>
          <button
            type="button"
            onClick={async () => {
              const updated = await api.scanLocalRuntimes();
              setLocalRuntimes(updated);
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-white/6 hover:bg-white/10 px-3 py-1.5 text-xs text-slate-200 cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Scan Local Ports
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3.5">
          {localRuntimes.map((rt) => {
            const detailEntries = Object.entries(rt.model_details ?? {});
            const modelNames =
              detailEntries.length > 0 ? detailEntries.map(([m]) => m) : rt.models;

            return (
              <div
                key={rt.runtime_id}
                className="rounded-xl bg-slate-950/75 border border-white/6 p-4 flex flex-col justify-between gap-3"
              >
                <div>
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
                      <span className="text-xs font-semibold text-white truncate">
                        {rt.display_name}
                      </span>
                      {rt.mode && (
                        <span className="rounded bg-indigo-500/15 border border-indigo-500/25 px-1.5 py-0.5 font-mono text-[10px] text-indigo-300">
                          {rt.mode}
                        </span>
                      )}
                      {rt.total_slots != null && rt.total_slots > 0 && (
                        <span className="rounded bg-white/6 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                          {rt.total_slots} slots
                        </span>
                      )}
                    </div>
                    <span
                      className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-mono ${
                        rt.healthy
                          ? 'bg-emerald-500/15 text-emerald-300'
                          : rt.auth_required
                            ? 'bg-amber-500/15 text-amber-300'
                            : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {rt.healthy
                        ? `${rt.models.length} ready`
                        : rt.auth_required
                          ? 'API key required'
                          : 'offline'}
                    </span>
                  </div>

                  <div className="mt-1 flex items-center justify-between gap-2 font-mono text-[11px] text-slate-500">
                    <span className="truncate">{rt.base_url}</span>
                    {rt.server_version && (
                      <span className="shrink-0 text-[10px] text-slate-500">
                        {rt.server_version}
                      </span>
                    )}
                  </div>

                  {rt.auth_required && (
                    <div className="mt-2.5 rounded-lg bg-amber-500/10 border border-amber-500/25 p-2.5 text-[11px] text-amber-200 flex items-center justify-between gap-2">
                      <span>Server requires Bearer <code className="font-mono">--api-key</code></span>
                      <button
                        type="button"
                        onClick={() => {
                          setLocalId(rt.runtime_id);
                          setLocalName(rt.display_name);
                          setLocalUrl(rt.base_url);
                          setShowLocalModal(true);
                        }}
                        className="rounded bg-amber-500/20 hover:bg-amber-500/30 px-2 py-1 text-[10px] font-medium text-amber-100 cursor-pointer shrink-0"
                      >
                        Set Key
                      </button>
                    </div>
                  )}

                  {modelNames.length > 0 && (
                    <div className="mt-3 space-y-1.5 max-h-44 overflow-y-auto pr-0.5">
                      {modelNames.map((mod) => {
                        const info = rt.model_details?.[mod];
                        const statusStr =
                          info?.status ??
                          (rt.loaded_models.includes(mod) ? 'loaded' : 'ready');
                        const isLoaded =
                          statusStr === 'loaded' || statusStr === 'loading';
                        const busy = localControlBusy === `${rt.runtime_id}:${mod}`;

                        return (
                          <div
                            key={mod}
                            className="rounded-md bg-white/[0.03] border border-white/6 px-2.5 py-1.5 flex items-center justify-between gap-2 text-[11px]"
                          >
                            <div className="min-w-0 flex items-center gap-1.5">
                              <span
                                className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                                  isLoaded ? 'bg-emerald-400' : 'bg-slate-500'
                                }`}
                              />
                              <span className="font-mono text-slate-200 truncate">
                                {mod}
                              </span>
                              {info?.context_window ? (
                                <span className="shrink-0 rounded bg-white/6 px-1 py-0.2 font-mono text-[10px] text-slate-400">
                                  {Math.round(info.context_window / 1024)}k
                                </span>
                              ) : null}
                              {info?.supports_vision && (
                                <span className="shrink-0 rounded bg-sky-500/15 px-1 py-0.2 font-mono text-[10px] text-sky-300">
                                  vision
                                </span>
                              )}
                            </div>

                            <div className="flex items-center gap-1.5 shrink-0">
                              <span className="font-mono text-[10px] text-slate-400">
                                {statusStr}
                              </span>
                              {rt.supports_load_control && (
                                <button
                                  type="button"
                                  disabled={busy}
                                  onClick={() =>
                                    void handleLocalModelControl(
                                      rt.runtime_id,
                                      mod,
                                      isLoaded ? 'unload' : 'load',
                                    )
                                  }
                                  className={`rounded px-2 py-0.5 text-[10px] font-medium cursor-pointer ${
                                    isLoaded
                                      ? 'bg-rose-500/15 hover:bg-rose-500/25 text-rose-300'
                                      : 'bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300'
                                  }`}
                                >
                                  {busy ? '…' : isLoaded ? 'Unload' : 'Load'}
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                <div className="pt-2 border-t border-white/5 flex items-center justify-between text-[11px]">
                  <span className="font-mono text-slate-500">
                    slug: {rt.runtime_id}/&lt;model&gt;
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setLocalId(rt.runtime_id);
                      setLocalName(rt.display_name);
                      setLocalUrl(rt.base_url);
                      setShowLocalModal(true);
                    }}
                    className="text-slate-400 hover:text-white cursor-pointer"
                  >
                    Configure
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* OAuth / Setup-Token Wizard Modal */}
      {activeOAuth && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-base font-semibold text-white">
                  {activeOAuth.display_name}
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  {activeOAuth.description}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setActiveOAuth(null);
                  setWizardSession(null);
                }}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {!wizardSession ? (
              <form onSubmit={handleStartOAuth} className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="col-span-2">
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      Seat Label
                    </label>
                    <input
                      type="text"
                      value={seatLabel}
                      onChange={(e) => setSeatLabel(e.target.value)}
                      required
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      Priority
                    </label>
                    <input
                      type="number"
                      value={seatPriority}
                      onChange={(e) => setSeatPriority(Number(e.target.value))}
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                    />
                  </div>
                </div>

                {activeOAuth.provider_id === 'anthropic-token' && (
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      Claude Setup Token (<code className="text-indigo-300">sk-ant-oat01-...</code>)
                    </label>
                    <textarea
                      rows={3}
                      value={setupTokenInput}
                      onChange={(e) => setSetupTokenInput(e.target.value)}
                      placeholder="Run `claude setup-token` in your terminal and paste the sk-ant-oat01-... token here"
                      required
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                    />
                    <p className="mt-1 text-[11px] text-slate-400 flex items-center gap-1">
                      <Shield className="h-3 w-3 text-emerald-400" />
                      Stored strictly inside QuixiGate's AES-256-GCM vault — never reads external ~/.claude files.
                    </p>
                  </div>
                )}

                {activeOAuth.provider_id === 'openai-codex' && (
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={preferDeviceCode}
                      onChange={(e) => setPreferDeviceCode(e.target.checked)}
                    />
                    Use Device Code flow (ideal for headless / remote sessions)
                  </label>
                )}

                {activeOAuth.provider_id === 'minimax-portal' && (
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      MiniMax Region
                    </label>
                    <select
                      value={regionInput}
                      onChange={(e) => setRegionInput(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                    >
                      <option value="global">Global (api.minimax.io)</option>
                      <option value="cn">Mainland China (api.minimaxi.com)</option>
                    </select>
                  </div>
                )}

                {activeOAuth.provider_id === 'github-copilot' && (
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      GitHub Enterprise Domain (optional)
                    </label>
                    <input
                      type="text"
                      value={enterpriseDomain}
                      onChange={(e) => setEnterpriseDomain(e.target.value)}
                      placeholder="Leave blank for github.com"
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                    />
                  </div>
                )}

                {activeOAuth.provider_id === 'chutes' && (
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      Chutes OAuth Client ID (optional override)
                    </label>
                    <input
                      type="text"
                      value={customClientId}
                      onChange={(e) => setCustomClientId(e.target.value)}
                      placeholder="Defaults to CHUTES_CLIENT_ID / quixigate"
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                    />
                  </div>
                )}

                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setActiveOAuth(null)}
                    className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={wizardBusy}
                    className="rounded-lg bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 px-4 py-2 text-xs font-medium text-white cursor-pointer"
                  >
                    {wizardBusy
                      ? 'Starting…'
                      : activeOAuth.flow_style === 'setup_token'
                        ? 'Validate & Save Token'
                        : 'Start Sign-In'}
                  </button>
                </div>
              </form>
            ) : (
              <div className="space-y-4">
                {wizardSession.user_code && (
                  <div className="rounded-xl bg-slate-950 border border-indigo-500/30 p-4 text-center space-y-2">
                    <div className="text-xs text-slate-400">
                      Enter this verification code in your browser:
                    </div>
                    <div className="flex items-center justify-center gap-2">
                      <span className="font-mono text-2xl font-bold tracking-widest text-indigo-300">
                        {wizardSession.user_code}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          void navigator.clipboard.writeText(wizardSession.user_code ?? '');
                          setCopiedCode(true);
                          setTimeout(() => setCopiedCode(false), 1800);
                        }}
                        className="rounded-md bg-white/8 p-1.5 text-slate-200 cursor-pointer"
                      >
                        {copiedCode ? (
                          <Check className="h-4 w-4 text-emerald-400" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                  </div>
                )}

                {(wizardSession.authorize_url || wizardSession.verification_uri) && (
                  <a
                    href={wizardSession.authorize_url ?? wizardSession.verification_uri ?? '#'}
                    target="_blank"
                    rel="noreferrer"
                    className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2.5 text-xs font-medium text-white"
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    Open Vendor Authorization Page
                  </a>
                )}

                {wizardSession.accepts_manual_code && (
                  <form onSubmit={handleCompleteManualCode} className="space-y-2 pt-2 border-t border-white/8">
                    <label className="block text-xs text-slate-300">
                      Or paste callback URL / code manually (for remote port-forwarding):
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={manualCallbackCode}
                        onChange={(e) => setManualCallbackCode(e.target.value)}
                        placeholder="http://localhost:1455/auth/callback?code=..."
                        className="flex-1 rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                      />
                      <button
                        type="submit"
                        disabled={wizardBusy}
                        className="rounded-lg bg-emerald-500 hover:bg-emerald-400 px-3.5 py-2 text-xs font-medium text-slate-950 cursor-pointer"
                      >
                        Complete
                      </button>
                    </div>
                  </form>
                )}

                <div className="text-xs text-slate-400 flex items-center gap-2">
                  <RefreshCw className="h-3.5 w-3.5 animate-spin text-indigo-400" />
                  Status: <span className="font-mono text-white">{wizardSession.status}</span>
                  {wizardSession.error && (
                    <span className="text-rose-300">({wizardSession.error})</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Add Built-in API Provider Modal */}
      {showApiModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">
                Add API Key / Custom OpenAI-Compatible Endpoint
              </h3>
              <button
                type="button"
                onClick={() => setShowApiModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={handleSaveApiCredential} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Filter {apiCatalog.length} Built-In aisdk Providers
                </label>
                <input
                  type="text"
                  value={apiSearch}
                  onChange={(e) => setApiSearch(e.target.value)}
                  placeholder="Search e.g. openai, anthropic, groq, deepseek, togetherai, cerebras..."
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white mb-2"
                />
                <select
                  value={selectedApiProvider}
                  onChange={(e) => setSelectedApiProvider(e.target.value)}
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                >
                  {filteredApiProviders.map((p) => (
                    <option key={p.provider_id} value={p.provider_id}>
                      {p.display_name} ({p.provider_id})
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="col-span-2">
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Label
                  </label>
                  <input
                    type="text"
                    value={apiLabel}
                    onChange={(e) => setApiLabel(e.target.value)}
                    placeholder={`${selectedApiProvider} production key`}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Priority
                  </label>
                  <input
                    type="number"
                    value={apiPriority}
                    onChange={(e) => setApiPriority(Number(e.target.value))}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  API Key (encrypted at rest in vault)
                </label>
                <input
                  type="password"
                  value={apiKeyVal}
                  onChange={(e) => setApiKeyVal(e.target.value)}
                  placeholder="sk-..."
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Custom Base URL (optional)
                </label>
                <input
                  type="text"
                  value={apiBaseUrl}
                  onChange={(e) => setApiBaseUrl(e.target.value)}
                  placeholder="Leave empty for vendor default"
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>

              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={apiAllowOverflow}
                  onChange={(e) => setApiAllowOverflow(e.target.checked)}
                />
                Allow automatic paid overflow when $0 OAuth seats are on cooldown
              </label>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowApiModal(false)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={apiBusy}
                  className="rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
                >
                  {apiBusy ? 'Saving…' : 'Save API Key'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Custom Local Runtime Modal */}
      {showLocalModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">
                Add / Configure Local Server
              </h3>
              <button
                type="button"
                onClick={() => setShowLocalModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {LOCAL_RUNTIME_PRESETS.map((preset) => (
                <button
                  key={preset.runtime_id}
                  type="button"
                  onClick={() => {
                    setLocalId(preset.runtime_id);
                    setLocalName(preset.display_name);
                    setLocalUrl(preset.base_url);
                  }}
                  className={`rounded-md border px-2.5 py-1 text-[11px] font-medium cursor-pointer ${
                    localId === preset.runtime_id
                      ? 'bg-emerald-500/15 border-emerald-500/35 text-emerald-300'
                      : 'bg-white/5 border-white/10 text-slate-300 hover:bg-white/10'
                  }`}
                >
                  {preset.display_name}
                </button>
              ))}
            </div>

            <form onSubmit={handleRegisterLocal} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Runtime Slug
                </label>
                <input
                  type="text"
                  value={localId}
                  onChange={(e) => setLocalId(e.target.value)}
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Display Name
                </label>
                <input
                  type="text"
                  value={localName}
                  onChange={(e) => setLocalName(e.target.value)}
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  OpenAI-Compatible Base URL
                </label>
                <input
                  type="text"
                  value={localUrl}
                  onChange={(e) => setLocalUrl(e.target.value)}
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Optional Bearer API Key (e.g. <code className="text-emerald-300">llama-server --api-key</code>)
                </label>
                <input
                  type="password"
                  value={localApiKey}
                  onChange={(e) => setLocalApiKey(e.target.value)}
                  placeholder="Leave blank if local server requires no auth"
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowLocalModal(false)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-emerald-500 hover:bg-emerald-400 px-4 py-2 text-xs font-medium text-slate-950 cursor-pointer"
                >
                  Probe & Save
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* OpenRouter API Key / OIDC Workload Identity Modal */}
      {orModeModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[#0d111a] border border-violet-500/30 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">
                {orModeModal === 'api_key'
                  ? 'Connect OpenRouter via API Key'
                  : 'Connect OpenRouter via OIDC (RFC 8693)'}
              </h3>
              <button
                type="button"
                onClick={() => setOrModeModal(null)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={handleSubmitOpenRouterModal} className="space-y-3">
              {orModeModal === 'api_key' ? (
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    OpenRouter API Key (<code className="text-violet-300">sk-or-v1-...</code>)
                  </label>
                  <input
                    type="password"
                    required
                    value={orKeyInput}
                    onChange={(e) => setOrKeyInput(e.target.value)}
                    placeholder="sk-or-v1-..."
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
              ) : (
                <>
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">
                      OIDC Subject JWT (or leave blank to read from env var below)
                    </label>
                    <input
                      type="password"
                      value={orOidcToken}
                      onChange={(e) => setOrOidcToken(e.target.value)}
                      placeholder="eyJhbGciOiJSUzI1NiIs..."
                      className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                    />
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Subject Token Env Var (auto-refresh)
                      </label>
                      <input
                        type="text"
                        value={orOidcEnvVar}
                        onChange={(e) => setOrOidcEnvVar(e.target.value)}
                        placeholder={orStatus?.oidc_env_detected ?? 'OPENROUTER_OIDC_TOKEN'}
                        className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Federation Policy ID (optional)
                      </label>
                      <input
                        type="text"
                        value={orFederationPolicyId}
                        onChange={(e) => setOrFederationPolicyId(e.target.value)}
                        placeholder="pol_..."
                        className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                      />
                    </div>
                  </div>
                </>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Seat Label (optional)
                  </label>
                  <input
                    type="text"
                    value={orLabelInput}
                    onChange={(e) => setOrLabelInput(e.target.value)}
                    placeholder="OpenRouter Primary"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Data Region
                  </label>
                  <select
                    value={orRegion}
                    onChange={(e) => setOrRegion(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="global">Global (openrouter.ai)</option>
                    <option value="us">US (us.openrouter.ai)</option>
                    <option value="eu">EU (eu.openrouter.ai)</option>
                  </select>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setOrModeModal(null)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={orBusy}
                  className="rounded-lg bg-violet-500 hover:bg-violet-400 px-4 py-2 text-xs font-semibold text-white cursor-pointer"
                >
                  {orBusy ? 'Validating & Syncing…' : 'Connect & Sync @Presets'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
