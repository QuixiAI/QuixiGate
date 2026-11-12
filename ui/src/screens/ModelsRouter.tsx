import { useEffect, useMemo, useState } from 'react';
import {
  Check,
  Copy,
  GitBranch,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { api, formatNanos, isModelRoutable } from '../api';
import type {
  CatalogModel,
  CredentialRecord,
  ModelRoute,
  PoolStrategy,
} from '../types';

type FilterChip = 'all' | 'ready' | 'local' | 'presets' | 'tools' | 'reasoning' | 'vision';

export function ModelsRouter({
  onOpenPlaygroundModel,
  onNavigateProviders,
}: {
  onOpenPlaygroundModel: (modelId: string) => void;
  onNavigateProviders?: () => void;
}) {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [routes, setRoutes] = useState<ModelRoute[]>([]);
  const [credentials, setCredentials] = useState<CredentialRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [query, setQuery] = useState('');
  const [chip, setChip] = useState<FilterChip>('ready');
  const [drawerModel, setDrawerModel] = useState<CatalogModel | null>(null);
  const [snippetTab, setSnippetTab] = useState<'cli' | 'openai' | 'anthropic' | 'aisdk'>('cli');
  const [copiedSnippet, setCopiedSnippet] = useState(false);

  // Route Alias Builder State
  const [showAliasModal, setShowAliasModal] = useState(false);
  const [aliasName, setAliasName] = useState('');
  const [aliasTargetsText, setAliasTargetsText] = useState(
    'anthropic/claude-sonnet-4-5\nopenai/gpt-5',
  );
  const [aliasStrategy, setAliasStrategy] = useState<PoolStrategy>('fill_first');
  const [aliasOverflow, setAliasOverflow] = useState(true);
  const [banner, setBanner] = useState<string | null>(null);

  async function load() {
    try {
      const bundle = await api.getModelsBundle();
      setModels(bundle.models);
      setRoutes(bundle.routes);
      setCredentials(bundle.credentials);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const enabledProviders = useMemo(() => {
    const s = new Set<string>();
    for (const c of credentials) {
      if (c.enabled) s.add(c.provider_id);
    }
    return s;
  }, [credentials]);

  function isModelReady(m: CatalogModel): boolean {
    return isModelRoutable(m, enabledProviders);
  }

  const readyCount = useMemo(
    () => models.filter((m) => isModelRoutable(m, enabledProviders)).length,
    [models, enabledProviders],
  );
  const presetCount = useMemo(
    () =>
      enabledProviders.has('openrouter')
        ? models.filter((m) => m.is_preset || m.canonical_id.startsWith('@preset/')).length
        : 0,
    [models, enabledProviders],
  );

  const filteredModels = useMemo(() => {
    const q = query.trim().toLowerCase();
    return models
      .filter((m) => {
        if (chip === 'ready' && !isModelReady(m)) return false;
        if (chip === 'local' && !m.is_local) return false;
        if (chip === 'presets' && !(m.is_preset || m.canonical_id.startsWith('@preset/'))) {
          return false;
        }
        if (chip === 'tools' && !m.supports_tools) return false;
        if (chip === 'reasoning' && !m.supports_reasoning) return false;
        if (chip === 'vision' && !m.supports_vision) return false;
        if (!q) return true;
        return (
          m.canonical_id.toLowerCase().includes(q) ||
          m.display_name.toLowerCase().includes(q) ||
          m.author.toLowerCase().includes(q)
        );
      })
      .slice(0, 120);
  }, [models, query, chip, enabledProviders]);

  async function handleSaveRoute(e: React.FormEvent) {
    e.preventDefault();
    const targets = aliasTargetsText
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [modelPart, pin] = line.split('@');
        return {
          model: modelPart.trim(),
          provider_id: pin ? pin.trim() : null,
        };
      });
    await api.upsertRoute({
      alias: aliasName.trim(),
      targets,
      pool_strategy: aliasStrategy,
      allow_paid_overflow: aliasOverflow,
    });
    setBanner(`Saved Smart Route alias "${aliasName.trim()}"`);
    setShowAliasModal(false);
    setAliasName('');
    await load();
  }

  function buildSnippet(modelId: string): string {
    const origin = window.location.origin;
    if (snippetTab === 'cli') {
      return `# Launch coding CLIs pointed at QuixiGate (zero dotfile mutation!)\nquixigate run claude -- --model ${modelId}\nquixigate run codex -- -m ${modelId}\nquixigate run gemini -- -m ${modelId}\nquixigate run opencode -- -m ${modelId}`;
    }
    if (snippetTab === 'openai') {
      return `import OpenAI from 'openai';\n\nconst client = new OpenAI({\n  baseURL: '${origin}/v1',\n  apiKey: process.env.QUIXIGATE_KEY ?? 'local',\n});\n\nconst res = await client.chat.completions.create({\n  model: '${modelId}',\n  messages: [{ role: 'user', content: 'Hello from QuixiGate!' }],\n});`;
    }
    if (snippetTab === 'anthropic') {
      return `import Anthropic from '@anthropic-ai/sdk';\n\nconst anthropic = new Anthropic({\n  baseURL: '${origin}',\n  apiKey: process.env.QUIXIGATE_KEY ?? 'local',\n});\n\nconst msg = await anthropic.messages.create({\n  model: '${modelId}',\n  max_tokens: 1024,\n  messages: [{ role: 'user', content: 'Hello from QuixiGate!' }],\n});`;
    }
    return `import { generateText } from 'ai';\nimport { createOpenAI } from '@ai-sdk/openai';\n\nconst quixigate = createOpenAI({\n  baseURL: '${origin}/v1',\n  apiKey: process.env.QUIXIGATE_KEY ?? 'local',\n});\n\nconst { text } = await generateText({\n  model: quixigate('${modelId}'),\n  prompt: 'Hello from QuixiGate!',\n});`;
  }

  if (loading) {
    return <div className="p-8 text-sm text-slate-400">Indexing models.dev catalog…</div>;
  }

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-[1440px] mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Model Explorer & Smart Router
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            {readyCount.toLocaleString()} ready models
            {enabledProviders.has('openrouter') ? ` · ${presetCount} @presets` : ''} across Local Runtimes, OAuth Subscriptions{enabledProviders.has('openrouter') ? ', OpenRouter' : ''}, &amp;{' '}
            <code className="text-indigo-300 text-xs">author/model@provider:online</code> Smart Routes.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          {enabledProviders.has('openrouter') && (
            <button
              type="button"
              disabled={syncing}
              onClick={async () => {
                setSyncing(true);
                try {
                  const res = await api.syncOpenRouter();
                  setBanner(
                    `Synced ${res.synced_models} OpenRouter models & ${res.synced_presets} @presets`,
                  );
                  await load();
                } catch (err) {
                  setBanner(
                    err instanceof Error ? err.message : 'OpenRouter sync failed',
                  );
                } finally {
                  setSyncing(false);
                }
              }}
              className="inline-flex items-center gap-2 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/35 px-3.5 py-2 text-xs font-medium text-violet-200 cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${syncing ? 'animate-spin' : ''}`} />
              Sync OpenRouter
            </button>
          )}
          <button
            type="button"
            disabled={syncing}
            onClick={async () => {
              setSyncing(true);
              try {
                const res = await api.syncModelsDev();
                setBanner(`Synced ${res.synced_models} models from models.dev`);
                await load();
              } catch (err) {
                setBanner(err instanceof Error ? err.message : 'Offline snapshot active');
              } finally {
                setSyncing(false);
              }
            }}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 px-3.5 py-2 text-xs font-medium text-slate-200 cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${syncing ? 'animate-spin' : ''}`} />
            Sync models.dev
          </button>
          <button
            type="button"
            onClick={() => setShowAliasModal(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-3.5 py-2 text-xs font-medium text-white cursor-pointer"
          >
            <Plus className="h-3.5 w-3.5" />
            New Smart Alias
          </button>
        </div>
      </div>

      {banner && (
        <div className="rounded-xl bg-indigo-500/10 border border-indigo-500/30 px-4 py-2.5 text-xs text-indigo-200 flex items-center justify-between">
          <span>{banner}</span>
          <button
            type="button"
            onClick={() => setBanner(null)}
            className="text-slate-400 hover:text-white cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Custom Smart Router Aliases */}
      <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
        <div className="flex items-center gap-2 mb-3">
          <GitBranch className="h-4 w-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-white">
            Smart Routing Aliases ({routes.length})
          </h2>
        </div>

        {routes.length === 0 ? (
          <p className="text-xs text-slate-400">
            No custom aliases yet — create aliases like <code className="text-indigo-300">coding</code> or{' '}
            <code className="text-indigo-300">fast</code> that chain multiple seats & fallback models.
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {routes.map((r) => (
              <div
                key={r.alias}
                className="rounded-lg bg-slate-950/80 border border-white/8 p-3.5 flex flex-col justify-between gap-3"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-semibold text-indigo-300">
                      {r.alias}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="rounded bg-white/6 px-2 py-0.5 font-mono text-[10px] text-slate-300">
                        {r.pool_strategy}
                      </span>
                      <button
                        type="button"
                        onClick={() => onOpenPlaygroundModel(r.alias)}
                        className="rounded bg-indigo-500/15 hover:bg-indigo-500/30 px-2 py-0.5 text-[11px] text-indigo-300 cursor-pointer"
                      >
                        Try
                      </button>
                      <button
                        type="button"
                        onClick={async () => {
                          await api.deleteRoute(r.alias);
                          await load();
                        }}
                        className="text-rose-400 hover:text-rose-300 p-1 cursor-pointer"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {r.targets.map((t, idx) => (
                      <span
                        key={`${t.model}-${idx}`}
                        className="rounded bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-300"
                      >
                        {idx + 1}. {t.model}
                        {t.provider_id ? `@${t.provider_id}` : ''}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="text-[11px] text-slate-500">
                  Paid API overflow: {r.allow_paid_overflow ? 'Allowed' : 'Blocked ($0 only)'}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Search + Filter Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[260px]">
          <Search className="h-4 w-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by author/model, family, or slug (e.g. claude-sonnet, gpt-5, gemini-2.5, qwen)..."
            className="w-full rounded-xl bg-[#0d111a] border border-white/10 pl-10 pr-4 py-2.5 text-xs text-white focus:outline-none focus:border-indigo-400"
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(
            [
              ['ready', `Ready Now (${readyCount})`],
              ['local', 'Local ($0)'],
              ...(enabledProviders.has('openrouter')
                ? ([['presets', `@Presets (${presetCount})`]] as [FilterChip, string][])
                : []),
              ['tools', 'Tools'],
              ['reasoning', 'Reasoning'],
              ['vision', 'Vision'],
              ['all', `Full Catalog (${models.length.toLocaleString()})`],
            ] as [FilterChip, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setChip(id)}
              className={`rounded-lg px-3 py-2 text-xs font-medium transition cursor-pointer ${
                chip === id
                  ? 'bg-indigo-500 text-white'
                  : 'bg-[#0d111a] border border-white/8 text-slate-300 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Catalog Table */}
      <div className="rounded-xl bg-[#0d111a] border border-white/8 overflow-hidden">
        {filteredModels.length === 0 ? (
          <div className="py-14 px-6 text-center space-y-3">
            <p className="text-sm font-medium text-white">
              {chip === 'ready' && readyCount === 0
                ? 'No ready models yet — connect a provider seat or start a local server'
                : 'No models matched your current search or filter'}
            </p>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              {chip === 'ready' && readyCount === 0
                ? 'QuixiGate only marks models Ready Now when backed by an active OAuth seat, API key, or running local runtime (Ollama, LM Studio, llama.cpp, vLLM).'
                : 'Try adjusting your search query or switching filter chips.'}
            </p>
            {chip === 'ready' && readyCount === 0 && (
              <div className="flex items-center justify-center gap-2.5 pt-1">
                {onNavigateProviders && (
                  <button
                    type="button"
                    onClick={onNavigateProviders}
                    className="rounded-lg bg-indigo-500 hover:bg-indigo-400 px-3.5 py-2 text-xs font-semibold text-white cursor-pointer"
                  >
                    Connect Provider or OAuth Seat
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setChip('all')}
                  className="rounded-lg bg-white/8 hover:bg-white/12 border border-white/10 px-3.5 py-2 text-xs font-medium text-slate-200 cursor-pointer"
                >
                  Browse Reference Catalog ({models.length.toLocaleString()})
                </button>
              </div>
            )}
          </div>
        ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/70 text-slate-400 border-b border-white/8">
              <tr>
                <th className="py-3 px-4 font-medium">Model</th>
                <th className="py-3 px-4 font-medium">Capabilities</th>
                <th className="py-3 px-4 font-medium">Context</th>
                <th className="py-3 px-4 font-medium">Input / 1M</th>
                <th className="py-3 px-4 font-medium">Output / 1M</th>
                <th className="py-3 px-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filteredModels.map((m) => {
                const ready = isModelReady(m);
                return (
                  <tr
                    key={m.canonical_id}
                    onClick={() => setDrawerModel(m)}
                    className="hover:bg-white/[0.03] cursor-pointer"
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span
                          className={`h-2 w-2 rounded-full ${
                            ready ? 'bg-emerald-400' : 'bg-slate-600'
                          }`}
                        />
                        <span className="font-medium text-white">
                          {m.display_name}
                        </span>
                        {m.is_local && (
                          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                            LOCAL
                          </span>
                        )}
                        {m.is_preset && (
                          <span className="rounded bg-violet-500/20 border border-violet-500/30 px-1.5 py-0.5 font-mono text-[10px] text-violet-200">
                            PRESET
                          </span>
                        )}
                      </div>
                      <div className="font-mono text-[11px] text-slate-400 mt-0.5 pl-4">
                        {m.canonical_id}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1">
                        {m.supports_tools && (
                          <span className="rounded bg-white/6 px-1.5 py-0.5 text-[10px] text-slate-300">
                            tools
                          </span>
                        )}
                        {m.supports_reasoning && (
                          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">
                            reasoning
                          </span>
                        )}
                        {m.supports_vision && (
                          <span className="rounded bg-sky-500/15 px-1.5 py-0.5 text-[10px] text-sky-300">
                            vision
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300 tabular-nums">
                      {Math.round(m.context_window / 1000)}k
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300 tabular-nums">
                      {m.is_local
                        ? '$0.00'
                        : m.is_preset && m.input_nanos_per_m === 0
                          ? 'preset'
                          : formatNanos(m.input_nanos_per_m)}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300 tabular-nums">
                      {m.is_local
                        ? '$0.00'
                        : m.is_preset && m.output_nanos_per_m === 0
                          ? 'preset'
                          : formatNanos(m.output_nanos_per_m)}
                    </td>
                    <td
                      className="py-3 px-4 text-right"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        type="button"
                        onClick={() => onOpenPlaygroundModel(m.canonical_id)}
                        className="inline-flex items-center gap-1 rounded-md bg-indigo-500/15 hover:bg-indigo-500 hover:text-white px-2.5 py-1 text-[11px] font-medium text-indigo-300 transition cursor-pointer"
                      >
                        <Sparkles className="h-3 w-3" />
                        Chat
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        )}
      </div>

      {/* Model Detail & SDK Snippet Drawer */}
      {drawerModel && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-xs flex justify-end">
          <div className="w-full max-w-xl bg-[#0d111a] border-l border-white/12 h-full overflow-y-auto p-6 space-y-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-white">
                  {drawerModel.display_name}
                </h3>
                <p className="font-mono text-xs text-indigo-300 mt-0.5">
                  {drawerModel.canonical_id}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setDrawerModel(null)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-[11px] text-slate-400">Context Window</div>
                <div className="font-mono text-sm font-semibold text-white mt-0.5">
                  {drawerModel.context_window.toLocaleString()} tok
                </div>
              </div>
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-[11px] text-slate-400">Input / 1M</div>
                <div className="font-mono text-sm font-semibold text-white mt-0.5">
                  {formatNanos(drawerModel.input_nanos_per_m)}
                </div>
              </div>
              <div className="rounded-lg bg-slate-950 border border-white/8 p-3">
                <div className="text-[11px] text-slate-400">Output / 1M</div>
                <div className="font-mono text-sm font-semibold text-white mt-0.5">
                  {formatNanos(drawerModel.output_nanos_per_m)}
                </div>
              </div>
            </div>

            <div>
              <div className="text-xs font-medium text-slate-300 mb-2">
                Supported Providers & Modifiers
              </div>
              <div className="flex flex-wrap gap-1.5">
                {Object.keys(drawerModel.providers).map((p) => (
                  <span
                    key={p}
                    className="rounded bg-indigo-500/15 border border-indigo-500/25 px-2 py-0.5 font-mono text-[11px] text-indigo-300"
                  >
                    @{p}
                  </span>
                ))}
                <span className="rounded bg-emerald-500/15 px-2 py-0.5 font-mono text-[11px] text-emerald-300">
                  :online
                </span>
                <span className="rounded bg-amber-500/15 px-2 py-0.5 font-mono text-[11px] text-amber-300">
                  :free
                </span>
                <span className="rounded bg-violet-500/15 px-2 py-0.5 font-mono text-[11px] text-violet-300">
                  :nitro
                </span>
                <span className="rounded bg-sky-500/15 px-2 py-0.5 font-mono text-[11px] text-sky-300">
                  :floor
                </span>
              </div>
            </div>

            {/* SDK & CLI Snippets */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex gap-1.5">
                  {(
                    [
                      ['cli', 'Coding CLIs'],
                      ['openai', 'OpenAI SDK'],
                      ['anthropic', 'Anthropic SDK'],
                      ['aisdk', 'Vercel AI SDK'],
                    ] as const
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      onClick={() => setSnippetTab(id)}
                      className={`rounded-md px-2.5 py-1 text-xs font-medium cursor-pointer ${
                        snippetTab === id
                          ? 'bg-indigo-500 text-white'
                          : 'bg-white/6 text-slate-300'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard.writeText(
                      buildSnippet(drawerModel.canonical_id),
                    );
                    setCopiedSnippet(true);
                    setTimeout(() => setCopiedSnippet(false), 1600);
                  }}
                  className="inline-flex items-center gap-1 rounded-md bg-white/8 px-2.5 py-1 text-xs text-slate-200 cursor-pointer"
                >
                  {copiedSnippet ? (
                    <Check className="h-3.5 w-3.5 text-emerald-400" />
                  ) : (
                    <Copy className="h-3.5 w-3.5" />
                  )}
                  Copy
                </button>
              </div>
              <pre className="rounded-xl bg-slate-950 border border-white/10 p-4 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
                {buildSnippet(drawerModel.canonical_id)}
              </pre>
            </div>

            <button
              type="button"
              onClick={() => {
                onOpenPlaygroundModel(drawerModel.canonical_id);
                setDrawerModel(null);
              }}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2.5 text-sm font-medium text-white cursor-pointer"
            >
              <Sparkles className="h-4 w-4" />
              Test in Playground / Arena
            </button>
          </div>
        </div>
      )}

      {/* New Smart Alias Modal */}
      {showAliasModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">
                Create Smart Routing Alias
              </h3>
              <button
                type="button"
                onClick={() => setShowAliasModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={handleSaveRoute} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Alias Slug (used directly as <code className="text-indigo-300">model</code> in SDKs/CLIs)
                </label>
                <input
                  type="text"
                  value={aliasName}
                  onChange={(e) => setAliasName(e.target.value)}
                  placeholder="coding"
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Ordered Fallback Targets (one per line, optional <code className="text-indigo-300">@provider</code>)
                </label>
                <textarea
                  rows={4}
                  value={aliasTargetsText}
                  onChange={(e) => setAliasTargetsText(e.target.value)}
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Pool Selection Strategy
                  </label>
                  <select
                    value={aliasStrategy}
                    onChange={(e) => setAliasStrategy(e.target.value as PoolStrategy)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="fill_first">fill_first (Sticky Cache-Warm)</option>
                    <option value="least_used">least_used (Quota Balanced)</option>
                    <option value="round_robin">round_robin</option>
                    <option value="random">random</option>
                  </select>
                </div>
                <div className="flex items-end pb-2">
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={aliasOverflow}
                      onChange={(e) => setAliasOverflow(e.target.checked)}
                    />
                    Allow Paid API Overflow
                  </label>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAliasModal(false)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
                >
                  Save Route Alias
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
