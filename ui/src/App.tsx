import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  Flame,
  GitBranch,
  KeyRound,
  LayoutDashboard,
  Lock,
  LogOut,
  MessageSquareCode,
  Server,
  ShieldCheck,
} from 'lucide-react';
import { api, setOnUnauthorized } from './api';
import { AuthGate } from './components/AuthGate';
import { CommandCenter } from './screens/CommandCenter';
import { ModelsRouter } from './screens/ModelsRouter';
import { PlaygroundArena } from './screens/PlaygroundArena';
import { ProvidersSeats } from './screens/ProvidersSeats';
import { TracesSettings } from './screens/TracesSettings';
import { UsersTeamsKeys } from './screens/UsersTeamsKeys';
import type { AuthStatusResponse } from './types';

export type NavTab =
  | 'overview'
  | 'chat'
  | 'providers'
  | 'models'
  | 'keys'
  | 'traces'
  | 'settings';

const TAB_TO_PATH: Record<NavTab, string> = {
  overview: '/ui',
  chat: '/ui/chat',
  providers: '/ui/providers',
  models: '/ui/models',
  keys: '/ui/keys',
  traces: '/ui/traces',
  settings: '/ui/settings',
};

function pathToTab(pathname: string): NavTab {
  const withoutBase = pathname.replace(/^\/ui(?=\/|$)/, '');
  const clean = withoutBase.replace(/\/+$/, '') || '/';
  switch (clean) {
    case '/chat':
    case '/playground':
      return 'chat';
    case '/providers':
      return 'providers';
    case '/models':
      return 'models';
    case '/keys':
    case '/users':
      return 'keys';
    case '/traces':
      return 'traces';
    case '/settings':
      return 'settings';
    default:
      return 'overview';
  }
}

const NAV_ITEMS: {
  id: NavTab;
  label: string;
  subtitle: string;
  icon: typeof LayoutDashboard;
}[] = [
  {
    id: 'overview',
    label: 'Command Center',
    subtitle: 'Telemetry & savings',
    icon: LayoutDashboard,
  },
  {
    id: 'chat',
    label: 'Playground & Arena',
    subtitle: 'Stream & compare',
    icon: MessageSquareCode,
  },
  {
    id: 'providers',
    label: 'Providers & OAuth',
    subtitle: '9 OAuth + 73 APIs',
    icon: Server,
  },
  {
    id: 'models',
    label: 'Models & Router',
    subtitle: 'Catalog & aliases',
    icon: GitBranch,
  },
  {
    id: 'keys',
    label: 'Users, Teams & Keys',
    subtitle: 'sk-qx-… & budgets',
    icon: KeyRound,
  },
  {
    id: 'traces',
    label: 'Live Traces',
    subtitle: 'SSE wire inspector',
    icon: Activity,
  },
  {
    id: 'settings',
    label: 'Phoenix & Settings',
    subtitle: 'OpenInference OTLP',
    icon: Flame,
  },
];

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>(() =>
    pathToTab(window.location.pathname),
  );
  const [playgroundModel, setPlaygroundModel] = useState<string | null>(null);
  const [authStatus, setAuthStatus] = useState<AuthStatusResponse | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  const refreshAuth = useCallback(async () => {
    try {
      const status = await api.getAuthStatus();
      setAuthStatus(status);
    } catch {
      // Keep non-blocking fallback if server is briefly restarting
    } finally {
      setAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshAuth();
    setOnUnauthorized(() => {
      void refreshAuth();
    });
    return () => setOnUnauthorized(null);
  }, [refreshAuth]);

  useEffect(() => {
    function handlePopState() {
      setActiveTab(pathToTab(window.location.pathname));
    }
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  function navigate(tabInput: string) {
    const tab = (
      tabInput in TAB_TO_PATH ? tabInput : 'overview'
    ) as NavTab;
    setActiveTab(tab);
    const targetPath = TAB_TO_PATH[tab];
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, '', targetPath);
    }
  }

  function handleOpenPlaygroundModel(modelId: string) {
    setPlaygroundModel(modelId);
    navigate('chat');
  }

  async function handleLogout() {
    try {
      await api.logout();
    } finally {
      await refreshAuth();
    }
  }

  if (authLoading && !authStatus) {
    return (
      <div className="min-h-screen bg-[#07090e] flex items-center justify-center text-sm text-slate-400">
        Connecting to QuixiGate…
      </div>
    );
  }

  if (authStatus && authStatus.require_auth && !authStatus.authenticated) {
    return <AuthGate status={authStatus} onAuthenticated={() => void refreshAuth()} />;
  }

  return (
    <div className="h-screen overflow-hidden bg-[#07090e] text-slate-100 flex">
      {/* Persistent Left Sidebar */}
      <aside className="w-64 h-full shrink-0 border-r border-white/8 bg-[#0a0d14] flex flex-col justify-between overflow-y-auto">
        <div>
          {/* Brand Header */}
          <div className="px-5 py-5 border-b border-white/8 flex items-center justify-between">
            <button
              type="button"
              onClick={() => navigate('overview')}
              className="flex items-center gap-3 text-left cursor-pointer"
            >
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 via-violet-500 to-sky-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 font-mono font-bold text-sm text-white">
                QX
              </div>
              <div>
                <div className="text-sm font-semibold tracking-tight text-white flex items-center gap-1.5">
                  QuixiGate
                  <span className="rounded bg-indigo-500/20 border border-indigo-400/30 px-1.5 py-0.2 text-[10px] font-mono text-indigo-300">
                    v0.1
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 font-mono">
                  127.0.0.1:4000
                </div>
              </div>
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1" aria-label="Main Navigation">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => navigate(item.id)}
                  className={`w-full flex items-center gap-3 rounded-xl px-3 py-2.5 text-left transition cursor-pointer ${
                    isActive
                      ? 'bg-indigo-500/15 border border-indigo-500/30 text-white shadow-sm'
                      : 'border border-transparent text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 shrink-0 ${
                      isActive ? 'text-indigo-400' : 'text-slate-400'
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-medium truncate">{item.label}</div>
                    <div className="text-[11px] text-slate-500 truncate">
                      {item.subtitle}
                    </div>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Gateway Security Footer */}
        <div className="p-3.5 border-t border-white/8 space-y-2.5">
          <div className="rounded-xl bg-slate-950/80 border border-white/6 px-3 py-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-medium text-slate-200">
                {authStatus?.require_auth ? (
                  <>
                    <Lock className="h-3.5 w-3.5 text-amber-400" />
                    <span>Strict Session</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                    <span>Loopback Trust</span>
                  </>
                )}
              </div>
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
            </div>
            <div className="mt-1 text-[11px] text-slate-400 truncate font-mono">
              {authStatus?.user
                ? `${authStatus.user.username} (${authStatus.user.role})`
                : 'AES-256-GCM Vault'}
            </div>
          </div>

          {authStatus?.require_auth && authStatus.authenticated && (
            <button
              type="button"
              onClick={() => void handleLogout()}
              className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 px-3 py-2 text-xs font-medium text-slate-300 transition cursor-pointer"
            >
              <LogOut className="h-3.5 w-3.5" />
              Sign Out
            </button>
          )}
        </div>
      </aside>

      {/* Active Screen Viewport */}
      <main className="flex-1 min-w-0 h-full overflow-y-auto">
        {activeTab === 'overview' && (
          <CommandCenter
            onNavigate={navigate}
            onOpenPlaygroundModel={handleOpenPlaygroundModel}
          />
        )}
        {activeTab === 'chat' && (
          <PlaygroundArena
            initialModel={playgroundModel}
            onNavigateProviders={() => navigate('providers')}
          />
        )}
        {activeTab === 'providers' && <ProvidersSeats />}
        {activeTab === 'models' && (
          <ModelsRouter
            onOpenPlaygroundModel={handleOpenPlaygroundModel}
            onNavigateProviders={() => navigate('providers')}
          />
        )}
        {activeTab === 'keys' && <UsersTeamsKeys />}
        {activeTab === 'traces' && (
          <TracesSettings key="traces" initialSubTab="traces" />
        )}
        {activeTab === 'settings' && (
          <TracesSettings key="settings" initialSubTab="settings" />
        )}
      </main>
    </div>
  );
}
export default App;
