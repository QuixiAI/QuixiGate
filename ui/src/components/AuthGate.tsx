import { useState } from 'react';
import { Lock, ShieldCheck } from 'lucide-react';
import { api } from '../api';
import type { AuthStatusResponse } from '../types';

export function AuthGate({
  status,
  onAuthenticated,
}: {
  status: AuthStatusResponse;
  onAuthenticated: () => void;
}) {
  const isLocalPeer = status.direct_loopback ?? status.bound_loopback;
  const needsBootstrap = !status.has_password_user && isLocalPeer;
  const [username, setUsername] = useState('local');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (needsBootstrap) {
        await api.bootstrapPassword(password, username);
      } else {
        await api.login(username, password);
      }
      onAuthenticated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#07090e] flex items-center justify-center p-6">
      <div className="w-full max-w-md rounded-2xl bg-[#0d111a] border border-white/10 p-8 shadow-2xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-11 w-11 rounded-xl bg-indigo-500/15 border border-indigo-400/30 flex items-center justify-center text-indigo-400">
            {needsBootstrap ? <ShieldCheck className="h-5 w-5" /> : <Lock className="h-5 w-5" />}
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">
              {needsBootstrap ? 'Initialize Operator Password' : 'Sign in to QuixiGate'}
            </h1>
            <p className="text-xs text-slate-400">
              {needsBootstrap
                ? 'Set an initial Argon2id password for this gateway.'
                : 'Strict HttpOnly session authentication is active.'}
            </p>
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg bg-rose-500/10 border border-rose-500/30 px-3.5 py-2.5 text-xs text-rose-300">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full rounded-lg bg-slate-950 border border-white/10 px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-400"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              {needsBootstrap ? 'Password' : 'Password or Master Key'}
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={
                needsBootstrap
                  ? 'Minimum 6 characters'
                  : 'Password or sk-qx-master-…'
              }
              required
              className="w-full rounded-lg bg-slate-950 border border-white/10 px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-400"
            />
          </div>
          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 px-4 py-2.5 text-sm font-medium text-white transition cursor-pointer"
          >
            {busy
              ? 'Authenticating…'
              : needsBootstrap
                ? 'Save Password & Continue'
                : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}
