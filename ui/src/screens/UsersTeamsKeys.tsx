import { useEffect, useState } from 'react';
import {
  Ban,
  Check,
  Copy,
  KeyRound,
  Plus,
  RefreshCw,
  RotateCw,
  Trash2,
  UserPlus,
  Users,
  X,
} from 'lucide-react';
import { api, formatNanos, formatRelativeTime } from '../api';
import type { BudgetWindow, Team, User, UserRole, VirtualKey } from '../types';

export function UsersTeamsKeys() {
  const [keys, setKeys] = useState<VirtualKey[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(true);
  const [mintedSecret, setMintedSecret] = useState<{
    label: string;
    rawKey: string;
  } | null>(null);
  const [copiedSecret, setCopiedSecret] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Create Key Modal
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [keyName, setKeyName] = useState('');
  const [keyUserId, setKeyUserId] = useState('local');
  const [keyTeamId, setKeyTeamId] = useState('');
  const [keyBudgetUsd, setKeyBudgetUsd] = useState('25.00');
  const [keyWindow, setKeyWindow] = useState<BudgetWindow>('monthly');
  const [keyRpm, setKeyRpm] = useState('');
  const [keyTpm, setKeyTpm] = useState('');
  const [keyMaxConc, setKeyMaxConc] = useState('');
  const [keyAllowedModels, setKeyAllowedModels] = useState('');

  // Create User Modal
  const [showUserModal, setShowUserModal] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newDisplayName, setNewDisplayName] = useState('');
  const [newUserRole, setNewUserRole] = useState<UserRole>('member');
  const [newUserPassword, setNewUserPassword] = useState('');
  const [newUserBudgetUsd, setNewUserBudgetUsd] = useState('');
  const [newUserWindow, setNewUserWindow] = useState<BudgetWindow>('monthly');

  // Create Team Modal
  const [showTeamModal, setShowTeamModal] = useState(false);
  const [newTeamName, setNewTeamName] = useState('');
  const [newTeamBudgetUsd, setNewTeamBudgetUsd] = useState('');
  const [newTeamWindow, setNewTeamWindow] = useState<BudgetWindow>('monthly');
  const [selectedTeamMembers, setSelectedTeamMembers] = useState<string[]>(['local']);

  async function loadAll() {
    try {
      const [k, u, t] = await Promise.all([
        api.getKeys(),
        api.getUsers(),
        api.getTeams(),
      ]);
      setKeys(k);
      setUsers(u);
      setTeams(t);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load access control data');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadAll();
  }, []);

  async function handleMintKey(e: React.FormEvent) {
    e.preventDefault();
    try {
      const allowed = keyAllowedModels
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const res = await api.createKey({
        name: keyName.trim(),
        user_id: keyUserId,
        team_id: keyTeamId || undefined,
        budget_usd: keyBudgetUsd.trim() || undefined,
        budget_window: keyWindow,
        rpm_limit: keyRpm ? Number(keyRpm) : null,
        tpm_limit: keyTpm ? Number(keyTpm) : null,
        max_concurrent: keyMaxConc ? Number(keyMaxConc) : null,
        allowed_models: allowed,
      });
      setMintedSecret({ label: res.key.name, rawKey: res.raw_key });
      setShowKeyModal(false);
      setKeyName('');
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to mint virtual key');
    }
  }

  async function handleCreateUser(e: React.FormEvent) {
    e.preventDefault();
    try {
      const created = await api.createUser({
        username: newUsername.trim(),
        display_name: newDisplayName.trim() || newUsername.trim(),
        role: newUserRole,
        password: newUserPassword.trim() || undefined,
        budget_usd: newUserBudgetUsd.trim() || undefined,
        budget_window: newUserWindow,
      });
      setKeyUserId(created.id);
      setShowUserModal(false);
      setNewUsername('');
      setNewDisplayName('');
      setNewUserPassword('');
      setNewUserBudgetUsd('');
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create user');
    }
  }

  async function handleCreateTeam(e: React.FormEvent) {
    e.preventDefault();
    try {
      const created = await api.createTeam({
        name: newTeamName.trim(),
        budget_usd: newTeamBudgetUsd.trim() || undefined,
        budget_window: newTeamWindow,
        member_ids: selectedTeamMembers,
      });
      setKeyTeamId(created.id);
      setShowTeamModal(false);
      setNewTeamName('');
      setNewTeamBudgetUsd('');
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create team');
    }
  }

  function renderBudgetBar(spendNanos: number, limitNanos: number | null) {
    if (!limitNanos || limitNanos <= 0) {
      return (
        <span className="font-mono text-xs text-slate-300 tabular-nums">
          {formatNanos(spendNanos)} <span className="text-slate-500">/ unlimited</span>
        </span>
      );
    }
    const pct = Math.min(100, Math.round((spendNanos / limitNanos) * 100));
    const barColor =
      pct >= 95 ? 'bg-rose-500' : pct >= 75 ? 'bg-amber-500' : 'bg-emerald-500';
    return (
      <div className="space-y-1 min-w-[180px]">
        <div className="flex items-center justify-between font-mono text-[11px] tabular-nums">
          <span className="text-white">{formatNanos(spendNanos)}</span>
          <span className="text-slate-400">/ {formatNanos(limitNanos)} ({pct}%)</span>
        </div>
        <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
          <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
      </div>
    );
  }

  if (loading) {
    return <div className="p-8 text-sm text-slate-400">Loading virtual keys & budgets…</div>;
  }

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-[1440px] mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Users, Teams & <code className="text-indigo-300 text-xl">sk-qx-…</code> Keys
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            SHA-256 hashed virtual keys, zero-race pre-flight{' '}
            <code className="text-indigo-300 text-xs">NanoUsd</code> budget reservations, RPM/TPM limiters, and grace-window key rotation.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => void loadAll()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 px-3 py-2 text-xs text-slate-200 cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => setShowUserModal(true)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 px-3.5 py-2 text-xs font-medium text-slate-200 cursor-pointer"
          >
            <UserPlus className="h-3.5 w-3.5 text-sky-400" />
            New User
          </button>
          <button
            type="button"
            onClick={() => setShowTeamModal(true)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 px-3.5 py-2 text-xs font-medium text-slate-200 cursor-pointer"
          >
            <Users className="h-3.5 w-3.5 text-emerald-400" />
            New Team
          </button>
          <button
            type="button"
            onClick={() => setShowKeyModal(true)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
          >
            <Plus className="h-3.5 w-3.5" />
            Mint Virtual Key
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-4 py-3 text-xs text-rose-200 flex items-center justify-between">
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-slate-400 hover:text-white cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* One-Time Minted Key Secret Banner */}
      {mintedSecret && (
        <div className="rounded-2xl bg-emerald-500/10 border border-emerald-500/35 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
              <KeyRound className="h-4 w-4" />
              <span>
                Copy secret for "{mintedSecret.label}" now — QuixiGate only stores its SHA-256 digest!
              </span>
            </div>
            <button
              type="button"
              onClick={() => setMintedSecret(null)}
              className="text-slate-400 hover:text-white cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex items-center gap-2">
            <code className="flex-1 rounded-xl bg-slate-950 border border-emerald-500/30 px-4 py-2.5 text-xs font-mono text-emerald-200 select-all break-all">
              {mintedSecret.rawKey}
            </code>
            <button
              type="button"
              onClick={() => {
                void navigator.clipboard.writeText(mintedSecret.rawKey);
                setCopiedSecret(true);
                setTimeout(() => setCopiedSecret(false), 1800);
              }}
              className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 px-4 py-2.5 text-xs font-semibold text-slate-950 cursor-pointer"
            >
              {copiedSecret ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              {copiedSecret ? 'Copied!' : 'Copy Key'}
            </button>
          </div>
        </div>
      )}

      {/* Virtual Keys Table */}
      <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
        <h2 className="text-sm font-semibold text-white mb-4">
          Virtual Keys ({keys.length})
        </h2>
        {keys.length === 0 ? (
          <div className="py-6 text-center text-xs text-slate-400">
            No virtual keys minted yet. On localhost, SDKs work with zero keys unless{' '}
            <code className="text-indigo-300">--require-auth</code> is enabled.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-400 border-b border-white/8">
                <tr>
                  <th className="py-2.5 pr-4 font-medium">Name & Prefix</th>
                  <th className="py-2.5 pr-4 font-medium">Owner / Team</th>
                  <th className="py-2.5 pr-4 font-medium">Budget Usage</th>
                  <th className="py-2.5 pr-4 font-medium">Rate Limits</th>
                  <th className="py-2.5 pr-4 font-medium">Status</th>
                  <th className="py-2.5 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/6">
                {keys.map((vk) => {
                  const owner = users.find((u) => u.id === vk.user_id);
                  const team = teams.find((t) => t.id === vk.team_id);
                  const isRevoked = Boolean(vk.revoked_at_ms);

                  return (
                    <tr key={vk.id} className="hover:bg-white/[0.02]">
                      <td className="py-3 pr-4">
                        <div className="font-medium text-white">{vk.name}</div>
                        <div className="font-mono text-[11px] text-indigo-300">
                          {vk.key_prefix}…
                        </div>
                      </td>
                      <td className="py-3 pr-4 text-slate-300">
                        <div>{owner?.display_name ?? vk.user_id}</div>
                        {team && (
                          <div className="text-[11px] text-emerald-300">
                            Team: {team.name}
                          </div>
                        )}
                      </td>
                      <td className="py-3 pr-4">
                        {renderBudgetBar(vk.spend_nanos, vk.budget_limit_nanos)}
                        <div className="text-[10px] text-slate-500 mt-0.5">
                          Window: {vk.budget_window}
                        </div>
                      </td>
                      <td className="py-3 pr-4 font-mono text-[11px] text-slate-300">
                        <div>RPM: {vk.rpm_limit ?? '∞'}</div>
                        <div>TPM: {vk.tpm_limit ?? '∞'}</div>
                      </td>
                      <td className="py-3 pr-4">
                        {isRevoked ? (
                          <span className="rounded-full bg-rose-500/15 px-2 py-0.5 text-[11px] text-rose-300">
                            revoked
                          </span>
                        ) : (
                          <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-300">
                            active
                          </span>
                        )}
                      </td>
                      <td className="py-3 text-right">
                        <div className="inline-flex items-center gap-1.5">
                          {!isRevoked && (
                            <>
                              <button
                                type="button"
                                title="Rotate secret (5m grace window)"
                                onClick={async () => {
                                  const res = await api.rotateKey(vk.id, 300);
                                  setMintedSecret({
                                    label: `${res.key.name} (Rotated)`,
                                    rawKey: res.raw_key,
                                  });
                                  await loadAll();
                                }}
                                className="inline-flex items-center gap-1 rounded-md bg-white/6 hover:bg-white/12 px-2.5 py-1 text-[11px] text-slate-200 cursor-pointer"
                              >
                                <RotateCw className="h-3 w-3" />
                                Rotate
                              </button>
                              <button
                                type="button"
                                onClick={async () => {
                                  await api.revokeKey(vk.id);
                                  await loadAll();
                                }}
                                className="inline-flex items-center gap-1 rounded-md bg-amber-500/15 hover:bg-amber-500/25 px-2.5 py-1 text-[11px] text-amber-300 cursor-pointer"
                              >
                                <Ban className="h-3 w-3" />
                                Revoke
                              </button>
                            </>
                          )}
                          <button
                            type="button"
                            onClick={async () => {
                              await api.deleteKey(vk.id);
                              await loadAll();
                            }}
                            className="rounded-md bg-rose-500/10 hover:bg-rose-500/20 p-1.5 text-rose-300 cursor-pointer"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Users & Teams Dual Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Users */}
        <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
          <h2 className="text-sm font-semibold text-white mb-3">
            Users ({users.length})
          </h2>
          <div className="space-y-2.5">
            {users.map((u) => (
              <div
                key={u.id}
                className="rounded-lg bg-slate-950/75 border border-white/6 p-3.5 flex items-center justify-between gap-3"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-white">
                      {u.display_name}
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      @{u.username}
                    </span>
                    <span className="rounded bg-indigo-500/15 px-1.5 py-0.5 text-[10px] font-mono text-indigo-300">
                      {u.role}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">
                    Created {formatRelativeTime(u.created_at_ms)}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {renderBudgetBar(u.spend_nanos, u.budget_limit_nanos)}
                  {u.id !== 'local' && (
                    <button
                      type="button"
                      onClick={async () => {
                        await api.deleteUser(u.id);
                        await loadAll();
                      }}
                      className="text-rose-400 hover:text-rose-300 p-1 cursor-pointer"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Teams */}
        <div className="rounded-xl bg-[#0d111a] border border-white/8 p-5">
          <h2 className="text-sm font-semibold text-white mb-3">
            Teams ({teams.length})
          </h2>
          {teams.length === 0 ? (
            <p className="text-xs text-slate-400 py-4">
              Create teams to enforce shared monthly or daily budgets across multiple developer keys.
            </p>
          ) : (
            <div className="space-y-2.5">
              {teams.map((t) => (
                <div
                  key={t.id}
                  className="rounded-lg bg-slate-950/75 border border-white/6 p-3.5 flex items-center justify-between gap-3"
                >
                  <div>
                    <div className="text-xs font-semibold text-white">{t.name}</div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {t.member_ids.length} members · {t.budget_window} window
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {renderBudgetBar(t.spend_nanos, t.budget_limit_nanos)}
                    <button
                      type="button"
                      onClick={async () => {
                        await api.deleteTeam(t.id);
                        await loadAll();
                      }}
                      className="text-rose-400 hover:text-rose-300 p-1 cursor-pointer"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Mint Virtual Key Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">
                Mint Virtual Key (<code className="text-indigo-300">sk-qx-…</code>)
              </h3>
              <button
                type="button"
                onClick={() => setShowKeyModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={handleMintKey} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Key Label
                </label>
                <input
                  type="text"
                  value={keyName}
                  onChange={(e) => setKeyName(e.target.value)}
                  placeholder="ci-agent-key"
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Owner User
                  </label>
                  <select
                    value={keyUserId}
                    onChange={(e) => setKeyUserId(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    {users.map((u) => (
                      <option key={u.id} value={u.id}>
                        {u.display_name} (@{u.username})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Team (optional)
                  </label>
                  <select
                    value={keyTeamId}
                    onChange={(e) => setKeyTeamId(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="">No team</option>
                    {teams.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Hard Budget Limit (USD)
                  </label>
                  <input
                    type="text"
                    value={keyBudgetUsd}
                    onChange={(e) => setKeyBudgetUsd(e.target.value)}
                    placeholder="25.00 (leave blank for unlimited)"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Reset Window
                  </label>
                  <select
                    value={keyWindow}
                    onChange={(e) => setKeyWindow(e.target.value as BudgetWindow)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="hourly">Hourly</option>
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                    <option value="total">Total (Non-resetting)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    RPM Limit
                  </label>
                  <input
                    type="number"
                    value={keyRpm}
                    onChange={(e) => setKeyRpm(e.target.value)}
                    placeholder="Unlimited"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    TPM Limit
                  </label>
                  <input
                    type="number"
                    value={keyTpm}
                    onChange={(e) => setKeyTpm(e.target.value)}
                    placeholder="Unlimited"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Max Concurrent
                  </label>
                  <input
                    type="number"
                    value={keyMaxConc}
                    onChange={(e) => setKeyMaxConc(e.target.value)}
                    placeholder="Unlimited"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Allowed Model Globs (comma-separated, blank = all)
                </label>
                <input
                  type="text"
                  value={keyAllowedModels}
                  onChange={(e) => setKeyAllowedModels(e.target.value)}
                  placeholder="anthropic/*, coding, ollama/*"
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowKeyModal(false)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
                >
                  Mint sk-qx-… Key
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Create User Modal */}
      {showUserModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">Create User</h3>
              <button
                type="button"
                onClick={() => setShowUserModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <form onSubmit={handleCreateUser} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Username
                </label>
                <input
                  type="text"
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  placeholder="alice"
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Display Name
                </label>
                <input
                  type="text"
                  value={newDisplayName}
                  onChange={(e) => setNewDisplayName(e.target.value)}
                  placeholder="Alice Chen"
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Role
                  </label>
                  <select
                    value={newUserRole}
                    onChange={(e) => setNewUserRole(e.target.value as UserRole)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="member">Member</option>
                    <option value="admin">Admin</option>
                    <option value="viewer">Viewer</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Budget Limit (USD)
                  </label>
                  <input
                    type="text"
                    value={newUserBudgetUsd}
                    onChange={(e) => setNewUserBudgetUsd(e.target.value)}
                    placeholder="100.00"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Budget Window
                </label>
                <select
                  value={newUserWindow}
                  onChange={(e) => setNewUserWindow(e.target.value as BudgetWindow)}
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                >
                  <option value="monthly">Monthly</option>
                  <option value="weekly">Weekly</option>
                  <option value="daily">Daily</option>
                  <option value="total">Total</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Password (optional, for UI login)
                </label>
                <input
                  type="password"
                  value={newUserPassword}
                  onChange={(e) => setNewUserPassword(e.target.value)}
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowUserModal(false)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
                >
                  Create User
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Create Team Modal */}
      {showTeamModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-2xl bg-[#0d111a] border border-white/12 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">Create Team</h3>
              <button
                type="button"
                onClick={() => setShowTeamModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <form onSubmit={handleCreateTeam} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Team Name
                </label>
                <input
                  type="text"
                  value={newTeamName}
                  onChange={(e) => setNewTeamName(e.target.value)}
                  placeholder="Core Platform"
                  required
                  className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Shared Budget (USD)
                  </label>
                  <input
                    type="text"
                    value={newTeamBudgetUsd}
                    onChange={(e) => setNewTeamBudgetUsd(e.target.value)}
                    placeholder="250.00"
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Reset Window
                  </label>
                  <select
                    value={newTeamWindow}
                    onChange={(e) => setNewTeamWindow(e.target.value as BudgetWindow)}
                    className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
                  >
                    <option value="monthly">Monthly</option>
                    <option value="weekly">Weekly</option>
                    <option value="daily">Daily</option>
                    <option value="total">Total</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">
                  Team Members
                </label>
                <div className="space-y-1.5 max-h-36 overflow-y-auto rounded-lg bg-slate-950 border border-white/10 p-2.5">
                  {users.map((u) => (
                    <label
                      key={u.id}
                      className="flex items-center gap-2 text-xs text-slate-200 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedTeamMembers.includes(u.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedTeamMembers((prev) => [...prev, u.id]);
                          } else {
                            setSelectedTeamMembers((prev) =>
                              prev.filter((id) => id !== u.id),
                            );
                          }
                        }}
                      />
                      <span>
                        {u.display_name} (@{u.username})
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowTeamModal(false)}
                  className="rounded-lg bg-white/6 px-3.5 py-2 text-xs text-slate-300 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-medium text-white cursor-pointer"
                >
                  Create Team
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
