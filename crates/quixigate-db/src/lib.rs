//! Authoritative SQLite (`WAL`) runtime database, exact `NanoUsd` spend ledger,
//! `sk-qx-...` virtual key verifier, Argon2id user/session store, and live metrics.

pub mod config_sync;
pub mod limiter;
pub mod schema;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
pub use limiter::ConcurrencyGuard;
use limiter::RuntimeLimiter;
use quixigate_core::*;
use quixigate_vault::Vault;
use parking_lot::{Mutex, RwLock};
use rand::RngCore;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use uuid::Uuid;

const SETTINGS_KEY: &str = "gateway_settings";
const SESSION_TTL_MS: i64 = 30 * 24 * 60 * 60 * 1000;

#[derive(Clone)]
pub struct Database {
    inner: Arc<DatabaseInner>,
}

struct DatabaseInner {
    conn: Mutex<Connection>,
    db_path: PathBuf,
    vault: Vault,
    limiter: Arc<RuntimeLimiter>,
    settings_cache: RwLock<GatewaySettings>,
}

/// Authenticated caller identity resolved from `sk-qx-...` virtual key, master key, session, or loopback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedPrincipal {
    pub user: User,
    pub virtual_key: Option<VirtualKey>,
    pub via_session: bool,
    pub via_loopback_trust: bool,
    #[serde(default)]
    pub via_master_key: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jwt_team_id: Option<String>,
}

impl AuthenticatedPrincipal {
    pub fn effective_team_id(&self) -> Option<String> {
        self.virtual_key
            .as_ref()
            .and_then(|k| k.team_id.clone())
            .or_else(|| self.jwt_team_id.clone())
    }
}

/// RAII reservation of estimated `NanoUsd` across virtual key, user, team, model, and tag budgets.
pub struct BudgetGuard {
    db: Database,
    scopes: Vec<String>,
    durable_scopes: Vec<String>,
    effective_tags: Vec<String>,
    suppress_log: bool,
    reserved: NanoUsd,
    user_id: Option<String>,
    team_id: Option<String>,
    virtual_key_id: Option<String>,
    committed: bool,
}

impl BudgetGuard {
    pub fn effective_tags(&self) -> &[String] {
        &self.effective_tags
    }

    pub fn suppress_log(&self) -> bool {
        self.suppress_log
    }

    pub fn set_suppress_log(&mut self, suppress: bool) {
        self.suppress_log = suppress;
    }

    /// Commits exact `actual_cost` to SQLite (`users`, `teams`, `virtual_keys`, `scoped_spend`)
    /// and releases in-flight reservation.
    pub fn commit(mut self, actual_cost: NanoUsd) -> Result<()> {
        for scope in &self.scopes {
            self.db.inner.limiter.add_in_flight(scope, -self.reserved.nanos());
        }
        self.committed = true;
        if actual_cost.nanos() > 0 {
            self.db.record_spend_with_scopes(
                self.user_id.as_deref(),
                self.team_id.as_deref(),
                self.virtual_key_id.as_deref(),
                &self.durable_scopes,
                actual_cost,
            )?;
        }
        Ok(())
    }
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        if !self.committed {
            for scope in &self.scopes {
                self.db.inner.limiter.add_in_flight(scope, -self.reserved.nanos());
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSpendSummary {
    pub provider_id: String,
    pub requests: u64,
    pub tokens: u64,
    pub spend_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpendSummary {
    pub model: String,
    pub requests: u64,
    pub tokens: u64,
    pub avg_ttft_ms: Option<u64>,
    pub spend_nanos: NanoUsd,
    pub api_equiv_nanos: NanoUsd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub active_requests: u64,
    pub total_requests_24h: u64,
    pub error_requests_24h: u64,
    pub cache_hits_24h: u64,
    pub input_tokens_24h: u64,
    pub output_tokens_24h: u64,
    pub cached_tokens_24h: u64,
    pub avg_tok_per_sec_24h: f64,
    pub ttft_p50_ms: Option<u64>,
    pub ttft_p95_ms: Option<u64>,
    pub spend_24h_nanos: NanoUsd,
    pub spend_30d_nanos: NanoUsd,
    pub savings_30d_nanos: NanoUsd,
    pub by_provider: Vec<ProviderSpendSummary>,
    pub by_model: Vec<ModelSpendSummary>,
}

#[derive(Debug, Clone, Default)]
pub struct LogFilter {
    pub model: Option<String>,
    pub provider_id: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub virtual_key_id: Option<String>,
    pub tag: Option<String>,
    pub status_min: Option<u16>,
    pub since_ms: Option<i64>,
    pub until_ms: Option<i64>,
    pub limit: usize,
    pub offset: usize,
}

impl Database {
    pub fn open(db_path: &Path, vault: Vault) -> Result<Self> {
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                QuixigateError::Storage(format!("cannot create {}: {e}", parent.display()))
            })?;
        }
        let conn = Connection::open(db_path).map_err(|e| {
            QuixigateError::Storage(format!("cannot open SQLite {}: {e}", db_path.display()))
        })?;
        schema::init_schema(&conn)?;

        let settings = load_settings_from_conn(&conn)?;
        Ok(Self {
            inner: Arc::new(DatabaseInner {
                conn: Mutex::new(conn),
                db_path: db_path.to_owned(),
                vault,
                limiter: RuntimeLimiter::new(),
                settings_cache: RwLock::new(settings),
            }),
        })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| QuixigateError::Storage(format!("cannot open in-memory SQLite: {e}")))?;
        schema::init_schema(&conn)?;
        let settings = load_settings_from_conn(&conn)?;
        Ok(Self {
            inner: Arc::new(DatabaseInner {
                conn: Mutex::new(conn),
                db_path: PathBuf::from(":memory:"),
                vault: Vault::ephemeral(),
                limiter: RuntimeLimiter::new(),
                settings_cache: RwLock::new(settings),
            }),
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.inner.db_path
    }

    pub fn vault(&self) -> &Vault {
        &self.inner.vault
    }

    pub fn limiter(&self) -> &Arc<RuntimeLimiter> {
        &self.inner.limiter
    }

    // ── Settings ─────────────────────────────────────────────────────────────

    pub fn get_settings(&self) -> Result<GatewaySettings> {
        Ok(self.inner.settings_cache.read().clone())
    }

    pub fn update_settings(&self, settings: &GatewaySettings) -> Result<()> {
        let json = serde_json::to_string(settings)
            .map_err(|e| QuixigateError::Storage(format!("settings serialize failed: {e}")))?;
        let now = now_ms();
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "INSERT INTO settings (key, value_json, updated_at_ms) VALUES (?1, ?2, ?3)
                 ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at_ms = excluded.updated_at_ms",
                params![SETTINGS_KEY, json, now],
            )
            .map_err(storage_err)?;
        }
        *self.inner.settings_cache.write() = settings.clone();
        Ok(())
    }

    // ── Users & Sessions ─────────────────────────────────────────────────────

    pub fn list_users(&self) -> Result<Vec<User>> {
        let conn = self.inner.conn.lock();
        self.refresh_expired_windows_locked(&conn, now_ms())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, username, display_name, role, password_hash, budget_limit_nanos,
                        budget_window, spend_nanos, window_reset_at_ms, created_at_ms, blocked
                 FROM users ORDER BY created_at_ms ASC",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], map_user_row)
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(rows)
    }

    pub fn get_user(&self, id: &str) -> Result<User> {
        let trimmed = id.trim();
        let conn = self.inner.conn.lock();
        self.refresh_expired_windows_locked(&conn, now_ms())?;
        conn.query_row(
            "SELECT id, username, display_name, role, password_hash, budget_limit_nanos,
                    budget_window, spend_nanos, window_reset_at_ms, created_at_ms, blocked
             FROM users WHERE id = ?1 OR username = ?1",
            params![trimmed],
            map_user_row,
        )
        .optional()
        .map_err(storage_err)?
        .ok_or_else(|| QuixigateError::NotFound(format!("user '{trimmed}' not found")))
    }

    pub fn set_user_blocked(&self, id: &str, blocked: bool) -> Result<User> {
        let existing = self.get_user(id)?;
        if blocked && (existing.id == "local" || existing.username == "local") {
            return Err(QuixigateError::Validation(
                "the built-in 'local' operator user cannot be blocked".to_owned(),
            ));
        }
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE users SET blocked = ?1 WHERE id = ?2",
                params![i32::from(blocked), existing.id],
            )
            .map_err(storage_err)?;
        }
        self.get_user(&existing.id)
    }

    /// Resolves or auto-provisions an external JWT/OIDC subject as a local SQLite `User`.
    pub fn ensure_external_user(
        &self,
        subject: &str,
        display_name: Option<&str>,
        role: UserRole,
    ) -> Result<User> {
        let trimmed = subject.trim();
        if trimmed.is_empty() {
            return Err(QuixigateError::Unauthorized(
                "JWT subject claim is empty".to_owned(),
            ));
        }
        if let Ok(existing) = self.get_user(trimmed) {
            if existing.blocked {
                return Err(QuixigateError::Forbidden(format!(
                    "user '{}' is blocked",
                    existing.username
                )));
            }
            return Ok(existing);
        }
        self.create_user(
            trimmed,
            display_name.unwrap_or(trimmed),
            role,
            None,
            None,
            BudgetWindow::Monthly,
        )
    }

    pub fn create_user(
        &self,
        username: &str,
        display_name: &str,
        role: UserRole,
        password: Option<&str>,
        budget_limit: Option<NanoUsd>,
        budget_window: BudgetWindow,
    ) -> Result<User> {
        let username = username.trim();
        if username.is_empty() {
            return Err(QuixigateError::Validation("username cannot be empty".to_owned()));
        }
        let id = format!("usr_{}", Uuid::new_v4().simple());
        let now = now_ms();
        let reset_at = budget_window.next_reset_ms(now);
        let password_hash = match password.filter(|p| !p.is_empty()) {
            Some(pw) => Some(hash_password(pw)?),
            None => None,
        };
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "INSERT INTO users (id, username, display_name, role, password_hash,
                                    budget_limit_nanos, budget_window, spend_nanos,
                                    window_reset_at_ms, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, ?8, ?9)",
                params![
                    id,
                    username,
                    if display_name.trim().is_empty() {
                        username
                    } else {
                        display_name.trim()
                    },
                    role.as_str(),
                    password_hash,
                    budget_limit.map(NanoUsd::nanos),
                    budget_window.as_str(),
                    reset_at,
                    now
                ],
            )
            .map_err(|e| QuixigateError::Validation(format!("could not create user '{username}': {e}")))?;
        }
        self.get_user(&id)
    }

    pub fn update_user(
        &self,
        id: &str,
        display_name: Option<&str>,
        role: Option<UserRole>,
        password: Option<&str>,
        budget_limit: Option<Option<NanoUsd>>,
        budget_window: Option<BudgetWindow>,
    ) -> Result<User> {
        let existing = self.get_user(id)?;
        let new_name = display_name.unwrap_or(&existing.display_name).trim();
        let new_role = role.unwrap_or(existing.role);
        let new_limit = budget_limit.unwrap_or(existing.budget_limit_nanos);
        let new_window = budget_window.unwrap_or(existing.budget_window);
        let reset_at = if budget_window.is_some() {
            new_window.next_reset_ms(now_ms())
        } else {
            existing.window_reset_at_ms
        };
        let pw_hash_update = match password {
            Some(pw) if !pw.is_empty() => Some(hash_password(pw)?),
            _ => None,
        };
        {
            let conn = self.inner.conn.lock();
            if let Some(hash) = pw_hash_update {
                conn.execute(
                    "UPDATE users SET display_name = ?1, role = ?2, password_hash = ?3,
                                      budget_limit_nanos = ?4, budget_window = ?5, window_reset_at_ms = ?6
                     WHERE id = ?7",
                    params![
                        new_name,
                        new_role.as_str(),
                        hash,
                        new_limit.map(NanoUsd::nanos),
                        new_window.as_str(),
                        reset_at,
                        existing.id
                    ],
                )
                .map_err(storage_err)?;
            } else {
                conn.execute(
                    "UPDATE users SET display_name = ?1, role = ?2,
                                      budget_limit_nanos = ?3, budget_window = ?4, window_reset_at_ms = ?5
                     WHERE id = ?6",
                    params![
                        new_name,
                        new_role.as_str(),
                        new_limit.map(NanoUsd::nanos),
                        new_window.as_str(),
                        reset_at,
                        existing.id
                    ],
                )
                .map_err(storage_err)?;
            }
        }
        self.get_user(&existing.id)
    }

    pub fn delete_user(&self, id: &str) -> Result<()> {
        let existing = self.get_user(id)?;
        if existing.id == "local" || existing.username == "local" {
            return Err(QuixigateError::Validation(
                "the built-in 'local' operator user cannot be deleted".to_owned(),
            ));
        }
        let conn = self.inner.conn.lock();
        conn.execute("DELETE FROM users WHERE id = ?1", params![existing.id])
            .map_err(storage_err)?;
        Ok(())
    }

    pub fn has_any_password_user(&self) -> Result<bool> {
        let conn = self.inner.conn.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM users WHERE password_hash IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .map_err(storage_err)?;
        Ok(count > 0)
    }

    pub fn authenticate_password(&self, username: &str, password: &str) -> Result<(User, String)> {
        let (user, stored_hash) = {
            let conn = self.inner.conn.lock();
            conn.query_row(
                "SELECT id, username, display_name, role, password_hash, budget_limit_nanos,
                        budget_window, spend_nanos, window_reset_at_ms, created_at_ms, blocked
                 FROM users WHERE username = ?1",
                params![username.trim()],
                |row| {
                    let hash: Option<String> = row.get(4)?;
                    Ok((map_user_row(row)?, hash))
                },
            )
            .optional()
            .map_err(storage_err)?
            .ok_or_else(|| QuixigateError::Unauthorized("invalid username or password".to_owned()))?
        };
        if user.blocked {
            return Err(QuixigateError::Forbidden(format!(
                "user '{}' is blocked",
                user.username
            )));
        }
        let Some(hash_str) = stored_hash else {
            return Err(QuixigateError::Unauthorized(
                "user has no password configured".to_owned(),
            ));
        };
        let parsed = PasswordHash::new(&hash_str)
            .map_err(|_| QuixigateError::Internal("stored password hash invalid".to_owned()))?;
        Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .map_err(|_| QuixigateError::Unauthorized("invalid username or password".to_owned()))?;

        let token = self.create_session(&user.id)?;
        Ok((user, token))
    }

    pub fn create_session(&self, user_id: &str) -> Result<String> {
        let mut raw = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut raw);
        let token = format!(
            "{}{}",
            quixigate_core::brand::SESSION_TOKEN_PREFIX,
            URL_SAFE_NO_PAD.encode(raw)
        );
        let digest = sha256_bytes(token.as_bytes());
        let now = now_ms();
        let expires = now + SESSION_TTL_MS;
        let conn = self.inner.conn.lock();
        conn.execute(
            "INSERT INTO sessions (token_sha256, user_id, expires_at_ms, created_at_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![digest.as_slice(), user_id, expires, now],
        )
        .map_err(storage_err)?;
        Ok(token)
    }

    pub fn validate_session(&self, token: &str) -> Result<User> {
        let digest = sha256_bytes(token.as_bytes());
        let now = now_ms();
        let user_id: String = {
            let conn = self.inner.conn.lock();
            conn.query_row(
                "SELECT user_id FROM sessions WHERE token_sha256 = ?1 AND expires_at_ms > ?2",
                params![digest.as_slice(), now],
                |r| r.get(0),
            )
            .optional()
            .map_err(storage_err)?
            .ok_or_else(|| QuixigateError::Unauthorized("session expired or invalid".to_owned()))?
        };
        let user = self.get_user(&user_id)?;
        if user.blocked {
            return Err(QuixigateError::Forbidden(format!(
                "user '{}' is blocked",
                user.username
            )));
        }
        Ok(user)
    }

    pub fn revoke_session(&self, token: &str) -> Result<()> {
        let digest = sha256_bytes(token.as_bytes());
        let conn = self.inner.conn.lock();
        conn.execute(
            "DELETE FROM sessions WHERE token_sha256 = ?1",
            params![digest.as_slice()],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    // ── Teams ────────────────────────────────────────────────────────────────

    pub fn list_teams(&self) -> Result<Vec<Team>> {
        let conn = self.inner.conn.lock();
        self.refresh_expired_windows_locked(&conn, now_ms())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, budget_limit_nanos, budget_window, spend_nanos,
                        window_reset_at_ms, created_at_ms, tags_json, model_budgets_json
                 FROM teams ORDER BY created_at_ms ASC",
            )
            .map_err(storage_err)?;
        let mut teams = stmt
            .query_map([], |row| {
                let tags_json: String = row.get(7)?;
                let mb_json: String = row.get(8)?;
                Ok(Team {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    budget_limit_nanos: row.get::<_, Option<i64>>(2)?.map(NanoUsd),
                    budget_window: BudgetWindow::parse(&row.get::<_, String>(3)?),
                    spend_nanos: NanoUsd(row.get(4)?),
                    window_reset_at_ms: row.get(5)?,
                    member_ids: Vec::new(),
                    members: Vec::new(),
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    model_budgets_nanos: parse_model_budgets_json(&mb_json),
                    created_at_ms: row.get(6)?,
                })
            })
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;

        for team in &mut teams {
            let mut mstmt = conn
                .prepare(
                    "SELECT user_id, role FROM team_members WHERE team_id = ?1 ORDER BY user_id ASC",
                )
                .map_err(storage_err)?;
            let members = mstmt
                .query_map(params![team.id], |r| {
                    Ok(TeamMember {
                        user_id: r.get(0)?,
                        role: UserRole::parse(&r.get::<_, String>(1)?),
                    })
                })
                .map_err(storage_err)?
                .collect::<std::result::Result<Vec<TeamMember>, _>>()
                .map_err(storage_err)?;
            team.member_ids = members.iter().map(|m| m.user_id.clone()).collect();
            team.members = members;
        }
        Ok(teams)
    }

    pub fn create_team(
        &self,
        name: &str,
        budget_limit: Option<NanoUsd>,
        budget_window: BudgetWindow,
        member_ids: &[String],
    ) -> Result<Team> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(QuixigateError::Validation("team name cannot be empty".to_owned()));
        }
        let id = format!("team_{}", Uuid::new_v4().simple());
        let now = now_ms();
        let reset_at = budget_window.next_reset_ms(now);
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "INSERT INTO teams (id, name, budget_limit_nanos, budget_window, spend_nanos, window_reset_at_ms, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6)",
                params![
                    id,
                    trimmed,
                    budget_limit.map(NanoUsd::nanos),
                    budget_window.as_str(),
                    reset_at,
                    now
                ],
            )
            .map_err(|e| QuixigateError::Validation(format!("could not create team '{trimmed}': {e}")))?;

            for uid in member_ids {
                let _ = conn.execute(
                    "INSERT OR IGNORE INTO team_members (team_id, user_id, role) VALUES (?1, ?2, 'member')",
                    params![id, uid],
                );
            }
        }
        self.list_teams()?
            .into_iter()
            .find(|t| t.id == id)
            .ok_or(QuixigateError::NotFound(id))
    }

    pub fn update_team(
        &self,
        id: &str,
        name: Option<&str>,
        budget_limit: Option<Option<NanoUsd>>,
        budget_window: Option<BudgetWindow>,
        member_ids: Option<&[String]>,
    ) -> Result<Team> {
        let existing = self
            .list_teams()?
            .into_iter()
            .find(|t| t.id == id)
            .ok_or_else(|| QuixigateError::NotFound(format!("team '{id}' not found")))?;
        let new_name = name.unwrap_or(&existing.name).trim();
        let new_limit = budget_limit.unwrap_or(existing.budget_limit_nanos);
        let new_window = budget_window.unwrap_or(existing.budget_window);
        let reset_at = if budget_window.is_some() {
            new_window.next_reset_ms(now_ms())
        } else {
            existing.window_reset_at_ms
        };
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE teams SET name = ?1, budget_limit_nanos = ?2, budget_window = ?3, window_reset_at_ms = ?4
                 WHERE id = ?5",
                params![
                    new_name,
                    new_limit.map(NanoUsd::nanos),
                    new_window.as_str(),
                    reset_at,
                    id
                ],
            )
            .map_err(storage_err)?;
            if let Some(members) = member_ids {
                conn.execute("DELETE FROM team_members WHERE team_id = ?1", params![id])
                    .map_err(storage_err)?;
                for uid in members {
                    let _ = conn.execute(
                        "INSERT OR IGNORE INTO team_members (team_id, user_id, role) VALUES (?1, ?2, 'member')",
                        params![id, uid],
                    );
                }
            }
        }
        self.list_teams()?
            .into_iter()
            .find(|t| t.id == id)
            .ok_or_else(|| QuixigateError::NotFound(id.to_owned()))
    }

    pub fn delete_team(&self, id: &str) -> Result<()> {
        let conn = self.inner.conn.lock();
        conn.execute("DELETE FROM teams WHERE id = ?1", params![id])
            .map_err(storage_err)?;
        Ok(())
    }

    pub fn get_team(&self, id: &str) -> Result<Team> {
        self.list_teams()?
            .into_iter()
            .find(|t| t.id == id || t.name == id)
            .ok_or_else(|| QuixigateError::NotFound(format!("team '{id}' not found")))
    }

    pub fn add_team_members(&self, id: &str, user_ids: &[String]) -> Result<Team> {
        let pairs: Vec<(String, UserRole)> = user_ids
            .iter()
            .map(|u| (u.clone(), UserRole::Member))
            .collect();
        self.add_team_members_with_roles(id, &pairs)
    }

    pub fn add_team_members_with_roles(
        &self,
        id: &str,
        members: &[(String, UserRole)],
    ) -> Result<Team> {
        let existing = self.get_team(id)?;
        {
            let conn = self.inner.conn.lock();
            for (uid, role) in members {
                let trimmed = uid.trim();
                if !trimmed.is_empty() {
                    let resolved_uid = conn
                        .query_row(
                            "SELECT id FROM users WHERE id = ?1 OR username = ?1",
                            params![trimmed],
                            |r| r.get::<_, String>(0),
                        )
                        .optional()
                        .map_err(storage_err)?
                        .unwrap_or_else(|| trimmed.to_owned());
                    let _ = conn.execute(
                        "INSERT INTO team_members (team_id, user_id, role) VALUES (?1, ?2, ?3)
                         ON CONFLICT(team_id, user_id) DO UPDATE SET role = excluded.role",
                        params![existing.id, resolved_uid, role.as_str()],
                    );
                }
            }
        }
        self.get_team(&existing.id)
    }

    /// Resolves a team by `id` or `name` and ensures `user_id` is recorded in `team_members`
    /// without overwriting an existing `'admin'` team role (`INSERT OR IGNORE`).
    pub fn ensure_team_membership(
        &self,
        team_id_or_name: &str,
        user_id: &str,
    ) -> Result<Option<String>> {
        let trimmed = team_id_or_name.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let Ok(team) = self.get_team(trimmed) else {
            return Ok(None);
        };
        {
            let conn = self.inner.conn.lock();
            let _ = conn.execute(
                "INSERT OR IGNORE INTO team_members (team_id, user_id, role) VALUES (?1, ?2, 'member')",
                params![team.id, user_id],
            );
        }
        Ok(Some(team.id))
    }

    pub fn set_team_policy(
        &self,
        id: &str,
        tags: Option<&[String]>,
        model_budgets: Option<&BTreeMap<String, NanoUsd>>,
    ) -> Result<Team> {
        let existing = self.get_team(id)?;
        let next_tags = match tags {
            Some(t) => serde_json::to_string(&normalize_string_list(t)).unwrap_or_else(|_| "[]".to_owned()),
            None => serde_json::to_string(&existing.tags).unwrap_or_else(|_| "[]".to_owned()),
        };
        let next_mb = match model_budgets {
            Some(m) => serialize_model_budgets_json(m),
            None => serialize_model_budgets_json(&existing.model_budgets_nanos),
        };
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE teams SET tags_json = ?1, model_budgets_json = ?2 WHERE id = ?3",
                params![next_tags, next_mb, existing.id],
            )
            .map_err(storage_err)?;
        }
        self.get_team(&existing.id)
    }

    pub fn is_team_admin(&self, team_id: &str, user_id: &str) -> Result<bool> {
        let conn = self.inner.conn.lock();
        let role: Option<String> = conn
            .query_row(
                "SELECT role FROM team_members
                 WHERE (team_id = ?1 OR team_id IN (SELECT id FROM teams WHERE name = ?1))
                   AND user_id = ?2",
                params![team_id.trim(), user_id.trim()],
                |r| r.get(0),
            )
            .optional()
            .map_err(storage_err)?;
        Ok(role.as_deref() == Some("admin"))
    }

    pub fn is_team_member(&self, team_id: &str, user_id: &str) -> Result<bool> {
        let conn = self.inner.conn.lock();
        let present: Option<i32> = conn
            .query_row(
                "SELECT 1 FROM team_members
                 WHERE (team_id = ?1 OR team_id IN (SELECT id FROM teams WHERE name = ?1))
                   AND user_id = ?2",
                params![team_id.trim(), user_id.trim()],
                |r| r.get(0),
            )
            .optional()
            .map_err(storage_err)?;
        Ok(present.is_some())
    }

    pub fn list_admin_team_ids(&self, user_id: &str) -> Result<Vec<String>> {
        let conn = self.inner.conn.lock();
        let mut stmt = conn
            .prepare("SELECT team_id FROM team_members WHERE user_id = ?1 AND role = 'admin'")
            .map_err(storage_err)?;
        let ids = stmt
            .query_map(params![user_id.trim()], |r| r.get::<_, String>(0))
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(ids)
    }

    pub fn remove_team_member(&self, id: &str, user_id: &str) -> Result<Team> {
        let existing = self.get_team(id)?;
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "DELETE FROM team_members WHERE team_id = ?1 AND user_id = ?2",
                params![existing.id, user_id.trim()],
            )
            .map_err(storage_err)?;
        }
        self.get_team(&existing.id)
    }

    // ── Virtual Keys (`sk-qx-...`) ───────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn create_virtual_key(
        &self,
        name: &str,
        user_id: &str,
        team_id: Option<&str>,
        budget_limit: Option<NanoUsd>,
        budget_window: BudgetWindow,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u64>,
        max_concurrent: Option<u32>,
        allowed_models: &[String],
        log_payloads: Option<bool>,
        expires_at_ms: Option<i64>,
    ) -> Result<(VirtualKey, String)> {
        self.create_virtual_key_with_token(
            name,
            user_id,
            team_id,
            budget_limit,
            budget_window,
            rpm_limit,
            tpm_limit,
            max_concurrent,
            allowed_models,
            log_payloads,
            expires_at_ms,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_virtual_key_with_token(
        &self,
        name: &str,
        user_id: &str,
        team_id: Option<&str>,
        budget_limit: Option<NanoUsd>,
        budget_window: BudgetWindow,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u64>,
        max_concurrent: Option<u32>,
        allowed_models: &[String],
        log_payloads: Option<bool>,
        expires_at_ms: Option<i64>,
        custom_raw_token: Option<&str>,
    ) -> Result<(VirtualKey, String)> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(QuixigateError::Validation("key name cannot be empty".to_owned()));
        }
        // Confirm user exists and resolve canonical user.id.
        let owner = self.get_user(user_id)?;
        let id = format!("key_{}", Uuid::new_v4().simple());
        let (raw_key, prefix, digest) = match custom_raw_token.map(str::trim).filter(|s| !s.is_empty()) {
            Some(custom) => {
                if custom.len() < 16 || !custom.chars().all(|c| c.is_ascii_graphic()) {
                    return Err(QuixigateError::Validation(
                        "custom virtual key token must be at least 16 printable ASCII characters"
                            .to_owned(),
                    ));
                }
                let pfx = format!("{}…{}", &custom[..12.min(custom.len())], &custom[custom.len() - 4..]);
                let dig = sha256_bytes(custom.as_bytes());
                (custom.to_owned(), pfx, dig)
            }
            None => generate_virtual_key_material(),
        };
        let now = now_ms();
        let reset_at = budget_window.next_reset_ms(now);
        let models_json = serde_json::to_string(allowed_models).unwrap_or_else(|_| "[]".to_owned());

        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "INSERT INTO virtual_keys (
                    id, key_prefix, key_sha256, name, user_id, team_id,
                    budget_limit_nanos, budget_window, spend_nanos, window_reset_at_ms,
                    rpm_limit, tpm_limit, max_concurrent, allowed_models_json,
                    log_payloads, expires_at_ms, revoked_at_ms, created_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?10, ?11, ?12, ?13, ?14, ?15, NULL, ?16)",
                params![
                    id,
                    prefix,
                    digest.as_slice(),
                    trimmed,
                    owner.id,
                    team_id,
                    budget_limit.map(NanoUsd::nanos),
                    budget_window.as_str(),
                    reset_at,
                    rpm_limit,
                    tpm_limit.map(|v| v as i64),
                    max_concurrent,
                    models_json,
                    log_payloads.map(i32::from),
                    expires_at_ms,
                    now
                ],
            )
            .map_err(storage_err)?;
        }
        let vk = self.get_virtual_key(&id)?;
        Ok((vk, raw_key))
    }

    pub fn set_virtual_key_policy(
        &self,
        id_or_token: &str,
        tags: Option<&[String]>,
        model_budgets: Option<&BTreeMap<String, NanoUsd>>,
        allowed_cidrs: Option<&[String]>,
        allow_spend_read: Option<bool>,
    ) -> Result<VirtualKey> {
        let id = self.resolve_virtual_key_id(id_or_token)?;
        let existing = self.get_virtual_key(&id)?;
        let next_tags = match tags {
            Some(t) => serde_json::to_string(&normalize_string_list(t)).unwrap_or_else(|_| "[]".to_owned()),
            None => serde_json::to_string(&existing.tags).unwrap_or_else(|_| "[]".to_owned()),
        };
        let next_mb = match model_budgets {
            Some(m) => serialize_model_budgets_json(m),
            None => serialize_model_budgets_json(&existing.model_budgets_nanos),
        };
        let next_cidrs = match allowed_cidrs {
            Some(c) => serde_json::to_string(&normalize_string_list(c)).unwrap_or_else(|_| "[]".to_owned()),
            None => serde_json::to_string(&existing.allowed_cidrs).unwrap_or_else(|_| "[]".to_owned()),
        };
        let next_spend_read = allow_spend_read.unwrap_or(existing.allow_spend_read);
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE virtual_keys
                 SET tags_json = ?1, model_budgets_json = ?2, allowed_cidrs_json = ?3, allow_spend_read = ?4
                 WHERE id = ?5",
                params![
                    next_tags,
                    next_mb,
                    next_cidrs,
                    i32::from(next_spend_read),
                    id
                ],
            )
            .map_err(storage_err)?;
        }
        self.get_virtual_key(&id)
    }

    pub fn list_virtual_keys(&self) -> Result<Vec<VirtualKey>> {
        let conn = self.inner.conn.lock();
        self.refresh_expired_windows_locked(&conn, now_ms())?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {VIRTUAL_KEY_COLUMNS} FROM virtual_keys ORDER BY created_at_ms DESC"
            ))
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], map_virtual_key_row)
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(rows)
    }

    pub fn get_virtual_key(&self, id: &str) -> Result<VirtualKey> {
        let conn = self.inner.conn.lock();
        self.refresh_expired_windows_locked(&conn, now_ms())?;
        conn.query_row(
            &format!("SELECT {VIRTUAL_KEY_COLUMNS} FROM virtual_keys WHERE id = ?1"),
            params![id],
            map_virtual_key_row,
        )
        .optional()
        .map_err(storage_err)?
        .ok_or_else(|| QuixigateError::NotFound(format!("virtual key '{id}' not found")))
    }

    /// Rotates a virtual key's secret, keeping the previous secret valid for `grace_seconds`
    /// so active agents and servers can switch keys with zero downtime.
    pub fn rotate_virtual_key(&self, id: &str, grace_seconds: u64) -> Result<(VirtualKey, String)> {
        let resolved = self.resolve_virtual_key_id(id)?;
        let (new_raw, new_prefix, new_digest) = generate_virtual_key_material();
        let now = now_ms();
        {
            let conn = self.inner.conn.lock();
            let old_digest: Vec<u8> = conn
                .query_row(
                    "SELECT key_sha256 FROM virtual_keys WHERE id = ?1 AND revoked_at_ms IS NULL",
                    params![resolved],
                    |r| r.get(0),
                )
                .optional()
                .map_err(storage_err)?
                .ok_or_else(|| {
                    QuixigateError::NotFound(format!("active virtual key '{id}' not found"))
                })?;

            if grace_seconds > 0 {
                let valid_until = now + (grace_seconds as i64) * 1000;
                conn.execute(
                    "INSERT OR REPLACE INTO virtual_key_rotations (old_key_sha256, virtual_key_id, valid_until_ms)
                     VALUES (?1, ?2, ?3)",
                    params![old_digest, resolved, valid_until],
                )
                .map_err(storage_err)?;
            }

            conn.execute(
                "UPDATE virtual_keys SET key_prefix = ?1, key_sha256 = ?2 WHERE id = ?3",
                params![new_prefix, new_digest.as_slice(), resolved],
            )
            .map_err(storage_err)?;
        }
        Ok((self.get_virtual_key(&resolved)?, new_raw))
    }

    pub fn revoke_virtual_key(&self, id: &str) -> Result<VirtualKey> {
        let resolved = self.resolve_virtual_key_id(id)?;
        let now = now_ms();
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE virtual_keys SET revoked_at_ms = ?1 WHERE id = ?2",
                params![now, resolved],
            )
            .map_err(storage_err)?;
            conn.execute(
                "DELETE FROM virtual_key_rotations WHERE virtual_key_id = ?1",
                params![resolved],
            )
            .map_err(storage_err)?;
        }
        self.get_virtual_key(&resolved)
    }

    pub fn unrevoke_virtual_key(&self, id: &str) -> Result<VirtualKey> {
        let resolved = self.resolve_virtual_key_id(id)?;
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE virtual_keys SET revoked_at_ms = NULL WHERE id = ?1",
                params![resolved],
            )
            .map_err(storage_err)?;
        }
        self.get_virtual_key(&resolved)
    }

    /// Resolves a virtual key ID (`key_...`), raw token (`sk-qx-...`), or unique alias name to its canonical `key_...` ID.
    pub fn resolve_virtual_key_id(&self, id_or_token: &str) -> Result<String> {
        let trimmed = id_or_token.trim();
        if trimmed.is_empty() {
            return Err(QuixigateError::Validation("key identifier cannot be empty".to_owned()));
        }
        let conn = self.inner.conn.lock();
        if let Some(id) = conn
            .query_row(
                "SELECT id FROM virtual_keys WHERE id = ?1",
                params![trimmed],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(storage_err)?
        {
            return Ok(id);
        }
        if quixigate_core::brand::is_virtual_key_token(trimmed) {
            let digest = sha256_bytes(trimmed.as_bytes());
            if let Some(id) = conn
                .query_row(
                    "SELECT id FROM virtual_keys WHERE key_sha256 = ?1",
                    params![digest.as_slice()],
                    |r| r.get::<_, String>(0),
                )
                .optional()
                .map_err(storage_err)?
            {
                return Ok(id);
            }
        }
        if let Some(id) = conn
            .query_row(
                "SELECT id FROM virtual_keys WHERE name = ?1 ORDER BY created_at_ms DESC LIMIT 1",
                params![trimmed],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(storage_err)?
        {
            return Ok(id);
        }
        Err(QuixigateError::NotFound(format!(
            "virtual key '{trimmed}' not found"
        )))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_virtual_key(
        &self,
        id_or_token: &str,
        name: Option<&str>,
        user_id: Option<&str>,
        team_id: Option<Option<&str>>,
        budget_limit: Option<Option<NanoUsd>>,
        budget_window: Option<BudgetWindow>,
        rpm_limit: Option<Option<u32>>,
        tpm_limit: Option<Option<u64>>,
        max_concurrent: Option<Option<u32>>,
        allowed_models: Option<&[String]>,
        log_payloads: Option<Option<bool>>,
        expires_at_ms: Option<Option<i64>>,
    ) -> Result<VirtualKey> {
        let id = self.resolve_virtual_key_id(id_or_token)?;
        let existing = self.get_virtual_key(&id)?;
        let next_name = name.unwrap_or(&existing.name).trim();
        if next_name.is_empty() {
            return Err(QuixigateError::Validation("key name cannot be empty".to_owned()));
        }
        let next_user_id = match user_id.map(str::trim).filter(|s| !s.is_empty()) {
            Some(uid) => {
                let _ = self.get_user(uid)?;
                uid.to_owned()
            }
            None => existing.user_id,
        };
        let next_team_id = match team_id {
            Some(opt) => opt.map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned),
            None => existing.team_id,
        };
        let next_budget = budget_limit.unwrap_or(existing.budget_limit_nanos);
        let next_window = budget_window.unwrap_or(existing.budget_window);
        let next_reset = if budget_window.is_some() {
            next_window.next_reset_ms(now_ms())
        } else {
            existing.window_reset_at_ms
        };
        let next_rpm = rpm_limit.unwrap_or(existing.rpm_limit);
        let next_tpm = tpm_limit.unwrap_or(existing.tpm_limit);
        let next_conc = max_concurrent.unwrap_or(existing.max_concurrent);
        let next_models = match allowed_models {
            Some(m) => serde_json::to_string(m).unwrap_or_else(|_| "[]".to_owned()),
            None => serde_json::to_string(&existing.allowed_models).unwrap_or_else(|_| "[]".to_owned()),
        };
        let next_log = log_payloads.unwrap_or(existing.log_payloads);
        let next_exp = expires_at_ms.unwrap_or(existing.expires_at_ms);

        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE virtual_keys
                 SET name = ?1, user_id = ?2, team_id = ?3, budget_limit_nanos = ?4,
                     budget_window = ?5, window_reset_at_ms = ?6, rpm_limit = ?7,
                     tpm_limit = ?8, max_concurrent = ?9, allowed_models_json = ?10,
                     log_payloads = ?11, expires_at_ms = ?12
                 WHERE id = ?13",
                params![
                    next_name,
                    next_user_id,
                    next_team_id,
                    next_budget.map(NanoUsd::nanos),
                    next_window.as_str(),
                    next_reset,
                    next_rpm,
                    next_tpm.map(|v| v as i64),
                    next_conc,
                    next_models,
                    next_log.map(i32::from),
                    next_exp,
                    id,
                ],
            )
            .map_err(storage_err)?;
        }
        self.get_virtual_key(&id)
    }

    pub fn delete_virtual_key(&self, id: &str) -> Result<()> {
        let target = self
            .resolve_virtual_key_id(id)
            .unwrap_or_else(|_| id.trim().to_owned());
        let conn = self.inner.conn.lock();
        conn.execute("DELETE FROM virtual_keys WHERE id = ?1", params![target])
            .map_err(storage_err)?;
        conn.execute(
            "DELETE FROM virtual_key_rotations WHERE virtual_key_id = ?1",
            params![target],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    /// Authenticates a `sk-qx-...` bearer token using constant-time SHA-256 verification.
    pub fn verify_virtual_key(&self, raw_token: &str) -> Result<AuthenticatedPrincipal> {
        let trimmed = raw_token.trim();
        if !quixigate_core::brand::is_virtual_key_token(trimmed) {
            return Err(QuixigateError::Unauthorized(
                "invalid virtual key prefix (expected sk-qx-...)".to_owned(),
            ));
        }
        let candidate_digest = sha256_bytes(trimmed.as_bytes());
        let now = now_ms();

        let vk = {
            let conn = self.inner.conn.lock();
            self.refresh_expired_windows_locked(&conn, now)?;

            let primary: Option<(VirtualKey, Vec<u8>)> = conn
                .query_row(
                    &format!(
                        "SELECT {VIRTUAL_KEY_COLUMNS}, key_sha256 FROM virtual_keys WHERE key_sha256 = ?1"
                    ),
                    params![candidate_digest.as_slice()],
                    |row| Ok((map_virtual_key_row(row)?, row.get::<_, Vec<u8>>(21)?)),
                )
                .optional()
                .map_err(storage_err)?;

            if let Some((vk, stored_digest)) = primary {
                if stored_digest.ct_eq(&candidate_digest).unwrap_u8() != 1 {
                    return Err(QuixigateError::Unauthorized("invalid virtual key".to_owned()));
                }
                vk
            } else {
                // Check rotation grace table.
                let rotated_id: Option<String> = conn
                    .query_row(
                        "SELECT virtual_key_id FROM virtual_key_rotations
                         WHERE old_key_sha256 = ?1 AND valid_until_ms > ?2",
                        params![candidate_digest.as_slice(), now],
                        |r| r.get(0),
                    )
                    .optional()
                    .map_err(storage_err)?;
                let Some(vk_id) = rotated_id else {
                    return Err(QuixigateError::Unauthorized("invalid virtual key".to_owned()));
                };
                conn.query_row(
                    &format!("SELECT {VIRTUAL_KEY_COLUMNS} FROM virtual_keys WHERE id = ?1"),
                    params![vk_id],
                    map_virtual_key_row,
                )
                .map_err(storage_err)?
            }
        };

        if vk.revoked_at_ms.is_some() {
            return Err(QuixigateError::Unauthorized(
                "virtual key has been revoked".to_owned(),
            ));
        }
        if let Some(exp) = vk.expires_at_ms
            && exp <= now
        {
            return Err(QuixigateError::Unauthorized(
                "virtual key has expired".to_owned(),
            ));
        }

        let user = self.get_user(&vk.user_id)?;
        if user.blocked {
            return Err(QuixigateError::Forbidden(format!(
                "user '{}' is blocked",
                user.username
            )));
        }
        Ok(AuthenticatedPrincipal {
            user,
            virtual_key: Some(vk),
            via_session: false,
            via_loopback_trust: false,
            via_master_key: false,
            jwt_team_id: None,
        })
    }

    // ── Budget Reservation & Rate Limit Gate ─────────────────────────────────

    /// Enforces model allowlists, team-scoped routes, RPM/TPM/concurrency limits, and reserves
    /// `estimated_paid_nanos` across virtual key, user, team, per-model, and tag budgets atomically.
    pub fn authorize_and_reserve(
        &self,
        principal: &AuthenticatedPrincipal,
        requested_model: &str,
        estimated_tokens: u64,
        estimated_paid_nanos: NanoUsd,
    ) -> Result<(ConcurrencyGuard, BudgetGuard)> {
        self.authorize_and_reserve_with_tags(
            principal,
            requested_model,
            estimated_tokens,
            estimated_paid_nanos,
            &[],
        )
    }

    pub fn authorize_and_reserve_with_tags(
        &self,
        principal: &AuthenticatedPrincipal,
        requested_model: &str,
        estimated_tokens: u64,
        estimated_paid_nanos: NanoUsd,
        request_tags: &[String],
    ) -> Result<(ConcurrencyGuard, BudgetGuard)> {
        if principal.user.blocked {
            return Err(QuixigateError::Forbidden(format!(
                "user '{}' is blocked",
                principal.user.username
            )));
        }

        let base_model = ModelSpec::parse(requested_model).base_model;

        // Enforce team-scoped custom model routes (`model_routes.team_id`).
        if let Some(route) = self.get_model_route(&base_model)?
            && let Some(route_tid) = route.team_id.as_deref().filter(|s| !s.is_empty())
            && !principal.via_master_key
            && principal.user.role != UserRole::Admin
        {
            let eff_tid = principal.effective_team_id();
            let eff_matches_team = eff_tid.as_deref() == Some(route_tid);
            let user_in_team = self.is_team_member(route_tid, &principal.user.id)?;
            if !(eff_matches_team || (principal.virtual_key.is_none() && user_in_team)) {
                return Err(QuixigateError::Forbidden(format!(
                    "model '{requested_model}' is scoped to team '{route_tid}'"
                )));
            }
        }

        let concurrency_guard = if let Some(vk) = &principal.virtual_key {
            if !vk.allowed_models.is_empty()
                && !vk
                    .allowed_models
                    .iter()
                    .any(|pat| glob_matches(pat, requested_model) || glob_matches(pat, &base_model))
            {
                return Err(QuixigateError::Forbidden(format!(
                    "virtual key '{}' is not permitted to access model '{requested_model}'",
                    vk.name
                )));
            }
            let scope = format!("key:{}", vk.id);
            self.inner.limiter.check_and_acquire(
                &scope,
                vk.rpm_limit,
                vk.tpm_limit,
                vk.max_concurrent,
                estimated_tokens,
            )?
        } else {
            self.inner
                .limiter
                .check_and_acquire("global:loopback", None, None, None, estimated_tokens)?
        };

        let mut scopes = Vec::new();
        let mut durable_scopes = Vec::new();
        let mut effective_tags: Vec<String> = request_tags
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .fold(Vec::new(), |mut acc, s| {
                if !acc.iter().any(|existing| existing == s) {
                    acc.push(s.to_owned());
                }
                acc
            });
        if estimated_paid_nanos.nanos() >= 0 {
            let settings = self.inner.settings_cache.read().clone();
            let conn = self.inner.conn.lock();
            self.refresh_expired_windows_locked(&conn, now_ms())?;

            if let Some(vk) = &principal.virtual_key {
                for t in &vk.tags {
                    if !effective_tags.contains(t) {
                        effective_tags.push(t.clone());
                    }
                }
                let (spend, limit): (i64, Option<i64>) = conn
                    .query_row(
                        "SELECT spend_nanos, budget_limit_nanos FROM virtual_keys WHERE id = ?1",
                        params![vk.id],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .map_err(storage_err)?;
                let key_scope = format!("budget:key:{}", vk.id);
                self.inner.limiter.check_scope_budget(
                    &format!("virtual key '{}'", vk.name),
                    &key_scope,
                    NanoUsd(spend),
                    limit.map(NanoUsd),
                    estimated_paid_nanos,
                )?;
                scopes.push(key_scope);

                // Per-model max budget on Virtual Key (`model_max_budget`)
                for (pat, model_limit) in &vk.model_budgets_nanos {
                    if glob_matches(pat, requested_model) || glob_matches(pat, &base_model) {
                        let db_scope = format!("key_model:{}:{pat}", vk.id);
                        let lim_scope = format!("budget:{db_scope}");
                        let cur_spend = query_scoped_spend_locked(&conn, &db_scope)?;
                        self.inner.limiter.check_scope_budget(
                            &format!("virtual key '{}' model '{pat}'", vk.name),
                            &lim_scope,
                            NanoUsd(cur_spend),
                            Some(*model_limit),
                            estimated_paid_nanos,
                        )?;
                        scopes.push(lim_scope);
                        durable_scopes.push(db_scope);
                    }
                }
            }

            let (u_spend, u_limit): (i64, Option<i64>) = conn
                .query_row(
                    "SELECT spend_nanos, budget_limit_nanos FROM users WHERE id = ?1",
                    params![principal.user.id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .map_err(storage_err)?;
            let user_scope = format!("budget:user:{}", principal.user.id);
            self.inner.limiter.check_scope_budget(
                &format!("user '{}'", principal.user.username),
                &user_scope,
                NanoUsd(u_spend),
                u_limit.map(NanoUsd),
                estimated_paid_nanos,
            )?;
            scopes.push(user_scope);

            let team_id = principal.effective_team_id();
            if let Some(tid) = &team_id
                && let Some((t_name, t_spend, t_limit, t_tags_json, t_mb_json)) = conn
                    .query_row(
                        "SELECT name, spend_nanos, budget_limit_nanos, tags_json, model_budgets_json
                         FROM teams WHERE id = ?1",
                        params![tid],
                        |r| {
                            Ok((
                                r.get::<_, String>(0)?,
                                r.get::<_, i64>(1)?,
                                r.get::<_, Option<i64>>(2)?,
                                r.get::<_, String>(3)?,
                                r.get::<_, String>(4)?,
                            ))
                        },
                    )
                    .optional()
                    .map_err(storage_err)?
            {
                let team_scope = format!("budget:team:{tid}");
                self.inner.limiter.check_scope_budget(
                    &format!("team '{t_name}'"),
                    &team_scope,
                    NanoUsd(t_spend),
                    t_limit.map(NanoUsd),
                    estimated_paid_nanos,
                )?;
                scopes.push(team_scope);

                let t_tags: Vec<String> = serde_json::from_str(&t_tags_json).unwrap_or_default();
                for t in t_tags {
                    if !effective_tags.contains(&t) {
                        effective_tags.push(t);
                    }
                }
                let t_mb = parse_model_budgets_json(&t_mb_json);
                for (pat, model_limit) in &t_mb {
                    if glob_matches(pat, requested_model) || glob_matches(pat, &base_model) {
                        let db_scope = format!("team_model:{tid}:{pat}");
                        let lim_scope = format!("budget:{db_scope}");
                        let cur_spend = query_scoped_spend_locked(&conn, &db_scope)?;
                        self.inner.limiter.check_scope_budget(
                            &format!("team '{t_name}' model '{pat}'"),
                            &lim_scope,
                            NanoUsd(cur_spend),
                            Some(*model_limit),
                            estimated_paid_nanos,
                        )?;
                        scopes.push(lim_scope);
                        durable_scopes.push(db_scope);
                    }
                }
            }

            // Global tag-based max budgets (`tag_budgets_nanos`)
            for tag in &effective_tags {
                if let Some(tag_limit) = settings.tag_budgets_nanos.get(tag) {
                    let db_scope = format!("tag:{tag}");
                    let lim_scope = format!("budget:{db_scope}");
                    let cur_spend = query_scoped_spend_locked(&conn, &db_scope)?;
                    self.inner.limiter.check_scope_budget(
                        &format!("tag '{tag}'"),
                        &lim_scope,
                        NanoUsd(cur_spend),
                        Some(*tag_limit),
                        estimated_paid_nanos,
                    )?;
                    scopes.push(lim_scope);
                    durable_scopes.push(db_scope);
                }
            }

            for scope in &scopes {
                self.inner
                    .limiter
                    .add_in_flight(scope, estimated_paid_nanos.nanos());
            }
        }

        let budget_guard = BudgetGuard {
            db: self.clone(),
            scopes,
            durable_scopes,
            effective_tags,
            suppress_log: false,
            reserved: estimated_paid_nanos,
            user_id: Some(principal.user.id.clone()),
            team_id: principal.effective_team_id(),
            virtual_key_id: principal.virtual_key.as_ref().map(|k| k.id.clone()),
            committed: false,
        };
        Ok((concurrency_guard, budget_guard))
    }

    pub fn record_spend(
        &self,
        user_id: Option<&str>,
        team_id: Option<&str>,
        virtual_key_id: Option<&str>,
        cost: NanoUsd,
    ) -> Result<()> {
        self.record_spend_with_scopes(user_id, team_id, virtual_key_id, &[], cost)
    }

    pub fn record_spend_with_scopes(
        &self,
        user_id: Option<&str>,
        team_id: Option<&str>,
        virtual_key_id: Option<&str>,
        durable_scopes: &[String],
        cost: NanoUsd,
    ) -> Result<()> {
        if cost.nanos() <= 0 {
            return Ok(());
        }
        let nanos = cost.nanos();
        let now = now_ms();
        let conn = self.inner.conn.lock();
        if let Some(uid) = user_id {
            conn.execute(
                "UPDATE users SET spend_nanos = spend_nanos + ?1 WHERE id = ?2",
                params![nanos, uid],
            )
            .map_err(storage_err)?;
        }
        if let Some(tid) = team_id {
            conn.execute(
                "UPDATE teams SET spend_nanos = spend_nanos + ?1 WHERE id = ?2",
                params![nanos, tid],
            )
            .map_err(storage_err)?;
        }
        if let Some(kid) = virtual_key_id {
            conn.execute(
                "UPDATE virtual_keys SET spend_nanos = spend_nanos + ?1 WHERE id = ?2",
                params![nanos, kid],
            )
            .map_err(storage_err)?;
        }
        for scope in durable_scopes {
            conn.execute(
                "INSERT INTO scoped_spend (scope_key, spend_nanos, updated_at_ms)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(scope_key) DO UPDATE SET
                    spend_nanos = spend_nanos + excluded.spend_nanos,
                    updated_at_ms = excluded.updated_at_ms",
                params![scope, nanos, now],
            )
            .map_err(storage_err)?;
        }
        Ok(())
    }

    pub fn reset_spend(
        &self,
        user_id: Option<&str>,
        team_id: Option<&str>,
        virtual_key_id: Option<&str>,
        reset_all: bool,
    ) -> Result<()> {
        let resolved_key = match virtual_key_id {
            Some(k) => Some(self.resolve_virtual_key_id(k)?),
            None => None,
        };
        let conn = self.inner.conn.lock();
        if reset_all {
            conn.execute("UPDATE users SET spend_nanos = 0", [])
                .map_err(storage_err)?;
            conn.execute("UPDATE teams SET spend_nanos = 0", [])
                .map_err(storage_err)?;
            conn.execute("UPDATE virtual_keys SET spend_nanos = 0", [])
                .map_err(storage_err)?;
            conn.execute("DELETE FROM scoped_spend", [])
                .map_err(storage_err)?;
            return Ok(());
        }
        if let Some(uid) = user_id {
            conn.execute(
                "UPDATE users SET spend_nanos = 0 WHERE id = ?1 OR username = ?1",
                params![uid.trim()],
            )
            .map_err(storage_err)?;
        }
        if let Some(tid) = team_id {
            let trimmed = tid.trim();
            conn.execute(
                "UPDATE teams SET spend_nanos = 0 WHERE id = ?1 OR name = ?1",
                params![trimmed],
            )
            .map_err(storage_err)?;
            conn.execute(
                "DELETE FROM scoped_spend WHERE scope_key LIKE ?1",
                params![format!("team_model:{trimmed}:%")],
            )
            .map_err(storage_err)?;
        }
        if let Some(kid) = resolved_key {
            conn.execute(
                "UPDATE virtual_keys SET spend_nanos = 0 WHERE id = ?1",
                params![kid],
            )
            .map_err(storage_err)?;
            conn.execute(
                "DELETE FROM scoped_spend WHERE scope_key LIKE ?1",
                params![format!("key_model:{kid}:%")],
            )
            .map_err(storage_err)?;
        }
        Ok(())
    }

    fn refresh_expired_windows_locked(&self, conn: &Connection, now: i64) -> Result<()> {
        for table in ["users", "teams", "virtual_keys"] {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT id, budget_window FROM {table}
                     WHERE window_reset_at_ms IS NOT NULL AND window_reset_at_ms <= ?1"
                ))
                .map_err(storage_err)?;
            let expired: Vec<(String, String)> = stmt
                .query_map(params![now], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(storage_err)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(storage_err)?;
            drop(stmt);
            for (id, win_str) in expired {
                let next_reset = BudgetWindow::parse(&win_str).next_reset_ms(now);
                conn.execute(
                    &format!(
                        "UPDATE {table} SET spend_nanos = 0, window_reset_at_ms = ?1 WHERE id = ?2"
                    ),
                    params![next_reset, id],
                )
                .map_err(storage_err)?;
            }
        }
        Ok(())
    }

    // ── Credentials & Multi-Seat Pool State ──────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_credential(
        &self,
        id: Option<&str>,
        provider_id: &str,
        label: &str,
        auth_kind: AuthKind,
        base_url: Option<&str>,
        priority: i32,
        enabled: bool,
        allow_api_overflow: bool,
        secret: &CredentialSecret,
    ) -> Result<CredentialRecord> {
        let cred_id = id
            .filter(|s| !s.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("cred_{}", Uuid::new_v4().simple()));
        let encrypted = self.inner.vault.encrypt_json(&cred_id, secret)?;
        let expires_at = secret.expires_at_ms();
        let now = now_ms();
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "INSERT INTO credentials (
                    id, provider_id, label, auth_kind, base_url, priority,
                    enabled, allow_api_overflow, encrypted_payload, status,
                    status_reason, cooldown_until_ms, model_cooldowns_json,
                    expires_at_ms, created_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'ok', NULL, NULL, '{}', ?10, ?11)
                 ON CONFLICT(id) DO UPDATE SET
                    provider_id = excluded.provider_id,
                    label = excluded.label,
                    auth_kind = excluded.auth_kind,
                    base_url = excluded.base_url,
                    priority = excluded.priority,
                    enabled = excluded.enabled,
                    allow_api_overflow = excluded.allow_api_overflow,
                    encrypted_payload = excluded.encrypted_payload,
                    status = 'ok',
                    status_reason = NULL,
                    cooldown_until_ms = NULL,
                    expires_at_ms = excluded.expires_at_ms",
                params![
                    cred_id,
                    provider_id.trim(),
                    label.trim(),
                    auth_kind.as_str(),
                    base_url.filter(|u| !u.trim().is_empty()),
                    priority,
                    i32::from(enabled),
                    i32::from(allow_api_overflow),
                    encrypted,
                    expires_at,
                    now
                ],
            )
            .map_err(storage_err)?;
        }
        self.get_credential(&cred_id)
    }

    /// Atomically persists a rotated OAuth token pair before returning to callers.
    pub fn update_credential_secret(
        &self,
        id: &str,
        secret: &CredentialSecret,
        new_base_url: Option<&str>,
    ) -> Result<()> {
        let encrypted = self.inner.vault.encrypt_json(id, secret)?;
        let expires_at = secret.expires_at_ms();
        let conn = self.inner.conn.lock();
        let updated = if let Some(url) = new_base_url {
            conn.execute(
                "UPDATE credentials
                 SET encrypted_payload = ?1, expires_at_ms = ?2, base_url = ?3,
                     status = CASE WHEN status = 'dead' THEN status ELSE 'ok' END
                 WHERE id = ?4",
                params![encrypted, expires_at, url, id],
            )
            .map_err(storage_err)?
        } else {
            conn.execute(
                "UPDATE credentials
                 SET encrypted_payload = ?1, expires_at_ms = ?2,
                     status = CASE WHEN status = 'dead' THEN status ELSE 'ok' END
                 WHERE id = ?3",
                params![encrypted, expires_at, id],
            )
            .map_err(storage_err)?
        };
        if updated == 0 {
            return Err(QuixigateError::NotFound(format!("credential '{id}' not found")));
        }
        Ok(())
    }

    pub fn update_credential_meta(
        &self,
        id: &str,
        label: Option<&str>,
        base_url: Option<Option<&str>>,
        priority: Option<i32>,
        enabled: Option<bool>,
        allow_api_overflow: Option<bool>,
    ) -> Result<CredentialRecord> {
        let existing = self.get_credential(id)?;
        let next_label = label.unwrap_or(&existing.label).trim();
        let next_base = match base_url {
            Some(v) => v.filter(|s| !s.trim().is_empty()).map(str::to_owned),
            None => existing.base_url,
        };
        let next_prio = priority.unwrap_or(existing.priority);
        let next_enabled = enabled.unwrap_or(existing.enabled);
        let next_overflow = allow_api_overflow.unwrap_or(existing.allow_api_overflow);
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE credentials
                 SET label = ?1, base_url = ?2, priority = ?3, enabled = ?4, allow_api_overflow = ?5
                 WHERE id = ?6",
                params![
                    next_label,
                    next_base,
                    next_prio,
                    i32::from(next_enabled),
                    i32::from(next_overflow),
                    id
                ],
            )
            .map_err(storage_err)?;
        }
        self.get_credential(id)
    }

    pub fn list_credentials(&self) -> Result<Vec<CredentialRecord>> {
        let conn = self.inner.conn.lock();
        let now = now_ms();
        // Auto-clear expired credential-wide cooldowns.
        conn.execute(
            "UPDATE credentials SET status = 'ok', status_reason = NULL, cooldown_until_ms = NULL
             WHERE status = 'exhausted' AND cooldown_until_ms IS NOT NULL AND cooldown_until_ms <= ?1",
            params![now],
        )
        .map_err(storage_err)?;

        let mut stmt = conn
            .prepare(
                "SELECT id, provider_id, label, auth_kind, base_url, priority, enabled,
                        allow_api_overflow, status, status_reason, cooldown_until_ms,
                        model_cooldowns_json, quota_snapshot_json, expires_at_ms,
                        last_used_at_ms, created_at_ms
                 FROM credentials ORDER BY priority ASC, created_at_ms ASC",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], map_credential_row)
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(rows)
    }

    pub fn get_credential(&self, id: &str) -> Result<CredentialRecord> {
        let conn = self.inner.conn.lock();
        conn.query_row(
            "SELECT id, provider_id, label, auth_kind, base_url, priority, enabled,
                    allow_api_overflow, status, status_reason, cooldown_until_ms,
                    model_cooldowns_json, quota_snapshot_json, expires_at_ms,
                    last_used_at_ms, created_at_ms
             FROM credentials WHERE id = ?1",
            params![id],
            map_credential_row,
        )
        .optional()
        .map_err(storage_err)?
        .ok_or_else(|| QuixigateError::NotFound(format!("credential '{id}' not found")))
    }

    pub fn get_credential_with_secret(
        &self,
        id: &str,
    ) -> Result<(CredentialRecord, CredentialSecret)> {
        let (record, blob): (CredentialRecord, Vec<u8>) = {
            let conn = self.inner.conn.lock();
            conn.query_row(
                "SELECT id, provider_id, label, auth_kind, base_url, priority, enabled,
                        allow_api_overflow, status, status_reason, cooldown_until_ms,
                        model_cooldowns_json, quota_snapshot_json, expires_at_ms,
                        last_used_at_ms, created_at_ms, encrypted_payload
                 FROM credentials WHERE id = ?1",
                params![id],
                |row| Ok((map_credential_row(row)?, row.get::<_, Vec<u8>>(16)?)),
            )
            .optional()
            .map_err(storage_err)?
            .ok_or_else(|| QuixigateError::NotFound(format!("credential '{id}' not found")))?
        };
        let secret = self.inner.vault.decrypt_json(&record.id, &blob)?;
        Ok((record, secret))
    }

    pub fn delete_credential(&self, id: &str) -> Result<()> {
        let conn = self.inner.conn.lock();
        conn.execute("DELETE FROM credentials WHERE id = ?1", params![id])
            .map_err(storage_err)?;
        Ok(())
    }

    pub fn mark_credential_used(&self, id: &str) -> Result<()> {
        let conn = self.inner.conn.lock();
        conn.execute(
            "UPDATE credentials SET last_used_at_ms = ?1 WHERE id = ?2",
            params![now_ms(), id],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    pub fn record_credential_cooldown(
        &self,
        id: &str,
        until_ms: i64,
        reason: &str,
    ) -> Result<()> {
        let conn = self.inner.conn.lock();
        conn.execute(
            "UPDATE credentials
             SET status = 'exhausted', status_reason = ?1, cooldown_until_ms = ?2
             WHERE id = ?3 AND status != 'dead'",
            params![reason, until_ms, id],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    pub fn record_model_cooldown(&self, id: &str, model: &str, until_ms: i64) -> Result<()> {
        let conn = self.inner.conn.lock();
        let raw: Option<String> = conn
            .query_row(
                "SELECT model_cooldowns_json FROM credentials WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .optional()
            .map_err(storage_err)?;
        let Some(raw_json) = raw else {
            return Ok(());
        };
        let mut map: BTreeMap<String, i64> = serde_json::from_str(&raw_json).unwrap_or_default();
        let now = now_ms();
        map.retain(|_, &mut v| v > now);
        let entry = map.entry(model.to_owned()).or_insert(0);
        *entry = (*entry).max(until_ms);
        let updated = serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_owned());
        conn.execute(
            "UPDATE credentials SET model_cooldowns_json = ?1 WHERE id = ?2",
            params![updated, id],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    pub fn mark_credential_dead(&self, id: &str, reason: &str) -> Result<()> {
        let conn = self.inner.conn.lock();
        conn.execute(
            "UPDATE credentials SET status = 'dead', status_reason = ?1 WHERE id = ?2",
            params![reason, id],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    pub fn reset_credential_health(&self, id: &str) -> Result<CredentialRecord> {
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "UPDATE credentials
                 SET status = 'ok', status_reason = NULL, cooldown_until_ms = NULL, model_cooldowns_json = '{}'
                 WHERE id = ?1",
                params![id],
            )
            .map_err(storage_err)?;
        }
        self.get_credential(id)
    }

    pub fn save_quota_snapshot(&self, id: &str, snapshot: &QuotaSnapshot) -> Result<()> {
        let json = serde_json::to_string(snapshot)
            .map_err(|e| QuixigateError::Storage(format!("quota snapshot serialize failed: {e}")))?;
        let conn = self.inner.conn.lock();
        conn.execute(
            "UPDATE credentials SET quota_snapshot_json = ?1 WHERE id = ?2",
            params![json, id],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    // ── Model Routes & Aliases ───────────────────────────────────────────────

    pub fn list_model_routes(&self) -> Result<Vec<ModelRoute>> {
        let conn = self.inner.conn.lock();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {MODEL_ROUTE_COLUMNS} FROM model_routes ORDER BY alias ASC"
            ))
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], map_model_route_row)
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(rows)
    }

    pub fn get_model_route(&self, alias: &str) -> Result<Option<ModelRoute>> {
        let conn = self.inner.conn.lock();
        conn.query_row(
            &format!("SELECT {MODEL_ROUTE_COLUMNS} FROM model_routes WHERE alias = ?1"),
            params![alias],
            map_model_route_row,
        )
        .optional()
        .map_err(storage_err)
    }

    pub fn upsert_model_route(&self, route: &ModelRoute) -> Result<ModelRoute> {
        let alias = route.alias.trim();
        if alias.is_empty() {
            return Err(QuixigateError::Validation("route alias cannot be empty".to_owned()));
        }
        let targets_json = serde_json::to_string(&route.targets)
            .map_err(|e| QuixigateError::Validation(format!("invalid route targets: {e}")))?;
        let strategy = serde_json::to_value(route.pool_strategy)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "fill_first".to_owned());
        let now = now_ms();
        let created_at = route.created_at_ms.unwrap_or(now);
        let team_id = route
            .team_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        {
            let conn = self.inner.conn.lock();
            conn.execute(
                "INSERT INTO model_routes (
                    alias, targets_json, pool_strategy, allow_paid_overflow,
                    input_nanos_per_m, output_nanos_per_m, cache_read_nanos_per_m,
                    team_id, created_by, updated_by, created_at_ms, updated_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                 ON CONFLICT(alias) DO UPDATE SET
                    targets_json = excluded.targets_json,
                    pool_strategy = excluded.pool_strategy,
                    allow_paid_overflow = excluded.allow_paid_overflow,
                    input_nanos_per_m = excluded.input_nanos_per_m,
                    output_nanos_per_m = excluded.output_nanos_per_m,
                    cache_read_nanos_per_m = excluded.cache_read_nanos_per_m,
                    team_id = COALESCE(excluded.team_id, model_routes.team_id),
                    created_by = COALESCE(model_routes.created_by, excluded.created_by),
                    updated_by = COALESCE(excluded.updated_by, model_routes.updated_by),
                    created_at_ms = COALESCE(model_routes.created_at_ms, excluded.created_at_ms),
                    updated_at_ms = excluded.updated_at_ms",
                params![
                    alias,
                    targets_json,
                    strategy,
                    i32::from(route.allow_paid_overflow),
                    route.input_nanos_per_m,
                    route.output_nanos_per_m,
                    route.cache_read_nanos_per_m,
                    team_id,
                    route.created_by,
                    route.updated_by,
                    created_at,
                    now
                ],
            )
            .map_err(storage_err)?;
        }
        self.get_model_route(alias)?
            .ok_or_else(|| QuixigateError::NotFound(alias.to_owned()))
    }

    pub fn delete_model_route(&self, alias: &str) -> Result<()> {
        let conn = self.inner.conn.lock();
        conn.execute("DELETE FROM model_routes WHERE alias = ?1", params![alias])
            .map_err(storage_err)?;
        Ok(())
    }

    // ── Exact Response Cache Persistence ─────────────────────────────────────

    pub fn get_cached_response(&self, cache_key: &str) -> Result<Option<String>> {
        let now = now_ms();
        let conn = self.inner.conn.lock();
        conn.query_row(
            "SELECT response_json FROM response_cache WHERE cache_key = ?1 AND expires_at_ms > ?2",
            params![cache_key, now],
            |r| r.get(0),
        )
        .optional()
        .map_err(storage_err)
    }

    pub fn put_cached_response(
        &self,
        cache_key: &str,
        model: &str,
        response_json: &str,
        ttl_secs: u64,
    ) -> Result<()> {
        let now = now_ms();
        let expires = now + (ttl_secs as i64) * 1000;
        let conn = self.inner.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO response_cache (cache_key, model, response_json, expires_at_ms, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![cache_key, model, response_json, expires, now],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    pub fn clear_response_cache(&self) -> Result<usize> {
        let conn = self.inner.conn.lock();
        conn.execute("DELETE FROM response_cache", [])
            .map_err(storage_err)
    }

    pub fn count_cached_responses(&self) -> Result<usize> {
        let now = now_ms();
        let conn = self.inner.conn.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM response_cache WHERE expires_at_ms > ?1",
                params![now],
                |r| r.get(0),
            )
            .map_err(storage_err)?;
        Ok(count.max(0) as usize)
    }

    // ── Request Logs & Dashboard Analytics ───────────────────────────────────

    pub fn insert_request_log(&self, log: &RequestLog) -> Result<()> {
        let max_rows = self.inner.settings_cache.read().max_trace_rows.max(100);
        let tags_json = serde_json::to_string(&log.tags).unwrap_or_else(|_| "[]".to_owned());
        let conn = self.inner.conn.lock();
        conn.execute(
            "INSERT INTO request_logs (
                id, timestamp_ms, ingress_protocol, requested_model, resolved_model,
                provider_id, credential_id, user_id, team_id, virtual_key_id, tags_json,
                status_code, stream, fallback_attempts, cache_hit, ttft_ms,
                duration_ms, input_tokens, cached_input_tokens, reasoning_tokens,
                output_tokens, cost_nanos, api_equiv_nanos, otel_trace_id,
                error_message, request_body_json, response_body_json
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
                ?21, ?22, ?23, ?24, ?25, ?26, ?27
             )",
            params![
                log.id,
                log.timestamp_ms,
                log.ingress_protocol.as_str(),
                log.requested_model,
                log.resolved_model,
                log.provider_id,
                log.credential_id,
                log.user_id,
                log.team_id,
                log.virtual_key_id,
                tags_json,
                log.status_code,
                i32::from(log.stream),
                log.fallback_attempts,
                i32::from(log.cache_hit),
                log.ttft_ms.map(|v| v as i64),
                log.duration_ms as i64,
                log.input_tokens as i64,
                log.cached_input_tokens as i64,
                log.reasoning_tokens as i64,
                log.output_tokens as i64,
                log.cost_nanos.nanos(),
                log.api_equiv_nanos.nanos(),
                log.otel_trace_id,
                log.error_message,
                log.request_body_json,
                log.response_body_json,
            ],
        )
        .map_err(storage_err)?;

        // Periodic ring-buffer prune so SQLite stays fast indefinitely.
        conn.execute(
            "DELETE FROM request_logs WHERE id IN (
                SELECT id FROM request_logs ORDER BY timestamp_ms DESC LIMIT -1 OFFSET ?1
             )",
            params![max_rows as i64],
        )
        .map_err(storage_err)?;
        Ok(())
    }

    pub fn list_request_logs(&self, filter: &LogFilter) -> Result<Vec<RequestLog>> {
        let limit = filter.limit.clamp(1, 500) as i64;
        let offset = filter.offset as i64;
        let resolved_key_filter = filter.virtual_key_id.as_deref().map(|k| {
            self.resolve_virtual_key_id(k)
                .unwrap_or_else(|_| k.trim().to_owned())
        });
        let tag_needle = filter
            .tag
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|t| format!("\"{}\"", t.replace('"', "")));
        let conn = self.inner.conn.lock();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {REQUEST_LOG_COLUMNS}
                 FROM request_logs
                 WHERE (?1 IS NULL OR resolved_model LIKE '%' || ?1 || '%' OR requested_model LIKE '%' || ?1 || '%')
                   AND (?2 IS NULL OR provider_id = ?2)
                   AND (?3 IS NULL OR status_code >= ?3)
                   AND (?4 IS NULL OR user_id = ?4)
                   AND (?5 IS NULL OR team_id = ?5)
                   AND (?6 IS NULL OR virtual_key_id = ?6)
                   AND (?7 IS NULL OR timestamp_ms >= ?7)
                   AND (?8 IS NULL OR timestamp_ms <= ?8)
                   AND (?9 IS NULL OR tags_json LIKE '%' || ?9 || '%')
                 ORDER BY timestamp_ms DESC
                 LIMIT ?10 OFFSET ?11"
            ))
            .map_err(storage_err)?;
        let rows = stmt
            .query_map(
                params![
                    filter.model,
                    filter.provider_id,
                    filter.status_min.map(i32::from),
                    filter.user_id,
                    filter.team_id,
                    resolved_key_filter,
                    filter.since_ms,
                    filter.until_ms,
                    tag_needle,
                    limit,
                    offset
                ],
                map_request_log_row,
            )
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(rows)
    }

    pub fn clear_request_logs(&self, before_ms: Option<i64>) -> Result<usize> {
        let conn = self.inner.conn.lock();
        match before_ms {
            Some(cutoff) => conn
                .execute(
                    "DELETE FROM request_logs WHERE timestamp_ms <= ?1",
                    params![cutoff],
                )
                .map_err(storage_err),
            None => conn
                .execute("DELETE FROM request_logs", [])
                .map_err(storage_err),
        }
    }

    pub fn get_request_log(&self, id: &str) -> Result<RequestLog> {
        let conn = self.inner.conn.lock();
        conn.query_row(
            &format!("SELECT {REQUEST_LOG_COLUMNS} FROM request_logs WHERE id = ?1"),
            params![id],
            map_request_log_row,
        )
        .optional()
        .map_err(storage_err)?
        .ok_or_else(|| QuixigateError::NotFound(format!("trace '{id}' not found")))
    }

    /// Aggregates `request_logs` into exact `NanoUsd` spend & savings breakdowns grouped by
    /// `"model"`, `"provider"`, `"tag"`, `"key"` (`"api_key"`), `"team"`, `"user"`, or `"day"` (`"date"`).
    pub fn get_spend_report(
        &self,
        group_by: &str,
        filter: &LogFilter,
    ) -> Result<Vec<SpendReportRow>> {
        let mut scan_filter = filter.clone();
        scan_filter.limit = 500;
        scan_filter.offset = 0;
        let resolved_key_filter = scan_filter.virtual_key_id.as_deref().map(|k| {
            self.resolve_virtual_key_id(k)
                .unwrap_or_else(|_| k.trim().to_owned())
        });
        let tag_needle = scan_filter
            .tag
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|t| format!("\"{}\"", t.replace('"', "")));

        let rows = {
            let conn = self.inner.conn.lock();
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT {REQUEST_LOG_COLUMNS}
                     FROM request_logs
                     WHERE (?1 IS NULL OR resolved_model LIKE '%' || ?1 || '%' OR requested_model LIKE '%' || ?1 || '%')
                       AND (?2 IS NULL OR provider_id = ?2)
                       AND (?3 IS NULL OR status_code >= ?3)
                       AND (?4 IS NULL OR user_id = ?4)
                       AND (?5 IS NULL OR team_id = ?5)
                       AND (?6 IS NULL OR virtual_key_id = ?6)
                       AND (?7 IS NULL OR timestamp_ms >= ?7)
                       AND (?8 IS NULL OR timestamp_ms <= ?8)
                       AND (?9 IS NULL OR tags_json LIKE '%' || ?9 || '%')
                     ORDER BY timestamp_ms DESC"
                ))
                .map_err(storage_err)?;
            stmt.query_map(
                params![
                    scan_filter.model,
                    scan_filter.provider_id,
                    scan_filter.status_min.map(i32::from),
                    scan_filter.user_id,
                    scan_filter.team_id,
                    resolved_key_filter,
                    scan_filter.since_ms,
                    scan_filter.until_ms,
                    tag_needle,
                ],
                map_request_log_row,
            )
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?
        };

        let mode = group_by.trim().to_ascii_lowercase();
        let mut agg: BTreeMap<String, SpendReportRow> = BTreeMap::new();

        for r in rows {
            let keys: Vec<String> = match mode.as_str() {
                "tag" | "tags" => {
                    if r.tags.is_empty() {
                        vec!["(untagged)".to_owned()]
                    } else {
                        r.tags.clone()
                    }
                }
                "provider" | "providers" => vec![r.provider_id.clone()],
                "key" | "keys" | "api_key" | "virtual_key" => {
                    vec![r.virtual_key_id.clone().unwrap_or_else(|| "(no-key)".to_owned())]
                }
                "team" | "teams" | "team_id" => {
                    vec![r.team_id.clone().unwrap_or_else(|| "(no-team)".to_owned())]
                }
                "user" | "users" | "user_id" => {
                    vec![r.user_id.clone().unwrap_or_else(|| "(anonymous)".to_owned())]
                }
                "day" | "date" | "daily" => {
                    let day = chrono::TimeZone::timestamp_millis_opt(&chrono::Utc, r.timestamp_ms)
                        .single()
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_else(|| "unknown".to_owned());
                    vec![day]
                }
                _ => vec![r.resolved_model.clone()],
            };

            let savings = if r.cost_nanos.nanos() == 0 {
                r.api_equiv_nanos
            } else {
                NanoUsd::ZERO
            };

            for k in keys {
                let entry = agg.entry(k.clone()).or_insert(SpendReportRow {
                    group_key: k,
                    requests: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_tokens: 0,
                    total_tokens: 0,
                    spend_nanos: NanoUsd::ZERO,
                    api_equiv_nanos: NanoUsd::ZERO,
                    savings_nanos: NanoUsd::ZERO,
                });
                entry.requests += 1;
                entry.input_tokens += r.input_tokens;
                entry.output_tokens += r.output_tokens;
                entry.cached_tokens += r.cached_input_tokens;
                entry.total_tokens += r.input_tokens + r.output_tokens;
                entry.spend_nanos = NanoUsd(entry.spend_nanos.nanos().saturating_add(r.cost_nanos.nanos()));
                entry.api_equiv_nanos =
                    NanoUsd(entry.api_equiv_nanos.nanos().saturating_add(r.api_equiv_nanos.nanos()));
                entry.savings_nanos =
                    NanoUsd(entry.savings_nanos.nanos().saturating_add(savings.nanos()));
            }
        }

        let mut out: Vec<SpendReportRow> = agg.into_values().collect();
        out.sort_by(|a, b| {
            b.spend_nanos
                .nanos()
                .cmp(&a.spend_nanos.nanos())
                .then_with(|| b.requests.cmp(&a.requests))
                .then_with(|| a.group_key.cmp(&b.group_key))
        });
        Ok(out)
    }

    // ── Management Audit Logs (`audit_logs`) ─────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn insert_audit_log(
        &self,
        actor_id: &str,
        actor_role: &str,
        action: &str,
        table_name: &str,
        object_id: &str,
        before_value: Option<serde_json::Value>,
        updated_values: Option<serde_json::Value>,
    ) -> Result<AuditLogEntry> {
        let entry = AuditLogEntry {
            id: format!("aud_{}", Uuid::new_v4().simple()),
            timestamp_ms: now_ms(),
            actor_id: actor_id.to_owned(),
            actor_role: actor_role.to_owned(),
            action: action.to_owned(),
            table_name: table_name.to_owned(),
            object_id: object_id.to_owned(),
            before_value,
            updated_values,
        };
        let before_json = entry
            .before_value
            .as_ref()
            .and_then(|v| serde_json::to_string(v).ok());
        let after_json = entry
            .updated_values
            .as_ref()
            .and_then(|v| serde_json::to_string(v).ok());
        let max_rows = self.inner.settings_cache.read().max_trace_rows.max(500) as i64;
        let conn = self.inner.conn.lock();
        conn.execute(
            "INSERT INTO audit_logs (
                id, timestamp_ms, actor_id, actor_role, action, table_name,
                object_id, before_value_json, updated_values_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                entry.id,
                entry.timestamp_ms,
                entry.actor_id,
                entry.actor_role,
                entry.action,
                entry.table_name,
                entry.object_id,
                before_json,
                after_json,
            ],
        )
        .map_err(storage_err)?;
        let _ = conn.execute(
            "DELETE FROM audit_logs WHERE id IN (
                SELECT id FROM audit_logs ORDER BY timestamp_ms DESC LIMIT -1 OFFSET ?1
             )",
            params![max_rows],
        );
        Ok(entry)
    }

    pub fn list_audit_logs(
        &self,
        table_name: Option<&str>,
        action: Option<&str>,
        actor_id: Option<&str>,
        object_id: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<AuditLogEntry>> {
        let lim = limit.clamp(1, 500) as i64;
        let off = offset as i64;
        let conn = self.inner.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, timestamp_ms, actor_id, actor_role, action, table_name,
                        object_id, before_value_json, updated_values_json
                 FROM audit_logs
                 WHERE (?1 IS NULL OR table_name = ?1)
                   AND (?2 IS NULL OR action = ?2)
                   AND (?3 IS NULL OR actor_id = ?3)
                   AND (?4 IS NULL OR object_id = ?4)
                 ORDER BY timestamp_ms DESC
                 LIMIT ?5 OFFSET ?6",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map(
                params![table_name, action, actor_id, object_id, lim, off],
                map_audit_log_row,
            )
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        Ok(rows)
    }

    pub fn get_audit_log(&self, id: &str) -> Result<AuditLogEntry> {
        let conn = self.inner.conn.lock();
        conn.query_row(
            "SELECT id, timestamp_ms, actor_id, actor_role, action, table_name,
                    object_id, before_value_json, updated_values_json
             FROM audit_logs WHERE id = ?1",
            params![id.trim()],
            map_audit_log_row,
        )
        .optional()
        .map_err(storage_err)?
        .ok_or_else(|| QuixigateError::NotFound(format!("audit log '{id}' not found")))
    }

    pub fn get_dashboard_metrics(&self) -> Result<DashboardMetrics> {
        let now = now_ms();
        let since_24h = now - 24 * 60 * 60 * 1000;
        let since_30d = now - 30 * 24 * 60 * 60 * 1000;
        let conn = self.inner.conn.lock();

        let (total_24h, err_24h, hits_24h, in_24h, out_24h, cached_24h, dur_24h, spend_24h): (
            i64,
            i64,
            i64,
            i64,
            i64,
            i64,
            i64,
            i64,
        ) = conn
            .query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(cache_hit), 0),
                        COALESCE(SUM(input_tokens), 0),
                        COALESCE(SUM(output_tokens), 0),
                        COALESCE(SUM(cached_input_tokens), 0),
                        COALESCE(SUM(CASE WHEN output_tokens > 0 THEN duration_ms ELSE 0 END), 0),
                        COALESCE(SUM(cost_nanos), 0)
                 FROM request_logs WHERE timestamp_ms >= ?1",
                params![since_24h],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                    ))
                },
            )
            .map_err(storage_err)?;

        let (spend_30d, savings_30d): (i64, i64) = conn
            .query_row(
                "SELECT COALESCE(SUM(cost_nanos), 0),
                        COALESCE(SUM(CASE WHEN cost_nanos = 0 THEN api_equiv_nanos ELSE 0 END), 0)
                 FROM request_logs WHERE timestamp_ms >= ?1",
                params![since_30d],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(storage_err)?;

        let mut ttft_stmt = conn
            .prepare(
                "SELECT ttft_ms FROM request_logs
                 WHERE timestamp_ms >= ?1 AND ttft_ms IS NOT NULL
                 ORDER BY ttft_ms ASC",
            )
            .map_err(storage_err)?;
        let ttfts: Vec<u64> = ttft_stmt
            .query_map(params![since_24h], |r| Ok(r.get::<_, i64>(0)?.max(0) as u64))
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        drop(ttft_stmt);

        let percentile = |pct: usize| -> Option<u64> {
            if ttfts.is_empty() {
                None
            } else {
                let idx = ((ttfts.len() - 1) * pct) / 100;
                Some(ttfts[idx])
            }
        };

        let avg_tok_per_sec_24h = if dur_24h > 0 && out_24h > 0 {
            (out_24h as f64) / ((dur_24h as f64) / 1000.0)
        } else {
            0.0
        };

        let mut prov_stmt = conn
            .prepare(
                "SELECT provider_id, COUNT(*),
                        COALESCE(SUM(input_tokens + output_tokens), 0),
                        COALESCE(SUM(cost_nanos), 0),
                        COALESCE(SUM(api_equiv_nanos), 0)
                 FROM request_logs WHERE timestamp_ms >= ?1
                 GROUP BY provider_id ORDER BY COUNT(*) DESC LIMIT 12",
            )
            .map_err(storage_err)?;
        let by_provider = prov_stmt
            .query_map(params![since_30d], |r| {
                Ok(ProviderSpendSummary {
                    provider_id: r.get(0)?,
                    requests: r.get::<_, i64>(1)? as u64,
                    tokens: r.get::<_, i64>(2)? as u64,
                    spend_nanos: NanoUsd(r.get(3)?),
                    api_equiv_nanos: NanoUsd(r.get(4)?),
                })
            })
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;
        drop(prov_stmt);

        let mut mod_stmt = conn
            .prepare(
                "SELECT resolved_model, COUNT(*),
                        COALESCE(SUM(input_tokens + output_tokens), 0),
                        AVG(ttft_ms),
                        COALESCE(SUM(cost_nanos), 0),
                        COALESCE(SUM(api_equiv_nanos), 0)
                 FROM request_logs WHERE timestamp_ms >= ?1
                 GROUP BY resolved_model ORDER BY COUNT(*) DESC LIMIT 12",
            )
            .map_err(storage_err)?;
        let by_model = mod_stmt
            .query_map(params![since_30d], |r| {
                Ok(ModelSpendSummary {
                    model: r.get(0)?,
                    requests: r.get::<_, i64>(1)? as u64,
                    tokens: r.get::<_, i64>(2)? as u64,
                    avg_ttft_ms: r.get::<_, Option<f64>>(3)?.map(|v| v.round() as u64),
                    spend_nanos: NanoUsd(r.get(4)?),
                    api_equiv_nanos: NanoUsd(r.get(5)?),
                })
            })
            .map_err(storage_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(storage_err)?;

        Ok(DashboardMetrics {
            active_requests: self.inner.limiter.active_requests(),
            total_requests_24h: total_24h as u64,
            error_requests_24h: err_24h as u64,
            cache_hits_24h: hits_24h as u64,
            input_tokens_24h: in_24h as u64,
            output_tokens_24h: out_24h as u64,
            cached_tokens_24h: cached_24h as u64,
            avg_tok_per_sec_24h,
            ttft_p50_ms: percentile(50),
            ttft_p95_ms: percentile(95),
            spend_24h_nanos: NanoUsd(spend_24h),
            spend_30d_nanos: NanoUsd(spend_30d),
            savings_30d_nanos: NanoUsd(savings_30d),
            by_provider,
            by_model,
        })
    }
}

fn load_settings_from_conn(conn: &Connection) -> Result<GatewaySettings> {
    let raw: Option<String> = conn
        .query_row(
            "SELECT value_json FROM settings WHERE key = ?1",
            params![SETTINGS_KEY],
            |r| r.get(0),
        )
        .optional()
        .map_err(storage_err)?;
    match raw {
        Some(json) => Ok(serde_json::from_str(&json).unwrap_or_default()),
        None => Ok(GatewaySettings::default()),
    }
}

fn hash_password(password: &str) -> Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| QuixigateError::Internal(format!("password hashing failed: {e}")))
}

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    Sha256::digest(input).into()
}

fn generate_virtual_key_material() -> (String, String, [u8; 32]) {
    let mut secret = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut secret);
    let encoded = URL_SAFE_NO_PAD.encode(secret);
    let raw = format!("{}{encoded}", quixigate_core::brand::VIRTUAL_KEY_PREFIX);
    let prefix = format!("{}…{}", &raw[..12], &raw[raw.len() - 4..]);
    let digest = sha256_bytes(raw.as_bytes());
    (raw, prefix, digest)
}

fn glob_matches(pattern: &str, candidate: &str) -> bool {
    let pat = pattern.trim();
    if pat == "*" || pat.eq_ignore_ascii_case(candidate) {
        return true;
    }
    if let Some(prefix) = pat.strip_suffix('*') {
        return candidate.starts_with(prefix);
    }
    false
}

const VIRTUAL_KEY_COLUMNS: &str = "id, key_prefix, name, user_id, team_id, budget_limit_nanos, \
    budget_window, spend_nanos, window_reset_at_ms, rpm_limit, tpm_limit, max_concurrent, \
    allowed_models_json, log_payloads, expires_at_ms, revoked_at_ms, created_at_ms, \
    tags_json, model_budgets_json, allowed_cidrs_json, allow_spend_read";

const MODEL_ROUTE_COLUMNS: &str = "alias, targets_json, pool_strategy, allow_paid_overflow, \
    input_nanos_per_m, output_nanos_per_m, cache_read_nanos_per_m, updated_at_ms, \
    team_id, created_by, updated_by, created_at_ms";

const REQUEST_LOG_COLUMNS: &str = "id, timestamp_ms, ingress_protocol, requested_model, resolved_model, \
    provider_id, credential_id, user_id, team_id, virtual_key_id, \
    status_code, stream, fallback_attempts, cache_hit, ttft_ms, \
    duration_ms, input_tokens, cached_input_tokens, reasoning_tokens, \
    output_tokens, cost_nanos, api_equiv_nanos, otel_trace_id, \
    error_message, request_body_json, response_body_json, tags_json";

fn normalize_string_list(items: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for item in items {
        let trimmed = item.trim();
        if !trimmed.is_empty() && !out.iter().any(|existing: &String| existing == trimmed) {
            out.push(trimmed.to_owned());
        }
    }
    out
}

fn parse_model_budgets_json(raw: &str) -> BTreeMap<String, NanoUsd> {
    if let Ok(nanos_map) = serde_json::from_str::<BTreeMap<String, i64>>(raw) {
        return nanos_map.into_iter().map(|(k, v)| (k, NanoUsd(v))).collect();
    }
    serde_json::from_str::<BTreeMap<String, NanoUsd>>(raw).unwrap_or_default()
}

fn serialize_model_budgets_json(map: &BTreeMap<String, NanoUsd>) -> String {
    let raw_map: BTreeMap<&String, i64> = map.iter().map(|(k, v)| (k, v.nanos())).collect();
    serde_json::to_string(&raw_map).unwrap_or_else(|_| "{}".to_owned())
}

fn query_scoped_spend_locked(conn: &Connection, scope_key: &str) -> Result<i64> {
    Ok(conn
        .query_row(
            "SELECT spend_nanos FROM scoped_spend WHERE scope_key = ?1",
            params![scope_key],
            |r| r.get::<_, i64>(0),
        )
        .optional()
        .map_err(storage_err)?
        .unwrap_or(0))
}

fn storage_err(err: rusqlite::Error) -> QuixigateError {
    QuixigateError::Storage(err.to_string())
}

fn map_user_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<User> {
    let pw: Option<String> = row.get(4)?;
    Ok(User {
        id: row.get(0)?,
        username: row.get(1)?,
        display_name: row.get(2)?,
        role: UserRole::parse(&row.get::<_, String>(3)?),
        has_password: pw.is_some(),
        blocked: row.get::<_, i32>(10).unwrap_or(0) != 0,
        budget_limit_nanos: row.get::<_, Option<i64>>(5)?.map(NanoUsd),
        budget_window: BudgetWindow::parse(&row.get::<_, String>(6)?),
        spend_nanos: NanoUsd(row.get(7)?),
        window_reset_at_ms: row.get(8)?,
        created_at_ms: row.get(9)?,
    })
}

fn map_virtual_key_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<VirtualKey> {
    let models_json: String = row.get(12)?;
    let allowed_models: Vec<String> = serde_json::from_str(&models_json).unwrap_or_default();
    let tags_json: String = row.get(17).unwrap_or_else(|_| "[]".to_owned());
    let mb_json: String = row.get(18).unwrap_or_else(|_| "{}".to_owned());
    let cidrs_json: String = row.get(19).unwrap_or_else(|_| "[]".to_owned());
    let allow_spend_read: i32 = row.get(20).unwrap_or(1);
    Ok(VirtualKey {
        id: row.get(0)?,
        key_prefix: row.get(1)?,
        name: row.get(2)?,
        user_id: row.get(3)?,
        team_id: row.get(4)?,
        budget_limit_nanos: row.get::<_, Option<i64>>(5)?.map(NanoUsd),
        budget_window: BudgetWindow::parse(&row.get::<_, String>(6)?),
        spend_nanos: NanoUsd(row.get(7)?),
        window_reset_at_ms: row.get(8)?,
        rpm_limit: row.get(9)?,
        tpm_limit: row.get::<_, Option<i64>>(10)?.map(|v| v.max(0) as u64),
        max_concurrent: row.get(11)?,
        allowed_models,
        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
        model_budgets_nanos: parse_model_budgets_json(&mb_json),
        allowed_cidrs: serde_json::from_str(&cidrs_json).unwrap_or_default(),
        allow_spend_read: allow_spend_read != 0,
        log_payloads: row.get::<_, Option<i32>>(13)?.map(|v| v != 0),
        expires_at_ms: row.get(14)?,
        revoked_at_ms: row.get(15)?,
        created_at_ms: row.get(16)?,
    })
}

fn map_credential_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CredentialRecord> {
    let model_cd_json: String = row.get(11)?;
    let quota_json: Option<String> = row.get(12)?;
    Ok(CredentialRecord {
        id: row.get(0)?,
        provider_id: row.get(1)?,
        label: row.get(2)?,
        auth_kind: AuthKind::parse(&row.get::<_, String>(3)?),
        base_url: row.get(4)?,
        priority: row.get(5)?,
        enabled: row.get::<_, i32>(6)? != 0,
        allow_api_overflow: row.get::<_, i32>(7)? != 0,
        status: CredentialStatus::parse(&row.get::<_, String>(8)?),
        status_reason: row.get(9)?,
        cooldown_until_ms: row.get(10)?,
        model_cooldowns: serde_json::from_str(&model_cd_json).unwrap_or_default(),
        quota_snapshot: quota_json.and_then(|j| serde_json::from_str(&j).ok()),
        expires_at_ms: row.get(13)?,
        last_used_at_ms: row.get(14)?,
        created_at_ms: row.get(15)?,
    })
}

fn map_model_route_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRoute> {
    let targets_json: String = row.get(1)?;
    let strategy_str: String = row.get(2)?;
    let pool_strategy = serde_json::from_value(serde_json::Value::String(strategy_str))
        .unwrap_or(PoolStrategy::FillFirst);
    Ok(ModelRoute {
        alias: row.get(0)?,
        targets: serde_json::from_str(&targets_json).unwrap_or_default(),
        pool_strategy,
        allow_paid_overflow: row.get::<_, i32>(3)? != 0,
        input_nanos_per_m: row.get(4)?,
        output_nanos_per_m: row.get(5)?,
        cache_read_nanos_per_m: row.get(6)?,
        updated_at_ms: row.get(7)?,
        team_id: row.get(8).ok().flatten(),
        created_by: row.get(9).ok().flatten(),
        updated_by: row.get(10).ok().flatten(),
        created_at_ms: row.get(11).ok().flatten(),
    })
}

fn map_request_log_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RequestLog> {
    let tags_json: String = row.get(26).unwrap_or_else(|_| "[]".to_owned());
    Ok(RequestLog {
        id: row.get(0)?,
        timestamp_ms: row.get(1)?,
        ingress_protocol: IngressProtocol::parse(&row.get::<_, String>(2)?),
        requested_model: row.get(3)?,
        resolved_model: row.get(4)?,
        provider_id: row.get(5)?,
        credential_id: row.get(6)?,
        user_id: row.get(7)?,
        team_id: row.get(8)?,
        virtual_key_id: row.get(9)?,
        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
        status_code: row.get(10)?,
        stream: row.get::<_, i32>(11)? != 0,
        fallback_attempts: row.get(12)?,
        cache_hit: row.get::<_, i32>(13)? != 0,
        ttft_ms: row.get::<_, Option<i64>>(14)?.map(|v| v.max(0) as u64),
        duration_ms: row.get::<_, i64>(15)?.max(0) as u64,
        input_tokens: row.get::<_, i64>(16)?.max(0) as u64,
        cached_input_tokens: row.get::<_, i64>(17)?.max(0) as u64,
        reasoning_tokens: row.get::<_, i64>(18)?.max(0) as u64,
        output_tokens: row.get::<_, i64>(19)?.max(0) as u64,
        cost_nanos: NanoUsd(row.get(20)?),
        api_equiv_nanos: NanoUsd(row.get(21)?),
        otel_trace_id: row.get(22)?,
        error_message: row.get(23)?,
        request_body_json: row.get(24)?,
        response_body_json: row.get(25)?,
    })
}

fn map_audit_log_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuditLogEntry> {
    let before_raw: Option<String> = row.get(7)?;
    let after_raw: Option<String> = row.get(8)?;
    Ok(AuditLogEntry {
        id: row.get(0)?,
        timestamp_ms: row.get(1)?,
        actor_id: row.get(2)?,
        actor_role: row.get(3)?,
        action: row.get(4)?,
        table_name: row.get(5)?,
        object_id: row.get(6)?,
        before_value: before_raw.and_then(|s| serde_json::from_str(&s).ok()),
        updated_values: after_raw.and_then(|s| serde_json::from_str(&s).ok()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_key_rotation_grace_budget_gate_and_config_sync_work() {
        let db = Database::open_in_memory().unwrap();
        let user = db
            .create_user(
                "alice",
                "Alice",
                UserRole::Member,
                Some("hunter2-strong!"),
                Some(NanoUsd::parse_usd("5.00").unwrap()),
                BudgetWindow::Monthly,
            )
            .unwrap();
        let (_, session_tok) = db
            .authenticate_password("alice", "hunter2-strong!")
            .unwrap();
        assert_eq!(db.validate_session(&session_tok).unwrap().id, user.id);

        let (vk, raw_key_1) = db
            .create_virtual_key(
                "cursor-key",
                &user.id,
                None,
                Some(NanoUsd::parse_usd("1.00").unwrap()),
                BudgetWindow::Monthly,
                Some(2),
                None,
                Some(1),
                &["anthropic/*".to_owned()],
                None,
                None,
            )
            .unwrap();

        // Rotate with 60s grace window: both old and new keys authenticate!
        let (_, raw_key_2) = db.rotate_virtual_key(&vk.id, 60).unwrap();
        let p_old = db.verify_virtual_key(&raw_key_1).unwrap();
        let p_new = db.verify_virtual_key(&raw_key_2).unwrap();
        assert_eq!(p_old.virtual_key.as_ref().unwrap().id, vk.id);
        assert_eq!(p_new.virtual_key.as_ref().unwrap().id, vk.id);

        // Model allowlist blocks disallowed model.
        assert!(matches!(
            db.authorize_and_reserve(&p_new, "openai/gpt-5.2", 100, NanoUsd::ZERO),
            Err(QuixigateError::Forbidden(_))
        ));

        // Reserve $0.70 in-flight -> second parallel $0.40 reservation is blocked by zero-race budget gate!
        let (_conc1, b1) = db
            .authorize_and_reserve(
                &p_new,
                "anthropic/claude-sonnet-4-5",
                100,
                NanoUsd::parse_usd("0.70").unwrap(),
            )
            .unwrap();
        // Max concurrent = 1 also blocks while _conc1 is held!
        assert!(matches!(
            db.authorize_and_reserve(
                &p_new,
                "anthropic/claude-sonnet-4-5",
                100,
                NanoUsd::parse_usd("0.10").unwrap(),
            ),
            Err(QuixigateError::RateLimited { .. })
        ));
        drop(_conc1);

        // Now concurrency is free, but $0.70 is still reserved in-flight -> $0.40 exceeds $1.00 key budget!
        assert!(matches!(
            db.authorize_and_reserve(
                &p_new,
                "anthropic/claude-sonnet-4-5",
                100,
                NanoUsd::parse_usd("0.40").unwrap(),
            ),
            Err(QuixigateError::BudgetExceeded(_))
        ));

        // Commit actual spend of $0.65 -> remaining budget is $0.35.
        b1.commit(NanoUsd::parse_usd("0.65").unwrap()).unwrap();
        assert_eq!(
            db.get_virtual_key(&vk.id).unwrap().spend_nanos,
            NanoUsd::parse_usd("0.65").unwrap()
        );

        // Test TOML export & import with ${ENV_VAR:-default} expansion.
        let toml_cfg = r#"
[settings]
require_auth = false
openinference_project_name = "phoenix-prod"

[[credentials]]
provider_id = "openai"
label = "OpenAI Paid"
auth_kind = "api_key"
api_key = "${QUIXIGATE_TEST_NONEXISTENT_ENV:-sk-fallback-test}"
priority = 200

[[routes]]
alias = "coding"
pool_strategy = "fill_first"
allow_paid_overflow = true
targets = [{ model = "anthropic/claude-sonnet-4-5" }, { model = "openai/gpt-5.2" }]
"#;
        let count = db.import_config(toml_cfg, Some("toml")).unwrap();
        assert_eq!(count, 2);
        assert_eq!(
            db.get_settings().unwrap().openinference_project_name,
            "phoenix-prod"
        );
        let exported = db.export_config("toml", false).unwrap();
        assert!(exported.contains("OPENAI_API_KEY"));
    }
}
