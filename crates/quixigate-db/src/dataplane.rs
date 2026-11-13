//! Lock-free in-memory data-plane snapshot (`ArcSwap<ControlSnapshot>`), atomic `NanoUsd`
//! spend buckets, and batched background SQLite WAL write-behind engine.
//!
//! Inference requests never lock SQLite or run AES-256-GCM decryption on the hot path.
//! All Admin / Control-Plane reads and writes synchronize seamlessly via `lock_conn_synced()`
//! and `reload_snapshot()`, preserving 100% immediate read-after-write consistency.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::{Condvar, Mutex};
use quixigate_core::*;
use quixigate_vault::Vault;
use rusqlite::{Connection, params};

use crate::{
    MODEL_ROUTE_COLUMNS, VIRTUAL_KEY_COLUMNS, glob_matches, map_credential_row, map_end_user_row,
    map_model_route_row, map_user_row, map_virtual_key_row, parse_model_budgets_json, storage_err,
};

const FLUSH_INTERVAL: Duration = Duration::from_millis(15);
const FLUSH_BATCH_TRIGGER: usize = 256;

// ── Atomic Live Spend Bucket ─────────────────────────────────────────────────

pub struct SpendBucket {
    spend_nanos: AtomicI64,
    /// `0` indicates `None` (lifetime / non-expiring window).
    window_reset_at_ms: AtomicI64,
    window: BudgetWindow,
    table_meta: Option<(&'static str, &'static str, String)>,
}

impl SpendBucket {
    pub fn new(
        initial_spend: i64,
        window_reset_at_ms: Option<i64>,
        window: BudgetWindow,
        table_meta: Option<(&'static str, &'static str, String)>,
    ) -> Self {
        Self {
            spend_nanos: AtomicI64::new(initial_spend),
            window_reset_at_ms: AtomicI64::new(window_reset_at_ms.unwrap_or(0)),
            window,
            table_meta,
        }
    }

    #[inline]
    pub fn load_spend(&self, now_ms: i64, wb: &WriteBehindEngine) -> i64 {
        let reset = self.window_reset_at_ms.load(Ordering::Acquire);
        if reset > 0 && now_ms >= reset {
            let next_reset = self.window.next_reset_ms(now_ms).unwrap_or(0);
            if self
                .window_reset_at_ms
                .compare_exchange(reset, next_reset, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.spend_nanos.store(0, Ordering::Release);
                if let Some((table, id_col, id)) = &self.table_meta {
                    wb.push(WriteBehindOp::WindowReset {
                        table,
                        id_col,
                        id: id.clone(),
                        next_reset_ms: (next_reset > 0).then_some(next_reset),
                    });
                }
                return 0;
            }
        }
        self.spend_nanos.load(Ordering::Acquire)
    }

    #[inline]
    pub fn add_spend(&self, delta_nanos: i64) {
        if delta_nanos > 0 {
            self.spend_nanos.fetch_add(delta_nanos, Ordering::AcqRel);
        }
    }

    pub fn sync_from_authoritative(&self, spend_nanos: i64, window_reset_at_ms: Option<i64>) {
        self.window_reset_at_ms
            .store(window_reset_at_ms.unwrap_or(0), Ordering::Release);
        self.spend_nanos.store(spend_nanos, Ordering::Release);
    }
}

#[derive(Default)]
pub struct SpendRegistry {
    buckets: DashMap<String, Arc<SpendBucket>>,
}

impl SpendRegistry {
    pub fn upsert_authoritative(
        &self,
        key: String,
        spend_nanos: i64,
        window_reset_at_ms: Option<i64>,
        window: BudgetWindow,
        table_meta: Option<(&'static str, &'static str, String)>,
    ) -> Arc<SpendBucket> {
        if let Some(existing) = self.buckets.get(&key) {
            // If window changed, replace bucket; otherwise sync atomic values in place.
            if existing.window == window {
                existing.sync_from_authoritative(spend_nanos, window_reset_at_ms);
                return Arc::clone(existing.value());
            }
        }
        let bucket = Arc::new(SpendBucket::new(
            spend_nanos,
            window_reset_at_ms,
            window,
            table_meta,
        ));
        self.buckets.insert(key, Arc::clone(&bucket));
        bucket
    }

    #[inline]
    pub fn get_or_init_scoped(&self, db_scope: &str, initial_spend: i64) -> Arc<SpendBucket> {
        let map_key = format!("scope:{db_scope}");
        if let Some(existing) = self.buckets.get(&map_key) {
            return Arc::clone(existing.value());
        }
        self.buckets
            .entry(map_key)
            .or_insert_with(|| {
                Arc::new(SpendBucket::new(
                    initial_spend,
                    None,
                    BudgetWindow::Total,
                    None,
                ))
            })
            .clone()
    }

    pub fn clear_all(&self) {
        for entry in self.buckets.iter() {
            entry.value().spend_nanos.store(0, Ordering::Release);
        }
    }

    pub fn sync_scoped_keys(&self, seen_scope_keys: &HashSet<String>) {
        for entry in self.buckets.iter() {
            if let Some(raw_scope) = entry.key().strip_prefix("scope:")
                && !seen_scope_keys.contains(raw_scope)
            {
                entry.value().spend_nanos.store(0, Ordering::Release);
            }
        }
    }
}

// ── Cached Control-Plane Snapshot Types ──────────────────────────────────────

pub struct CachedUser {
    pub user: User,
    pub limiter_budget_scope: Arc<str>,
    pub label: String,
    pub spend_bucket: Arc<SpendBucket>,
}

pub struct CachedTeam {
    pub id: String,
    pub name: String,
    pub blocked: bool,
    pub disable_logging: bool,
    pub budget_limit_nanos: Option<NanoUsd>,
    pub allowed_models: Vec<String>,
    pub tags: Vec<String>,
    pub model_budgets_nanos: BTreeMap<String, NanoUsd>,
    pub member_ids: HashSet<String>,
    pub limiter_budget_scope: Arc<str>,
    pub label: String,
    pub spend_bucket: Arc<SpendBucket>,
}

pub struct CachedVirtualKey {
    pub vk: VirtualKey,
    pub key_sha256: [u8; 32],
    pub rate_scope: Arc<str>,
    pub limiter_budget_scope: Arc<str>,
    pub label: String,
    pub spend_bucket: Arc<SpendBucket>,
}

pub struct CachedEndUser {
    pub user_id: String,
    pub blocked: bool,
    pub allowed_models: Vec<String>,
    pub effective_limit_nanos: Option<NanoUsd>,
    pub limiter_budget_scope: Arc<str>,
    pub label: String,
    pub spend_bucket: Arc<SpendBucket>,
}

pub struct CachedCredential {
    pub record: CredentialRecord,
    pub encrypted_blob: Vec<u8>,
    pub secret: Option<Arc<CredentialSecret>>,
    pub last_used_at_ms: AtomicI64,
}

impl CachedCredential {
    #[inline]
    pub fn snapshot_record(&self, now_ms: i64) -> CredentialRecord {
        let mut rec = self.record.clone();
        let lu = self.last_used_at_ms.load(Ordering::Relaxed);
        if lu > 0 {
            rec.last_used_at_ms = Some(lu);
        }
        if rec.status == CredentialStatus::Exhausted
            && rec.cooldown_until_ms.is_some_and(|until| until <= now_ms)
        {
            rec.status = CredentialStatus::Ok;
            rec.status_reason = None;
            rec.cooldown_until_ms = None;
        }
        rec
    }
}

static NEXT_SNAPSHOT_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
pub struct ControlSnapshot {
    pub generation: u64,
    pub settings: Arc<GatewaySettings>,
    pub users_by_id: HashMap<String, Arc<CachedUser>>,
    pub users_by_username: HashMap<String, Arc<CachedUser>>,
    pub teams_by_id: HashMap<String, Arc<CachedTeam>>,
    pub keys_by_sha256: HashMap<[u8; 32], Arc<CachedVirtualKey>>,
    pub keys_by_id: HashMap<String, Arc<CachedVirtualKey>>,
    pub rotated_keys_by_sha256: HashMap<[u8; 32], (String, i64)>,
    pub routes_by_alias: HashMap<String, Arc<ModelRoute>>,
    pub access_groups: HashMap<String, Vec<String>>,
    pub end_users_by_id: HashMap<String, Arc<CachedEndUser>>,
    pub tag_limits_nanos: HashMap<String, i64>,
    pub credentials: Vec<Arc<CachedCredential>>,
    pub credentials_by_id: HashMap<String, Arc<CachedCredential>>,
}

impl ControlSnapshot {
    #[inline]
    pub fn model_matches_allowlist(
        &self,
        patterns: &[String],
        requested_model: &str,
        base_model: &str,
    ) -> bool {
        for pat in patterns {
            if glob_matches(pat, requested_model) || glob_matches(pat, base_model) {
                return true;
            }
            let ag_key = pat
                .trim()
                .strip_prefix("access_group:")
                .unwrap_or(pat.trim());
            if let Some(ag_models) = self.access_groups.get(ag_key)
                && ag_models
                    .iter()
                    .any(|m| glob_matches(m, requested_model) || glob_matches(m, base_model))
            {
                return true;
            }
        }
        false
    }
}

// ── Asynchronous Write-Behind Batcher ────────────────────────────────────────

pub enum WriteBehindOp {
    Spend {
        user_id: Option<String>,
        team_id: Option<String>,
        virtual_key_id: Option<String>,
        end_user_id: Option<String>,
        durable_scopes: Vec<String>,
        nanos: i64,
        now_ms: i64,
    },
    WindowReset {
        table: &'static str,
        id_col: &'static str,
        id: String,
        next_reset_ms: Option<i64>,
    },
    CredentialUsed {
        id: String,
        at_ms: i64,
    },
    RequestLog {
        log: Box<RequestLog>,
        max_rows: usize,
    },
}

struct QueueInner {
    ops: Vec<WriteBehindOp>,
    shutdown: bool,
}

pub struct WriteBehindEngine {
    queue: Mutex<QueueInner>,
    cv: Condvar,
    pending_count: AtomicUsize,
    worker_busy: AtomicBool,
}

impl WriteBehindEngine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(QueueInner {
                ops: Vec::with_capacity(512),
                shutdown: false,
            }),
            cv: Condvar::new(),
            pending_count: AtomicUsize::new(0),
            worker_busy: AtomicBool::new(false),
        })
    }

    #[inline]
    pub fn has_pending(&self) -> bool {
        self.pending_count.load(Ordering::Acquire) > 0 || self.worker_busy.load(Ordering::Acquire)
    }

    #[inline]
    pub fn push(&self, op: WriteBehindOp) {
        let prev = self.pending_count.fetch_add(1, Ordering::AcqRel);
        let should_wake = {
            let mut q = self.queue.lock();
            q.ops.push(op);
            prev == 0 || q.ops.len() >= FLUSH_BATCH_TRIGGER
        };
        if should_wake {
            self.cv.notify_one();
        }
    }

    pub fn take_pending(&self) -> Vec<WriteBehindOp> {
        let mut q = self.queue.lock();
        if q.ops.is_empty() {
            return Vec::new();
        }
        let batch = std::mem::replace(&mut q.ops, Vec::with_capacity(256));
        self.pending_count.fetch_sub(batch.len(), Ordering::AcqRel);
        batch
    }

    pub fn signal_shutdown(&self) {
        let mut q = self.queue.lock();
        q.shutdown = true;
        self.cv.notify_all();
    }

    pub fn spawn_flusher_thread(
        self: &Arc<Self>,
        conn_mutex: Arc<Mutex<Connection>>,
    ) -> JoinHandle<()> {
        let engine = Arc::clone(self);
        std::thread::Builder::new()
            .name("quixigate-wal-flusher".to_owned())
            .spawn(move || {
                loop {
                    let should_exit = {
                        let mut q = engine.queue.lock();
                        if q.ops.is_empty() && !q.shutdown {
                            engine.cv.wait_for(&mut q, FLUSH_INTERVAL);
                        }
                        q.shutdown && q.ops.is_empty()
                    };
                    if should_exit {
                        break;
                    }
                    engine.worker_busy.store(true, Ordering::Release);
                    {
                        let conn = conn_mutex.lock();
                        let batch = engine.take_pending();
                        if !batch.is_empty() {
                            let _ = apply_ops_locked(&conn, &batch);
                        }
                    }
                    engine.worker_busy.store(false, Ordering::Release);
                }
            })
            .expect("spawn quixigate-wal-flusher thread")
    }
}

pub fn apply_ops_locked(conn: &Connection, ops: &[WriteBehindOp]) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }
    conn.execute_batch("SAVEPOINT qg_wb;")
        .map_err(storage_err)?;

    let res = (|| -> Result<()> {
        let mut max_trace_rows = 0usize;
        let mut inserted_logs = 0usize;
        let mut cred_used: HashMap<&str, i64> = HashMap::new();

        for op in ops {
            match op {
                WriteBehindOp::WindowReset {
                    table,
                    id_col,
                    id,
                    next_reset_ms,
                } => {
                    let sql = format!(
                        "UPDATE {table} SET spend_nanos = 0, window_reset_at_ms = ?1 WHERE {id_col} = ?2"
                    );
                    let mut stmt = conn.prepare_cached(&sql).map_err(storage_err)?;
                    stmt.execute(params![next_reset_ms, id])
                        .map_err(storage_err)?;
                }
                WriteBehindOp::Spend {
                    user_id,
                    team_id,
                    virtual_key_id,
                    end_user_id,
                    durable_scopes,
                    nanos,
                    now_ms,
                } => {
                    if let Some(uid) = user_id {
                        let mut stmt = conn
                            .prepare_cached(
                                "UPDATE users SET spend_nanos = spend_nanos + ?1 WHERE id = ?2",
                            )
                            .map_err(storage_err)?;
                        stmt.execute(params![nanos, uid]).map_err(storage_err)?;
                    }
                    if let Some(tid) = team_id {
                        let mut stmt = conn
                            .prepare_cached(
                                "UPDATE teams SET spend_nanos = spend_nanos + ?1 WHERE id = ?2",
                            )
                            .map_err(storage_err)?;
                        stmt.execute(params![nanos, tid]).map_err(storage_err)?;
                    }
                    if let Some(kid) = virtual_key_id {
                        let mut stmt = conn
                            .prepare_cached(
                                "UPDATE virtual_keys SET spend_nanos = spend_nanos + ?1 WHERE id = ?2",
                            )
                            .map_err(storage_err)?;
                        stmt.execute(params![nanos, kid]).map_err(storage_err)?;
                    }
                    if let Some(eu_id) = end_user_id {
                        let mut stmt = conn
                            .prepare_cached(
                                "UPDATE end_users SET spend_nanos = spend_nanos + ?1, updated_at_ms = ?2 WHERE user_id = ?3",
                            )
                            .map_err(storage_err)?;
                        stmt.execute(params![nanos, now_ms, eu_id])
                            .map_err(storage_err)?;
                    }
                    if !durable_scopes.is_empty() {
                        let mut stmt = conn
                            .prepare_cached(
                                "INSERT INTO scoped_spend (scope_key, spend_nanos, updated_at_ms)
                                 VALUES (?1, ?2, ?3)
                                 ON CONFLICT(scope_key) DO UPDATE SET
                                    spend_nanos = spend_nanos + excluded.spend_nanos,
                                    updated_at_ms = excluded.updated_at_ms",
                            )
                            .map_err(storage_err)?;
                        for scope in durable_scopes {
                            stmt.execute(params![scope, nanos, now_ms])
                                .map_err(storage_err)?;
                        }
                    }
                }
                WriteBehindOp::CredentialUsed { id, at_ms } => {
                    let slot = cred_used.entry(id.as_str()).or_insert(0);
                    if *at_ms > *slot {
                        *slot = *at_ms;
                    }
                }
                WriteBehindOp::RequestLog { log, max_rows } => {
                    max_trace_rows = max_trace_rows.max(*max_rows);
                    inserted_logs += 1;
                    let tags_json =
                        serde_json::to_string(&log.tags).unwrap_or_else(|_| "[]".to_owned());
                    let mut stmt = conn
                        .prepare_cached(
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
                        )
                        .map_err(storage_err)?;
                    stmt.execute(params![
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
                    ])
                    .map_err(storage_err)?;
                }
            }
        }

        if !cred_used.is_empty() {
            let mut stmt = conn
                .prepare_cached("UPDATE credentials SET last_used_at_ms = ?1 WHERE id = ?2")
                .map_err(storage_err)?;
            for (cid, at_ms) in cred_used {
                stmt.execute(params![at_ms, cid]).map_err(storage_err)?;
            }
        }

        if inserted_logs > 0 && max_trace_rows > 0 {
            let count: i64 = conn
                .prepare_cached("SELECT COUNT(*) FROM request_logs")
                .map_err(storage_err)?
                .query_row([], |r| r.get(0))
                .unwrap_or(0);
            if count > max_trace_rows as i64 {
                let mut prune_stmt = conn
                    .prepare_cached(
                        "DELETE FROM request_logs WHERE id IN (
                            SELECT id FROM request_logs ORDER BY timestamp_ms DESC LIMIT -1 OFFSET ?1
                         )",
                    )
                    .map_err(storage_err)?;
                prune_stmt
                    .execute(params![max_trace_rows as i64])
                    .map_err(storage_err)?;
            }
        }

        Ok(())
    })();

    match res {
        Ok(()) => {
            conn.execute_batch("RELEASE qg_wb;").map_err(storage_err)?;
            Ok(())
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK TO qg_wb; RELEASE qg_wb;");
            Err(err)
        }
    }
}

// ── Snapshot Builder from Authoritative SQLite ───────────────────────────────

pub fn build_control_snapshot_locked(
    conn: &Connection,
    vault: &Vault,
    settings: GatewaySettings,
    prev: Option<&ControlSnapshot>,
    spend_registry: &SpendRegistry,
) -> Result<ControlSnapshot> {
    // 1. Load scoped_spend into SpendRegistry so model/tag buckets are pre-seeded,
    // and zero any previously cached scope buckets whose rows were deleted.
    {
        let mut seen_scopes = HashSet::new();
        let mut stmt = conn
            .prepare_cached("SELECT scope_key, spend_nanos FROM scoped_spend")
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            .map_err(storage_err)?;
        for row in rows {
            let (scope_key, spend) = row.map_err(storage_err)?;
            seen_scopes.insert(scope_key.clone());
            spend_registry.upsert_authoritative(
                format!("scope:{scope_key}"),
                spend,
                None,
                BudgetWindow::Total,
                None,
            );
        }
        spend_registry.sync_scoped_keys(&seen_scopes);
    }

    // 2. Users
    let mut users_by_id = HashMap::new();
    let mut users_by_username = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, username, display_name, role, password_hash, budget_limit_nanos,
                        budget_window, spend_nanos, window_reset_at_ms, created_at_ms, blocked
                 FROM users",
            )
            .map_err(storage_err)?;
        let rows = stmt.query_map([], map_user_row).map_err(storage_err)?;
        for r in rows {
            let u = r.map_err(storage_err)?;
            let bucket = spend_registry.upsert_authoritative(
                format!("user:{}", u.id),
                u.spend_nanos.nanos(),
                u.window_reset_at_ms,
                u.budget_window,
                Some(("users", "id", u.id.clone())),
            );
            let cached = Arc::new(CachedUser {
                limiter_budget_scope: Arc::from(format!("budget:user:{}", u.id)),
                label: format!("user '{}'", u.username),
                user: u,
                spend_bucket: bucket,
            });
            users_by_id.insert(cached.user.id.clone(), Arc::clone(&cached));
            users_by_username.insert(cached.user.username.clone(), cached);
        }
    }

    // 3. Teams + team_members
    let mut member_map: HashMap<String, HashSet<String>> = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached("SELECT team_id, user_id FROM team_members")
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(storage_err)?;
        for r in rows {
            let (tid, uid) = r.map_err(storage_err)?;
            member_map.entry(tid).or_default().insert(uid);
        }
    }

    let mut teams_by_id = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, name, budget_limit_nanos, budget_window, spend_nanos,
                        window_reset_at_ms, blocked, disable_logging, allowed_models_json,
                        tags_json, model_budgets_json
                 FROM teams",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<i64>>(2)?.map(NanoUsd),
                    BudgetWindow::parse(&r.get::<_, String>(3)?),
                    r.get::<_, i64>(4)?,
                    r.get::<_, Option<i64>>(5)?,
                    r.get::<_, i32>(6).unwrap_or(0) != 0,
                    r.get::<_, i32>(7).unwrap_or(0) != 0,
                    r.get::<_, String>(8).unwrap_or_else(|_| "[]".to_owned()),
                    r.get::<_, String>(9).unwrap_or_else(|_| "[]".to_owned()),
                    r.get::<_, String>(10).unwrap_or_else(|_| "{}".to_owned()),
                ))
            })
            .map_err(storage_err)?;
        for r in rows {
            let (
                id,
                name,
                budget_limit_nanos,
                window,
                spend,
                reset_at,
                blocked,
                disable_logging,
                allowed_json,
                tags_json,
                mb_json,
            ) = r.map_err(storage_err)?;
            let bucket = spend_registry.upsert_authoritative(
                format!("team:{id}"),
                spend,
                reset_at,
                window,
                Some(("teams", "id", id.clone())),
            );
            let cached = Arc::new(CachedTeam {
                limiter_budget_scope: Arc::from(format!("budget:team:{id}")),
                label: format!("team '{name}'"),
                member_ids: member_map.remove(&id).unwrap_or_default(),
                allowed_models: serde_json::from_str(&allowed_json).unwrap_or_default(),
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                model_budgets_nanos: parse_model_budgets_json(&mb_json),
                id: id.clone(),
                name,
                blocked,
                disable_logging,
                budget_limit_nanos,
                spend_bucket: bucket,
            });
            teams_by_id.insert(id, cached);
        }
    }

    // 4. Virtual Keys + Rotation Grace Hashes
    let mut keys_by_sha256 = HashMap::new();
    let mut keys_by_id = HashMap::new();
    {
        let sql = format!("SELECT {VIRTUAL_KEY_COLUMNS}, key_sha256 FROM virtual_keys");
        let mut stmt = conn.prepare_cached(&sql).map_err(storage_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((map_virtual_key_row(row)?, row.get::<_, Vec<u8>>(21)?))
            })
            .map_err(storage_err)?;
        for r in rows {
            let (vk, raw_digest) = r.map_err(storage_err)?;
            let Ok(digest): std::result::Result<[u8; 32], _> = raw_digest.try_into() else {
                continue;
            };
            let bucket = spend_registry.upsert_authoritative(
                format!("vk:{}", vk.id),
                vk.spend_nanos.nanos(),
                vk.window_reset_at_ms,
                vk.budget_window,
                Some(("virtual_keys", "id", vk.id.clone())),
            );
            let cached = Arc::new(CachedVirtualKey {
                rate_scope: Arc::from(format!("key:{}", vk.id)),
                limiter_budget_scope: Arc::from(format!("budget:key:{}", vk.id)),
                label: format!("virtual key '{}'", vk.name),
                key_sha256: digest,
                vk,
                spend_bucket: bucket,
            });
            keys_by_sha256.insert(digest, Arc::clone(&cached));
            keys_by_id.insert(cached.vk.id.clone(), cached);
        }
    }

    let mut rotated_keys_by_sha256 = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached(
                "SELECT old_key_sha256, virtual_key_id, valid_until_ms FROM virtual_key_rotations",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, Vec<u8>>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })
            .map_err(storage_err)?;
        for r in rows {
            let (raw_dig, vk_id, until_ms) = r.map_err(storage_err)?;
            if let Ok(dig) = raw_dig.try_into() {
                rotated_keys_by_sha256.insert(dig, (vk_id, until_ms));
            }
        }
    }

    // 5. Model Routes
    let mut routes_by_alias = HashMap::new();
    {
        let sql = format!("SELECT {MODEL_ROUTE_COLUMNS} FROM model_routes");
        let mut stmt = conn.prepare_cached(&sql).map_err(storage_err)?;
        let rows = stmt
            .query_map([], map_model_route_row)
            .map_err(storage_err)?;
        for r in rows {
            let route = r.map_err(storage_err)?;
            routes_by_alias.insert(route.alias.clone(), Arc::new(route));
        }
    }

    // 6. Access Groups
    let mut access_groups = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached("SELECT access_group, model_names_json FROM access_groups")
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(storage_err)?;
        for r in rows {
            let (ag, models_json) = r.map_err(storage_err)?;
            let models: Vec<String> = serde_json::from_str(&models_json).unwrap_or_default();
            access_groups.insert(ag, models);
        }
    }

    // 7. Budgets + End-Users + Tag Limits
    let mut budgets_max: HashMap<String, i64> = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached("SELECT budget_id, max_budget_nanos FROM budgets")
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<i64>>(1)?))
            })
            .map_err(storage_err)?;
        for r in rows {
            let (bid, max_b) = r.map_err(storage_err)?;
            if let Some(m) = max_b {
                budgets_max.insert(bid, m);
            }
        }
    }

    let mut end_users_by_id = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached(
                "SELECT user_id, alias, blocked, budget_id, budget_limit_nanos, budget_window,
                        spend_nanos, window_reset_at_ms, allowed_models_json, metadata_json,
                        created_at_ms, updated_at_ms
                 FROM end_users",
            )
            .map_err(storage_err)?;
        let rows = stmt.query_map([], map_end_user_row).map_err(storage_err)?;
        for r in rows {
            let eu = r.map_err(storage_err)?;
            let linked = eu
                .budget_id
                .as_deref()
                .and_then(|bid| budgets_max.get(bid).copied())
                .map(NanoUsd);
            let eff_limit = eu.budget_limit_nanos.or(linked);
            let bucket = spend_registry.upsert_authoritative(
                format!("eu:{}", eu.user_id),
                eu.spend_nanos.nanos(),
                eu.window_reset_at_ms,
                eu.budget_window,
                Some(("end_users", "user_id", eu.user_id.clone())),
            );
            let cached = Arc::new(CachedEndUser {
                limiter_budget_scope: Arc::from(format!("budget:end_user:{}", eu.user_id)),
                label: format!("end_user '{}'", eu.user_id),
                user_id: eu.user_id.clone(),
                blocked: eu.blocked,
                allowed_models: eu.allowed_models,
                effective_limit_nanos: eff_limit,
                spend_bucket: bucket,
            });
            end_users_by_id.insert(eu.user_id, cached);
        }
    }

    let mut tag_limits_nanos = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached(
                "SELECT t.tag_name, COALESCE(t.max_budget_nanos, b.max_budget_nanos)
                 FROM tags t
                 LEFT JOIN budgets b ON t.budget_id = b.budget_id",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<i64>>(1)?))
            })
            .map_err(storage_err)?;
        for r in rows {
            let (tname, lim) = r.map_err(storage_err)?;
            if let Some(l) = lim {
                tag_limits_nanos.insert(tname, l);
            }
        }
    }

    // 8. Credentials + Pre-Decrypted Secrets (reusing unchanged decrypted blobs!)
    let mut credentials = Vec::new();
    let mut credentials_by_id = HashMap::new();
    {
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, provider_id, label, auth_kind, base_url, priority, enabled,
                        allow_api_overflow, status, status_reason, cooldown_until_ms,
                        model_cooldowns_json, quota_snapshot_json, expires_at_ms,
                        last_used_at_ms, created_at_ms, encrypted_payload
                 FROM credentials
                 ORDER BY priority ASC, created_at_ms ASC",
            )
            .map_err(storage_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((map_credential_row(row)?, row.get::<_, Vec<u8>>(16)?))
            })
            .map_err(storage_err)?;
        for r in rows {
            let (rec, blob) = r.map_err(storage_err)?;
            let reused_secret = prev
                .and_then(|p| p.credentials_by_id.get(&rec.id))
                .filter(|old| old.encrypted_blob == blob && old.secret.is_some())
                .and_then(|old| old.secret.clone());
            let secret = match reused_secret {
                Some(s) => Some(s),
                None => vault
                    .decrypt_json::<CredentialSecret>(&rec.id, &blob)
                    .ok()
                    .map(Arc::new),
            };
            let lu = rec.last_used_at_ms.unwrap_or(0);
            let cached = Arc::new(CachedCredential {
                record: rec,
                encrypted_blob: blob,
                secret,
                last_used_at_ms: AtomicI64::new(lu),
            });
            credentials_by_id.insert(cached.record.id.clone(), Arc::clone(&cached));
            credentials.push(cached);
        }
    }

    Ok(ControlSnapshot {
        generation: NEXT_SNAPSHOT_GENERATION.fetch_add(1, Ordering::Relaxed),
        settings: Arc::new(settings),
        users_by_id,
        users_by_username,
        teams_by_id,
        keys_by_sha256,
        keys_by_id,
        rotated_keys_by_sha256,
        routes_by_alias,
        access_groups,
        end_users_by_id,
        tag_limits_nanos,
        credentials,
        credentials_by_id,
    })
}
