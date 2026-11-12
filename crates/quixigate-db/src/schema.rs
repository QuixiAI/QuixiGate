//! SQLite WAL schema initialization for QuixiGate.

use quixigate_core::{QuixigateError, Result, now_ms};
use rusqlite::{Connection, params};

pub const SCHEMA_SQL: &str = r#"
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY,
    username            TEXT NOT NULL UNIQUE,
    display_name        TEXT NOT NULL,
    role                TEXT NOT NULL CHECK (role IN ('admin', 'member', 'viewer')),
    password_hash       TEXT,
    blocked             INTEGER NOT NULL DEFAULT 0,
    budget_limit_nanos  INTEGER,
    budget_window       TEXT NOT NULL DEFAULT 'monthly' CHECK (budget_window IN ('hourly','daily','weekly','monthly','total')),
    spend_nanos         INTEGER NOT NULL DEFAULT 0,
    window_reset_at_ms  INTEGER,
    created_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_sha256        BLOB PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at_ms       INTEGER NOT NULL,
    created_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    budget_limit_nanos  INTEGER,
    budget_window       TEXT NOT NULL DEFAULT 'monthly' CHECK (budget_window IN ('hourly','daily','weekly','monthly','total')),
    spend_nanos         INTEGER NOT NULL DEFAULT 0,
    window_reset_at_ms  INTEGER,
    tags_json           TEXT NOT NULL DEFAULT '[]',
    model_budgets_json  TEXT NOT NULL DEFAULT '{}',
    created_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS team_members (
    team_id             TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role                TEXT NOT NULL DEFAULT 'member',
    PRIMARY KEY (team_id, user_id)
);

CREATE TABLE IF NOT EXISTS virtual_keys (
    id                  TEXT PRIMARY KEY,
    key_prefix          TEXT NOT NULL,
    key_sha256          BLOB NOT NULL UNIQUE,
    name                TEXT NOT NULL,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    team_id             TEXT REFERENCES teams(id) ON DELETE SET NULL,
    budget_limit_nanos  INTEGER,
    budget_window       TEXT NOT NULL DEFAULT 'monthly' CHECK (budget_window IN ('hourly','daily','weekly','monthly','total')),
    spend_nanos         INTEGER NOT NULL DEFAULT 0,
    window_reset_at_ms  INTEGER,
    rpm_limit           INTEGER,
    tpm_limit           INTEGER,
    max_concurrent      INTEGER,
    allowed_models_json TEXT NOT NULL DEFAULT '[]',
    tags_json           TEXT NOT NULL DEFAULT '[]',
    model_budgets_json  TEXT NOT NULL DEFAULT '{}',
    allowed_cidrs_json  TEXT NOT NULL DEFAULT '[]',
    allow_spend_read    INTEGER NOT NULL DEFAULT 1,
    log_payloads        INTEGER,
    expires_at_ms       INTEGER,
    revoked_at_ms       INTEGER,
    created_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS virtual_key_rotations (
    old_key_sha256      BLOB PRIMARY KEY,
    virtual_key_id      TEXT NOT NULL REFERENCES virtual_keys(id) ON DELETE CASCADE,
    valid_until_ms      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS credentials (
    id                   TEXT PRIMARY KEY,
    provider_id          TEXT NOT NULL,
    label                TEXT NOT NULL,
    auth_kind            TEXT NOT NULL CHECK (auth_kind IN ('oauth', 'setup_token', 'api_key', 'local_noauth')),
    base_url             TEXT,
    priority             INTEGER NOT NULL DEFAULT 100,
    enabled              INTEGER NOT NULL DEFAULT 1,
    allow_api_overflow   INTEGER NOT NULL DEFAULT 1,
    encrypted_payload    BLOB NOT NULL,
    status               TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'exhausted', 'dead')),
    status_reason        TEXT,
    cooldown_until_ms    INTEGER,
    model_cooldowns_json TEXT NOT NULL DEFAULT '{}',
    quota_snapshot_json  TEXT,
    expires_at_ms        INTEGER,
    last_used_at_ms      INTEGER,
    created_at_ms        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_credentials_provider ON credentials(provider_id, priority ASC);

CREATE TABLE IF NOT EXISTS model_routes (
    alias                  TEXT PRIMARY KEY,
    targets_json           TEXT NOT NULL,
    pool_strategy          TEXT NOT NULL DEFAULT 'fill_first',
    allow_paid_overflow    INTEGER NOT NULL DEFAULT 1,
    input_nanos_per_m      INTEGER,
    output_nanos_per_m     INTEGER,
    cache_read_nanos_per_m INTEGER,
    team_id                TEXT,
    created_by             TEXT,
    updated_by             TEXT,
    created_at_ms          INTEGER,
    updated_at_ms          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key                 TEXT PRIMARY KEY,
    value_json          TEXT NOT NULL,
    updated_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS response_cache (
    cache_key           TEXT PRIMARY KEY,
    model               TEXT NOT NULL,
    response_json       TEXT NOT NULL,
    expires_at_ms       INTEGER NOT NULL,
    created_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS request_logs (
    id                  TEXT PRIMARY KEY,
    timestamp_ms        INTEGER NOT NULL,
    ingress_protocol    TEXT NOT NULL,
    requested_model     TEXT NOT NULL,
    resolved_model      TEXT NOT NULL,
    provider_id         TEXT NOT NULL,
    credential_id       TEXT,
    user_id             TEXT,
    team_id             TEXT,
    virtual_key_id      TEXT,
    tags_json           TEXT NOT NULL DEFAULT '[]',
    status_code         INTEGER NOT NULL,
    stream              INTEGER NOT NULL,
    fallback_attempts   INTEGER NOT NULL DEFAULT 0,
    cache_hit           INTEGER NOT NULL DEFAULT 0,
    ttft_ms             INTEGER,
    duration_ms         INTEGER NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cost_nanos          INTEGER NOT NULL DEFAULT 0,
    api_equiv_nanos     INTEGER NOT NULL DEFAULT 0,
    otel_trace_id       TEXT,
    error_message       TEXT,
    request_body_json   TEXT,
    response_body_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_request_logs_ts ON request_logs(timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_request_logs_user ON request_logs(user_id, timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_request_logs_key ON request_logs(virtual_key_id, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS scoped_spend (
    scope_key           TEXT PRIMARY KEY,
    spend_nanos         INTEGER NOT NULL DEFAULT 0,
    updated_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id                  TEXT PRIMARY KEY,
    timestamp_ms        INTEGER NOT NULL,
    actor_id            TEXT NOT NULL,
    actor_role          TEXT NOT NULL,
    action              TEXT NOT NULL,
    table_name          TEXT NOT NULL,
    object_id           TEXT NOT NULL,
    before_value_json   TEXT,
    updated_values_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_ts ON audit_logs(timestamp_ms DESC);
"#;

fn ensure_column(conn: &Connection, table: &str, column: &str, ddl: &str) -> Result<()> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|e| QuixigateError::Storage(format!("PRAGMA table_info({table}) failed: {e}")))?;
    let cols = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| QuixigateError::Storage(format!("reading columns for {table} failed: {e}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| QuixigateError::Storage(format!("reading columns for {table} failed: {e}")))?;
    if !cols.iter().any(|c| c.eq_ignore_ascii_case(column)) {
        conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {ddl};"))
            .map_err(|e| {
                QuixigateError::Storage(format!("migration ALTER TABLE {table}.{column} failed: {e}"))
            })?;
    }
    Ok(())
}

pub fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;",
    )
    .map_err(|e| QuixigateError::Storage(format!("SQLite pragma setup failed: {e}")))?;

    conn.execute_batch(SCHEMA_SQL)
        .map_err(|e| QuixigateError::Storage(format!("SQLite schema migration failed: {e}")))?;

    // Idempotent online migrations for upgraded SQLite databases
    ensure_column(conn, "users", "blocked", "INTEGER NOT NULL DEFAULT 0")?;
    ensure_column(conn, "teams", "tags_json", "TEXT NOT NULL DEFAULT '[]'")?;
    ensure_column(conn, "teams", "model_budgets_json", "TEXT NOT NULL DEFAULT '{}'")?;
    ensure_column(conn, "virtual_keys", "tags_json", "TEXT NOT NULL DEFAULT '[]'")?;
    ensure_column(conn, "virtual_keys", "model_budgets_json", "TEXT NOT NULL DEFAULT '{}'")?;
    ensure_column(conn, "virtual_keys", "allowed_cidrs_json", "TEXT NOT NULL DEFAULT '[]'")?;
    ensure_column(conn, "virtual_keys", "allow_spend_read", "INTEGER NOT NULL DEFAULT 1")?;
    ensure_column(conn, "model_routes", "team_id", "TEXT")?;
    ensure_column(conn, "model_routes", "created_by", "TEXT")?;
    ensure_column(conn, "model_routes", "updated_by", "TEXT")?;
    ensure_column(conn, "model_routes", "created_at_ms", "INTEGER")?;
    ensure_column(conn, "request_logs", "tags_json", "TEXT NOT NULL DEFAULT '[]'")?;

    let now = now_ms();
    conn.execute(
        "INSERT OR IGNORE INTO users (id, username, display_name, role, created_at_ms)
         VALUES ('local', 'local', 'Local Operator', 'admin', ?1)",
        params![now],
    )
    .map_err(|e| QuixigateError::Storage(format!("failed seeding local user: {e}")))?;

    Ok(())
}
