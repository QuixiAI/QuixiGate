//! `quixigate` — Pure-Rust AI Gateway CLI & Embedded Single-Binary Server.

#[cfg(feature = "demo")]
mod demo_upstreams;

use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use directories::BaseDirs;
#[cfg(feature = "subscription-oauth")]
use quixigate_auth::providers::validate_anthropic_setup_token;
use quixigate_auth::providers::{
    OPENROUTER_OIDC_TOKEN_URL, exchange_openrouter_oidc_token, inspect_openrouter_account,
    openrouter_base_url_for_region,
};
use quixigate_auth::quota::{QuotaEndpoints, poll_and_save_quota};
use quixigate_auth::refresh::CredentialRefresher;
use quixigate_auth::{OAuthWizardManager, StartOAuthRequest, WizardStatus};
use quixigate_core::{AuthKind, BudgetWindow, CredentialSecret, NanoUsd};
use quixigate_db::Database;
use quixigate_providers::local_discovery::{
    LocalEndpointSpec, LocalRuntimeScanner, normalize_local_runtime_id,
};
use quixigate_proxy::admin::{sync_local_scanner_from_db, sync_openrouter_for_state};
use quixigate_proxy::build_router;
use quixigate_proxy::codecs::common::is_catalog_model_routable;
use quixigate_proxy::state::AppState;
use quixigate_router::GatewayRouter;
use quixigate_router::catalog::{MODELS_DEV_URL, ModelCatalog};
use quixigate_telemetry::TelemetryHub;
use quixigate_vault::Vault;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "quixigate",
    version,
    about = "QuixiGate — Pure-Rust Polyglot AI Gateway, Multi-Seat OAuth Pool & OpenInference Proxy"
)]
struct Cli {
    /// Gateway state directory (`vault.key` + `quixigate.db`).
    #[arg(long, global = true, env = "QUIXIGATE_DATA_DIR")]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the QuixiGate HTTP/SSE/WebSocket gateway and optional `/ui` dashboard.
    Serve {
        #[arg(long, default_value = "127.0.0.1", env = "QUIXIGATE_HOST")]
        host: String,
        #[arg(short, long, default_value_t = 4000, env = "QUIXIGATE_PORT")]
        port: u16,
        /// Enforce strict session / key authentication even on 127.0.0.1 loopback.
        #[arg(long)]
        require_auth: bool,
        /// Disable passwordless local container trust (`host.docker.internal` & local Docker/Podman/OrbStack bridges).
        #[arg(long)]
        no_trust_containers: bool,
        /// Additional CIDR/IP ranges permitted without an API key when `--require-auth` is not set (e.g. `100.64.0.0/10`).
        #[arg(
            long = "trusted-cidr",
            value_delimiter = ',',
            env = "QUIXIGATE_TRUSTED_CIDRS"
        )]
        trusted_cidrs: Vec<String>,
        /// Master Key for remote & admin management (falls back to `LITELLM_MASTER_KEY`; auto-generated and printed to stdout if unset).
        #[arg(long, env = "QUIXIGATE_MASTER_KEY")]
        master_key: Option<String>,
        /// Explicitly enable the optional `/ui` convenience Web UI.
        #[arg(long, conflicts_with = "no_ui")]
        ui: bool,
        /// Disable the `/ui` Web UI at runtime (LiteLLM parity: `--disable-admin-ui` / `DISABLE_ADMIN_UI=true`).
        #[arg(long, visible_alias = "disable-admin-ui")]
        no_ui: bool,
        /// Open `/ui` (or `/docs` if UI is disabled) in the default browser after startup.
        #[arg(long)]
        open: bool,
        /// Optional override URL for local Ollama discovery.
        #[arg(long, env = "QUIXIGATE_OLLAMA_URL")]
        ollama_url: Option<String>,
        /// Optional override URL for local llama.cpp (`llama-server`) discovery.
        #[arg(long, env = "QUIXIGATE_LLAMACPP_URL")]
        llamacpp_url: Option<String>,
        /// Optional override URL for local LM Studio discovery.
        #[arg(long, env = "QUIXIGATE_LMSTUDIO_URL")]
        lmstudio_url: Option<String>,
    },

    /// Authenticate a Local Runtime, OAuth Subscription seat, OpenRouter (PKCE / OIDC / Key / Env), or Cloud API key.
    Login {
        /// Optional Provider ID (e.g. `llamacpp`, `ollama`, `openai-codex`, `anthropic-token`, `google-gemini-cli`, `github-copilot`, `openrouter`; omit for interactive 3-pillar menu).
        provider: Option<String>,
        /// Human-readable seat label.
        #[arg(short, long)]
        label: Option<String>,
        /// Prefer RFC 8628 headless device-code flow when supported.
        #[arg(long)]
        device: bool,
        /// Store a static API key in the AES-256-GCM vault.
        #[arg(long)]
        api_key: Option<String>,
        /// Import API key from environment variable (defaults to `OPENROUTER_API_KEY` or `<PROVIDER>_API_KEY`).
        #[arg(long)]
        from_env: bool,
        /// Explicit environment variable name to read when `--from-env` is set.
        #[arg(long)]
        env_var: Option<String>,
        /// OpenRouter RFC 8693 OIDC subject JWT for Workload Identity exchange.
        #[arg(long)]
        oidc_token: Option<String>,
        /// Environment variable holding an OpenRouter OIDC subject JWT (auto-refreshed before expiry).
        #[arg(long)]
        oidc_token_env: Option<String>,
        /// Optional OpenRouter OIDC federation policy ID.
        #[arg(long)]
        federation_policy_id: Option<String>,
        /// Optional OIDC token endpoint override (defaults to `https://openrouter.ai/api/v1/oauth/token`).
        #[arg(long)]
        token_endpoint: Option<String>,
        /// Store an Anthropic `claude setup-token` (`sk-ant-oat01-...`).
        #[arg(long)]
        setup_token: Option<String>,
        /// Custom upstream base URL override.
        #[arg(long)]
        base_url: Option<String>,
        /// Seat priority (lower = drained first; Local/OAuth defaults to 50, OpenRouter/API keys to 200).
        #[arg(long)]
        priority: Option<i32>,
        /// Region (`global`/`us`/`eu` for OpenRouter; `global`/`cn` for MiniMax).
        #[arg(long)]
        region: Option<String>,
        /// GitHub Enterprise domain for Copilot.
        #[arg(long)]
        enterprise_domain: Option<String>,
    },

    /// Display seat health, subscription quota windows, local runtimes, and `NanoUsd` savings.
    Status,

    /// Launch a coding CLI (`claude`, `codex`, `gemini`, `opencode`) wired to QuixiGate via scoped child-process env vars only (zero dotfile mutation).
    Run {
        /// Target agent (`claude`, `codex`, `gemini`, `opencode`, or binary path).
        agent: String,
        /// Existing QuixiGate URL (starts an ephemeral in-process gateway if unreachable).
        #[arg(long, default_value = "http://127.0.0.1:4000", env = "QUIXIGATE_URL")]
        gateway: String,
        /// Optional model override passed to the child environment.
        #[arg(short, long)]
        model: Option<String>,
        /// Optional `sk-qx-...` virtual key token (defaults to `sk-qx-loopback` on loopback).
        #[arg(short, long, env = "QUIXIGATE_KEY")]
        key: Option<String>,
        /// Arguments forwarded directly to the child CLI process.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Search and inspect routable models across Local Runtimes, OAuth Subscriptions, OpenRouter `@preset/*`, and custom routes.
    Models {
        /// Optional search substring (`claude`, `@preset/`, `qwen`, etc.).
        query: Option<String>,
        /// Refresh the catalog from `https://models.dev/api.json` (and OpenRouter if connected).
        #[arg(long)]
        sync: bool,
        /// Include unconfigured reference catalog models (`models.dev`), not just routable models.
        #[arg(long)]
        all: bool,
        #[arg(short, long, default_value_t = 40)]
        limit: usize,
    },

    /// Manage `sk-qx-...` virtual keys, budgets, and zero-downtime key rotation.
    Keys {
        #[command(subcommand)]
        action: KeysAction,
    },

    /// Export or import declarative TOML / YAML gateway config with `${ENV}` expansion.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Manage 1-click Local/Subscription -> OpenRouter Cloud Burst mode and attribution headers.
    Burst {
        #[command(subcommand)]
        action: Option<BurstAction>,
        /// Live gateway URL to hot-reload settings if running.
        #[arg(long, default_value = "http://127.0.0.1:4000", env = "QUIXIGATE_URL")]
        gateway: String,
    },

    /// Run self-diagnostics on vault permissions, SQLite WAL health, seats, and local ports.
    Doctor {
        #[arg(long, default_value_t = 4000)]
        port: u16,
    },

    #[cfg(feature = "demo")]
    #[command(hide = true, name = "__demo-upstreams")]
    DemoUpstreams {
        #[arg(long, default_value_t = 4100)]
        port: u16,
        #[arg(long, default_value_t = 6006)]
        otlp_port: u16,
        #[arg(long)]
        seed_data_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum KeysAction {
    /// List all `sk-qx-...` virtual keys and current `NanoUsd` spend.
    List,
    /// Mint a new `sk-qx-...` virtual key.
    Create {
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "local")]
        user: String,
        #[arg(long)]
        team: Option<String>,
        #[arg(long)]
        budget_usd: Option<String>,
        #[arg(long, default_value = "monthly")]
        window: String,
        #[arg(long)]
        rpm: Option<u32>,
        #[arg(long)]
        tpm: Option<u64>,
        #[arg(long)]
        max_concurrent: Option<u32>,
        #[arg(long = "model")]
        allowed_models: Vec<String>,
    },
    /// Rotate a key secret while keeping the previous token valid during `grace_seconds`.
    Rotate {
        id: String,
        #[arg(long, default_value_t = 300)]
        grace_seconds: u64,
    },
    /// Immediately revoke a virtual key.
    Revoke { id: String },
    /// Permanently delete a virtual key record.
    Delete { id: String },
}

#[derive(Subcommand, Debug)]
enum ConfigAction {
    /// Export gateway settings, seats, and routes as TOML or YAML.
    Export {
        #[arg(short, long, default_value = "toml")]
        format: String,
        #[arg(long)]
        include_secrets: bool,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Import a TOML or YAML config file with `${ENV_VAR}` expansion.
    Import {
        path: PathBuf,
        #[arg(short, long)]
        format: Option<String>,
    },
}

#[derive(Subcommand, Debug, Clone)]
enum BurstAction {
    /// Enable automatic OpenRouter Cloud Burst when local GPUs or subscription seats hit 429/5xx.
    Enable {
        /// Fallback OpenRouter model/router or @preset/* (defaults to `openrouter/auto`).
        #[arg(short, long)]
        model: Option<String>,
    },
    /// Disable automatic OpenRouter Cloud Burst (strict local/subscription isolation).
    Disable,
    /// Show current Cloud Burst status, OpenRouter seat health, and attribution headers.
    Status,
}

fn resolve_data_dir(cli_dir: Option<PathBuf>) -> PathBuf {
    if let Some(d) = cli_dir {
        return d;
    }
    if let Some(base) = BaseDirs::new() {
        return base.home_dir().join(quixigate_core::brand::DATA_DIR_NAME);
    }
    PathBuf::from(quixigate_core::brand::DATA_DIR_NAME)
}

fn open_db(data_dir: &Path) -> anyhow::Result<(Vault, Database)> {
    let vault = Vault::load_or_init(data_dir)
        .with_context(|| format!("failed to initialize vault in {}", data_dir.display()))?;
    let db_path = data_dir.join(quixigate_core::brand::DB_FILE_NAME);
    let db = Database::open(&db_path, vault.clone())
        .with_context(|| format!("failed to open SQLite database {}", db_path.display()))?;
    Ok((vault, db))
}

fn build_quota_endpoints() -> QuotaEndpoints {
    if let Some(base) = quixigate_core::brand::env_opt("QUIXIGATE_QUOTA_BASE_URL") {
        let b = base.trim_end_matches('/');
        return QuotaEndpoints {
            codex_wham_usage_url: format!("{b}/wham/usage"),
            anthropic_oauth_usage_url: format!("{b}/api/oauth/usage"),
            gemini_quota_url: format!("{b}/v1internal:retrieveUserQuota"),
            openrouter_credits_url: format!("{b}/api/v1/credits"),
        };
    }
    QuotaEndpoints::default()
}

fn assemble_gateway_state(
    db: Database,
    bound_loopback: bool,
    ollama_override: Option<&str>,
    llamacpp_override: Option<&str>,
    lmstudio_override: Option<&str>,
) -> AppState {
    let refresher = CredentialRefresher::new(db.clone());
    let catalog = ModelCatalog::embedded();
    let scanner = LocalRuntimeScanner::new();
    sync_local_scanner_from_db(&db, &scanner);
    if let Some(url) = ollama_override.filter(|u| !u.trim().is_empty()) {
        scanner.add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: "ollama".to_owned(),
            display_name: "Ollama".to_owned(),
            base_url: url.trim().to_owned(),
        });
    }
    if let Some(url) = llamacpp_override.filter(|u| !u.trim().is_empty()) {
        scanner.add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: "llamacpp".to_owned(),
            display_name: "llama.cpp Server".to_owned(),
            base_url: url.trim().to_owned(),
        });
    }
    if let Some(url) = lmstudio_override.filter(|u| !u.trim().is_empty()) {
        scanner.add_or_update_endpoint(LocalEndpointSpec {
            runtime_id: "lmstudio".to_owned(),
            display_name: "LM Studio".to_owned(),
            base_url: url.trim().to_owned(),
        });
    }
    let mut router = GatewayRouter::new(db.clone(), refresher.clone(), catalog, scanner);
    if let Some(search_url) = quixigate_core::brand::env_opt("QUIXIGATE_WEB_SEARCH_URL") {
        router = router.with_web_search_url(&search_url);
    }
    let telemetry = TelemetryHub::new();
    let quota_endpoints = build_quota_endpoints();
    let wizard = OAuthWizardManager::new(db.clone(), refresher)
        .with_quota_endpoints(quota_endpoints.clone());
    AppState::new(db, router, telemetry, wizard, bound_loopback)
        .with_quota_endpoints(quota_endpoints)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("quixigate=info")),
        )
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();
    let data_dir = resolve_data_dir(cli.data_dir);

    match cli.command {
        #[cfg(feature = "demo")]
        Commands::DemoUpstreams {
            port,
            otlp_port,
            seed_data_dir,
        } => {
            demo_upstreams::run_demo_servers(port, otlp_port, seed_data_dir.as_deref()).await?;
        }
        Commands::Serve {
            host,
            port,
            require_auth,
            no_trust_containers,
            trusted_cidrs,
            master_key,
            ui,
            no_ui,
            open,
            ollama_url,
            llamacpp_url,
            lmstudio_url,
        } => {
            cmd_serve(
                &data_dir,
                &host,
                port,
                require_auth,
                no_trust_containers,
                &trusted_cidrs,
                master_key.as_deref(),
                ui,
                no_ui,
                open,
                ollama_url.as_deref(),
                llamacpp_url.as_deref(),
                lmstudio_url.as_deref(),
            )
            .await?;
        }
        Commands::Login {
            provider,
            label,
            device,
            api_key,
            from_env,
            env_var,
            oidc_token,
            oidc_token_env,
            federation_policy_id,
            token_endpoint,
            setup_token,
            base_url,
            priority,
            region,
            enterprise_domain,
        } => {
            let (_vault, db) = open_db(&data_dir)?;
            cmd_login(
                db,
                LoginOptions {
                    provider,
                    label,
                    device,
                    api_key,
                    from_env,
                    env_var,
                    oidc_token,
                    oidc_token_env,
                    federation_policy_id,
                    token_endpoint,
                    setup_token,
                    base_url,
                    priority,
                    region,
                    enterprise_domain,
                },
            )
            .await?;
        }
        Commands::Status => {
            let (vault, db) = open_db(&data_dir)?;
            cmd_status(&data_dir, &vault, &db).await?;
        }
        Commands::Run {
            agent,
            gateway,
            model,
            key,
            args,
        } => {
            let code = cmd_run(
                &data_dir,
                &agent,
                &gateway,
                model.as_deref(),
                key.as_deref(),
                &args,
            )
            .await?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Commands::Models {
            query,
            sync,
            all,
            limit,
        } => {
            let (_vault, db) = open_db(&data_dir)?;
            cmd_models(&db, query.as_deref(), sync, all, limit).await?;
        }
        Commands::Keys { action } => {
            let (_vault, db) = open_db(&data_dir)?;
            cmd_keys(&db, action)?;
        }
        Commands::Config { action } => {
            let (_vault, db) = open_db(&data_dir)?;
            cmd_config(&db, action)?;
        }
        Commands::Burst { action, gateway } => {
            let (_vault, db) = open_db(&data_dir)?;
            cmd_burst(&db, action.unwrap_or(BurstAction::Status), &gateway).await?;
        }
        Commands::Doctor { port } => {
            cmd_doctor(&data_dir, port).await?;
        }
    }

    Ok(())
}

fn resolve_ui_override(cli_ui: bool, cli_no_ui: bool) -> Option<bool> {
    if cli_no_ui {
        return Some(false);
    }
    if cli_ui {
        return Some(true);
    }
    if let Ok(disable_raw) = std::env::var("DISABLE_ADMIN_UI") {
        let v = disable_raw.trim().to_ascii_lowercase();
        if matches!(v.as_str(), "1" | "true" | "yes") {
            return Some(false);
        }
        if matches!(v.as_str(), "0" | "false" | "no") {
            return Some(true);
        }
    }
    if let Ok(enabled_raw) = std::env::var("QUIXIGATE_UI_ENABLED") {
        let v = enabled_raw.trim().to_ascii_lowercase();
        if matches!(v.as_str(), "1" | "true" | "yes") {
            return Some(true);
        }
        if matches!(v.as_str(), "0" | "false" | "no") {
            return Some(false);
        }
    }
    None
}

fn resolve_master_key(cli_or_quixigate_env: Option<&str>) -> (String, bool) {
    if let Some(k) = cli_or_quixigate_env
        .map(str::trim)
        .filter(|k| !k.is_empty())
    {
        return (k.to_owned(), false);
    }
    if let Ok(litellm_key) = std::env::var("LITELLM_MASTER_KEY") {
        let trimmed = litellm_key.trim();
        if !trimmed.is_empty() {
            return (trimmed.to_owned(), false);
        }
    }
    (quixigate_proxy::state::generate_master_key(), true)
}

#[allow(clippy::too_many_arguments)]
async fn cmd_serve(
    data_dir: &Path,
    host: &str,
    port: u16,
    require_auth: bool,
    no_trust_containers: bool,
    trusted_cidrs: &[String],
    master_key_arg: Option<&str>,
    cli_ui: bool,
    cli_no_ui: bool,
    open_browser: bool,
    ollama_url: Option<&str>,
    llamacpp_url: Option<&str>,
    lmstudio_url: Option<&str>,
) -> anyhow::Result<()> {
    let (vault, db) = open_db(data_dir)?;
    if require_auth || no_trust_containers || !trusted_cidrs.is_empty() {
        let mut settings = db.get_settings()?;
        let mut changed = false;
        if require_auth && !settings.require_auth {
            settings.require_auth = true;
            changed = true;
        }
        if no_trust_containers && settings.trust_local_containers {
            settings.trust_local_containers = false;
            changed = true;
        }
        for cidr in trusted_cidrs
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            if !settings.trusted_cidrs.iter().any(|c| c == cidr) {
                settings.trusted_cidrs.push(cidr.to_owned());
                changed = true;
            }
        }
        if changed {
            db.update_settings(&settings)?;
        }
    }

    let listener = TcpListener::bind((host, port))
        .await
        .with_context(|| format!("failed to bind {host}:{port}"))?;
    let local_addr: SocketAddr = listener.local_addr()?;
    let bound_loopback = local_addr.ip().is_loopback();
    let (master_key, master_key_generated) = resolve_master_key(master_key_arg);
    let ui_override = resolve_ui_override(cli_ui, cli_no_ui);

    let state = assemble_gateway_state(
        db.clone(),
        bound_loopback,
        ollama_url,
        llamacpp_url,
        lmstudio_url,
    )
    .with_master_key(&master_key)
    .with_ui_override(ui_override);

    if db.get_settings()?.local_discovery_enabled {
        state.router.local_scanner().scan_all().await;
    }

    // If OpenRouter is connected, hydrate live `/models` pricing & `@preset/*` presets on boot.
    if db
        .list_credentials()
        .map(|cs| {
            cs.iter()
                .any(|c| c.enabled && c.provider_id == "openrouter")
        })
        .unwrap_or(false)
    {
        let boot_state = state.clone();
        tokio::spawn(async move {
            let _ = sync_openrouter_for_state(&boot_state).await;
        });
    }

    // Periodic local runtime discovery (every 30s).
    {
        let bg_state = state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(30));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                if bg_state
                    .db
                    .get_settings()
                    .map(|s| s.local_discovery_enabled)
                    .unwrap_or(true)
                {
                    sync_local_scanner_from_db(&bg_state.db, bg_state.router.local_scanner());
                    bg_state.router.local_scanner().scan_all().await;
                }
            }
        });
    }

    // Periodic OAuth subscription quota & OpenRouter credit/@preset sync (every 10m).
    {
        let bg_state = state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(600));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                if let Ok(creds) = bg_state.db.list_credentials() {
                    let mut has_openrouter = false;
                    for c in creds.into_iter().filter(|c| c.enabled) {
                        if c.provider_id == "openrouter" {
                            has_openrouter = true;
                        } else if c.auth_kind.is_zero_marginal_cost() {
                            let _ = poll_and_save_quota(
                                &bg_state.db,
                                bg_state.router.refresher(),
                                &c.id,
                                &bg_state.quota_endpoints,
                            )
                            .await;
                        }
                    }
                    if has_openrouter {
                        let _ = sync_openrouter_for_state(&bg_state).await;
                    }
                }
            }
        });
    }

    let seats = db.list_credentials()?.len();
    let healthy_locals: Vec<String> = state
        .router
        .local_scanner()
        .list_statuses()
        .into_iter()
        .filter(|r| r.healthy)
        .map(|r| format!("{} ({} models)", r.display_name, r.models.len()))
        .collect();

    let base_url = format!("http://{local_addr}");
    let ui_enabled = state.is_ui_enabled();
    println!("╭──────────────────────────────────────────────────────────────╮");
    println!(
        "│  QuixiGate v{} — Pure-Rust Polyglot AI Gateway            │",
        env!("CARGO_PKG_VERSION")
    );
    println!("╰──────────────────────────────────────────────────────────────╯");
    println!("  API & OpenAPI    : {base_url}  ({base_url}/openapi.json)");
    if ui_enabled {
        println!("  Admin Web UI     : {base_url}/ui  (optional convenience UI)");
    } else {
        println!("  Admin Web UI     : disabled (headless Management API mode)");
    }
    println!("  OpenAI Chat/Resp : {base_url}/v1/chat/completions | /v1/responses");
    println!("  Anthropic / Gem  : {base_url}/v1/messages | /v1beta/models/*");
    if master_key_generated {
        println!("  Master Key       : {master_key}  (generated; set QUIXIGATE_MASTER_KEY to pin)");
    } else {
        println!("  Master Key       : configured via QUIXIGATE_MASTER_KEY / LITELLM_MASTER_KEY");
    }
    let cur_settings = db.get_settings()?;
    println!(
        "  Auth Policy      : {}",
        if cur_settings.require_auth {
            "Strict (--require-auth: key or session required on all connections)"
        } else if cur_settings.trust_local_containers {
            "Loopback & local containers passwordless (127.0.0.1/::1, host.docker.internal) · Remote IPs require key"
        } else {
            "Loopback passwordless (127.0.0.1/::1) · Remote & container IPs require key"
        }
    );
    println!(
        "  Data Directory   : {} (vault: {})",
        data_dir.display(),
        vault.source().as_str()
    );
    println!("  Configured Seats : {seats}");
    if !healthy_locals.is_empty() {
        println!("  Local Runtimes   : {}", healthy_locals.join(", "));
    }
    let _ = io::stdout().flush();

    if open_browser && std::env::var("QUIXIGATE_NO_BROWSER").is_err() {
        let target = if ui_enabled {
            format!("{base_url}/ui")
        } else {
            format!("{base_url}/docs")
        };
        let _ = open::that(&target);
    }

    let app = build_router(state);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        println!("\nShutting down QuixiGate cleanly…");
    })
    .await?;

    Ok(())
}

struct LoginOptions {
    provider: Option<String>,
    label: Option<String>,
    device: bool,
    api_key: Option<String>,
    from_env: bool,
    env_var: Option<String>,
    oidc_token: Option<String>,
    oidc_token_env: Option<String>,
    federation_policy_id: Option<String>,
    token_endpoint: Option<String>,
    setup_token: Option<String>,
    base_url: Option<String>,
    priority: Option<i32>,
    region: Option<String>,
    enterprise_domain: Option<String>,
}

fn default_env_var_for_provider(provider_id: &str) -> String {
    match provider_id {
        "openrouter" => "OPENROUTER_API_KEY".to_owned(),
        "openai" => "OPENAI_API_KEY".to_owned(),
        "anthropic" => "ANTHROPIC_API_KEY".to_owned(),
        "google" | "gemini" => "GEMINI_API_KEY".to_owned(),
        "xai" => "XAI_API_KEY".to_owned(),
        other => format!(
            "{}_API_KEY",
            other.to_ascii_uppercase().replace(['-', '.'], "_")
        ),
    }
}

async fn run_oauth_wizard_cli(
    db: Database,
    provider_id: &str,
    label: Option<String>,
    priority: Option<i32>,
    device: bool,
    region: Option<String>,
    enterprise_domain: Option<String>,
) -> anyhow::Result<quixigate_core::CredentialRecord> {
    let refresher = CredentialRefresher::new(db.clone());
    let wizard =
        OAuthWizardManager::new(db, refresher).with_quota_endpoints(build_quota_endpoints());
    let session = wizard
        .start_flow(StartOAuthRequest {
            provider_id: provider_id.to_owned(),
            label,
            priority,
            prefer_device_code: Some(device),
            setup_token: None,
            client_id: None,
            client_secret: None,
            region,
            enterprise_domain,
        })
        .await?;

    if let Some(code) = &session.user_code {
        println!("\n  One-Time Device Code: {code}");
        if let Some(uri) = &session.verification_uri {
            println!("  Verification URL    : {uri}");
            if std::env::var("QUIXIGATE_NO_BROWSER").is_err() {
                let _ = open::that(uri);
            }
        }
    } else if let Some(url) = &session.authorize_url {
        println!("\n  Open Authorization URL:\n  {url}\n");
        if std::env::var("QUIXIGATE_NO_BROWSER").is_err() {
            let _ = open::that(url);
        }
    }

    if session.accepts_manual_code {
        println!("Waiting for localhost callback (or press Ctrl+C to abort)…");
    }

    loop {
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let cur = wizard.get_session(&session.session_id)?;
        match cur.status {
            WizardStatus::Pending => continue,
            WizardStatus::Completed => {
                if let Some(cred) = cur.credential {
                    return Ok(cred);
                }
                bail!("OAuth flow completed without returning a credential record");
            }
            WizardStatus::Failed => {
                bail!(
                    "OAuth flow failed: {}",
                    cur.error.unwrap_or_else(|| "unknown error".to_owned())
                );
            }
        }
    }
}

async fn login_openrouter(db: Database, mut opts: LoginOptions) -> anyhow::Result<()> {
    let effective_base_url = opts
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| openrouter_base_url_for_region(opts.region.as_deref()).to_owned());

    // Determine whether a non-interactive method was explicitly specified via CLI flags
    let mut chosen_mode = if opts.oidc_token.is_some() || opts.oidc_token_env.is_some() {
        Some("oidc")
    } else if opts.from_env {
        Some("env")
    } else if opts.api_key.is_some() {
        Some("api_key")
    } else {
        None
    };

    if chosen_mode.is_none() {
        let existing: Vec<_> = db
            .list_credentials()?
            .into_iter()
            .filter(|c| c.enabled && c.provider_id == "openrouter")
            .collect();
        println!("\nQuixiGate · OpenRouter Connection");
        if let Some(first) = existing.first() {
            println!("Currently using saved OpenRouter seat '{}'.\n", first.label);
        } else {
            println!("OpenRouter is currently not connected (optional).\n");
        }

        let choices = vec![
            "Sign in with your browser (recommended) - Opens openrouter.ai/auth PKCE",
            "Sign in with OpenRouter OIDC - RFC 8693 Workload Identity / JWT exchange",
            "Enter an OpenRouter API key",
            "Use OPENROUTER_API_KEY from the environment",
        ];
        let picked = inquire::Select::new(
            "How would you like to connect to OpenRouter?",
            choices.clone(),
        )
        .prompt()?;
        chosen_mode = Some(if picked == choices[0] {
            "pkce"
        } else if picked == choices[1] {
            "oidc"
        } else if picked == choices[2] {
            "api_key"
        } else {
            "env"
        });
    }

    let http = reqwest::Client::new();
    let catalog = ModelCatalog::embedded();

    match chosen_mode.unwrap_or("pkce") {
        "pkce" => {
            let cred = run_oauth_wizard_cli(
                db.clone(),
                "openrouter",
                opts.label
                    .take()
                    .or_else(|| Some("OpenRouter (Browser PKCE)".to_owned())),
                Some(opts.priority.unwrap_or(200)),
                false,
                opts.region.take(),
                None,
            )
            .await?;
            let refresher = CredentialRefresher::new(db.clone());
            let _ = poll_and_save_quota(&db, &refresher, &cred.id, &build_quota_endpoints()).await;
            let bearer =
                db.get_credential_with_secret(&cred.id)
                    .ok()
                    .and_then(|(_, sec)| match sec {
                        CredentialSecret::ApiKey { key } => Some(key),
                        CredentialSecret::OpenRouterOidc { access, .. } => Some(access),
                        _ => None,
                    });
            let (model_cnt, preset_cnt) = catalog
                .sync_openrouter(&http, &effective_base_url, bearer.as_deref())
                .await
                .unwrap_or((0, 0));
            println!(
                "Connected OpenRouter seat '{}' ({}) at priority #{} · {model_cnt} models · {preset_cnt} @presets synced.",
                cred.label, cred.id, cred.priority
            );
        }
        "env" | "api_key" => {
            let mode = chosen_mode.unwrap_or("api_key");
            let raw_key = if mode == "env" {
                let var_name = opts
                    .env_var
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("OPENROUTER_API_KEY");
                std::env::var(var_name)
                    .ok()
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .with_context(|| {
                        format!("environment variable '{var_name}' is not set or empty")
                    })?
            } else {
                match opts.api_key.take() {
                    Some(k) => k.trim().to_owned(),
                    None => inquire::Password::new("Paste OpenRouter API key (sk-or-v1-...):")
                        .without_confirmation()
                        .prompt()?
                        .trim()
                        .to_owned(),
                }
            };
            if raw_key.is_empty() {
                bail!("OpenRouter API key cannot be empty");
            }
            let account = inspect_openrouter_account(&http, &effective_base_url, &raw_key).await?;
            let lbl = opts.label.or(account.key_label.clone()).unwrap_or_else(|| {
                if mode == "env" {
                    "OpenRouter (Env Key)".to_owned()
                } else {
                    "OpenRouter (API Key)".to_owned()
                }
            });
            let rec = db.upsert_credential(
                None,
                "openrouter",
                &lbl,
                AuthKind::ApiKey,
                Some(&effective_base_url),
                opts.priority.unwrap_or(200),
                true,
                true,
                &CredentialSecret::ApiKey {
                    key: raw_key.clone(),
                },
            )?;
            let refresher = CredentialRefresher::new(db.clone());
            let _ = poll_and_save_quota(&db, &refresher, &rec.id, &build_quota_endpoints()).await;
            let (model_cnt, preset_cnt) = catalog
                .sync_openrouter(&http, &effective_base_url, Some(&raw_key))
                .await
                .unwrap_or((0, 0));
            let credits_note = account
                .remaining_credits
                .map(|rem| format!(" · ${rem:.2} credits"))
                .unwrap_or_default();
            println!(
                "Connected OpenRouter seat '{}' ({}) at priority #{}{} · {model_cnt} models · {preset_cnt} @presets synced.",
                rec.label, rec.id, rec.priority, credits_note
            );
        }
        "oidc" => {
            let env_var = opts
                .oidc_token_env
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            let subject_token = if let Some(t) = opts
                .oidc_token
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                t.to_owned()
            } else if let Some(ref var_name) = env_var {
                std::env::var(var_name)
                    .ok()
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .with_context(|| {
                        format!("OIDC token environment variable '{var_name}' is not set or empty")
                    })?
            } else if let Ok(tok) = std::env::var("OPENROUTER_OIDC_TOKEN")
                && !tok.trim().is_empty()
            {
                tok.trim().to_owned()
            } else {
                inquire::Password::new("Paste OpenRouter OIDC subject JWT:")
                    .without_confirmation()
                    .prompt()?
                    .trim()
                    .to_owned()
            };

            let policy = match opts.federation_policy_id.clone() {
                Some(p) => Some(p),
                None if opts.oidc_token.is_none() && opts.oidc_token_env.is_none() => {
                    let entered =
                        inquire::Text::new("Federation Policy ID (optional, press Enter to skip):")
                            .prompt()
                            .unwrap_or_default();
                    (!entered.trim().is_empty()).then(|| entered.trim().to_owned())
                }
                None => None,
            };
            let token_endpoint = opts
                .token_endpoint
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(OPENROUTER_OIDC_TOKEN_URL);

            let secret = exchange_openrouter_oidc_token(
                &http,
                token_endpoint,
                &subject_token,
                policy.as_deref(),
                env_var.as_deref(),
            )
            .await?;
            let access_tok = match &secret {
                CredentialSecret::OpenRouterOidc { access, .. } => access.clone(),
                _ => String::new(),
            };
            let account = inspect_openrouter_account(&http, &effective_base_url, &access_tok)
                .await
                .unwrap_or_default();
            let lbl = opts
                .label
                .or(account.key_label)
                .unwrap_or_else(|| "OpenRouter (OIDC Workload Identity)".to_owned());
            let rec = db.upsert_credential(
                None,
                "openrouter",
                &lbl,
                AuthKind::Oauth,
                Some(&effective_base_url),
                opts.priority.unwrap_or(200),
                true,
                true,
                &secret,
            )?;
            let refresher = CredentialRefresher::new(db.clone());
            let _ = poll_and_save_quota(&db, &refresher, &rec.id, &build_quota_endpoints()).await;
            let (model_cnt, preset_cnt) = catalog
                .sync_openrouter(&http, &effective_base_url, Some(&access_tok))
                .await
                .unwrap_or((0, 0));
            println!(
                "Connected OpenRouter OIDC seat '{}' ({}) at priority #{} · {model_cnt} models · {preset_cnt} @presets synced.",
                rec.label, rec.id, rec.priority
            );
        }
        other => bail!("unsupported OpenRouter login mode '{other}'"),
    }
    Ok(())
}

async fn cmd_login(db: Database, mut opts: LoginOptions) -> anyhow::Result<()> {
    let raw_provider = match opts.provider.take() {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            let local_pillar =
                "Local Runtime (llama.cpp, Ollama, LM Studio, MLX, vLLM — $0 local inference)";
            #[cfg(feature = "subscription-oauth")]
            let oauth_pillar: Option<&str> = Some(
                "OAuth Subscription (ChatGPT/Codex, Claude Setup Token, Gemini CLI, Copilot, Grok, Qwen, MiniMax, Chutes)",
            );
            #[cfg(not(feature = "subscription-oauth"))]
            let oauth_pillar: Option<&str> = None;
            let or_pillar = "OpenRouter (Browser PKCE, OIDC Workload Identity, API Key, or OPENROUTER_API_KEY env)";
            let direct_pillar =
                "Direct Cloud API Key (OpenAI, Anthropic, Google, xAI, DeepSeek, Groq, ...)";
            let mut pillars: Vec<&str> = vec![local_pillar];
            if let Some(op) = oauth_pillar {
                pillars.push(op);
            }
            pillars.push(or_pillar);
            pillars.push(direct_pillar);
            let chosen =
                inquire::Select::new("Choose a provider pillar to connect to QuixiGate:", pillars)
                    .prompt()?;
            if chosen == local_pillar {
                let runtimes = [
                    (
                        "llamacpp",
                        "llama.cpp Server (default http://127.0.0.1:8080)",
                    ),
                    ("ollama", "Ollama (default http://127.0.0.1:11434)"),
                    ("lmstudio", "LM Studio (default http://127.0.0.1:1234)"),
                    ("mlx", "Apple MLX Server (default http://127.0.0.1:8081)"),
                    ("vllm", "vLLM Server (default http://127.0.0.1:8000)"),
                ];
                let labels: Vec<&str> = runtimes.iter().map(|(_, l)| *l).collect();
                let picked = inquire::Select::new("Select local runtime:", labels).prompt()?;
                let id = runtimes
                    .iter()
                    .find(|(_, l)| *l == picked)
                    .map(|(id, _)| *id)
                    .unwrap_or("llamacpp");
                if opts.base_url.is_none() {
                    let custom_url =
                        inquire::Text::new("Custom base URL (optional, press Enter for default):")
                            .prompt()
                            .unwrap_or_default();
                    if !custom_url.trim().is_empty() {
                        opts.base_url = Some(custom_url.trim().to_owned());
                    }
                }
                if opts.api_key.is_none() {
                    let maybe_key = inquire::Password::new(
                        "Optional --api-key (press Enter if unauthenticated):",
                    )
                    .without_confirmation()
                    .prompt()
                    .unwrap_or_default();
                    if !maybe_key.trim().is_empty() {
                        opts.api_key = Some(maybe_key.trim().to_owned());
                    } else {
                        let rec = db.upsert_credential(
                            None,
                            id,
                            opts.label.as_deref().unwrap_or(picked),
                            AuthKind::LocalNoauth,
                            opts.base_url.as_deref(),
                            opts.priority.unwrap_or(50),
                            true,
                            true,
                            &CredentialSecret::LocalNoauth,
                        )?;
                        println!(
                            "Registered local runtime '{}' ({}) at priority #{}.",
                            rec.label, rec.id, rec.priority
                        );
                        return Ok(());
                    }
                }
                id.to_owned()
            } else if Some(chosen) == oauth_pillar {
                #[cfg(feature = "subscription-oauth")]
                {
                    let subs = [
                        (
                            "openai-codex",
                            "ChatGPT Plus / Pro / Codex (PKCE / Device Code)",
                        ),
                        ("anthropic-token", "Claude Pro / Max (claude setup-token)"),
                        ("google-gemini-cli", "Google Gemini CLI / Code Assist"),
                        ("github-copilot", "GitHub Copilot (Device Code)"),
                        ("xai-oauth", "xAI Grok OAuth"),
                        ("qwen-portal", "Qwen Portal (Device Code)"),
                        ("minimax-portal", "MiniMax Portal (Device Code)"),
                        ("chutes", "Chutes OAuth (PKCE)"),
                    ];
                    let labels: Vec<&str> = subs.iter().map(|(_, l)| *l).collect();
                    let picked =
                        inquire::Select::new("Select OAuth subscription:", labels).prompt()?;
                    subs.iter()
                        .find(|(_, l)| *l == picked)
                        .map(|(id, _)| (*id).to_owned())
                        .unwrap_or_else(|| "openai-codex".to_owned())
                }
                #[cfg(not(feature = "subscription-oauth"))]
                {
                    "openrouter".to_owned()
                }
            } else if chosen == or_pillar {
                "openrouter".to_owned()
            } else {
                let p = inquire::Text::new(
                    "Provider ID (e.g. openai, anthropic, google, groq, deepseek):",
                )
                .prompt()?;
                if opts.api_key.is_none() && !opts.from_env {
                    let entered = inquire::Password::new("Paste API key:")
                        .without_confirmation()
                        .prompt()?;
                    opts.api_key = Some(entered);
                }
                p
            }
        }
    };

    let trimmed = raw_provider.trim();
    if trimmed.eq_ignore_ascii_case("openrouter") {
        return login_openrouter(db, opts).await;
    }

    let norm_local = normalize_local_runtime_id(trimmed);
    let is_local_prov = matches!(
        norm_local.as_str(),
        "llamacpp" | "ollama" | "lmstudio" | "mlx" | "vllm" | "local"
    );
    let provider_id = if is_local_prov {
        norm_local.as_str()
    } else {
        trimmed
    };

    if opts.from_env && opts.api_key.is_none() {
        let var_name = opts
            .env_var
            .clone()
            .unwrap_or_else(|| default_env_var_for_provider(provider_id));
        let val = std::env::var(&var_name)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .with_context(|| format!("environment variable '{var_name}' is not set or empty"))?;
        opts.api_key = Some(val);
    }

    if let Some(raw_key) = opts.api_key {
        let lbl = opts
            .label
            .unwrap_or_else(|| format!("{provider_id} API key"));
        let default_prio = if is_local_prov { 50 } else { 200 };
        let rec = db.upsert_credential(
            None,
            provider_id,
            &lbl,
            AuthKind::ApiKey,
            opts.base_url.as_deref(),
            opts.priority.unwrap_or(default_prio),
            true,
            true,
            &CredentialSecret::ApiKey {
                key: raw_key.trim().to_owned(),
            },
        )?;
        println!(
            "Saved encrypted API key seat '{}' ({}) at priority #{}.",
            rec.label, rec.id, rec.priority
        );
        return Ok(());
    }

    if provider_id == "anthropic-token" || opts.setup_token.is_some() {
        #[cfg(not(feature = "subscription-oauth"))]
        {
            bail!(
                "Consumer subscription OAuth ('anthropic-token') is disabled in this build (compiled without `subscription-oauth`)"
            );
        }
        #[cfg(feature = "subscription-oauth")]
        {
            let tok = match opts.setup_token {
                Some(t) => t,
                None => inquire::Password::new("Paste `claude setup-token` (sk-ant-oat01-...):")
                    .without_confirmation()
                    .prompt()?,
            };
            let secret = validate_anthropic_setup_token(&tok)?;
            let lbl = opts
                .label
                .unwrap_or_else(|| "Claude Setup Token".to_owned());
            let rec = db.upsert_credential(
                None,
                "anthropic-token",
                &lbl,
                AuthKind::SetupToken,
                opts.base_url.as_deref(),
                opts.priority.unwrap_or(50),
                true,
                true,
                &secret,
            )?;
            println!(
                "Saved Claude setup-token seat '{}' ({}) at priority #{}.",
                rec.label, rec.id, rec.priority
            );
            return Ok(());
        }
    }

    let cred = run_oauth_wizard_cli(
        db,
        provider_id,
        opts.label,
        opts.priority,
        opts.device,
        opts.region,
        opts.enterprise_domain,
    )
    .await?;
    println!(
        "Authenticated seat '{}' ({}) for {} at priority #{}.",
        cred.label, cred.id, cred.provider_id, cred.priority
    );
    Ok(())
}

async fn cmd_status(data_dir: &Path, vault: &Vault, db: &Database) -> anyhow::Result<()> {
    let metrics = db.get_dashboard_metrics()?;
    let creds = db.list_credentials()?;
    let scanner = LocalRuntimeScanner::new();
    sync_local_scanner_from_db(db, &scanner);
    scanner.scan_all().await;
    let locals = scanner.list_statuses();

    println!("QuixiGate Status ({})", data_dir.display());
    println!("  Vault Source       : {}", vault.source().as_str());
    println!(
        "  24h Requests       : {} ({} cache hits, {:.1} tok/s)",
        metrics.total_requests_24h, metrics.cache_hits_24h, metrics.avg_tok_per_sec_24h
    );
    println!(
        "  Spend / Savings    : 24h {} | 30d billed {} | 30d OAuth+Local saved {}",
        metrics.spend_24h_nanos.format_usd(),
        metrics.spend_30d_nanos.format_usd(),
        metrics.savings_30d_nanos.format_usd()
    );

    println!("\n1. Local Runtimes ($0 local inference):");
    for rt in &locals {
        let state_str = if rt.healthy {
            let mut extras = Vec::new();
            if let Some(mode) = &rt.mode {
                extras.push(mode.clone());
            }
            if let Some(slots) = rt.total_slots {
                extras.push(format!("{slots} slots"));
            }
            if !rt.loaded_models.is_empty() {
                extras.push(format!("{} loaded", rt.loaded_models.len()));
            }
            if extras.is_empty() {
                format!("ONLINE ({} models)", rt.models.len())
            } else {
                format!("ONLINE ({} models; {})", rt.models.len(), extras.join(", "))
            }
        } else if rt.auth_required {
            "AUTH REQUIRED (run `quixigate login llamacpp --api-key <KEY>`)".to_owned()
        } else {
            "offline".to_owned()
        };
        println!(
            "  {:<18} {:<30} {}",
            rt.display_name, rt.base_url, state_str
        );
    }

    let oauth_seats: Vec<_> = creds
        .iter()
        .filter(|c| c.provider_id != "openrouter" && c.auth_kind.is_zero_marginal_cost())
        .collect();
    println!("\n2. OAuth Subscriptions ({}):", oauth_seats.len());
    if oauth_seats.is_empty() {
        println!(
            "  (none configured — run `quixigate login` to add ChatGPT/Claude/Gemini/Copilot)"
        );
    } else {
        for c in &oauth_seats {
            let quotas = c
                .quota_snapshot
                .as_ref()
                .map(|q| {
                    q.windows
                        .iter()
                        .map(|w| format!("{}:{:.0}%", w.label, w.used_percent))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "—".to_owned());
            println!(
                "  #{:<3} {:<24} {:<18} [{:?}] quota: {}",
                c.priority, c.label, c.provider_id, c.status, quotas
            );
        }
    }

    let openrouter_seats: Vec<_> = creds
        .iter()
        .filter(|c| c.provider_id == "openrouter")
        .collect();
    println!("\n3. OpenRouter ({}):", openrouter_seats.len());
    if openrouter_seats.is_empty() {
        println!(
            "  (not connected — optional; run `quixigate login openrouter` for PKCE / OIDC / API key)"
        );
    } else {
        for c in &openrouter_seats {
            let credits = c
                .quota_snapshot
                .as_ref()
                .and_then(|q| {
                    q.balance_credits
                        .map(|bal| format!("${bal:.2} credits"))
                        .or_else(|| q.plan.clone())
                        .or_else(|| {
                            q.windows
                                .first()
                                .map(|w| format!("{} ({:.0}% used)", w.label, w.used_percent))
                        })
                })
                .unwrap_or_else(|| "active".to_owned());
            println!(
                "  #{:<3} {:<24} {:<18} [{:?}] {}",
                c.priority,
                c.label,
                c.auth_kind.as_str(),
                c.status,
                credits
            );
        }
    }

    let direct_api_seats: Vec<_> = creds
        .iter()
        .filter(|c| c.provider_id != "openrouter" && !c.auth_kind.is_zero_marginal_cost())
        .collect();
    if !direct_api_seats.is_empty() {
        println!(
            "\nDirect Cloud & Custom Seats ({}):",
            direct_api_seats.len()
        );
        for c in &direct_api_seats {
            println!(
                "  #{:<3} {:<24} {:<18} [{:?}]",
                c.priority, c.label, c.provider_id, c.status
            );
        }
    }

    Ok(())
}

async fn cmd_run(
    data_dir: &Path,
    agent: &str,
    gateway: &str,
    model: Option<&str>,
    key: Option<&str>,
    extra_args: &[String],
) -> anyhow::Result<i32> {
    let mut base_url = gateway.trim_end_matches('/').to_owned();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(700))
        .build()?;

    let reachable = http
        .get(format!("{base_url}/health/liveness"))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    // If no daemon is listening yet, spin up an ephemeral loopback QuixiGate gateway in-process
    // strictly for the lifetime of this child command!
    let _ephemeral_guard = if !reachable {
        let (_vault, db) = open_db(data_dir)?;
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let addr = listener.local_addr()?;
        base_url = format!("http://{addr}");
        let ollama_env = quixigate_core::brand::env_opt("QUIXIGATE_OLLAMA_URL");
        let llamacpp_env = quixigate_core::brand::env_opt("QUIXIGATE_LLAMACPP_URL");
        let lmstudio_env = quixigate_core::brand::env_opt("QUIXIGATE_LMSTUDIO_URL");
        let state = assemble_gateway_state(
            db,
            true,
            ollama_env.as_deref(),
            llamacpp_env.as_deref(),
            lmstudio_env.as_deref(),
        );
        let _ = quixigate_proxy::admin::sync_openrouter_for_state(&state).await;
        let app = build_router(state);
        Some(tokio::spawn(async move {
            let _ = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await;
        }))
    } else {
        None
    };

    let token = key.unwrap_or("quixigate-local");
    let normalized = Path::new(agent)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(agent)
        .to_ascii_lowercase();

    let binary = match normalized.as_str() {
        "claude-code" => "claude",
        "gemini-cli" => "gemini",
        _ => agent,
    };

    // Scoped child-process environment injection ONLY — zero dotfile writes on disk!
    let mut cmd = tokio::process::Command::new(binary);
    cmd.args(extra_args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("QUIXIGATE_ACTIVE", "1")
        .env("QUIXIGATE_GATEWAY_URL", &base_url);

    match normalized.as_str() {
        "claude" | "claude-code" => {
            cmd.env("ANTHROPIC_BASE_URL", &base_url)
                .env("ANTHROPIC_AUTH_TOKEN", token);
            if let Some(m) = model {
                cmd.env("ANTHROPIC_MODEL", m);
            }
        }
        "codex" => {
            cmd.env("OPENAI_BASE_URL", format!("{base_url}/v1"))
                .env("OPENAI_API_KEY", token);
            if let Some(m) = model {
                cmd.env("OPENAI_MODEL", m);
            }
        }
        "gemini" | "gemini-cli" => {
            cmd.env("GOOGLE_GEMINI_BASE_URL", &base_url)
                .env("GEMINI_API_KEY", token);
            if let Some(m) = model {
                cmd.env("GEMINI_MODEL", m);
            }
        }
        "ori" | "ori-code" => {
            cmd.env("OPENROUTER_BASE_URL", format!("{base_url}/api/v1"))
                .env("OPENROUTER_API_KEY", token)
                .env("OPENAI_BASE_URL", format!("{base_url}/v1"))
                .env("OPENAI_API_KEY", token);
            if let Some(m) = model {
                cmd.env("OPENROUTER_MODEL", m).env("OPENAI_MODEL", m);
            }
        }
        _ => {
            // opencode & generic polyglot tools
            cmd.env("OPENAI_BASE_URL", format!("{base_url}/v1"))
                .env("OPENAI_API_KEY", token)
                .env("ANTHROPIC_BASE_URL", &base_url)
                .env("ANTHROPIC_API_KEY", token)
                .env("OPENROUTER_BASE_URL", format!("{base_url}/api/v1"))
                .env("OPENROUTER_API_KEY", token);
            if let Some(m) = model {
                cmd.env("OPENAI_MODEL", m);
            }
        }
    }

    let status = cmd
        .status()
        .await
        .with_context(|| format!("failed to launch '{binary}'"))?;
    Ok(status.code().unwrap_or(1))
}

async fn cmd_models(
    db: &Database,
    query: Option<&str>,
    sync: bool,
    all: bool,
    limit: usize,
) -> anyhow::Result<()> {
    let catalog = ModelCatalog::embedded();
    let http = reqwest::Client::new();
    if sync {
        let count = catalog.sync_from_url(&http, MODELS_DEV_URL).await?;
        println!("Synced {count} models from {MODELS_DEV_URL}");
    }

    let creds = db.list_credentials()?;
    let configured_providers: std::collections::HashSet<String> = creds
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.provider_id.clone())
        .collect();
    let openrouter_connected = configured_providers.contains("openrouter");

    if openrouter_connected {
        let refresher = CredentialRefresher::new(db.clone());
        if let Some(or_seat) = creds
            .iter()
            .find(|c| c.enabled && c.provider_id == "openrouter")
            && let Ok((fresh_rec, sec)) = refresher.ensure_fresh(&or_seat.id).await
        {
            let bearer = match sec {
                CredentialSecret::ApiKey { key } => Some(key),
                CredentialSecret::OpenRouterOidc { access, .. } => Some(access),
                _ => None,
            };
            let base = fresh_rec
                .base_url
                .as_deref()
                .filter(|u| !u.trim().is_empty())
                .unwrap_or(openrouter_base_url_for_region(None));
            let _ = catalog
                .sync_openrouter(&http, base, bearer.as_deref())
                .await;
        }
    }

    let scanner = LocalRuntimeScanner::new();
    sync_local_scanner_from_db(db, &scanner);
    scanner.scan_all().await;
    let locals = scanner.list_statuses();
    let all_models = catalog.list_models_with_context(&locals, openrouter_connected);
    let routes = db.list_model_routes()?;

    let routable_models: Vec<_> = all_models
        .into_iter()
        .filter(|m| all || is_catalog_model_routable(m, &configured_providers))
        .collect();
    let preset_count = routable_models.iter().filter(|m| m.is_preset).count();
    let base_count = routable_models.len().saturating_sub(preset_count);
    println!(
        "Catalog Summary: {base_count} models · {preset_count} @presets{}",
        if all {
            " (full reference catalog)"
        } else {
            " (routable across active seats & local runtimes)"
        }
    );

    if !routes.is_empty() {
        println!("Custom Route Aliases ({}):", routes.len());
        for r in &routes {
            let chain = r
                .targets
                .iter()
                .map(|t| t.model.as_str())
                .collect::<Vec<_>>()
                .join(" -> ");
            println!("  {:<22} => {} ({:?})", r.alias, chain, r.pool_strategy);
        }
        println!();
    }

    let q = query.unwrap_or_default().to_ascii_lowercase();
    let filtered: Vec<_> = routable_models
        .into_iter()
        .filter(|m| {
            q.is_empty()
                || m.canonical_id.to_ascii_lowercase().contains(&q)
                || m.display_name.to_ascii_lowercase().contains(&q)
        })
        .take(limit)
        .collect();

    println!(
        "{:<38} {:>9} {:>12} {:>12}  FLAGS",
        "MODEL ID", "CONTEXT", "IN / 1M", "OUT / 1M"
    );
    println!("{}", "-".repeat(88));
    for m in filtered {
        let mut flags = Vec::new();
        if m.is_local {
            flags.push("LOCAL");
        }
        if m.is_preset {
            flags.push("PRESET");
        }
        if m.supports_tools {
            flags.push("tools");
        }
        if m.supports_reasoning {
            flags.push("reason");
        }
        if m.supports_vision {
            flags.push("vision");
        }
        let in_str = if m.is_preset && m.input_nanos_per_m.is_zero() {
            "preset".to_owned()
        } else {
            m.input_nanos_per_m.format_usd()
        };
        let out_str = if m.is_preset && m.output_nanos_per_m.is_zero() {
            "preset".to_owned()
        } else {
            m.output_nanos_per_m.format_usd()
        };
        println!(
            "{:<38} {:>8}k {:>12} {:>12}  {}",
            m.canonical_id,
            m.context_window / 1000,
            in_str,
            out_str,
            flags.join(",")
        );
    }
    Ok(())
}

fn cmd_keys(db: &Database, action: KeysAction) -> anyhow::Result<()> {
    match action {
        KeysAction::List => {
            let keys = db.list_virtual_keys()?;
            if keys.is_empty() {
                println!("No virtual keys minted yet.");
                return Ok(());
            }
            println!(
                "{:<36} {:<18} {:<16} {:<24} STATUS",
                "ID", "NAME", "PREFIX", "SPEND / BUDGET"
            );
            for k in keys {
                let budget = match k.budget_limit_nanos {
                    Some(lim) => format!(
                        "{} / {} ({})",
                        k.spend_nanos.format_usd(),
                        lim.format_usd(),
                        k.budget_window.as_str()
                    ),
                    None => format!("{} / unlimited", k.spend_nanos.format_usd()),
                };
                let st = if k.revoked_at_ms.is_some() {
                    "REVOKED"
                } else {
                    "ACTIVE"
                };
                println!(
                    "{:<36} {:<18} {:<16} {:<24} {}",
                    k.id, k.name, k.key_prefix, budget, st
                );
            }
        }
        KeysAction::Create {
            name,
            user,
            team,
            budget_usd,
            window,
            rpm,
            tpm,
            max_concurrent,
            allowed_models,
        } => {
            let limit = match budget_usd {
                Some(s) if !s.trim().is_empty() => Some(NanoUsd::parse_usd(&s)?),
                _ => None,
            };
            let (vk, raw) = db.create_virtual_key(
                &name,
                &user,
                team.as_deref(),
                limit,
                BudgetWindow::parse(&window),
                rpm,
                tpm,
                max_concurrent,
                &allowed_models,
                None,
                None,
            )?;
            println!("Created virtual key '{}' ({}):", vk.name, vk.id);
            println!("  Secret Token (save now, shown once): {raw}");
        }
        KeysAction::Rotate { id, grace_seconds } => {
            let (vk, raw) = db.rotate_virtual_key(&id, grace_seconds)?;
            println!(
                "Rotated virtual key '{}' ({}) with {}s grace window:",
                vk.name, vk.id, grace_seconds
            );
            println!("  New Secret Token: {raw}");
        }
        KeysAction::Revoke { id } => {
            let vk = db.revoke_virtual_key(&id)?;
            println!("Revoked virtual key '{}' ({}).", vk.name, vk.id);
        }
        KeysAction::Delete { id } => {
            db.delete_virtual_key(&id)?;
            println!("Deleted virtual key '{id}'.");
        }
    }
    Ok(())
}

fn cmd_config(db: &Database, action: ConfigAction) -> anyhow::Result<()> {
    match action {
        ConfigAction::Export {
            format,
            include_secrets,
            output,
        } => {
            let text = db.export_config(&format, include_secrets)?;
            if let Some(path) = output {
                std::fs::write(&path, &text)?;
                println!("Exported {format} config to {}", path.display());
            } else {
                io::stdout().write_all(text.as_bytes())?;
            }
        }
        ConfigAction::Import { path, format } => {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("cannot read {}", path.display()))?;
            let ext_fmt =
                format.or_else(|| path.extension().and_then(|e| e.to_str()).map(str::to_owned));
            let count = db.import_config(&raw, ext_fmt.as_deref())?;
            println!(
                "Imported {count} configuration items from {}.",
                path.display()
            );
        }
    }
    Ok(())
}

async fn cmd_doctor(data_dir: &Path, port: u16) -> anyhow::Result<()> {
    println!("QuixiGate Doctor — Diagnostic Report");
    println!("  Data Dir: {}", data_dir.display());

    let (vault, db) = open_db(data_dir)?;
    let probe = vault.encrypt("doctor", b"quixigate-doctor-check")?;
    let roundtrip = vault.decrypt("doctor", &probe)?;
    if roundtrip.as_slice() != b"quixigate-doctor-check" {
        bail!("Vault AEAD self-test failed!");
    }
    println!(
        "  [OK] Vault AES-256-GCM AEAD round-trip verified ({})",
        vault.source().as_str()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let key_file = data_dir.join("vault.key");
        if key_file.exists() {
            let mode = std::fs::metadata(&key_file)?.permissions().mode() & 0o777;
            if mode == 0o600 {
                println!("  [OK] vault.key permissions strictly 0600");
            } else {
                println!("  [WARN] vault.key mode is {:o} (expected 0600)", mode);
            }
        }
    }

    let settings = db.get_settings()?;
    let creds = db.list_credentials()?;
    let keys = db.list_virtual_keys()?;
    println!(
        "  [OK] SQLite WAL operational ({} seats, {} virtual keys, OTLP enabled={})",
        creds.len(),
        keys.len(),
        settings.otlp_enabled
    );

    match TcpListener::bind(("127.0.0.1", port)).await {
        Ok(l) => {
            drop(l);
            println!("  [OK] Port 127.0.0.1:{port} is available to bind");
        }
        Err(_) => {
            println!("  [INFO] Port 127.0.0.1:{port} is already bound (gateway may be running)");
        }
    }

    Ok(())
}

async fn cmd_burst(db: &Database, action: BurstAction, gateway_url: &str) -> anyhow::Result<()> {
    let mut settings = db.get_settings()?;
    match &action {
        BurstAction::Enable { model } => {
            settings.openrouter_burst_enabled = true;
            if let Some(m) = model.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                settings.openrouter_burst_model = m.to_owned();
            }
            db.update_settings(&settings)?;
            let gw = gateway_url.trim_end_matches('/');
            let _ = reqwest::Client::builder()
                .timeout(Duration::from_millis(500))
                .build()?
                .post(format!("{gw}/api/admin/burst"))
                .json(&serde_json::json!({
                    "enabled": true,
                    "burst_model": settings.openrouter_burst_model,
                }))
                .send()
                .await;
        }
        BurstAction::Disable => {
            settings.openrouter_burst_enabled = false;
            db.update_settings(&settings)?;
            let gw = gateway_url.trim_end_matches('/');
            let _ = reqwest::Client::builder()
                .timeout(Duration::from_millis(500))
                .build()?
                .post(format!("{gw}/api/admin/burst/disable"))
                .send()
                .await;
        }
        BurstAction::Status => {}
    }

    let creds = db.list_credentials()?;
    let or_seats = creds
        .iter()
        .filter(|c| c.enabled && c.provider_id == "openrouter")
        .count();
    let state_label = if settings.openrouter_burst_enabled {
        "ENABLED"
    } else {
        "DISABLED"
    };
    println!(
        "{} Cloud Burst: {state_label}",
        quixigate_core::brand::BRAND_NAME
    );
    println!(
        "  Burst Fallback Model : {}",
        settings.openrouter_burst_model
    );
    println!("  OpenRouter Seats     : {or_seats} connected");
    println!(
        "  Attribution Headers  : X-Title: {} | HTTP-Referer: {}",
        settings.openrouter_app_title, settings.openrouter_app_url
    );
    if settings.openrouter_burst_enabled && or_seats == 0 {
        println!(
            "  [TIP] Run `{} login openrouter` to connect an OpenRouter seat for live burst failover.",
            quixigate_core::brand::BIN_NAME
        );
    }
    Ok(())
}
