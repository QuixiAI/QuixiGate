//! End-to-end CLI verification for `quixigate` subcommands (`login`, `keys`,
//! `config export/import`, `status`, `models`, `doctor`, and `quixigate run`
//! scoped child-process env injection with strict zero-dotfile-mutation proof).

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use tempfile::TempDir;

fn quixigate_bin() -> &'static str {
    env!("CARGO_BIN_EXE_quixigate")
}

#[test]
fn cli_end_to_end_keys_config_doctor_and_zero_dotfile_mutation_run() {
    let temp = TempDir::new().unwrap();
    let data_dir = temp.path().join("quixigate-data");
    let fake_home = temp.path().join("fake-home");
    let claude_dir = fake_home.join(".claude");
    let fake_bin = temp.path().join("bin");

    fs::create_dir_all(&claude_dir).unwrap();
    fs::create_dir_all(&fake_bin).unwrap();

    // Place a canary dotfile inside ~/.claude/settings.json to prove `quixigate run claude`
    // NEVER touches or mutates global dotfiles on disk!
    let canary_path = claude_dir.join("settings.json");
    let canary_contents = "{\"untouched\":true}\n";
    fs::write(&canary_path, canary_contents).unwrap();

    // 1. `quixigate login` with API key and Anthropic setup-token
    let login_oai = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "login",
            "openai",
            "--api-key",
            "sk-cli-test-key",
            "--label",
            "CLI OpenAI Seat",
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(login_oai.status.success());

    let valid_oat = format!("sk-ant-oat01-{}", "b".repeat(84));
    let login_ant = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "login",
            "anthropic-token",
            "--setup-token",
            &valid_oat,
            "--label",
            "CLI Claude Max",
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(login_ant.status.success());

    // 2. `quixigate keys create`, `list`, `rotate`, `revoke`
    let key_create = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "keys",
            "create",
            "--name",
            "agent-dev-key",
            "--budget-usd",
            "15.00",
            "--rpm",
            "120",
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(key_create.status.success());
    let create_stdout = String::from_utf8_lossy(&key_create.stdout);
    assert!(create_stdout.contains("sk-qx-"));

    let key_list = Command::new(quixigate_bin())
        .args(["--data-dir", data_dir.to_str().unwrap(), "keys", "list"])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(key_list.status.success());
    assert!(String::from_utf8_lossy(&key_list.stdout).contains("agent-dev-key"));

    // 3. `quixigate config export` + `import` with `${ENV}` expansion
    let exported_toml = temp.path().join("exported.toml");
    let cfg_export = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "config",
            "export",
            "--format",
            "toml",
            "--output",
            exported_toml.to_str().unwrap(),
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(cfg_export.status.success());
    let exported_text = fs::read_to_string(&exported_toml).unwrap();
    assert!(
        exported_text.contains("${OPENAI_API_KEY}"),
        "Exported config must redact static secrets to ${{ENV}} placeholders by default"
    );

    let cfg_import = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "config",
            "import",
            exported_toml.to_str().unwrap(),
        ])
        .env("HOME", &fake_home)
        .env("OPENAI_API_KEY", "sk-expanded-from-env")
        .env("ANTHROPIC_TOKEN_TOKEN", &valid_oat)
        .output()
        .unwrap();
    assert!(cfg_import.status.success());

    // 4. `quixigate status`, `quixigate models`, and `quixigate doctor`
    let status_out = Command::new(quixigate_bin())
        .args(["--data-dir", data_dir.to_str().unwrap(), "status"])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(status_out.status.success());
    assert!(String::from_utf8_lossy(&status_out.stdout).contains("CLI Claude Max"));

    let models_out = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "models",
            "claude-sonnet",
            "--limit",
            "5",
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(models_out.status.success());
    assert!(String::from_utf8_lossy(&models_out.stdout).contains("claude-sonnet"));

    let doctor_out = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "doctor",
            "--port",
            "0",
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(doctor_out.status.success());
    let doctor_stdout = String::from_utf8_lossy(&doctor_out.stdout);
    assert!(doctor_stdout.contains("[OK] Vault AES-256-GCM AEAD round-trip verified"));
    assert!(doctor_stdout.contains("[OK] vault.key permissions strictly 0600"));

    // 5. `quixigate run claude -- --flag-test`: create a pure-POSIX shell stub `claude` binary
    // that prints its injected env vars and verifies the ephemeral QuixiGate gateway is live!
    let fake_claude = fake_bin.join("claude");
    fs::write(
        &fake_claude,
        "#!/bin/sh\necho \"BASE=$ANTHROPIC_BASE_URL\"\necho \"TOKEN=$ANTHROPIC_AUTH_TOKEN\"\necho \"MODEL=$ANTHROPIC_MODEL\"\necho \"ARGS=$*\"\ncurl -fsS \"$ANTHROPIC_BASE_URL/health/liveness\"\necho \"\"\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&fake_claude).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_claude, perms).unwrap();
    }

    let orig_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{orig_path}", fake_bin.display());

    let run_out = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "run",
            "claude",
            "--gateway",
            "http://127.0.0.1:9", // guaranteed unreachable -> triggers ephemeral in-process gateway!
            "--model",
            "anthropic/claude-sonnet-4-5",
            "--key",
            "sk-qx-ephemeral-test",
            "--",
            "--print",
            "hello",
        ])
        .env("HOME", &fake_home)
        .env("PATH", &new_path)
        .output()
        .unwrap();

    assert!(
        run_out.status.success(),
        "quixigate run failed: stdout={} stderr={}",
        String::from_utf8_lossy(&run_out.stdout),
        String::from_utf8_lossy(&run_out.stderr)
    );
    let run_stdout = String::from_utf8_lossy(&run_out.stdout);
    assert!(run_stdout.contains("BASE=http://127.0.0.1:"));
    assert!(run_stdout.contains("TOKEN=sk-qx-ephemeral-test"));
    assert!(run_stdout.contains("MODEL=anthropic/claude-sonnet-4-5"));
    assert!(run_stdout.contains("ARGS=--print hello"));
    assert!(run_stdout.contains("\"status\":\"ok\""));

    // Verify ZERO mutation of `~/.claude/`!
    assert_eq!(fs::read_to_string(&canary_path).unwrap(), canary_contents);
    let entries: Vec<_> = fs::read_dir(&claude_dir).unwrap().collect();
    assert_eq!(entries.len(), 1, "~/.claude must not gain any extra files");

    // 6. `quixigate serve --port 0 --no-ui` (without QUIXIGATE_MASTER_KEY / LITELLM_MASTER_KEY set):
    // must auto-generate and print `sk-qx-master-...` on stdout + run in headless API mode!
    let mut serve_child = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--no-ui",
        ])
        .env("HOME", &fake_home)
        .env_remove("QUIXIGATE_MASTER_KEY")
        .env_remove("LITELLM_MASTER_KEY")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    use std::io::{BufRead, BufReader};
    let stdout = serve_child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    let mut captured_banner = String::new();
    for line in reader.lines().take(9) {
        let l = line.unwrap();
        captured_banner.push_str(&l);
        captured_banner.push('\n');
        if l.contains("Auth Policy") {
            break;
        }
    }
    let _ = serve_child.kill();
    let _ = serve_child.wait();

    assert!(
        captured_banner.contains("Master Key       : sk-qx-master-"),
        "Expected auto-generated Master Key in stdout banner, got:\n{captured_banner}"
    );
    assert!(
        captured_banner.contains("Admin Web UI     : disabled"),
        "Expected headless UI disabled notice in stdout banner, got:\n{captured_banner}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_openrouter_login_from_env_and_optional_model_gating() {
    use axum::http::HeaderMap;
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::json;
    use tokio::net::TcpListener;

    let mock_or = Router::new()
        .route(
            "/api/v1/key",
            get(|headers: HeaderMap| async move {
                assert_eq!(
                    headers.get("authorization").and_then(|v| v.to_str().ok()),
                    Some("Bearer sk-or-v1-cli-env-key")
                );
                Json(json!({
                    "data": {
                        "label": "cli-env",
                        "limit": 25.0,
                        "limit_remaining": 20.0,
                        "usage": 5.0,
                        "is_free_tier": false
                    }
                }))
            }),
        )
        .route(
            "/api/v1/credits",
            get(|| async {
                Json(json!({
                    "data": {
                        "total_credits": 25.0,
                        "total_usage": 5.0
                    }
                }))
            }),
        )
        .route(
            "/api/v1/models",
            get(|| async {
                Json(json!({
                    "data": [{
                        "id": "deepseek/deepseek-r1",
                        "name": "DeepSeek R1",
                        "context_length": 163840,
                        "pricing": { "prompt": "0.00000055", "completion": "0.00000219" },
                        "supported_parameters": ["tools", "reasoning"]
                    }]
                }))
            }),
        )
        .route(
            "/api/v1/presets",
            get(|| async {
                Json(json!({
                    "data": [{
                        "slug": "code-reviewer",
                        "name": "Code Reviewer",
                        "description": "Strict code review preset",
                        "model": "anthropic/claude-opus-4-6"
                    }]
                }))
            }),
        );

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, mock_or).await;
    });
    let mock_base = format!("http://{addr}/api/v1");

    let temp = TempDir::new().unwrap();
    let data_dir = temp.path().join("quixigate-or-cli");
    let fake_home = temp.path().join("fake-home");
    let fake_bin = temp.path().join("bin");
    fs::create_dir_all(&fake_home).unwrap();
    fs::create_dir_all(&fake_bin).unwrap();

    // 1. Before OpenRouter is connected: `quixigate models` shows `0 @presets` and zero OpenRouter-only entries!
    let models_before = Command::new(quixigate_bin())
        .args(["--data-dir", data_dir.to_str().unwrap(), "models"])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(models_before.status.success());
    let before_out = String::from_utf8_lossy(&models_before.stdout);
    assert!(
        before_out.contains("· 0 @presets"),
        "expected 0 @presets when OpenRouter is not connected, got:\n{before_out}"
    );
    assert!(!before_out.contains("@preset/"));
    assert!(!before_out.contains("openrouter/auto"));

    // 2. `quixigate login openrouter --from-env --base-url <mock_base>` imports `OPENROUTER_API_KEY`,
    //    validates `/key` + `/credits`, and syncs `/models` + `/presets`!
    let login_or = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "login",
            "openrouter",
            "--from-env",
            "--base-url",
            &mock_base,
        ])
        .env("HOME", &fake_home)
        .env("OPENROUTER_API_KEY", "sk-or-v1-cli-env-key")
        .output()
        .unwrap();
    assert!(
        login_or.status.success(),
        "quixigate login openrouter --from-env failed: stdout={} stderr={}",
        String::from_utf8_lossy(&login_or.stdout),
        String::from_utf8_lossy(&login_or.stderr)
    );
    let login_out = String::from_utf8_lossy(&login_or.stdout);
    assert!(login_out.contains("$20.00 credits"));
    assert!(login_out.contains("1 @presets synced"));

    // 3. `quixigate status` displays the Three First-Class Pillars layout!
    let status_out = Command::new(quixigate_bin())
        .args(["--data-dir", data_dir.to_str().unwrap(), "status"])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(status_out.status.success());
    let status_text = String::from_utf8_lossy(&status_out.stdout);
    assert!(status_text.contains("1. Local Runtimes"));
    assert!(status_text.contains("2. OAuth Subscriptions"));
    assert!(status_text.contains("3. OpenRouter (1)"));

    // 4. `quixigate models @preset` now surfaces `@preset/code-reviewer` with `[PRESET]`!
    let models_after = Command::new(quixigate_bin())
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "models",
            "@preset",
        ])
        .env("HOME", &fake_home)
        .output()
        .unwrap();
    assert!(models_after.status.success());
    let after_out = String::from_utf8_lossy(&models_after.stdout);
    assert!(after_out.contains("· 1 @presets"));
    assert!(after_out.contains("@preset/code-reviewer"));
    assert!(after_out.contains("PRESET"));

    // 5. `quixigate run ori` injects `OPENROUTER_BASE_URL` + `OPENROUTER_MODEL` and syncs `@preset/*`
    //    on the ephemeral loopback gateway!
    #[cfg(unix)]
    {
        let mock_ori = fake_bin.join("ori");
        fs::write(
            &mock_ori,
            "#!/bin/sh\nset -e\necho \"OR_BASE=$OPENROUTER_BASE_URL\"\necho \"OR_MODEL=$OPENROUTER_MODEL\"\ncurl -fsS \"$OPENROUTER_BASE_URL/presets\"\necho \"\"\n",
        )
        .unwrap();
        fs::set_permissions(&mock_ori, fs::Permissions::from_mode(0o755)).unwrap();

        let existing_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{existing_path}", fake_bin.display());
        let run_ori = Command::new(quixigate_bin())
            .args([
                "--data-dir",
                data_dir.to_str().unwrap(),
                "run",
                "ori",
                "--gateway",
                "http://127.0.0.1:9",
                "--model",
                "@preset/code-reviewer",
            ])
            .env("HOME", &fake_home)
            .env("PATH", &new_path)
            .output()
            .unwrap();
        assert!(
            run_ori.status.success(),
            "quixigate run ori failed: stdout={} stderr={}",
            String::from_utf8_lossy(&run_ori.stdout),
            String::from_utf8_lossy(&run_ori.stderr)
        );
        let run_stdout = String::from_utf8_lossy(&run_ori.stdout);
        assert!(run_stdout.contains("OR_BASE=http://127.0.0.1:"));
        assert!(run_stdout.contains("/api/v1"));
        assert!(run_stdout.contains("OR_MODEL=@preset/code-reviewer"));
        assert!(run_stdout.contains("\"slug\":\"code-reviewer\""));
    }
}
