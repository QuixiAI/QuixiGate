//! Declarative TOML / YAML configuration import & export with `${ENV_VAR}` and
//! `${ENV_VAR:-default}` expansion.

use quixigate_core::{
    AuthKind, CredentialSecret, GatewaySettings, ModelRoute, QuixigateError, PoolStrategy, Result,
    RouteTarget, now_ms,
};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::Database;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeclarativeConfig {
    #[serde(default)]
    pub settings: GatewaySettings,
    #[serde(default)]
    pub credentials: Vec<DeclarativeCredential>,
    #[serde(default)]
    pub routes: Vec<DeclarativeRoute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclarativeCredential {
    pub id: Option<String>,
    pub provider_id: String,
    pub label: String,
    #[serde(default = "default_api_key_kind")]
    pub auth_kind: AuthKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub setup_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub allow_api_overflow: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclarativeRoute {
    pub alias: String,
    pub targets: Vec<RouteTarget>,
    #[serde(default)]
    pub pool_strategy: PoolStrategy,
    #[serde(default = "default_true")]
    pub allow_paid_overflow: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_nanos_per_m: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_nanos_per_m: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_nanos_per_m: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
}

const fn default_api_key_kind() -> AuthKind {
    AuthKind::ApiKey
}

const fn default_priority() -> i32 {
    100
}

const fn default_true() -> bool {
    true
}

/// Expands `${VAR}` and `${VAR:-fallback}` tokens using process environment variables.
pub fn expand_env_vars(raw: &str) -> Result<String> {
    let re = Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
        .expect("valid env var regex");
    let mut missing = Vec::new();
    let expanded = re.replace_all(raw, |caps: &regex::Captures<'_>| {
        let name = &caps[1];
        match std::env::var(name) {
            Ok(val) if !val.is_empty() => val,
            _ => {
                if let Some(default_val) = caps.get(2) {
                    default_val.as_str().to_owned()
                } else {
                    missing.push(name.to_owned());
                    String::new()
                }
            }
        }
    });
    if !missing.is_empty() {
        return Err(QuixigateError::Validation(format!(
            "missing required environment variables in config: {}",
            missing.join(", ")
        )));
    }
    Ok(expanded.into_owned())
}

impl Database {
    /// Exports runtime configuration as TOML or YAML (`format` = `"toml"` or `"yaml"`).
    /// Raw secrets are redacted to `${PROVIDER_API_KEY}` placeholders unless `include_secrets` is true.
    pub fn export_config(&self, format: &str, include_secrets: bool) -> Result<String> {
        let settings = self.get_settings()?;
        let creds = self.list_credentials()?;
        let routes = self.list_model_routes()?;

        let mut decl_creds = Vec::with_capacity(creds.len());
        for cred in creds {
            let mut api_key = None;
            let mut setup_token = None;
            if include_secrets {
                if let Ok((_, secret)) = self.get_credential_with_secret(&cred.id) {
                    match secret {
                        CredentialSecret::ApiKey { key } => api_key = Some(key),
                        CredentialSecret::AnthropicSetupToken { token } => setup_token = Some(token),
                        _ => {}
                    }
                }
            } else {
                let env_hint = cred.provider_id.to_ascii_uppercase().replace('-', "_");
                match cred.auth_kind {
                    AuthKind::ApiKey => api_key = Some(format!("${{{env_hint}_API_KEY}}")),
                    AuthKind::SetupToken => setup_token = Some(format!("${{{env_hint}_TOKEN}}")),
                    AuthKind::Oauth | AuthKind::LocalNoauth => {}
                }
            }

            decl_creds.push(DeclarativeCredential {
                id: Some(cred.id),
                provider_id: cred.provider_id,
                label: cred.label,
                auth_kind: cred.auth_kind,
                api_key,
                setup_token,
                base_url: cred.base_url,
                priority: cred.priority,
                enabled: cred.enabled,
                allow_api_overflow: cred.allow_api_overflow,
            });
        }

        let decl_routes = routes
            .into_iter()
            .map(|r| DeclarativeRoute {
                alias: r.alias,
                targets: r.targets,
                pool_strategy: r.pool_strategy,
                allow_paid_overflow: r.allow_paid_overflow,
                input_nanos_per_m: r.input_nanos_per_m,
                output_nanos_per_m: r.output_nanos_per_m,
                cache_read_nanos_per_m: r.cache_read_nanos_per_m,
                team_id: r.team_id,
            })
            .collect();

        let cfg = DeclarativeConfig {
            settings,
            credentials: decl_creds,
            routes: decl_routes,
        };

        match format.to_ascii_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml_ng::to_string(&cfg)
                .map_err(|e| QuixigateError::Validation(format!("YAML serialize failed: {e}"))),
            _ => toml::to_string_pretty(&cfg)
                .map_err(|e| QuixigateError::Validation(format!("TOML serialize failed: {e}"))),
        }
    }

    /// Imports declarative TOML or YAML config into SQLite after expanding `${ENV_VAR}` placeholders.
    pub fn import_config(&self, raw: &str, format_hint: Option<&str>) -> Result<usize> {
        let expanded = expand_env_vars(raw)?;
        let is_yaml = match format_hint {
            Some(f) => matches!(f.to_ascii_lowercase().as_str(), "yaml" | "yml"),
            None => expanded.trim_start().starts_with("settings:") || expanded.contains("\nroutes:\n"),
        };

        let cfg: DeclarativeConfig = if is_yaml {
            serde_yaml_ng::from_str(&expanded)
                .map_err(|e| QuixigateError::Validation(format!("invalid YAML config: {e}")))?
        } else {
            toml::from_str(&expanded).or_else(|toml_err| {
                serde_yaml_ng::from_str(&expanded).map_err(|_| {
                    QuixigateError::Validation(format!("invalid TOML/YAML config: {toml_err}"))
                })
            })?
        };

        self.update_settings(&cfg.settings)?;
        let mut applied = 0usize;

        for cred in cfg.credentials {
            let secret = match cred.auth_kind {
                AuthKind::ApiKey => {
                    let Some(key) = cred.api_key.filter(|k| !k.is_empty()) else {
                        continue;
                    };
                    CredentialSecret::ApiKey { key }
                }
                AuthKind::SetupToken => {
                    let Some(token) = cred.setup_token.filter(|t| !t.is_empty()) else {
                        continue;
                    };
                    CredentialSecret::AnthropicSetupToken { token }
                }
                AuthKind::LocalNoauth => CredentialSecret::LocalNoauth,
                AuthKind::Oauth => continue, // OAuth seats with rotating refresh tokens stay managed in vault.
            };
            self.upsert_credential(
                cred.id.as_deref(),
                &cred.provider_id,
                &cred.label,
                cred.auth_kind,
                cred.base_url.as_deref(),
                cred.priority,
                cred.enabled,
                cred.allow_api_overflow,
                &secret,
            )?;
            applied += 1;
        }

        for r in cfg.routes {
            let now = now_ms();
            self.upsert_model_route(&ModelRoute {
                alias: r.alias,
                targets: r.targets,
                pool_strategy: r.pool_strategy,
                allow_paid_overflow: r.allow_paid_overflow,
                input_nanos_per_m: r.input_nanos_per_m,
                output_nanos_per_m: r.output_nanos_per_m,
                cache_read_nanos_per_m: r.cache_read_nanos_per_m,
                team_id: r.team_id,
                created_by: Some("config_sync".to_owned()),
                updated_by: Some("config_sync".to_owned()),
                created_at_ms: Some(now),
                updated_at_ms: now,
            })?;
            applied += 1;
        }

        Ok(applied)
    }
}
