//! Zero-dependency in-process pre-flight guardrails:
//! - Secret / credential leak detection (`SecretScannerMode`: `Off` | `Redact` | `Block`)
//! - Case-insensitive banned keyword / phrase enforcement (`banned_keywords`)
//!
//! Runs directly inside `GatewayRouter` before cache lookup, upstream dispatch, or trace storage.

use std::sync::OnceLock;

use aisdk::core::language_model::LanguageModelResponseContentType;
use aisdk::core::{AssistantMessage, Message, SystemMessage, ToolResultInfo, UserMessage};
use quixigate_core::{GatewaySettings, QuixigateError, Result, SecretScannerMode};
use regex::Regex;
use serde_json::Value;

use crate::CanonicalRequest;

struct SecretPattern {
    kind: &'static str,
    regex: Regex,
}

fn secret_patterns() -> &'static [SecretPattern] {
    static PATTERNS: OnceLock<Vec<SecretPattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let specs: &[(&'static str, &str)] = &[
            (
                "PRIVATE_KEY",
                r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
            ),
            (
                "ANTHROPIC_API_KEY",
                r"\bsk-ant-(?:api\d*-)?[A-Za-z0-9_-]{20,}\b",
            ),
            (
                "OPENROUTER_API_KEY",
                r"\bsk-or-v1-[A-Za-z0-9_-]{24,}\b",
            ),
            (
                "QUIXIGATE_API_KEY",
                r"\bsk-qx-(?:master-)?[A-Za-z0-9_-]{20,}\b",
            ),
            (
                "OPENAI_API_KEY",
                r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b",
            ),
            (
                "GITHUB_TOKEN",
                r"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{36,})\b",
            ),
            ("AWS_ACCESS_KEY", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            ("SLACK_TOKEN", r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
            ("GOOGLE_API_KEY", r"\bAIza[0-9A-Za-z_-]{35}\b"),
            ("HUGGINGFACE_TOKEN", r"\bhf_[A-Za-z0-9]{30,}\b"),
        ];
        specs
            .iter()
            .map(|(kind, pat)| SecretPattern {
                kind,
                regex: Regex::new(pat).expect("valid built-in secret guardrail regex"),
            })
            .collect()
    })
}

/// Scans `text` for known API key / private-key formats and returns the first matched kind.
pub fn detect_first_secret(text: &str) -> Option<&'static str> {
    if text.len() < 16 {
        return None;
    }
    for pat in secret_patterns() {
        if pat.regex.is_match(text) {
            return Some(pat.kind);
        }
    }
    None
}

/// Replaces all detected secret tokens in `text` with `[REDACTED:<KIND>]`.
pub fn redact_secrets_in_str(text: &str) -> String {
    if text.len() < 16 {
        return text.to_owned();
    }
    let mut current = text.to_owned();
    for pat in secret_patterns() {
        if pat.regex.is_match(&current) {
            let replacement = format!("[REDACTED:{}]", pat.kind);
            current = pat
                .regex
                .replace_all(&current, replacement.as_str())
                .into_owned();
        }
    }
    current
}

fn check_banned_keywords(text: &str, banned_lower: &[String]) -> Result<()> {
    if banned_lower.is_empty() || text.is_empty() {
        return Ok(());
    }
    let haystack = text.to_lowercase();
    for kw in banned_lower {
        if haystack.contains(kw) {
            return Err(QuixigateError::Validation(format!(
                "guardrail blocked request: prompt matched banned keyword '{kw}'"
            )));
        }
    }
    Ok(())
}

fn apply_to_text(
    text: &str,
    mode: SecretScannerMode,
    banned_lower: &[String],
) -> Result<Option<String>> {
    check_banned_keywords(text, banned_lower)?;
    match mode {
        SecretScannerMode::Off => Ok(None),
        SecretScannerMode::Block => {
            if let Some(kind) = detect_first_secret(text) {
                return Err(QuixigateError::Validation(format!(
                    "guardrail blocked request: detected leaked secret ({kind}) in prompt"
                )));
            }
            Ok(None)
        }
        SecretScannerMode::Redact => {
            let redacted = redact_secrets_in_str(text);
            if redacted != text {
                Ok(Some(redacted))
            } else {
                Ok(None)
            }
        }
    }
}

/// Enforces `banned_keywords` and `secret_scanner_mode` across `CanonicalRequest` in-place
/// before cache lookup or upstream provider dispatch.
pub fn apply_request_guardrails(
    req: &mut CanonicalRequest,
    settings: &GatewaySettings,
) -> Result<()> {
    let mode = settings.secret_scanner_mode;
    let banned_lower: Vec<String> = settings
        .banned_keywords
        .iter()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    if mode == SecretScannerMode::Off && banned_lower.is_empty() {
        return Ok(());
    }

    if let Some(sys) = &req.system
        && let Some(redacted) = apply_to_text(sys, mode, &banned_lower)?
    {
        req.system = Some(redacted);
    }

    for msg in &mut req.messages {
        match msg {
            Message::System(s) => {
                if let Some(redacted) = apply_to_text(&s.content, mode, &banned_lower)? {
                    *msg = Message::System(SystemMessage::new(redacted));
                }
            }
            Message::Developer(d) => {
                if let Some(redacted) = apply_to_text(d, mode, &banned_lower)? {
                    *msg = Message::Developer(redacted);
                }
            }
            Message::User(u) => {
                if let Some(redacted) = apply_to_text(&u.content, mode, &banned_lower)? {
                    *msg = Message::User(UserMessage::new(redacted));
                }
            }
            Message::Assistant(a) => {
                if let LanguageModelResponseContentType::Text(t) = &a.content
                    && let Some(redacted) = apply_to_text(t, mode, &banned_lower)?
                {
                    *msg = Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::Text(redacted),
                        None,
                    ));
                }
            }
            Message::Tool(tr) => match &tr.output {
                Ok(Value::String(s)) => {
                    if let Some(redacted) = apply_to_text(s, mode, &banned_lower)? {
                        *msg = Message::Tool(ToolResultInfo {
                            tool: tr.tool.clone(),
                            output: Ok(Value::String(redacted)),
                        });
                    }
                }
                Ok(other) => {
                    let s = other.to_string();
                    let _ = apply_to_text(&s, mode, &banned_lower)?;
                }
                Err(_) => {}
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_scanner_redacts_and_blocks_and_checks_banned_keywords() {
        let leaked = "Here is my key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 and ghp_0123456789abcdef0123456789abcdef0123";
        assert_eq!(detect_first_secret(leaked), Some("OPENAI_API_KEY"));
        assert_eq!(
            detect_first_secret("token sk-qx-master-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"),
            Some("QUIXIGATE_API_KEY")
        );
        let scrubbed = redact_secrets_in_str(leaked);
        assert!(scrubbed.contains("[REDACTED:OPENAI_API_KEY]"));
        assert!(scrubbed.contains("[REDACTED:GITHUB_TOKEN]"));
        assert!(!scrubbed.contains("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));

        let mut req = CanonicalRequest {
            requested_model: "ollama/qwen3:8b".to_owned(),
            messages: vec![Message::User(UserMessage::new(leaked.to_owned()))],
            ..Default::default()
        };

        let mut settings = GatewaySettings {
            secret_scanner_mode: SecretScannerMode::Block,
            ..Default::default()
        };
        assert!(apply_request_guardrails(&mut req, &settings).is_err());

        settings.secret_scanner_mode = SecretScannerMode::Redact;
        apply_request_guardrails(&mut req, &settings).unwrap();
        match &req.messages[0] {
            Message::User(u) => assert!(u.content.contains("[REDACTED:OPENAI_API_KEY]")),
            _ => panic!("expected user message"),
        }

        settings.banned_keywords = vec!["project-nightfall".to_owned()];
        req.messages = vec![Message::User(UserMessage::new(
            "Tell me about PROJECT-NIGHTFALL".to_owned(),
        ))];
        assert!(apply_request_guardrails(&mut req, &settings).is_err());
    }
}
