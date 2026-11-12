//! Shared OAuth PKCE, JWT claim decoding, browser launch, and loopback callback server.

use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use quixigate_core::{QuixigateError, Result};
use rand::RngCore;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub fn headless() -> bool {
    let set = |name| std::env::var_os(name).is_some_and(|v| !v.is_empty());
    set("SSH_CONNECTION")
        || set("SSH_TTY")
        || (cfg!(target_os = "linux") && !set("DISPLAY") && !set("WAYLAND_DISPLAY"))
}

pub fn open_browser(url: &str) -> bool {
    !headless() && open::that_detached(url).is_ok()
}

pub fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

pub fn jwt_payload(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).ok()?;
    serde_json::from_slice(&decoded).ok()
}

/// Waits on `listener` for an OAuth browser redirect matching `expected_path`
/// and optional `expected_state`, returning the authorization `code`.
pub async fn wait_for_callback(
    listener: TcpListener,
    expected_path: &str,
    expected_state: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    tokio::time::timeout(timeout, async {
        loop {
            let (mut socket, _) = listener
                .accept()
                .await
                .map_err(|e| QuixigateError::Internal(format!("callback accept failed: {e}")))?;
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let target = req.split_whitespace().nth(1).unwrap_or("");
            let outcome = parse_callback_target(target, expected_path, expected_state);
            let (status, html) = match &outcome {
                Some(Ok(_)) => (
                    "200 OK",
                    "<!doctype html><html><body style=\"font-family:system-ui;background:#0b0f17;color:#f8fafc;display:grid;place-items:center;height:100vh;margin:0\"><div style=\"text-align:center\"><h2>Signed in to QuixiGate</h2><p>You can close this tab and return to QuixiGate.</p></div></body></html>",
                ),
                Some(Err(_)) => (
                    "400 Bad Request",
                    "<!doctype html><html><body style=\"font-family:system-ui;background:#0b0f17;color:#fca5a5;display:grid;place-items:center;height:100vh;margin:0\"><div style=\"text-align:center\"><h2>Sign-in failed</h2><p>Return to QuixiGate for details.</p></div></body></html>",
                ),
                None => ("404 Not Found", "Not found."),
            };
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
                html.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
            if let Some(res) = outcome {
                return res;
            }
        }
    })
    .await
    .map_err(|_| QuixigateError::Validation("timed out waiting for OAuth callback".to_owned()))?
}

/// Extracts `code` (and verifies `state` when supplied) from either a relative
/// target (`/auth/callback?code=...`), a full pasted redirect URL, or a bare code.
pub fn extract_code_from_input(raw_input: &str, expected_state: Option<&str>) -> Result<String> {
    let trimmed = raw_input.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let parsed = url::Url::parse(trimmed)
            .map_err(|e| QuixigateError::Validation(format!("invalid callback URL: {e}")))?;
        return extract_code_from_url(&parsed, expected_state);
    }
    if trimmed.starts_with('/') {
        let parsed = url::Url::parse(&format!("http://localhost{trimmed}"))
            .map_err(|e| QuixigateError::Validation(format!("invalid callback path: {e}")))?;
        return extract_code_from_url(&parsed, expected_state);
    }
    if trimmed.is_empty() {
        return Err(QuixigateError::Validation(
            "authorization code cannot be empty".to_owned(),
        ));
    }
    Ok(trimmed.to_owned())
}

fn parse_callback_target(
    target: &str,
    expected_path: &str,
    expected_state: Option<&str>,
) -> Option<Result<String>> {
    let parsed = url::Url::parse(&format!("http://localhost{target}")).ok()?;
    if parsed.path() != expected_path {
        return None;
    }
    Some(extract_code_from_url(&parsed, expected_state))
}

fn extract_code_from_url(parsed: &url::Url, expected_state: Option<&str>) -> Result<String> {
    let param = |name: &str| {
        parsed
            .query_pairs()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.into_owned())
    };
    if let Some(err) = param("error") {
        let desc = param("error_description").unwrap_or_default();
        return Err(QuixigateError::Unauthorized(format!(
            "OAuth authorization denied: {err} {desc}"
        )));
    }
    if let Some(exp_state) = expected_state
        && param("state").as_deref() != Some(exp_state)
    {
        return Err(QuixigateError::Validation(
            "OAuth state mismatch (CSRF protection)".to_owned(),
        ));
    }
    param("code").ok_or_else(|| {
        QuixigateError::Validation("OAuth callback contained no authorization code".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_rfc7636_vector_and_url_extraction() {
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        let code = extract_code_from_input(
            "http://localhost:1455/auth/callback?code=xyz123&state=st1",
            Some("st1"),
        )
        .unwrap();
        assert_eq!(code, "xyz123");
        assert!(
            extract_code_from_input(
                "http://localhost:1455/auth/callback?code=xyz123&state=wrong",
                Some("st1"),
            )
            .is_err()
        );
    }
}
