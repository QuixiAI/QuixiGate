//! Inbound OIDC Discovery (`/.well-known/openid-configuration`) & JWKS (`RS256`/`RS384`/`RS512`,
//! `PS256`/`PS384`/`PS512`, `ES256`/`ES384`, `EdDSA`) Bearer JWT verification for enterprise SSO
//! (Dex, Keycloak, Okta, Auth0, Entra ID, Google, Cloudflare Access) across inference and `/mcp`.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use parking_lot::RwLock;
use quixigate_core::{BudgetWindow, GatewaySettings, QuixigateError, Result, UserRole, now_ms};
use quixigate_db::AuthenticatedPrincipal;
use ring::signature::{self, RsaPublicKeyComponents, UnparsedPublicKey};
use serde_json::Value;

use crate::state::AppState;

const JWKS_TTL_MS: i64 = 15 * 60 * 1_000;
const KID_MISS_REFRESH_COOLDOWN_MS: i64 = 500;
const CLOCK_SKEW_SECS: i64 = 60;

const SUPPORTED_ASYMMETRIC_ALGS: &[&str] = &[
    "RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "EdDSA",
];

#[derive(Debug, Clone)]
enum JwkPublicKey {
    Rsa { n: Vec<u8>, e: Vec<u8> },
    EcP256 { uncompressed: [u8; 65] },
    EcP384 { uncompressed: [u8; 97] },
    Ed25519 { public: [u8; 32] },
}

#[derive(Debug, Clone)]
struct ParsedJwk {
    kid: Option<String>,
    alg: Option<String>,
    key: JwkPublicKey,
}

#[derive(Debug, Default, Clone)]
struct JwksSnapshot {
    config_fingerprint: String,
    discovered_issuer: Option<String>,
    keys: Vec<ParsedJwk>,
    fetched_at_ms: i64,
    last_attempt_ms: i64,
}

#[derive(Clone)]
pub struct OidcVerifier {
    inner: Arc<OidcInner>,
}

struct OidcInner {
    http: std::sync::OnceLock<reqwest::Client>,
    snapshot: RwLock<JwksSnapshot>,
    refresh_lock: tokio::sync::Mutex<()>,
}

impl Default for OidcVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl OidcVerifier {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(OidcInner {
                http: std::sync::OnceLock::new(),
                snapshot: RwLock::new(JwksSnapshot::default()),
                refresh_lock: tokio::sync::Mutex::new(()),
            }),
        }
    }

    fn http_client(&self) -> &reqwest::Client {
        self.inner.http.get_or_init(|| {
            quixigate_providers::shared_tls_client_builder()
                .timeout(Duration::from_secs(5))
                .connect_timeout(Duration::from_secs(3))
                .user_agent(format!(
                    "{}/{}",
                    quixigate_core::brand::BRAND_SLUG,
                    env!("CARGO_PKG_VERSION")
                ))
                .build()
                .unwrap_or_default()
        })
    }

    pub fn is_configured(settings: &GatewaySettings) -> bool {
        settings
            .jwt_jwks_url
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
            || settings
                .jwt_issuer_url
                .as_deref()
                .is_some_and(|s| !s.trim().is_empty())
    }

    /// Async pre-flight hook invoked in `peer_trust_middleware` so cold-start OIDC discovery
    /// and key-rotation (`kid` cache miss) refresh JWKS with single-flight coalescing while keeping
    /// handler authentication synchronous and lock-free on cached requests.
    pub async fn warm_for_token_if_needed(&self, settings: &GatewaySettings, raw_token: &str) {
        if !Self::is_configured(settings) {
            return;
        }
        let Some((_alg, kid)) = peek_asymmetric_jwt_header(raw_token) else {
            return;
        };
        let fp = config_fingerprint(settings);
        let now = now_ms();

        {
            let snap = self.inner.snapshot.read();
            let fresh = snap.config_fingerprint == fp
                && !snap.keys.is_empty()
                && now.saturating_sub(snap.fetched_at_ms) < JWKS_TTL_MS;
            if fresh {
                let kid_hit = match &kid {
                    Some(k) => snap
                        .keys
                        .iter()
                        .any(|jwk| jwk.kid.as_deref() == Some(k.as_str())),
                    None => true,
                };
                if kid_hit {
                    return;
                }
                if now.saturating_sub(snap.last_attempt_ms) < KID_MISS_REFRESH_COOLDOWN_MS {
                    return;
                }
            }
        }

        let _guard = self.inner.refresh_lock.lock().await;
        let now = now_ms();
        {
            let snap = self.inner.snapshot.read();
            let fresh = snap.config_fingerprint == fp
                && !snap.keys.is_empty()
                && now.saturating_sub(snap.fetched_at_ms) < JWKS_TTL_MS;
            if fresh {
                let kid_hit = match &kid {
                    Some(k) => snap
                        .keys
                        .iter()
                        .any(|jwk| jwk.kid.as_deref() == Some(k.as_str())),
                    None => true,
                };
                if kid_hit
                    || now.saturating_sub(snap.last_attempt_ms) < KID_MISS_REFRESH_COOLDOWN_MS
                {
                    return;
                }
            }
        }

        match fetch_jwks_snapshot(self.http_client(), settings, &fp).await {
            Ok(new_snap) => {
                *self.inner.snapshot.write() = new_snap;
            }
            Err(err) => {
                tracing::warn!("OIDC/JWKS refresh failed: {err}");
                self.inner.snapshot.write().last_attempt_ms = now_ms();
            }
        }
    }

    /// Verifies an inbound asymmetric JWT (`RS*`, `PS*`, `ES256`/`ES384`, `EdDSA`) against the
    /// cached JWKS, validates `exp`/`nbf`/`iss`/`aud`, and resolves/provisions the QuixiGate principal.
    pub fn verify_jwt(
        &self,
        state: &AppState,
        settings: &GatewaySettings,
        raw_token: &str,
    ) -> Result<AuthenticatedPrincipal> {
        let mut parts = raw_token.split('.');
        let (Some(header_b64), Some(payload_b64), Some(sig_b64), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(QuixigateError::Unauthorized(
                "malformed JWT structure".to_owned(),
            ));
        };

        let header_bytes = b64url_decode(header_b64).ok_or_else(|| {
            QuixigateError::Unauthorized("invalid JWT header base64url".to_owned())
        })?;
        let header: Value = serde_json::from_slice(&header_bytes)
            .map_err(|_| QuixigateError::Unauthorized("invalid JWT header JSON".to_owned()))?;

        let alg = header
            .get("alg")
            .and_then(Value::as_str)
            .ok_or_else(|| QuixigateError::Unauthorized("JWT header missing 'alg'".to_owned()))?;
        if !SUPPORTED_ASYMMETRIC_ALGS.contains(&alg) {
            return Err(QuixigateError::Unauthorized(format!(
                "unsupported OIDC JWT alg '{alg}'"
            )));
        }
        let kid = header
            .get("kid")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());

        let sig_bytes = b64url_decode(sig_b64).ok_or_else(|| {
            QuixigateError::Unauthorized("invalid JWT signature base64url".to_owned())
        })?;
        let signing_input = format!("{header_b64}.{payload_b64}");

        let discovered_issuer = {
            let snap = self.inner.snapshot.read();
            if snap.keys.is_empty() {
                return Err(QuixigateError::Unauthorized(
                    "OIDC JWKS keys unavailable or could not be fetched from configured issuer/JWKS URL"
                        .to_owned(),
                ));
            }
            let verified = snap.keys.iter().any(|jwk| {
                if let Some(req_kid) = kid
                    && let Some(jwk_kid) = jwk.kid.as_deref()
                    && jwk_kid != req_kid
                {
                    return false;
                }
                if let Some(jwk_alg) = jwk.alg.as_deref()
                    && jwk_alg != alg
                {
                    return false;
                }
                verify_jwk_signature(&jwk.key, alg, signing_input.as_bytes(), &sig_bytes)
            });
            if !verified {
                return Err(QuixigateError::Unauthorized(
                    "JWT signature verification failed against configured OIDC JWKS".to_owned(),
                ));
            }
            snap.discovered_issuer.clone()
        };

        let payload_bytes = b64url_decode(payload_b64).ok_or_else(|| {
            QuixigateError::Unauthorized("invalid JWT payload base64url".to_owned())
        })?;
        let payload: Value = serde_json::from_slice(&payload_bytes)
            .map_err(|_| QuixigateError::Unauthorized("invalid JWT payload JSON".to_owned()))?;

        let now_sec = now_ms() / 1_000;
        let exp = payload.get("exp").and_then(Value::as_i64).ok_or_else(|| {
            QuixigateError::Unauthorized("OIDC JWT missing required 'exp' claim".to_owned())
        })?;
        if exp.saturating_add(CLOCK_SKEW_SECS) < now_sec {
            return Err(QuixigateError::Unauthorized("JWT has expired".to_owned()));
        }
        if let Some(nbf) = payload.get("nbf").and_then(Value::as_i64)
            && nbf > now_sec.saturating_add(CLOCK_SKEW_SECS)
        {
            return Err(QuixigateError::Unauthorized(
                "JWT is not yet valid (nbf)".to_owned(),
            ));
        }

        // Validate `iss` if `jwt_issuer_url` is configured
        if let Some(cfg_iss) = settings
            .jwt_issuer_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let token_iss = payload
                .get("iss")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim_end_matches('/');
            let expected_cfg = cfg_iss.trim_end_matches('/');
            let expected_disc = discovered_issuer
                .as_deref()
                .map(|s| s.trim_end_matches('/'));
            if token_iss.is_empty()
                || (token_iss != expected_cfg && Some(token_iss) != expected_disc)
            {
                return Err(QuixigateError::Unauthorized(format!(
                    "JWT issuer '{token_iss}' does not match configured OIDC issuer '{expected_cfg}'"
                )));
            }
        }

        // Validate `aud` if `jwt_audience` is configured
        if let Some(cfg_aud) = settings
            .jwt_audience
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let accepted: Vec<&str> = cfg_aud
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            let token_auds = extract_claim_strings(&payload, "aud");
            if !token_auds
                .iter()
                .any(|ta| accepted.iter().any(|acc| acc == ta))
            {
                return Err(QuixigateError::Unauthorized(
                    "JWT 'aud' claim does not match configured jwt_audience".to_owned(),
                ));
            }
        }

        // Resolve user identity from `jwt_user_claim` (fallback: `sub` -> `email` -> `preferred_username`)
        let subject = extract_claim_strings(&payload, &settings.jwt_user_claim)
            .into_iter()
            .next()
            .or_else(|| extract_claim_strings(&payload, "sub").into_iter().next())
            .or_else(|| extract_claim_strings(&payload, "email").into_iter().next())
            .or_else(|| {
                extract_claim_strings(&payload, "preferred_username")
                    .into_iter()
                    .next()
            })
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                QuixigateError::Unauthorized(format!(
                    "OIDC JWT missing user claim '{}'",
                    settings.jwt_user_claim
                ))
            })?;

        let roles = extract_claim_strings(&payload, &settings.jwt_roles_claim);
        let grants_admin = roles.iter().any(|r| {
            settings
                .jwt_admin_roles
                .iter()
                .any(|adm| adm.eq_ignore_ascii_case(r.trim()))
        });

        let display_name = payload
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(&subject);

        let user = match state.db.get_user(&subject) {
            Ok(existing) => {
                if grants_admin && existing.role != UserRole::Admin {
                    state.db.update_user(
                        &existing.id,
                        None,
                        Some(UserRole::Admin),
                        None,
                        None,
                        None,
                    )?
                } else {
                    existing
                }
            }
            Err(_) if settings.jwt_auto_provision_users => state.db.create_user(
                &subject,
                display_name,
                if grants_admin {
                    UserRole::Admin
                } else {
                    UserRole::Member
                },
                None,
                None,
                BudgetWindow::Monthly,
            )?,
            Err(_) => {
                return Err(QuixigateError::Unauthorized(format!(
                    "OIDC user '{subject}' does not exist and jwt_auto_provision_users is disabled"
                )));
            }
        };

        if user.blocked {
            return Err(QuixigateError::Forbidden(format!(
                "user '{}' is blocked",
                user.username
            )));
        }

        // Map team claims (`jwt_team_claim`, `team_id`, `quixigate_team`) to existing QuixiGate teams
        let mut candidate_teams = Vec::new();
        if let Some(team_claim) = settings
            .jwt_team_claim
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            candidate_teams.extend(extract_claim_strings(&payload, team_claim));
        }
        for fallback_claim in ["quixigate_team", "team_id"] {
            for t in extract_claim_strings(&payload, fallback_claim) {
                if !candidate_teams.contains(&t) {
                    candidate_teams.push(t);
                }
            }
        }

        let mut primary_team_id = None;
        for candidate in &candidate_teams {
            if let Ok(Some(resolved_tid)) = state.db.ensure_team_membership(candidate, &user.id)
                && primary_team_id.is_none()
            {
                primary_team_id = Some(resolved_tid);
            }
        }

        Ok(AuthenticatedPrincipal {
            user,
            virtual_key: None,
            via_session: false,
            via_loopback_trust: false,
            via_master_key: false,
            jwt_team_id: primary_team_id,
        })
    }
}

pub fn peek_asymmetric_jwt_header(raw_token: &str) -> Option<(String, Option<String>)> {
    let mut parts = raw_token.split('.');
    let (header_b64, _payload_b64, _sig_b64, None) =
        (parts.next()?, parts.next()?, parts.next()?, parts.next())
    else {
        return None;
    };
    let header_bytes = b64url_decode(header_b64)?;
    let header: Value = serde_json::from_slice(&header_bytes).ok()?;
    let alg = header.get("alg")?.as_str()?;
    if !SUPPORTED_ASYMMETRIC_ALGS.contains(&alg) {
        return None;
    }
    let kid = header
        .get("kid")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Some((alg.to_owned(), kid))
}

fn config_fingerprint(settings: &GatewaySettings) -> String {
    format!(
        "{}|{}",
        settings.jwt_issuer_url.as_deref().unwrap_or("").trim(),
        settings.jwt_jwks_url.as_deref().unwrap_or("").trim()
    )
}

fn b64url_decode(input: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD
        .decode(input.trim())
        .or_else(|_| URL_SAFE.decode(input.trim()))
        .ok()
}

fn strip_leading_zero_bytes(bytes: &[u8]) -> Vec<u8> {
    let first_nonzero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
    if first_nonzero == bytes.len() {
        vec![0]
    } else {
        bytes[first_nonzero..].to_vec()
    }
}

fn parse_jwk(item: &Value) -> Option<ParsedJwk> {
    if item.get("use").and_then(Value::as_str) == Some("enc") {
        return None;
    }
    let kty = item.get("kty")?.as_str()?;
    let kid = item
        .get("kid")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let alg = item
        .get("alg")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);

    let key = match kty {
        "RSA" => {
            let n = strip_leading_zero_bytes(&b64url_decode(item.get("n")?.as_str()?)?);
            let e = strip_leading_zero_bytes(&b64url_decode(item.get("e")?.as_str()?)?);
            JwkPublicKey::Rsa { n, e }
        }
        "EC" => {
            let crv = item.get("crv")?.as_str()?;
            let x = b64url_decode(item.get("x")?.as_str()?)?;
            let y = b64url_decode(item.get("y")?.as_str()?)?;
            match crv {
                "P-256" if x.len() == 32 && y.len() == 32 => {
                    let mut uncompressed = [0u8; 65];
                    uncompressed[0] = 0x04;
                    uncompressed[1..33].copy_from_slice(&x);
                    uncompressed[33..65].copy_from_slice(&y);
                    JwkPublicKey::EcP256 { uncompressed }
                }
                "P-384" if x.len() == 48 && y.len() == 48 => {
                    let mut uncompressed = [0u8; 97];
                    uncompressed[0] = 0x04;
                    uncompressed[1..49].copy_from_slice(&x);
                    uncompressed[49..97].copy_from_slice(&y);
                    JwkPublicKey::EcP384 { uncompressed }
                }
                _ => return None,
            }
        }
        "OKP" => {
            if item.get("crv")?.as_str()? != "Ed25519" {
                return None;
            }
            let x = b64url_decode(item.get("x")?.as_str()?)?;
            let public: [u8; 32] = x.try_into().ok()?;
            JwkPublicKey::Ed25519 { public }
        }
        _ => return None,
    };

    Some(ParsedJwk { kid, alg, key })
}

fn verify_jwk_signature(key: &JwkPublicKey, alg: &str, msg: &[u8], sig: &[u8]) -> bool {
    match (key, alg) {
        (JwkPublicKey::Rsa { n, e }, "RS256") => RsaPublicKeyComponents { n, e }
            .verify(&signature::RSA_PKCS1_2048_8192_SHA256, msg, sig)
            .is_ok(),
        (JwkPublicKey::Rsa { n, e }, "RS384") => RsaPublicKeyComponents { n, e }
            .verify(&signature::RSA_PKCS1_2048_8192_SHA384, msg, sig)
            .is_ok(),
        (JwkPublicKey::Rsa { n, e }, "RS512") => RsaPublicKeyComponents { n, e }
            .verify(&signature::RSA_PKCS1_2048_8192_SHA512, msg, sig)
            .is_ok(),
        (JwkPublicKey::Rsa { n, e }, "PS256") => RsaPublicKeyComponents { n, e }
            .verify(&signature::RSA_PSS_2048_8192_SHA256, msg, sig)
            .is_ok(),
        (JwkPublicKey::Rsa { n, e }, "PS384") => RsaPublicKeyComponents { n, e }
            .verify(&signature::RSA_PSS_2048_8192_SHA384, msg, sig)
            .is_ok(),
        (JwkPublicKey::Rsa { n, e }, "PS512") => RsaPublicKeyComponents { n, e }
            .verify(&signature::RSA_PSS_2048_8192_SHA512, msg, sig)
            .is_ok(),
        (JwkPublicKey::EcP256 { uncompressed }, "ES256") => {
            UnparsedPublicKey::new(&signature::ECDSA_P256_SHA256_FIXED, uncompressed)
                .verify(msg, sig)
                .is_ok()
        }
        (JwkPublicKey::EcP384 { uncompressed }, "ES384") => {
            UnparsedPublicKey::new(&signature::ECDSA_P384_SHA384_FIXED, uncompressed)
                .verify(msg, sig)
                .is_ok()
        }
        (JwkPublicKey::Ed25519 { public }, "EdDSA") => {
            UnparsedPublicKey::new(&signature::ED25519, public)
                .verify(msg, sig)
                .is_ok()
        }
        _ => false,
    }
}

async fn fetch_json_bounded(
    http: &reqwest::Client,
    url: &str,
) -> std::result::Result<Value, String> {
    let trimmed = url.trim();
    if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
        return Err(format!("unsupported OIDC/JWKS URL scheme in '{trimmed}'"));
    }
    let resp = http
        .get(trimmed)
        .send()
        .await
        .map_err(|e| format!("GET {trimmed} failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {trimmed} returned HTTP {}", resp.status()));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reading {trimmed} failed: {e}"))?;
    if bytes.len() > 512 * 1024 {
        return Err(format!("{trimmed} exceeded 512 KiB JWKS size cap"));
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON from {trimmed}: {e}"))
}

async fn fetch_jwks_snapshot(
    http: &reqwest::Client,
    settings: &GatewaySettings,
    fingerprint: &str,
) -> std::result::Result<JwksSnapshot, String> {
    let mut jwks_urls: Vec<String> = settings
        .jwt_jwks_url
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();

    let mut discovered_issuer = None;
    if let Some(issuer_raw) = settings
        .jwt_issuer_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let base = issuer_raw.trim_end_matches('/');
        let discovery_url = if base.ends_with("/.well-known/openid-configuration") {
            base.to_owned()
        } else {
            format!("{base}/.well-known/openid-configuration")
        };
        match fetch_json_bounded(http, &discovery_url).await {
            Ok(doc) => {
                discovered_issuer = doc.get("issuer").and_then(Value::as_str).map(str::to_owned);
                if jwks_urls.is_empty()
                    && let Some(uri) = doc.get("jwks_uri").and_then(Value::as_str)
                {
                    jwks_urls.push(uri.trim().to_owned());
                }
            }
            Err(disc_err) => {
                if jwks_urls.is_empty() {
                    // Cloudflare Access fallback (`<team-domain>/cdn-cgi/access/certs`)
                    jwks_urls.push(format!("{base}/cdn-cgi/access/certs"));
                } else {
                    tracing::debug!(
                        "OIDC discovery skipped ({disc_err}); using explicit jwt_jwks_url"
                    );
                }
            }
        }
    }

    if jwks_urls.is_empty() {
        return Err("no JWKS URL configured or discovered".to_owned());
    }

    let mut parsed_keys = Vec::new();
    let mut last_err = None;
    for url in &jwks_urls {
        match fetch_json_bounded(http, url).await {
            Ok(doc) => {
                if let Some(arr) = doc.get("keys").and_then(Value::as_array) {
                    for item in arr {
                        if let Some(jwk) = parse_jwk(item) {
                            parsed_keys.push(jwk);
                        }
                    }
                }
            }
            Err(e) => last_err = Some(e),
        }
    }

    if parsed_keys.is_empty() {
        return Err(
            last_err.unwrap_or_else(|| "JWKS contained zero supported signing keys".to_owned())
        );
    }

    let now = now_ms();
    Ok(JwksSnapshot {
        config_fingerprint: fingerprint.to_owned(),
        discovered_issuer,
        keys: parsed_keys,
        fetched_at_ms: now,
        last_attempt_ms: now,
    })
}

fn extract_claim_strings(payload: &Value, claim_path: &str) -> Vec<String> {
    let trimmed = claim_path.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut cur = payload;
    for seg in trimmed.split('.') {
        let Some(next) = cur.get(seg) else {
            return Vec::new();
        };
        cur = next;
    }
    match cur {
        Value::String(s) => s
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
            .collect(),
        Value::Number(n) => vec![n.to_string()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}
