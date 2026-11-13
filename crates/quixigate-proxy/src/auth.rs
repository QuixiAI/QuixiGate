//! Request authentication & authorization across polyglot SDK headers,
//! `QUIXIGATE_MASTER_KEY` / `LITELLM_MASTER_KEY`, `sk-qx-...` virtual keys,
//! `HttpOnly` / bearer admin sessions (`qxs_...`), and per-connection TCP
//! loopback trust (`127.0.0.1` / `::1`) with reverse-proxy, DNS-rebinding, and CSRF guards.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use axum::Json;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use futures::StreamExt;
use quixigate_core::{QuixigateError, Result, UserRole, now_ms};
use quixigate_db::AuthenticatedPrincipal;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::state::AppState;

pub const SESSION_COOKIE_NAME: &str = quixigate_core::brand::SESSION_COOKIE_NAME;

/// Non-spoofable internal marker headers stripped and recomputed on every incoming request
/// by [`peer_trust_middleware`] from the real TCP `ConnectInfo<SocketAddr>`.
const INTERNAL_DIRECT_LOOPBACK_HEADER: &str = "x-quixigate-internal-direct-loopback";
const INTERNAL_PEER_CHECKED_HEADER: &str = "x-quixigate-internal-peer-checked";
const INTERNAL_PEER_IP_HEADER: &str = "x-quixigate-internal-peer-ip";

const PROXY_FORWARDING_HEADERS: &[&str] = &[
    "x-forwarded-for",
    "forwarded",
    "x-real-ip",
    "cf-connecting-ip",
    "true-client-ip",
    "fly-client-ip",
];

/// Standard container-to-host DNS aliases used by Docker Desktop, Linux Docker (`host-gateway`),
/// OrbStack, Podman, Colima/Lima, Rancher Desktop, and Minikube.
const CONTAINER_HOST_ALIASES: &[&str] = &[
    "host.docker.internal",
    "gateway.docker.internal",
    "docker.for.mac.localhost",
    "docker.for.mac.host.internal",
    "docker.for.win.localhost",
    "host.containers.internal",
    "host.orb.internal",
    "host.lima.internal",
    "host.rancher-desktop.internal",
    "host.minikube.internal",
];

/// Built-in local container bridge / user-mode NAT subnets:
/// - `172.16.0.0/12`: Linux `docker0` (`172.17.0.0/16`) + `docker compose` custom bridges (`172.18..31.x.x`)
/// - `192.168.65.0/24`: Docker Desktop VPNKit / `gvproxy` VM subnet
/// - `198.18.0.0/15`: OrbStack macOS local container/VM subnet
/// - `10.88.0.0/15`: Podman default CNI / Netavark bridges (`10.88.0.0/16` & `10.89.0.0/16`)
/// - `10.0.2.0/24` & `192.168.5.0/24`: Rootless SLIRP / QEMU / Lima / Colima user-v2 NAT
const LOCAL_CONTAINER_CIDRS: &[&str] = &[
    "172.16.0.0/12",
    "192.168.65.0/24",
    "198.18.0.0/15",
    "10.88.0.0/15",
    "10.0.2.0/24",
    "192.168.5.0/24",
];

fn ipv4_in_cidr(ip: Ipv4Addr, net: Ipv4Addr, prefix_len: u8) -> bool {
    if prefix_len > 32 {
        return false;
    }
    let mask = if prefix_len == 0 {
        0
    } else {
        !0u32 << (32 - prefix_len)
    };
    (u32::from(ip) & mask) == (u32::from(net) & mask)
}

fn ipv6_in_cidr(ip: Ipv6Addr, net: Ipv6Addr, prefix_len: u8) -> bool {
    if prefix_len > 128 {
        return false;
    }
    let mask = if prefix_len == 0 {
        0
    } else {
        !0u128 << (128 - prefix_len)
    };
    (u128::from(ip) & mask) == (u128::from(net) & mask)
}

pub fn matches_cidr_str(ip: IpAddr, cidr_raw: &str) -> bool {
    let trimmed = cidr_raw.trim();
    if trimmed.is_empty() {
        return false;
    }
    let ip = ip.to_canonical();
    if let Some((net_str, prefix_str)) = trimmed.split_once('/') {
        let Ok(prefix_len) = prefix_str.trim().parse::<u8>() else {
            return false;
        };
        let Ok(net_ip) = net_str.trim().parse::<IpAddr>() else {
            return false;
        };
        match (ip, net_ip.to_canonical()) {
            (IpAddr::V4(a), IpAddr::V4(b)) => ipv4_in_cidr(a, b, prefix_len),
            (IpAddr::V6(a), IpAddr::V6(b)) => ipv6_in_cidr(a, b, prefix_len),
            _ => false,
        }
    } else if let Ok(single_ip) = trimmed.parse::<IpAddr>() {
        ip == single_ip.to_canonical()
    } else {
        false
    }
}

pub fn is_local_container_ip(ip: IpAddr) -> bool {
    let canon = ip.to_canonical();
    LOCAL_CONTAINER_CIDRS
        .iter()
        .any(|cidr| matches_cidr_str(canon, cidr))
}

fn extract_host_only(host_raw: &str) -> &str {
    let trimmed = host_raw.trim();
    if let Some(rest) = trimmed.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        trimmed.split(':').next().unwrap_or(trimmed)
    }
}

fn is_trusted_hostname(host_raw: &str, trust_containers: bool, trusted_cidrs: &[String]) -> bool {
    let trimmed = host_raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    let host_only = extract_host_only(trimmed);
    let lower = host_only.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "localhost" | "127.0.0.1" | "::1" | "0.0.0.0"
    ) {
        return true;
    }
    if trust_containers && CONTAINER_HOST_ALIASES.contains(&lower.as_str()) {
        return true;
    }
    // Literal IP addresses in `Host` (never domain names, preserving DNS-rebinding defense)
    if let Ok(ip) = host_only.parse::<IpAddr>() {
        let canon = ip.to_canonical();
        if canon.is_loopback() {
            return true;
        }
        if trust_containers && is_local_container_ip(canon) {
            return true;
        }
        if trusted_cidrs.iter().any(|c| matches_cidr_str(canon, c)) {
            return true;
        }
    }
    false
}

fn is_trusted_origin(origin_raw: &str, trust_containers: bool, trusted_cidrs: &[String]) -> bool {
    let trimmed = origin_raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return false;
    };
    if !matches!(scheme.to_ascii_lowercase().as_str(), "http" | "https") {
        return false;
    }
    let authority = rest.split('/').next().unwrap_or(rest);
    is_trusted_hostname(authority, trust_containers, trusted_cidrs)
}

/// Outermost Axum middleware that strips any client-supplied internal headers, enforces
/// gateway-wide `allowed_cidrs` and live `max_request_size_mb`, and marks whether the request
/// arrived directly from a local loopback socket (`127.0.0.1` / `::1`) or an authorized local
/// container runtime (`host.docker.internal` / local Docker bridge) without reverse-proxy
/// forwarding headers, DNS rebinding, or cross-origin browser spoofing.
pub async fn peer_trust_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    req.headers_mut().remove(INTERNAL_DIRECT_LOOPBACK_HEADER);
    req.headers_mut().remove(INTERNAL_PEER_CHECKED_HEADER);
    req.headers_mut().remove(INTERNAL_PEER_IP_HEADER);

    let settings = state.db.get_settings().unwrap_or_default();
    let trust_containers = settings.trust_local_containers;
    let trusted_cidrs = &settings.trusted_cidrs;

    let peer_ip: Option<IpAddr> = match req.extensions().get::<ConnectInfo<SocketAddr>>() {
        Some(ConnectInfo(addr)) => Some(addr.ip().to_canonical()),
        None if state.bound_loopback => Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        None => None,
    };

    if let Some(ip) = peer_ip
        && let Ok(hv) = HeaderValue::from_str(&ip.to_string())
    {
        req.headers_mut().insert(INTERNAL_PEER_IP_HEADER, hv);
    }

    // Gateway-wide IP/CIDR allowlist (`settings.allowed_cidrs` / `QUIXIGATE_ALLOWED_IPS`).
    // Liveness probes remain reachable so container orchestrators don't restart pods.
    let path = req.uri().path();
    let is_liveness = matches!(path, "/health" | "/health/liveness");
    if !is_liveness && !settings.allowed_cidrs.is_empty() {
        let allowed = peer_ip.is_some_and(|ip| {
            settings
                .allowed_cidrs
                .iter()
                .any(|cidr| matches_cidr_str(ip, cidr))
        });
        if !allowed {
            let ip_label = peer_ip
                .map(|ip| ip.to_string())
                .unwrap_or_else(|| "unknown".to_owned());
            return (
                StatusCode::FORBIDDEN,
                Json(json!({
                    "type": "error",
                    "error": {
                        "message": format!("client IP '{ip_label}' is not permitted by gateway allowed_cidrs policy"),
                        "type": "permission_error",
                        "code": "ip_not_allowed"
                    }
                })),
            )
                .into_response();
        }
    }

    // Live-configurable request body limit (`settings.max_request_size_mb` / `MAX_REQUEST_SIZE_MB`).
    let max_mb = settings.max_request_size_mb.clamp(1, 512);
    let max_bytes = max_mb.saturating_mul(1024 * 1024);
    if let Some(cl_raw) = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        && let Ok(cl) = cl_raw.trim().parse::<usize>()
        && cl > max_bytes
    {
        return payload_too_large_response(max_mb);
    }

    let tcp_peer_trusted = match req.extensions().get::<ConnectInfo<SocketAddr>>() {
        Some(ConnectInfo(addr)) => {
            let ip = addr.ip().to_canonical();
            ip.is_loopback()
                || (trust_containers && is_local_container_ip(ip))
                || trusted_cidrs.iter().any(|c| matches_cidr_str(ip, c))
        }
        None => state.bound_loopback,
    };

    let has_proxy_header = PROXY_FORWARDING_HEADERS
        .iter()
        .any(|h| req.headers().contains_key(*h));

    let host_ok = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|h| is_trusted_hostname(h, trust_containers, trusted_cidrs));

    let origin_ok = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|o| is_trusted_origin(o, trust_containers, trusted_cidrs));

    req.headers_mut()
        .insert(INTERNAL_PEER_CHECKED_HEADER, HeaderValue::from_static("1"));

    if tcp_peer_trusted && !has_proxy_header && host_ok && origin_ok {
        req.headers_mut().insert(
            INTERNAL_DIRECT_LOOPBACK_HEADER,
            HeaderValue::from_static("1"),
        );
    }

    // Warm OIDC Discovery / JWKS cache asynchronously before synchronous route handlers run.
    let query_key_opt = req.uri().query().and_then(|q| {
        q.split('&').find_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            (k == "key").then_some(v)
        })
    });
    if let Some(tok) = extract_api_token(req.headers(), query_key_opt) {
        state.oidc.warm_for_token_if_needed(&settings, &tok).await;
    }

    // Enforce byte cap even on chunked transfer-encoded streams without `Content-Length`.
    let (parts, body) = req.into_parts();
    let exceeded_cap = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let exceeded_flag = exceeded_cap.clone();
    let mut seen_bytes = 0usize;
    let limited_stream = body.into_data_stream().map(move |chunk_res| {
        let chunk = chunk_res.map_err(std::io::Error::other)?;
        seen_bytes = seen_bytes.saturating_add(chunk.len());
        if seen_bytes > max_bytes {
            exceeded_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            return Err(std::io::Error::other(
                "request body exceeds max_request_size_mb",
            ));
        }
        Ok::<_, std::io::Error>(chunk)
    });
    let req = Request::from_parts(parts, Body::from_stream(limited_stream));

    let resp = next.run(req).await;
    if exceeded_cap.load(std::sync::atomic::Ordering::Relaxed)
        || resp.status() == StatusCode::PAYLOAD_TOO_LARGE
    {
        return payload_too_large_response(max_mb);
    }
    resp
}

fn payload_too_large_response(max_mb: usize) -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({
            "type": "error",
            "error": {
                "message": format!("request payload exceeds max_request_size_mb ({max_mb} MiB)"),
                "type": "invalid_request_error",
                "code": "payload_too_large"
            }
        })),
    )
        .into_response()
}

/// Split-plane middleware blocking data-plane LLM endpoints when `settings.disable_inference_endpoints` is true.
pub async fn enforce_inference_plane_enabled(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    if state
        .db
        .get_settings()
        .map(|s| s.disable_inference_endpoints)
        .unwrap_or(false)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "type": "error",
                "error": {
                    "message": "inference data-plane endpoints are disabled on this gateway instance (disable_inference_endpoints=true)",
                    "type": "permission_error",
                    "code": "inference_endpoints_disabled"
                }
            })),
        )
            .into_response();
    }
    next.run(req).await
}

/// Split-plane middleware blocking control-plane Admin & LiteLLM management endpoints when `settings.disable_admin_endpoints` is true.
pub async fn enforce_admin_plane_enabled(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    if state
        .db
        .get_settings()
        .map(|s| s.disable_admin_endpoints)
        .unwrap_or(false)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "type": "error",
                "error": {
                    "message": "admin and management control-plane endpoints are disabled on this gateway instance (disable_admin_endpoints=true)",
                    "type": "permission_error",
                    "code": "admin_endpoints_disabled"
                }
            })),
        )
            .into_response();
    }
    next.run(req).await
}

/// Extracts the verified peer IP recorded by [`peer_trust_middleware`].
pub fn verified_peer_ip(state: &AppState, headers: &HeaderMap) -> Option<IpAddr> {
    if let Some(ip_str) = headers
        .get(INTERNAL_PEER_IP_HEADER)
        .and_then(|v| v.to_str().ok())
        && let Ok(ip) = ip_str.parse::<IpAddr>()
    {
        return Some(ip.to_canonical());
    }
    if !headers.contains_key(INTERNAL_PEER_CHECKED_HEADER) && state.bound_loopback {
        return Some(IpAddr::V4(Ipv4Addr::LOCALHOST));
    }
    None
}

/// Returns true iff this request came from a verified direct local loopback connection.
pub fn is_direct_loopback(state: &AppState, headers: &HeaderMap) -> bool {
    if headers.contains_key(INTERNAL_PEER_CHECKED_HEADER) {
        headers
            .get(INTERNAL_DIRECT_LOOPBACK_HEADER)
            .and_then(|v| v.to_str().ok())
            == Some("1")
    } else {
        // Direct unit-test helper calls that did not pass through the router layer.
        state.bound_loopback
            && !PROXY_FORWARDING_HEADERS
                .iter()
                .any(|h| headers.contains_key(*h))
    }
}

/// Extracts raw token candidate from `Authorization: Bearer ...`, `x-quixigate-admin-key`,
/// `x-litellm-api-key`, Anthropic `x-api-key`, Google `x-goog-api-key`, or Gemini `?key=`.
pub fn extract_api_token(headers: &HeaderMap, query_key: Option<&str>) -> Option<String> {
    for hdr_name in ["x-quixigate-admin-key", "x-litellm-api-key"] {
        if let Some(val) = headers.get(hdr_name).and_then(|v| v.to_str().ok()) {
            let trimmed = val.trim();
            let tok = trimmed
                .strip_prefix("Bearer ")
                .or_else(|| trimmed.strip_prefix("bearer "))
                .map(str::trim)
                .unwrap_or(trimmed);
            if !tok.is_empty() {
                return Some(tok.to_owned());
            }
        }
    }
    if let Some(auth) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let trimmed = auth.trim();
        if let Some(rest) = trimmed
            .strip_prefix("Bearer ")
            .or_else(|| trimmed.strip_prefix("bearer "))
        {
            let tok = rest.trim();
            if !tok.is_empty() {
                return Some(tok.to_owned());
            }
        }
    }
    for hdr_name in [
        "x-api-key",
        "x-goog-api-key",
        "api-key",
        "x-litellm-api-key",
    ] {
        if let Some(val) = headers.get(hdr_name).and_then(|v| v.to_str().ok()) {
            let tok = val.trim();
            if !tok.is_empty() {
                return Some(tok.to_owned());
            }
        }
    }
    if let Some(qk) = query_key {
        let tok = qk.trim();
        if !tok.is_empty() {
            return Some(tok.to_owned());
        }
    }
    None
}

/// Extracts `quixigate_session` cookie value if present.
pub fn extract_session_cookie(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    for pair in raw.split(';') {
        if let Some((k, v)) = pair.trim().split_once('=')
            && k.trim() == SESSION_COOKIE_NAME
        {
            let val = v.trim();
            if !val.is_empty() {
                return Some(val.to_owned());
            }
        }
    }
    None
}

/// Builds a strict `Set-Cookie` header for `quixigate_session`.
pub fn format_session_cookie(token: &str, max_age_secs: u64, secure: bool) -> String {
    let secure_flag = if secure { "; Secure" } else { "" };
    format!(
        "{SESSION_COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age_secs}{secure_flag}"
    )
}

pub fn format_clear_session_cookie() -> String {
    format!("{SESSION_COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
}

/// RFC 2104 `HMAC-SHA256` computed in-process using `sha2::Sha256` (zero extra crates).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        k_block[..32].copy_from_slice(&hashed);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_digest);
    outer.finalize().into()
}

/// Signs an `HS256` JSON Web Token (`header.payload.signature`) for tests and CLI utilities.
pub fn sign_hs256_jwt(claims: &Value, secret: &[u8]) -> String {
    let header_b64 = URL_SAFE_NO_PAD.encode(r#"{"alg":"HS256","typ":"JWT"}"#.as_bytes());
    let payload_b64 = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let mac = hmac_sha256(secret, signing_input.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(mac);
    format!("{signing_input}.{sig_b64}")
}

fn verify_hs256_jwt(
    state: &AppState,
    raw_token: &str,
    secret: &[u8],
) -> Result<AuthenticatedPrincipal> {
    let mut parts = raw_token.split('.');
    let (Some(header_b64), Some(payload_b64), Some(sig_b64), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(QuixigateError::Unauthorized(
            "malformed JWT structure".to_owned(),
        ));
    };

    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|_| QuixigateError::Unauthorized("invalid JWT header base64url".to_owned()))?;
    let header_json: Value = serde_json::from_slice(&header_bytes)
        .map_err(|_| QuixigateError::Unauthorized("invalid JWT header JSON".to_owned()))?;
    if header_json.get("alg").and_then(Value::as_str) != Some("HS256") {
        return Err(QuixigateError::Unauthorized(
            "unsupported JWT alg (only HS256 is accepted)".to_owned(),
        ));
    }

    let sig_bytes = URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|_| QuixigateError::Unauthorized("invalid JWT signature base64url".to_owned()))?;
    let signing_input = format!("{header_b64}.{payload_b64}");
    let expected_mac = hmac_sha256(secret, signing_input.as_bytes());
    if sig_bytes.len() != 32 || sig_bytes.as_slice().ct_eq(&expected_mac).unwrap_u8() != 1 {
        return Err(QuixigateError::Unauthorized(
            "invalid JWT signature".to_owned(),
        ));
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| QuixigateError::Unauthorized("invalid JWT payload base64url".to_owned()))?;
    let payload: Value = serde_json::from_slice(&payload_bytes)
        .map_err(|_| QuixigateError::Unauthorized("invalid JWT payload JSON".to_owned()))?;

    let now_secs = now_ms() / 1000;
    if let Some(exp) = payload.get("exp").and_then(Value::as_i64)
        && now_secs >= exp
    {
        return Err(QuixigateError::Unauthorized("JWT has expired".to_owned()));
    }
    if let Some(nbf) = payload.get("nbf").and_then(Value::as_i64)
        && now_secs + 30 < nbf
    {
        return Err(QuixigateError::Unauthorized(
            "JWT is not yet valid (nbf)".to_owned(),
        ));
    }

    let sub = payload
        .get("sub")
        .or_else(|| payload.get("user_id"))
        .or_else(|| payload.get("email"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| QuixigateError::Unauthorized("JWT missing 'sub' claim".to_owned()))?;
    let email = payload.get("email").and_then(Value::as_str);
    let role_raw = payload
        .get("role")
        .or_else(|| payload.get("quixigate_role"))
        .or_else(|| payload.get("user_role"))
        .and_then(Value::as_str)
        .unwrap_or("member");
    let role = match role_raw.trim().to_ascii_lowercase().as_str() {
        "admin" | "proxy_admin" => UserRole::Admin,
        "viewer" | "internal_user_viewer" | "proxy_admin_viewer" => UserRole::Viewer,
        _ => UserRole::Member,
    };

    let user = state.db.ensure_external_user(sub, email, role)?;
    let jwt_team_id = payload
        .get("team_id")
        .or_else(|| payload.get("quixigate_team"))
        .and_then(Value::as_str)
        .and_then(|t| state.db.ensure_team_membership(t, &user.id).ok().flatten());
    Ok(AuthenticatedPrincipal {
        user,
        virtual_key: None,
        via_session: false,
        via_loopback_trust: false,
        via_master_key: false,
        jwt_team_id,
    })
}

fn enforce_principal_policies(
    state: &AppState,
    headers: &HeaderMap,
    principal: AuthenticatedPrincipal,
) -> Result<AuthenticatedPrincipal> {
    if !principal.via_master_key && principal.user.blocked {
        return Err(QuixigateError::Forbidden(format!(
            "user '{}' is blocked",
            principal.user.id
        )));
    }
    if let Some(vk) = &principal.virtual_key
        && !vk.allowed_cidrs.is_empty()
    {
        let peer_ip = verified_peer_ip(state, headers);
        let allowed = peer_ip.is_some_and(|ip| {
            vk.allowed_cidrs
                .iter()
                .any(|cidr| matches_cidr_str(ip, cidr))
        });
        if !allowed {
            let ip_label = peer_ip
                .map(|ip| ip.to_string())
                .unwrap_or_else(|| "unknown".to_owned());
            return Err(QuixigateError::Forbidden(format!(
                "client IP '{ip_label}' is not permitted by virtual key '{}' allowed_cidrs policy",
                vk.id
            )));
        }
    }
    Ok(principal)
}

/// Authenticates an inference, model-discovery, MCP, or management request:
/// 1. Valid `QUIXIGATE_MASTER_KEY` / `LITELLM_MASTER_KEY` (or startup-generated Master Key)
///    authenticates immediately in constant time as the root `Admin` principal.
/// 2. Any `sk-qx-...` token is ALWAYS verified (even on loopback!) so per-key budgets,
///    RPM/TPM limits, allowed-model globs, per-key `allowed_cidrs`, and spend attribution are strictly enforced.
/// 3. Any `qxs_...` bearer or `quixigate_session` cookie authenticates that session user.
/// 4. Inbound JWTs (`RS*`/`PS*`/`ES*`/`EdDSA` via OIDC JWKS, or `HS256` via `QUIXIGATE_JWT_SECRET`) are verified and mapped to QuixiGate users & teams.
/// 5. When `!settings.require_auth` and the TCP peer is a direct local loopback connection
///    (`127.0.0.1` / `::1`) without explicit invalid admin headers, grants zero-friction
///    loopback trust as `local`; all non-loopback connections ALWAYS require a valid key.
pub fn authenticate_inference(
    state: &AppState,
    headers: &HeaderMap,
    query_key: Option<&str>,
) -> Result<AuthenticatedPrincipal> {
    let settings = state.db.get_settings()?;
    let direct_loopback = is_direct_loopback(state, headers);
    let explicit_admin_hdr =
        headers.contains_key("x-quixigate-admin-key") || headers.contains_key("x-litellm-api-key");

    if let Some(raw_token) = extract_api_token(headers, query_key) {
        if state.verify_master_key(&raw_token) {
            let mut user = state.db.get_user("local")?;
            user.role = UserRole::Admin;
            return Ok(AuthenticatedPrincipal {
                user,
                virtual_key: None,
                via_session: false,
                via_loopback_trust: false,
                via_master_key: true,
                jwt_team_id: None,
            });
        }
        if quixigate_core::brand::is_virtual_key_token(&raw_token) {
            let principal = state.db.verify_virtual_key(&raw_token)?;
            return enforce_principal_policies(state, headers, principal);
        }
        if quixigate_core::brand::is_session_token(&raw_token) {
            let user = state.db.validate_session(&raw_token)?;
            return enforce_principal_policies(
                state,
                headers,
                AuthenticatedPrincipal {
                    user,
                    virtual_key: None,
                    via_session: true,
                    via_loopback_trust: false,
                    via_master_key: false,
                    jwt_team_id: None,
                },
            );
        }
        if raw_token.matches('.').count() == 2 {
            if crate::oidc::peek_asymmetric_jwt_header(&raw_token).is_some()
                && crate::oidc::OidcVerifier::is_configured(&settings)
            {
                let principal = state.oidc.verify_jwt(state, &settings, &raw_token)?;
                return enforce_principal_policies(state, headers, principal);
            }
            if let Some(secret) = state.jwt_secret() {
                let principal = verify_hs256_jwt(state, &raw_token, secret)?;
                return enforce_principal_policies(state, headers, principal);
            }
        }
        // Non-`sk-qx-` token supplied: reject immediately if remote, `--require-auth`, or sent via explicit admin key header.
        if settings.require_auth || !direct_loopback || explicit_admin_hdr {
            return Err(QuixigateError::Unauthorized(
                "invalid API key (supply Master Key, OIDC/JWT Bearer, or QuixiGate virtual key starting with 'sk-qx-')"
                    .to_owned(),
            ));
        }
    }

    if let Some(cookie_tok) = extract_session_cookie(headers)
        && let Ok(user) = state.db.validate_session(&cookie_tok)
    {
        return enforce_principal_policies(
            state,
            headers,
            AuthenticatedPrincipal {
                user,
                virtual_key: None,
                via_session: true,
                via_loopback_trust: false,
                via_master_key: false,
                jwt_team_id: None,
            },
        );
    }

    if !settings.require_auth && direct_loopback {
        let user = state.db.get_user("local")?;
        return enforce_principal_policies(
            state,
            headers,
            AuthenticatedPrincipal {
                user,
                virtual_key: None,
                via_session: false,
                via_loopback_trust: true,
                via_master_key: false,
                jwt_team_id: None,
            },
        );
    }

    Err(QuixigateError::Unauthorized(
        "supply Master Key, 'Authorization: Bearer sk-qx-...', or sign in".to_owned(),
    ))
}

/// Authenticates `/api/admin/*` and LiteLLM management routes. When `require_admin_role` is true,
/// requires `UserRole::Admin` (prevents `Member`/`Viewer` keys from mutating vault credentials).
pub fn authenticate_admin(
    state: &AppState,
    headers: &HeaderMap,
    require_admin_role: bool,
) -> Result<AuthenticatedPrincipal> {
    let principal = authenticate_inference(state, headers, None)?;
    if require_admin_role && principal.user.role != UserRole::Admin {
        return Err(QuixigateError::Forbidden(
            "administrator privileges required".to_owned(),
        ));
    }
    Ok(principal)
}

/// Authenticates either a global `UserRole::Admin` or a user who holds `role = 'admin'` inside `team_id`.
pub fn authenticate_team_admin_or_global(
    state: &AppState,
    headers: &HeaderMap,
    team_id: &str,
) -> Result<AuthenticatedPrincipal> {
    let principal = authenticate_inference(state, headers, None)?;
    if principal.user.role == UserRole::Admin {
        return Ok(principal);
    }
    if state.db.is_team_admin(team_id, &principal.user.id)? {
        return Ok(principal);
    }
    Err(QuixigateError::Forbidden(format!(
        "global administrator or team admin privileges for team '{team_id}' required"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4231_hmac_sha256_test_vector_matches() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex_out: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex_out,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
}
