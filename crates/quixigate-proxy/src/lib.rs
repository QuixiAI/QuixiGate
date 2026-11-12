//! Polyglot `axum` HTTP/SSE/WebSocket gateway, Management API (`/api/admin/*` +
//! LiteLLM-compatible endpoints), OpenAPI root (`/` & `/openapi.json`), and
//! strictly optional `/ui` convenience dashboard (`rust-embed`, `feature = "ui"`).

pub mod admin;
pub mod auth;
pub mod codecs;
pub mod litellm_mgmt;
pub mod mcp;
pub mod oidc;
pub mod openapi;
pub mod state;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use serde_json::json;
use state::AppState;
use tower_http::cors::{Any, CorsLayer};

#[cfg(feature = "ui")]
#[derive(rust_embed::RustEmbed)]
#[folder = "../../ui/dist"]
struct EmbeddedUiAssets;

/// Builds the complete QuixiGate `axum::Router` combining:
/// - Polyglot inference routes (`/v1/*`, `/v1beta/*`, `/api/v1/*`, `/api/chat`)
/// - Native `/api/admin/*` control plane + LiteLLM-compatible management routes (`/key/*`, `/user/*`, `/team/*`, `/model/*`, `/spend/*`, `/global/*`, `/cache/*`, `/config/yaml`)
/// - API root & OpenAPI 3.1 docs (`/`, `/docs`, `/openapi.json`, `/api/admin/openapi.json`)
/// - Optional `/ui` convenience SPA (only when compiled with `feature = "ui"` AND `state.is_ui_enabled()`)
pub fn build_router(state: AppState) -> Router {
    let inference_cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_private_network(true);

    let inference_routes = Router::new()
        .route(
            "/v1/chat/completions",
            post(codecs::openai_chat::handle_chat_completions),
        )
        .route(
            "/v1/responses",
            post(codecs::openai_responses::handle_responses_post)
                .get(codecs::openai_responses::handle_responses_ws),
        )
        .route("/v1/messages", post(codecs::anthropic::handle_messages))
        .route(
            "/v1/messages/count_tokens",
            post(codecs::anthropic::handle_count_tokens),
        )
        .route(
            "/v1beta/models/{*model_action}",
            post(codecs::gemini::handle_gemini_model_action),
        )
        .route("/v1/embeddings", post(codecs::misc::handle_embeddings))
        .route("/v1/models", get(codecs::misc::handle_list_models))
        .route("/api/v1/models", get(codecs::misc::handle_list_models))
        .route(
            "/api/v1/presets",
            get(codecs::misc::handle_openrouter_presets),
        )
        .route(
            "/api/v1/chat/completions",
            post(codecs::openai_chat::handle_chat_completions),
        )
        .route(
            "/api/v1/responses",
            post(codecs::openai_responses::handle_responses_post)
                .get(codecs::openai_responses::handle_responses_ws),
        )
        .route("/api/v1/key", get(codecs::misc::handle_openrouter_key))
        .route("/api/v1/auth/key", get(codecs::misc::handle_openrouter_key))
        .route(
            "/api/v1/credits",
            get(codecs::misc::handle_openrouter_credits),
        )
        .route("/api/chat", post(codecs::vercel_ui::handle_vercel_ui_chat))
        .route(
            "/mcp",
            get(mcp::handle_mcp_get)
                .post(mcp::handle_mcp_post)
                .delete(mcp::handle_mcp_delete),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::enforce_inference_plane_enabled,
        ))
        .layer(inference_cors);

    let admin_routes = Router::new()
        .route("/api/admin/openapi.json", get(openapi::handle_openapi_json))
        .route("/api/admin/auth/status", get(admin::get_auth_status))
        .route("/api/admin/auth/bootstrap", post(admin::post_auth_bootstrap))
        .route("/api/admin/auth/login", post(admin::post_auth_login))
        .route("/api/admin/auth/logout", post(admin::post_auth_logout))
        .route("/api/admin/auth/password", post(admin::post_auth_password))
        .route("/api/admin/overview", get(admin::get_overview))
        .route("/api/admin/doctor", get(admin::get_doctor))
        .route("/api/admin/spend/summary", get(admin::get_spend_summary))
        .route("/api/admin/spend/report", get(admin::get_spend_report))
        .route("/api/admin/spend/export", get(admin::get_spend_export))
        .route("/api/admin/spend/reset", post(admin::post_spend_reset))
        .route("/api/admin/audit-logs", get(admin::get_audit_logs))
        .route("/api/admin/audit-logs/{id}", get(admin::get_audit_log_by_id))
        .route("/api/admin/providers", get(admin::get_providers_catalog))
        .route(
            "/api/admin/credentials",
            get(admin::get_credentials).post(admin::post_credential),
        )
        .route(
            "/api/admin/credentials/{id}",
            get(admin::get_credential_by_id)
                .patch(admin::patch_credential)
                .delete(admin::delete_credential),
        )
        .route(
            "/api/admin/credentials/{id}/reset-health",
            post(admin::post_reset_credential_health),
        )
        .route(
            "/api/admin/credentials/{id}/refresh-quota",
            post(admin::post_refresh_credential_quota),
        )
        .route(
            "/api/admin/credentials/{id}/refresh-token",
            post(admin::post_refresh_credential_token),
        )
        .route(
            "/api/admin/credentials/{id}/test",
            post(admin::post_test_credential),
        )
        .route("/api/admin/oauth/start", post(admin::post_oauth_start))
        .route(
            "/api/admin/oauth/sessions/{id}",
            get(admin::get_oauth_session),
        )
        .route(
            "/api/admin/oauth/sessions/{id}/complete",
            post(admin::post_oauth_complete),
        )
        .route(
            "/api/admin/openrouter/status",
            get(admin::get_openrouter_status),
        )
        .route(
            "/api/admin/openrouter/connect",
            post(admin::post_openrouter_connect),
        )
        .route(
            "/api/admin/openrouter/sync",
            post(admin::post_openrouter_sync),
        )
        .route("/api/admin/models", get(admin::get_admin_models))
        .route("/api/admin/models/sync", post(admin::post_sync_models))
        .route(
            "/api/admin/routes",
            get(admin::get_routes).post(admin::post_route),
        )
        .route(
            "/api/admin/routes/{alias}",
            get(admin::get_route_by_alias)
                .put(admin::put_or_patch_route)
                .patch(admin::put_or_patch_route)
                .delete(admin::delete_route),
        )
        .route(
            "/api/admin/local-runtimes",
            get(admin::get_local_runtimes).post(admin::post_local_runtime),
        )
        .route(
            "/api/admin/local-runtimes/scan",
            post(admin::post_scan_local_runtimes),
        )
        .route(
            "/api/admin/local-runtimes/{id}",
            get(admin::get_local_runtime_by_id).delete(admin::delete_local_runtime),
        )
        .route(
            "/api/admin/local-runtimes/{id}/load",
            post(admin::post_load_local_model),
        )
        .route(
            "/api/admin/local-runtimes/{id}/unload",
            post(admin::post_unload_local_model),
        )
        .route(
            "/api/admin/users",
            get(admin::get_users).post(admin::post_user),
        )
        .route(
            "/api/admin/users/{id}",
            get(admin::get_user_by_id)
                .patch(admin::patch_user)
                .delete(admin::delete_user),
        )
        .route(
            "/api/admin/users/{id}/reset-spend",
            post(admin::post_reset_user_spend),
        )
        .route(
            "/api/admin/teams",
            get(admin::get_teams).post(admin::post_team),
        )
        .route(
            "/api/admin/teams/{id}",
            get(admin::get_team_by_id)
                .patch(admin::patch_team)
                .delete(admin::delete_team),
        )
        .route(
            "/api/admin/teams/{id}/members",
            post(admin::post_add_team_members),
        )
        .route(
            "/api/admin/teams/{id}/members/{user_id}",
            delete(admin::delete_team_member),
        )
        .route(
            "/api/admin/teams/{id}/reset-spend",
            post(admin::post_reset_team_spend),
        )
        .route(
            "/api/admin/keys",
            get(admin::get_keys).post(admin::post_key),
        )
        .route(
            "/api/admin/keys/{id}",
            get(admin::get_key_by_id)
                .patch(admin::patch_key)
                .delete(admin::delete_key),
        )
        .route("/api/admin/keys/{id}/rotate", post(admin::post_rotate_key))
        .route("/api/admin/keys/{id}/revoke", post(admin::post_revoke_key))
        .route(
            "/api/admin/keys/{id}/unrevoke",
            post(admin::post_unrevoke_key),
        )
        .route(
            "/api/admin/keys/{id}/reset-spend",
            post(admin::post_reset_key_spend),
        )
        .route(
            "/api/admin/traces",
            get(admin::get_traces).delete(admin::delete_traces),
        )
        .route("/api/admin/traces/stream", get(admin::get_traces_stream))
        .route("/api/admin/traces/{id}", get(admin::get_trace_by_id))
        .route(
            "/api/admin/settings",
            get(admin::get_settings)
                .put(admin::put_settings)
                .patch(admin::patch_settings),
        )
        .route(
            "/api/admin/ui",
            get(admin::get_ui_status)
                .post(admin::set_ui_status)
                .put(admin::set_ui_status)
                .patch(admin::set_ui_status),
        )
        .route("/api/admin/ui/enable", post(admin::post_enable_ui))
        .route("/api/admin/ui/disable", post(admin::post_disable_ui))
        .route(
            "/api/admin/burst",
            get(admin::get_burst_status)
                .post(admin::set_burst_status)
                .put(admin::set_burst_status)
                .patch(admin::set_burst_status),
        )
        .route("/api/admin/burst/enable", post(admin::post_enable_burst))
        .route("/api/admin/burst/disable", post(admin::post_disable_burst))
        .route(
            "/api/admin/mcp",
            get(admin::get_mcp_servers)
                .post(admin::post_mcp_server)
                .delete(admin::delete_mcp_server),
        )
        .route("/api/admin/mcp/probe", post(admin::post_probe_mcp_server))
        .route(
            "/api/admin/telemetry/phoenix-preset",
            post(admin::post_phoenix_preset),
        )
        .route(
            "/api/admin/telemetry/test-span",
            post(admin::post_test_span),
        )
        .route("/api/admin/cache", get(admin::get_cache_status))
        .route("/api/admin/cache/clear", post(admin::post_clear_cache))
        .route("/api/admin/config/export", get(admin::get_export_config))
        .route("/api/admin/config/import", post(admin::post_import_config));

    let control_plane_routes = admin_routes
        .merge(litellm_mgmt::router())
        .route("/ui", get(serve_ui_index))
        .route("/ui/", get(serve_ui_index))
        .route("/ui/{*path}", get(serve_ui_subpath))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::enforce_admin_plane_enabled,
        ));

    let peer_state = state.clone();

    Router::new()
        .route("/", get(openapi::handle_api_root))
        .route("/robots.txt", get(handle_robots_txt))
        .route("/docs", get(openapi::handle_docs))
        .route("/openapi.json", get(openapi::handle_openapi_json))
        .merge(inference_routes)
        .merge(control_plane_routes)
        .route("/health", get(codecs::misc::handle_liveness))
        .route("/health/liveness", get(codecs::misc::handle_liveness))
        .route("/health/readiness", get(codecs::misc::handle_readiness))
        .route("/metrics", get(codecs::misc::handle_prometheus_metrics))
        .fallback(fallback_not_found)
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(
            peer_state,
            auth::peer_trust_middleware,
        ))
}

async fn handle_robots_txt(State(state): State<AppState>) -> Response {
    let block = state
        .db
        .get_settings()
        .map(|s| s.block_robots)
        .unwrap_or(true);
    let body = if block {
        "User-agent: *\nDisallow: /\n"
    } else {
        "User-agent: *\nAllow: /\n"
    };
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

async fn fallback_not_found(uri: Uri) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "type": "error",
            "error": {
                "message": format!("route '{}' not found", uri.path()),
                "type": "not_found_error",
                "code": "not_found"
            }
        })),
    )
        .into_response()
}

fn ui_disabled_response() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "type": "error",
            "error": {
                "message": "Web UI is currently disabled; switch it on live via POST /api/admin/ui/enable or POST /config/ui {\"enabled\":true} (see /openapi.json)",
                "type": "not_found_error",
                "code": "ui_disabled"
            }
        })),
    )
        .into_response()
}

async fn serve_ui_index(State(state): State<AppState>) -> Response {
    if !state.is_ui_enabled() {
        return ui_disabled_response();
    }
    serve_ui_asset("index.html")
}

async fn serve_ui_subpath(
    State(state): State<AppState>,
    Path(subpath): Path<String>,
) -> Response {
    if !state.is_ui_enabled() {
        return ui_disabled_response();
    }
    serve_ui_asset(&subpath)
}

#[cfg(feature = "ui")]
fn serve_ui_asset(raw_subpath: &str) -> Response {
    use axum::http::{HeaderValue, header};

    let candidate = raw_subpath.trim_start_matches('/');
    let lookup = if candidate.is_empty() {
        "index.html"
    } else {
        candidate
    };

    if let Some(asset) = EmbeddedUiAssets::get(lookup) {
        let mime = mime_guess::from_path(lookup).first_or_octet_stream();
        let cache_header = if lookup.starts_with("assets/") {
            "public, max-age=31536000, immutable"
        } else {
            "no-cache"
        };
        let mut resp = (StatusCode::OK, asset.data.into_owned()).into_response();
        if let Ok(ct) = HeaderValue::from_str(mime.as_ref()) {
            resp.headers_mut().insert(header::CONTENT_TYPE, ct);
        }
        resp.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static(cache_header),
        );
        return resp;
    }

    // Do not serve `index.html` for missing static asset chunks (`/ui/assets/...`).
    if lookup.starts_with("assets/") {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "type": "error",
                "error": {
                    "message": format!("UI asset '/ui/{lookup}' not found"),
                    "type": "not_found_error",
                    "code": "not_found"
                }
            })),
        )
            .into_response();
    }

    // SPA client-side route fallback (`/ui/chat`, `/ui/models`, `/ui/providers`, `/ui/keys`, `/ui/traces`, `/ui/settings`)
    if let Some(index) = EmbeddedUiAssets::get("index.html") {
        let mut resp = (StatusCode::OK, index.data.into_owned()).into_response();
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        );
        resp.headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        return resp;
    }

    ui_disabled_response()
}

#[cfg(not(feature = "ui"))]
fn serve_ui_asset(_raw_subpath: &str) -> Response {
    ui_disabled_response()
}
