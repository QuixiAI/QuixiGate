//! Arize Phoenix **OpenInference** OTLP trace exporter (`HttpProtobuf`, `HttpJson`,
//! and gRPC envelope), Prometheus `/metrics` exposition, and live SSE trace bus.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use dashmap::DashMap;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value};
use opentelemetry_proto::tonic::resource::v1::Resource;
use opentelemetry_proto::tonic::trace::v1::{
    ResourceSpans, ScopeSpans, Span, Status, span::SpanKind, status::StatusCode,
};
use prost::Message as ProstMessage;
use quixigate_core::{
    GatewaySettings, IngressProtocol, OtlpProtocol, QuixigateError, RequestLog, Result, now_ms,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Default)]
pub struct OpenInferencePayload {
    pub invocation_parameters_json: Option<String>,
    /// `(role, content)` pairs when payload capture is enabled.
    pub input_messages: Vec<(String, String)>,
    pub output_text: Option<String>,
    /// `(id, name, arguments_json)` tool calls on output.
    pub output_tool_calls: Vec<(String, String, String)>,
    /// Raw JSON schemas of tools offered to the model.
    pub tool_schemas_json: Vec<String>,
    pub session_id: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSpanResult {
    pub ok: bool,
    pub trace_id: String,
    pub endpoint: String,
    pub status_code: u16,
    pub phoenix_url: String,
    pub message: String,
}

#[derive(Default)]
struct MetricsRegistry {
    active_requests: AtomicI64,
    requests_by_key: DashMap<(String, String, u16), AtomicU64>,
    input_tokens_by_provider: DashMap<String, AtomicU64>,
    output_tokens_by_provider: DashMap<String, AtomicU64>,
    cached_tokens_by_provider: DashMap<String, AtomicU64>,
    cache_hits_total: AtomicU64,
    spend_nanos_total: AtomicI64,
    savings_nanos_total: AtomicI64,
    ttft_sum_ms: AtomicU64,
    ttft_count: AtomicU64,
    duration_sum_ms: AtomicU64,
    duration_count: AtomicU64,
}

#[derive(Clone)]
pub struct TelemetryHub {
    http: reqwest::Client,
    http_grpc: reqwest::Client,
    trace_tx: broadcast::Sender<RequestLog>,
    metrics: Arc<MetricsRegistry>,
}

impl TelemetryHub {
    pub fn new() -> Self {
        let (trace_tx, _) = broadcast::channel(512);
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        let http_grpc = reqwest::Client::builder()
            .http2_prior_knowledge()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| http.clone());
        Self {
            http,
            http_grpc,
            trace_tx,
            metrics: Arc::new(MetricsRegistry::default()),
        }
    }

    pub fn subscribe_traces(&self) -> broadcast::Receiver<RequestLog> {
        self.trace_tx.subscribe()
    }

    pub fn inc_active(&self) {
        self.metrics.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_active(&self) {
        self.metrics.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn generate_trace_id() -> ([u8; 16], String) {
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        if bytes.iter().all(|&b| b == 0) {
            bytes[15] = 1;
        }
        let hex_str = hex::encode(bytes);
        (bytes, hex_str)
    }

    fn generate_span_id() -> [u8; 8] {
        let mut bytes = [0u8; 8];
        rand::thread_rng().fill_bytes(&mut bytes);
        if bytes.iter().all(|&b| b == 0) {
            bytes[7] = 1;
        }
        bytes
    }

    /// Updates Prometheus counters, broadcasts to SSE trace inspector, and asynchronously
    /// ships an Arize **OpenInference** OTLP span if `settings.otlp_enabled` is true.
    pub fn record_completed_request(
        &self,
        log: &RequestLog,
        settings: &GatewaySettings,
        capture_payloads: bool,
        oi: Option<OpenInferencePayload>,
    ) {
        // 1. Update lock-free Prometheus counters
        self.metrics
            .requests_by_key
            .entry((
                log.provider_id.clone(),
                log.resolved_model.clone(),
                log.status_code,
            ))
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        self.metrics
            .input_tokens_by_provider
            .entry(log.provider_id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(log.input_tokens, Ordering::Relaxed);

        self.metrics
            .output_tokens_by_provider
            .entry(log.provider_id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(log.output_tokens, Ordering::Relaxed);

        self.metrics
            .cached_tokens_by_provider
            .entry(log.provider_id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(log.cached_input_tokens, Ordering::Relaxed);

        if log.cache_hit {
            self.metrics
                .cache_hits_total
                .fetch_add(1, Ordering::Relaxed);
        }

        self.metrics
            .spend_nanos_total
            .fetch_add(log.cost_nanos.nanos(), Ordering::Relaxed);

        let saved = (log.api_equiv_nanos.nanos() - log.cost_nanos.nanos()).max(0);
        if saved > 0 {
            self.metrics
                .savings_nanos_total
                .fetch_add(saved, Ordering::Relaxed);
        }

        if let Some(ttft) = log.ttft_ms {
            self.metrics.ttft_sum_ms.fetch_add(ttft, Ordering::Relaxed);
            self.metrics.ttft_count.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics
            .duration_sum_ms
            .fetch_add(log.duration_ms, Ordering::Relaxed);
        self.metrics.duration_count.fetch_add(1, Ordering::Relaxed);

        // 2. Broadcast to live SSE subscribers
        let _ = self.trace_tx.send(log.clone());

        // 3. Export OTLP OpenInference span asynchronously if enabled
        if settings.otlp_enabled && !settings.otlp_endpoint.trim().is_empty() {
            let export_req =
                build_openinference_export_request(log, settings, capture_payloads, oi.as_ref());
            let client = self.http.clone();
            let grpc_client = self.http_grpc.clone();
            let settings_cloned = settings.clone();
            tokio::spawn(async move {
                if let Err(err) =
                    send_otlp_request(&client, &grpc_client, &settings_cloned, &export_req).await
                {
                    tracing::debug!("OTLP span export warning: {err}");
                }
            });
        }
    }

    /// Synchronously sends a verification OpenInference `LLM` span to Arize Phoenix / OTLP
    /// collector so the Web UI's **"Send Test Span"** button gives instant feedback.
    pub async fn send_test_span(&self, settings: &GatewaySettings) -> Result<TestSpanResult> {
        let endpoint = settings.otlp_endpoint.trim().to_owned();
        if endpoint.is_empty() {
            return Err(QuixigateError::Validation(
                "OTLP endpoint URL cannot be empty".to_owned(),
            ));
        }
        let (_, trace_hex) = Self::generate_trace_id();
        let now = now_ms();
        let synthetic_log = RequestLog {
            id: format!("req_test_{}", &trace_hex[..12]),
            timestamp_ms: now - 42,
            ingress_protocol: IngressProtocol::ChatCompletions,
            requested_model: "quixigate/phoenix-verification".to_owned(),
            resolved_model: "claude-sonnet-4-5".to_owned(),
            provider_id: "anthropic-token".to_owned(),
            credential_id: Some("seat_verification".to_owned()),
            user_id: Some("local".to_owned()),
            team_id: None,
            virtual_key_id: None,
            tags: Vec::new(),
            status_code: 200,
            stream: true,
            fallback_attempts: 0,
            cache_hit: false,
            ttft_ms: Some(18),
            duration_ms: 42,
            input_tokens: 24,
            cached_input_tokens: 8,
            reasoning_tokens: 0,
            output_tokens: 16,
            cost_nanos: quixigate_core::NanoUsd::ZERO,
            api_equiv_nanos: quixigate_core::NanoUsd(312_000),
            otel_trace_id: Some(trace_hex.clone()),
            error_message: None,
            request_body_json: Some(
                json!({ "messages": [{ "role": "user", "content": "Verify Arize Phoenix OpenInference connection from QuixiGate" }] })
                    .to_string(),
            ),
            response_body_json: Some(
                json!({ "content": "Arize Phoenix OpenInference telemetry verified cleanly!" })
                    .to_string(),
            ),
        };
        let oi = OpenInferencePayload {
            invocation_parameters_json: Some(
                json!({ "temperature": 0.0, "stream": true }).to_string(),
            ),
            input_messages: vec![(
                "user".to_owned(),
                "Verify Arize Phoenix OpenInference connection from QuixiGate".to_owned(),
            )],
            output_text: Some("Arize Phoenix OpenInference telemetry verified cleanly!".to_owned()),
            output_tool_calls: Vec::new(),
            tool_schemas_json: Vec::new(),
            session_id: Some("quixigate-phoenix-test".to_owned()),
            tags: vec!["quixigate".to_owned(), "verification".to_owned()],
        };

        let export_req =
            build_openinference_export_request(&synthetic_log, settings, true, Some(&oi));
        let status = send_otlp_request(&self.http, &self.http_grpc, settings, &export_req).await?;
        let phoenix_base = settings.phoenix_ui_url.trim_end_matches('/');
        Ok(TestSpanResult {
            ok: (200..300).contains(&status),
            trace_id: trace_hex.clone(),
            endpoint,
            status_code: status,
            phoenix_url: format!("{phoenix_base}/projects"),
            message: format!(
                "OpenInference LLM test span ({trace_hex}) accepted with HTTP {status}"
            ),
        })
    }

    /// Renders Prometheus text exposition (`text/plain; version=0.0.4`).
    pub fn render_prometheus_metrics(&self) -> String {
        let mut out = String::with_capacity(2048);
        let _ = writeln!(
            out,
            "# HELP quixigate_active_requests In-flight requests currently executing.\n# TYPE quixigate_active_requests gauge\nquixigate_active_requests {}",
            self.metrics.active_requests.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_cache_hits_total Exact response cache hits.\n# TYPE quixigate_cache_hits_total counter\nquixigate_cache_hits_total {}",
            self.metrics.cache_hits_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_spend_nanos_total Cumulative billed spend in NanoUsd (1e-9 USD).\n# TYPE quixigate_spend_nanos_total counter\nquixigate_spend_nanos_total {}",
            self.metrics.spend_nanos_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_savings_nanos_total Cumulative API-equivalent savings from OAuth seats & local models in NanoUsd.\n# TYPE quixigate_savings_nanos_total counter\nquixigate_savings_nanos_total {}",
            self.metrics.savings_nanos_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_ttft_milliseconds_sum Sum of time-to-first-token in ms.\n# TYPE quixigate_ttft_milliseconds_sum counter\nquixigate_ttft_milliseconds_sum {}",
            self.metrics.ttft_sum_ms.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_ttft_milliseconds_count Count of streaming TTFT measurements.\n# TYPE quixigate_ttft_milliseconds_count counter\nquixigate_ttft_milliseconds_count {}",
            self.metrics.ttft_count.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_duration_milliseconds_sum Sum of end-to-end request durations in ms.\n# TYPE quixigate_duration_milliseconds_sum counter\nquixigate_duration_milliseconds_sum {}",
            self.metrics.duration_sum_ms.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# HELP quixigate_duration_milliseconds_count Count of completed requests.\n# TYPE quixigate_duration_milliseconds_count counter\nquixigate_duration_milliseconds_count {}",
            self.metrics.duration_count.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP quixigate_requests_total Total gateway requests by provider, model, and HTTP status.\n# TYPE quixigate_requests_total counter"
        );
        for item in self.metrics.requests_by_key.iter() {
            let (prov, model, status) = item.key();
            let count = item.value().load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "quixigate_requests_total{{provider=\"{}\",model=\"{}\",status=\"{}\"}} {}",
                escape_prom(prov),
                escape_prom(model),
                status,
                count
            );
        }

        let _ = writeln!(
            out,
            "# HELP quixigate_tokens_total Total tokens processed by provider and direction.\n# TYPE quixigate_tokens_total counter"
        );
        for item in self.metrics.input_tokens_by_provider.iter() {
            let _ = writeln!(
                out,
                "quixigate_tokens_total{{provider=\"{}\",kind=\"input\"}} {}",
                escape_prom(item.key()),
                item.value().load(Ordering::Relaxed)
            );
        }
        for item in self.metrics.output_tokens_by_provider.iter() {
            let _ = writeln!(
                out,
                "quixigate_tokens_total{{provider=\"{}\",kind=\"output\"}} {}",
                escape_prom(item.key()),
                item.value().load(Ordering::Relaxed)
            );
        }
        for item in self.metrics.cached_tokens_by_provider.iter() {
            let _ = writeln!(
                out,
                "quixigate_tokens_total{{provider=\"{}\",kind=\"cached_input\"}} {}",
                escape_prom(item.key()),
                item.value().load(Ordering::Relaxed)
            );
        }
        out
    }
}

impl Default for TelemetryHub {
    fn default() -> Self {
        Self::new()
    }
}

fn escape_prom(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn kv_str(k: &str, v: impl Into<String>) -> KeyValue {
    KeyValue {
        key: k.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(v.into())),
        }),
    }
}

fn kv_int(k: &str, v: i64) -> KeyValue {
    KeyValue {
        key: k.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::IntValue(v)),
        }),
    }
}

fn kv_float(k: &str, v: f64) -> KeyValue {
    KeyValue {
        key: k.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::DoubleValue(v)),
        }),
    }
}

fn kv_bool(k: &str, v: bool) -> KeyValue {
    KeyValue {
        key: k.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::BoolValue(v)),
        }),
    }
}

fn kv_str_list(k: &str, items: &[String]) -> KeyValue {
    KeyValue {
        key: k.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::ArrayValue(
                opentelemetry_proto::tonic::common::v1::ArrayValue {
                    values: items
                        .iter()
                        .map(|s| AnyValue {
                            value: Some(any_value::Value::StringValue(s.clone())),
                        })
                        .collect(),
                },
            )),
        }),
    }
}

fn infer_llm_system(provider_id: &str, model: &str) -> &'static str {
    match provider_id {
        "openai" | "openai-codex" | "github-copilot" => "openai",
        "anthropic" | "anthropic-token" | "minimax-portal" => "anthropic",
        "google" | "google-gemini-cli" => "vertexai",
        "xai" | "xai-oauth" => "xai",
        "mistral" => "mistralai",
        _ if model.contains("claude") => "anthropic",
        _ if model.contains("gpt") => "openai",
        _ if model.contains("gemini") => "vertexai",
        _ => "openai",
    }
}

pub fn build_openinference_export_request(
    log: &RequestLog,
    settings: &GatewaySettings,
    capture_payloads: bool,
    oi: Option<&OpenInferencePayload>,
) -> ExportTraceServiceRequest {
    let trace_id = log
        .otel_trace_id
        .as_deref()
        .and_then(|h| hex::decode(h).ok())
        .filter(|b| b.len() == 16)
        .unwrap_or_else(|| TelemetryHub::generate_trace_id().0.to_vec());
    let span_id = TelemetryHub::generate_span_id().to_vec();

    let start_ns = (log.timestamp_ms.max(0) as u64).saturating_mul(1_000_000);
    let end_ns = start_ns.saturating_add(log.duration_ms.max(1).saturating_mul(1_000_000));

    let span_kind_label = if log.ingress_protocol == IngressProtocol::Embeddings {
        "EMBEDDING"
    } else {
        "LLM"
    };

    let project_name = if settings.openinference_project_name.trim().is_empty() {
        "quixigate"
    } else {
        settings.openinference_project_name.trim()
    };

    let total_tokens = log.input_tokens.saturating_add(log.output_tokens);
    let total_usd = (log.cost_nanos.nanos() as f64) / (quixigate_core::NANOS_PER_USD as f64);
    let equiv_usd = (log.api_equiv_nanos.nanos() as f64) / (quixigate_core::NANOS_PER_USD as f64);
    let (prompt_usd, completion_usd) = if total_tokens > 0 && total_usd > 0.0 {
        let p = total_usd * (log.input_tokens as f64 / total_tokens as f64);
        (p, (total_usd - p).max(0.0))
    } else {
        (0.0, 0.0)
    };

    let mut attrs = vec![
        kv_str("openinference.span.kind", span_kind_label),
        kv_str("openinference.project.name", project_name),
        kv_str("llm.provider", &log.provider_id),
        kv_str(
            "llm.system",
            infer_llm_system(&log.provider_id, &log.resolved_model),
        ),
        kv_str("llm.model_name", &log.resolved_model),
        kv_int("llm.token_count.prompt", log.input_tokens as i64),
        kv_int(
            "llm.token_count.prompt_details.cache_read",
            log.cached_input_tokens as i64,
        ),
        kv_int("llm.token_count.completion", log.output_tokens as i64),
        kv_int(
            "llm.token_count.completion_details.reasoning",
            log.reasoning_tokens as i64,
        ),
        kv_int("llm.token_count.total", total_tokens as i64),
        kv_float("llm.cost.prompt", prompt_usd),
        kv_float("llm.cost.completion", completion_usd),
        kv_float("llm.cost.total", total_usd),
        kv_float("quixigate.api_equiv_usd", equiv_usd),
        kv_str("quixigate.requested_model", &log.requested_model),
        kv_str("quixigate.ingress_protocol", log.ingress_protocol.as_str()),
        kv_bool("quixigate.stream", log.stream),
        kv_bool("quixigate.cache_hit", log.cache_hit),
        kv_int("quixigate.fallback_attempts", log.fallback_attempts as i64),
        kv_int("http.status_code", log.status_code as i64),
    ];

    if let Some(ttft) = log.ttft_ms {
        attrs.push(kv_int("quixigate.ttft_ms", ttft as i64));
    }
    if let Some(uid) = &log.user_id {
        attrs.push(kv_str("user.id", uid));
    }
    if let Some(cid) = &log.credential_id {
        attrs.push(kv_str("quixigate.credential_id", cid));
    }

    if let Some(meta) = oi {
        if let Some(inv) = &meta.invocation_parameters_json {
            attrs.push(kv_str("llm.invocation_parameters", inv));
        }
        if let Some(sid) = &meta.session_id {
            attrs.push(kv_str("session.id", sid));
        }
        if !meta.tags.is_empty() {
            attrs.push(kv_str_list("tag.tags", &meta.tags));
        }

        if capture_payloads {
            if let Some(req_json) = &log.request_body_json {
                attrs.push(kv_str("input.value", req_json));
                attrs.push(kv_str("input.mime_type", "application/json"));
            }
            if let Some(resp_json) = &log.response_body_json {
                attrs.push(kv_str("output.value", resp_json));
                attrs.push(kv_str("output.mime_type", "application/json"));
            } else if let Some(out_txt) = &meta.output_text {
                attrs.push(kv_str("output.value", out_txt));
                attrs.push(kv_str("output.mime_type", "text/plain"));
            }

            for (idx, (role, content)) in meta.input_messages.iter().enumerate() {
                attrs.push(kv_str(
                    &format!("llm.input_messages.{idx}.message.role"),
                    role,
                ));
                attrs.push(kv_str(
                    &format!("llm.input_messages.{idx}.message.content"),
                    content,
                ));
            }

            if meta.output_text.is_some() || !meta.output_tool_calls.is_empty() {
                attrs.push(kv_str("llm.output_messages.0.message.role", "assistant"));
                if let Some(txt) = &meta.output_text {
                    attrs.push(kv_str("llm.output_messages.0.message.content", txt));
                }
                for (j, (id, name, args)) in meta.output_tool_calls.iter().enumerate() {
                    attrs.push(kv_str(
                        &format!("llm.output_messages.0.message.tool_calls.{j}.tool_call.id"),
                        id,
                    ));
                    attrs.push(kv_str(
                        &format!(
                            "llm.output_messages.0.message.tool_calls.{j}.tool_call.function.name"
                        ),
                        name,
                    ));
                    attrs.push(kv_str(
                        &format!(
                            "llm.output_messages.0.message.tool_calls.{j}.tool_call.function.arguments"
                        ),
                        args,
                    ));
                }
            }

            for (idx, schema) in meta.tool_schemas_json.iter().enumerate() {
                attrs.push(kv_str(&format!("llm.tools.{idx}.tool.json_schema"), schema));
            }
        } else {
            attrs.push(kv_bool("quixigate.payloads_masked", true));
        }
    }

    let status = if log.status_code >= 400 {
        Status {
            message: log.error_message.clone().unwrap_or_default(),
            code: StatusCode::Error as i32,
        }
    } else {
        Status {
            message: String::new(),
            code: StatusCode::Ok as i32,
        }
    };

    let span = Span {
        trace_id,
        span_id,
        trace_state: String::new(),
        parent_span_id: Vec::new(),
        flags: 1,
        name: format!("{} {}", span_kind_label, log.resolved_model),
        kind: SpanKind::Client as i32,
        start_time_unix_nano: start_ns,
        end_time_unix_nano: end_ns,
        attributes: attrs,
        dropped_attributes_count: 0,
        events: Vec::new(),
        dropped_events_count: 0,
        links: Vec::new(),
        dropped_links_count: 0,
        status: Some(status),
    };

    ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: Some(Resource {
                attributes: vec![
                    kv_str("service.name", "quixigate"),
                    kv_str("service.version", env!("CARGO_PKG_VERSION")),
                    kv_str("openinference.project.name", project_name),
                ],
                dropped_attributes_count: 0,
            }),
            scope_spans: vec![ScopeSpans {
                scope: Some(InstrumentationScope {
                    name: "quixigate.openinference".to_owned(),
                    version: env!("CARGO_PKG_VERSION").to_owned(),
                    attributes: Vec::new(),
                    dropped_attributes_count: 0,
                }),
                spans: vec![span],
                schema_url: "https://arize.com/openinference/v1".to_owned(),
            }],
            schema_url: String::new(),
        }],
    }
}

async fn send_otlp_request(
    http: &reqwest::Client,
    http_grpc: &reqwest::Client,
    settings: &GatewaySettings,
    export_req: &ExportTraceServiceRequest,
) -> Result<u16> {
    let raw_endpoint = settings.otlp_endpoint.trim();
    let is_grpc = settings.otlp_protocol == OtlpProtocol::Grpc
        || raw_endpoint.starts_with("grpc://")
        || (raw_endpoint.contains(":4317") && !raw_endpoint.contains("/v1/traces"));

    let (url, content_type, payload) = if is_grpc {
        let base = raw_endpoint
            .trim_start_matches("grpc://")
            .trim_end_matches("/v1/traces")
            .trim_end_matches("/opentelemetry.proto.collector.trace.v1.TraceService/Export")
            .trim_end_matches('/');
        let http_base = if base.starts_with("http://") || base.starts_with("https://") {
            base.to_owned()
        } else {
            format!("http://{base}")
        };
        let url = format!("{http_base}/opentelemetry.proto.collector.trace.v1.TraceService/Export");
        let proto_bytes = export_req.encode_to_vec();
        let mut framed = Vec::with_capacity(5 + proto_bytes.len());
        framed.push(0u8); // uncompressed gRPC flag
        framed.extend_from_slice(&(proto_bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&proto_bytes);
        (url, "application/grpc+proto", framed)
    } else {
        let url = if raw_endpoint.ends_with("/v1/traces") {
            raw_endpoint.to_owned()
        } else {
            format!("{}/v1/traces", raw_endpoint.trim_end_matches('/'))
        };
        match settings.otlp_protocol {
            OtlpProtocol::HttpProtobuf | OtlpProtocol::Grpc => {
                (url, "application/x-protobuf", export_req.encode_to_vec())
            }
            OtlpProtocol::HttpJson => {
                let bytes = serde_json::to_vec(export_req).map_err(|e| {
                    QuixigateError::Internal(format!("OTLP JSON serialize failed: {e}"))
                })?;
                (url, "application/json", bytes)
            }
        }
    };

    let active_client = if is_grpc && !url.starts_with("https://") {
        http_grpc
    } else {
        http
    };
    let mut req = active_client
        .post(&url)
        .header("Content-Type", content_type);
    if is_grpc {
        req = req.header("te", "trailers");
    }
    if let Some(api_key) = settings
        .otlp_api_key
        .as_deref()
        .filter(|k| !k.trim().is_empty())
    {
        req = req
            .bearer_auth(api_key.trim())
            .header("api_key", api_key.trim());
    }

    let resp = req
        .body(payload)
        .send()
        .await
        .map_err(|e| QuixigateError::Upstream {
            status: 502,
            provider: "arize-phoenix-otlp".to_owned(),
            message: format!("could not reach OTLP collector at {url}: {e}"),
        })?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        let body = resp.text().await.unwrap_or_default();
        return Err(QuixigateError::Upstream {
            status,
            provider: "arize-phoenix-otlp".to_owned(),
            message: format!("OTLP collector rejected trace (HTTP {status}): {body}"),
        });
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode as HttpStatus};
    use axum::routing::post;
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn exports_wire_exact_protobuf_openinference_span_to_collector() {
        let captured: Arc<Mutex<Vec<ExportTraceServiceRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/v1/traces",
                post(
                    |State(store): State<Arc<Mutex<Vec<ExportTraceServiceRequest>>>>,
                     headers: HeaderMap,
                     body: Bytes| async move {
                        assert_eq!(
                            headers.get("content-type").and_then(|v| v.to_str().ok()),
                            Some("application/x-protobuf")
                        );
                        let decoded = ExportTraceServiceRequest::decode(body).unwrap();
                        store.lock().await.push(decoded);
                        HttpStatus::OK
                    },
                ),
            )
            .with_state(Arc::clone(&captured));

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let settings = GatewaySettings {
            otlp_enabled: true,
            otlp_endpoint: format!("http://{addr}/v1/traces"),
            otlp_protocol: OtlpProtocol::HttpProtobuf,
            openinference_project_name: "quixigate-test-proj".to_owned(),
            phoenix_ui_url: format!("http://{addr}"),
            ..Default::default()
        };

        let hub = TelemetryHub::new();
        let res = hub.send_test_span(&settings).await.unwrap();
        assert!(res.ok);
        assert_eq!(res.status_code, 200);

        let spans = captured.lock().await;
        assert_eq!(spans.len(), 1);
        let rs = &spans[0].resource_spans[0];
        let span = &rs.scope_spans[0].spans[0];
        let find_str = |k: &str| -> Option<String> {
            span.attributes
                .iter()
                .find(|kv| kv.key == k)
                .and_then(|kv| match kv.value.as_ref()?.value.as_ref()? {
                    any_value::Value::StringValue(s) => Some(s.clone()),
                    _ => None,
                })
        };
        assert_eq!(find_str("openinference.span.kind").as_deref(), Some("LLM"));
        assert_eq!(
            find_str("openinference.project.name").as_deref(),
            Some("quixigate-test-proj")
        );
        assert_eq!(
            find_str("llm.model_name").as_deref(),
            Some("claude-sonnet-4-5")
        );
        assert_eq!(
            find_str("llm.input_messages.0.message.role").as_deref(),
            Some("user")
        );

        let prom = hub.render_prometheus_metrics();
        assert!(prom.contains("quixigate_active_requests"));
    }

    #[tokio::test]
    async fn exports_wire_exact_grpc_h2c_openinference_span_to_4317_collector() {
        let captured: Arc<Mutex<Vec<ExportTraceServiceRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/opentelemetry.proto.collector.trace.v1.TraceService/Export",
                post(
                    |State(store): State<Arc<Mutex<Vec<ExportTraceServiceRequest>>>>,
                     headers: HeaderMap,
                     body: Bytes| async move {
                        assert_eq!(
                            headers.get("content-type").and_then(|v| v.to_str().ok()),
                            Some("application/grpc+proto")
                        );
                        assert_eq!(
                            headers.get("te").and_then(|v| v.to_str().ok()),
                            Some("trailers")
                        );
                        assert!(body.len() > 5);
                        assert_eq!(body[0], 0u8); // uncompressed gRPC flag
                        let msg_len =
                            u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
                        assert_eq!(body.len() - 5, msg_len);
                        let decoded = ExportTraceServiceRequest::decode(body.slice(5..)).unwrap();
                        store.lock().await.push(decoded);
                        (HttpStatus::OK, [("grpc-status", "0")])
                    },
                ),
            )
            .with_state(Arc::clone(&captured));

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let settings = GatewaySettings {
            otlp_enabled: true,
            otlp_endpoint: format!("http://{addr}"),
            otlp_protocol: OtlpProtocol::Grpc,
            openinference_project_name: "quixigate-grpc-proj".to_owned(),
            phoenix_ui_url: format!("http://{addr}"),
            ..Default::default()
        };

        let hub = TelemetryHub::new();
        let res = hub.send_test_span(&settings).await.unwrap();
        assert!(res.ok);
        assert_eq!(res.status_code, 200);

        let spans = captured.lock().await;
        assert_eq!(spans.len(), 1);
        let span = &spans[0].resource_spans[0].scope_spans[0].spans[0];
        let proj = span
            .attributes
            .iter()
            .find(|kv| kv.key == "openinference.project.name")
            .and_then(|kv| match kv.value.as_ref()?.value.as_ref()? {
                any_value::Value::StringValue(s) => Some(s.as_str()),
                _ => None,
            });
        assert_eq!(proj, Some("quixigate-grpc-proj"));
    }
}
