//! Unified `AnyLanguageModel` & `embed_texts` dispatcher across all built-in
//! `aisdk` providers plus QuixiGate's custom OAuth `LanguageModel` implementations.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use aisdk::core::capabilities::{
    AudioInputSupport, AudioOutputSupport, ImageInputSupport, ImageOutputSupport, ReasoningSupport,
    StructuredOutputSupport, TextInputSupport, TextOutputSupport, ToolCallSupport,
    VideoInputSupport, VideoOutputSupport,
};
use aisdk::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, ProviderStream,
};
use aisdk::core::{DynamicModel, LanguageModel, Tool};
use async_trait::async_trait;
use quixigate_core::{CredentialSecret, QuixigateError, Result};
use parking_lot::Mutex;
use serde_json::{Value, json};

#[cfg(feature = "subscription-oauth")]
use crate::anthropic_oauth::AnthropicOAuthModel;
#[cfg(feature = "subscription-oauth")]
use crate::codex::ChatGptCodexModel;
use crate::copilot::{GitHubCopilotModel, UpstreamResponseMeta};
#[cfg(feature = "subscription-oauth")]
use crate::google_code_assist::GoogleCodeAssistModel;

#[async_trait]
trait BoxedLanguageModel: Send + Sync + fmt::Debug {
    fn clone_box(&self) -> Box<dyn BoxedLanguageModel>;
    fn model_name(&self) -> String;
    async fn generate_text_box(
        &mut self,
        options: LanguageModelOptions,
    ) -> aisdk::Result<LanguageModelResponse>;
    async fn stream_text_box(
        &mut self,
        options: LanguageModelOptions,
    ) -> aisdk::Result<ProviderStream>;
}

#[async_trait]
impl<T> BoxedLanguageModel for T
where
    T: LanguageModel,
{
    fn clone_box(&self) -> Box<dyn BoxedLanguageModel> {
        Box::new(self.clone())
    }

    fn model_name(&self) -> String {
        self.name()
    }

    async fn generate_text_box(
        &mut self,
        options: LanguageModelOptions,
    ) -> aisdk::Result<LanguageModelResponse> {
        self.generate_text(options).await
    }

    async fn stream_text_box(
        &mut self,
        options: LanguageModelOptions,
    ) -> aisdk::Result<ProviderStream> {
        self.stream_text(options).await
    }
}

/// Type-erased `aisdk::core::LanguageModel` preserving `Clone`, `Debug`, and all
/// `aisdk` capability marker traits so `LanguageModelRequest` works identically
/// across all built-in `aisdk` providers and all custom OAuth backends.
pub struct AnyLanguageModel {
    inner: Box<dyn BoxedLanguageModel>,
    upstream_meta: Option<Arc<Mutex<UpstreamResponseMeta>>>,
}

impl AnyLanguageModel {
    pub fn new<M: LanguageModel>(model: M) -> Self {
        Self {
            inner: Box::new(model),
            upstream_meta: None,
        }
    }

    pub fn with_upstream_meta<M: LanguageModel>(
        model: M,
        upstream_meta: Arc<Mutex<UpstreamResponseMeta>>,
    ) -> Self {
        Self {
            inner: Box::new(model),
            upstream_meta: Some(upstream_meta),
        }
    }

    pub fn upstream_meta(&self) -> UpstreamResponseMeta {
        self.upstream_meta
            .as_ref()
            .map(|m| m.lock().clone())
            .unwrap_or_default()
    }
}

impl Clone for AnyLanguageModel {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_box(),
            upstream_meta: self.upstream_meta.clone(),
        }
    }
}

impl fmt::Debug for AnyLanguageModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl TextInputSupport for AnyLanguageModel {}
impl TextOutputSupport for AnyLanguageModel {}
impl ToolCallSupport for AnyLanguageModel {}
impl StructuredOutputSupport for AnyLanguageModel {}
impl ReasoningSupport for AnyLanguageModel {}
impl ImageInputSupport for AnyLanguageModel {}
impl VideoInputSupport for AnyLanguageModel {}
impl AudioInputSupport for AnyLanguageModel {}
impl ImageOutputSupport for AnyLanguageModel {}
impl VideoOutputSupport for AnyLanguageModel {}
impl AudioOutputSupport for AnyLanguageModel {}

#[async_trait]
impl LanguageModel for AnyLanguageModel {
    fn name(&self) -> String {
        self.inner.model_name()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> aisdk::Result<LanguageModelResponse> {
        self.inner.generate_text_box(options).await
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> aisdk::Result<ProviderStream> {
        self.inner.stream_text_box(options).await
    }
}

#[derive(Debug, Clone)]
pub struct ModelBuildRequest {
    pub provider_id: String,
    pub model_name: String,
    pub secret: Option<CredentialSecret>,
    pub base_url: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub body: Option<Value>,
    pub tools: Vec<Tool>,
    pub http: Option<reqwest::Client>,
}

macro_rules! aisdk_providers {
    ($($name:literal => $provider:ident, $label:literal;)*) => {
        /// `(provider_id, display_name)` for all built-in `aisdk` providers.
        pub const AISDK_BUILTIN_PROVIDERS: &[(&str, &str)] = &[$(($name, $label)),*];

        pub fn build_aisdk_builtin_provider(
            provider_id: &str,
            model_name: &str,
            api_key: Option<&str>,
            base_url: Option<&str>,
            headers: Option<HashMap<String, String>>,
            body: Option<Value>,
        ) -> Result<AnyLanguageModel> {
            match provider_id {
                $($name => {
                    let mut builder = aisdk::providers::$provider::<DynamicModel>::builder()
                        .model_name(model_name);
                    let effective_url = base_url
                        .filter(|u| !u.trim().is_empty())
                        .or_else(|| ($name == "privatemode-ai").then_some("http://127.0.0.1:8080/v1"));
                    if let Some(url) = effective_url {
                        builder = builder.base_url(url);
                    }
                    if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
                        builder = builder.api_key(key);
                    }
                    if let Some(hdrs) = headers.filter(|h| !h.is_empty()) {
                        builder = builder.headers(hdrs);
                    }
                    if let Some(extra) = body {
                        builder = builder.body(extra);
                    }
                    let built = builder.build().map_err(|err| {
                        QuixigateError::Upstream {
                            status: 400,
                            provider: provider_id.to_owned(),
                            message: format!("failed to configure {} ({}): {err}", $label, provider_id),
                        }
                    })?;
                    Ok(AnyLanguageModel::new(built))
                })*
                other => Err(QuixigateError::NotFound(format!("unknown aisdk provider '{other}'"))),
            }
        }
    };
}

aisdk_providers! {
    "openrouter" => Openrouter, "OpenRouter";
    "openai" => OpenAI, "OpenAI";
    "anthropic" => Anthropic, "Anthropic";
    "google" => Google, "Google Gemini";
    "xai" => XAI, "xAI Grok";
    "deepseek" => Deepseek, "DeepSeek";
    "mistral" => Mistral, "Mistral";
    "groq" => Groq, "Groq";
    "togetherai" => TogetherAI, "Together AI";
    "fireworks-ai" => FireworksAi, "Fireworks AI";
    "302ai" => Ai302, "302.AI";
    "abacus" => Abacus, "Abacus";
    "aihubmix" => Aihubmix, "AIHubMix";
    "alibaba" => Alibaba, "Alibaba";
    "alibaba-cn" => AlibabaCn, "Alibaba (China)";
    "amazon-bedrock" => AmazonBedrock, "Amazon Bedrock";
    "bailing" => Bailing, "Bailing";
    "baseten" => Baseten, "Baseten";
    "berget" => Berget, "Berget.AI";
    "chutes" => Chutes, "Chutes";
    "cloudflare-ai-gateway" => CloudflareAiGateway, "Cloudflare AI Gateway";
    "cloudflare-workers-ai" => CloudflareWorkersAi, "Cloudflare Workers AI";
    "cortecs" => Cortecs, "Cortecs";
    "fastrouter" => Fastrouter, "FastRouter";
    "firmware" => Firmware, "Firmware";
    "friendli" => Friendli, "Friendli";
    "github-copilot" => GithubCopilot, "GitHub Copilot";
    "github-models" => GithubModels, "GitHub Models";
    "helicone" => Helicone, "Helicone";
    "huggingface" => Huggingface, "Hugging Face";
    "iflowcn" => Iflowcn, "iFlow";
    "inception" => Inception, "Inception";
    "inference" => Inference, "Inference";
    "io-net" => IoNet, "IO.NET";
    "jiekou" => Jiekou, "Jiekou.AI";
    "kuae-cloud-coding-plan" => KuaeCloudCodingPlan, "KUAE Cloud Coding Plan";
    "llama" => Llama, "Llama";
    "lmstudio" => Lmstudio, "LM Studio";
    "lucidquery" => Lucidquery, "LucidQuery";
    "moark" => Moark, "Moark";
    "modelscope" => Modelscope, "ModelScope";
    "moonshotai" => Moonshotai, "Moonshot AI";
    "moonshotai-cn" => MoonshotaiCn, "Moonshot AI (China)";
    "morph" => Morph, "Morph";
    "nano-gpt" => NanoGpt, "NanoGPT";
    "nebius" => Nebius, "Nebius Token Factory";
    "nova" => Nova, "Nova";
    "novita-ai" => NovitaAi, "Novita AI";
    "nvidia" => Nvidia, "Nvidia";
    "ollama-cloud" => OllamaCloud, "Ollama Cloud";
    "opencode" => Opencode, "OpenCode";
    "ovhcloud" => Ovhcloud, "OVHcloud AI Endpoints";
    "poe" => Poe, "Poe";
    "privatemode-ai" => OpenAICompatible, "Privatemode AI";
    "requesty" => Requesty, "Requesty";
    "scaleway" => Scaleway, "Scaleway";
    "siliconflow" => Siliconflow, "SiliconFlow";
    "siliconflow-cn" => SiliconflowCn, "SiliconFlow (China)";
    "stackit" => Stackit, "STACKIT";
    "stepfun" => Stepfun, "StepFun (China)";
    "submodel" => Submodel, "submodel";
    "synthetic" => Synthetic, "Synthetic";
    "upstage" => Upstage, "Upstage";
    "vercel" => Vercel, "Vercel AI Gateway";
    "vultr" => Vultr, "Vultr";
    "wandb" => Wandb, "Weights & Biases";
    "xiaomi" => Xiaomi, "Xiaomi";
    "zai" => Zai, "Z.AI";
    "zai-coding-plan" => ZaiCodingPlan, "Z.AI Coding Plan";
    "zenmux" => Zenmux, "ZenMux";
    "zhipuai" => Zhipuai, "Zhipu AI";
    "zhipuai-coding-plan" => ZhipuaiCodingPlan, "Zhipu AI Coding Plan";
    "openaicompatible" => OpenAICompatible, "Custom (OpenAI-compatible)";
}

/// Constructs an `AnyLanguageModel` for any OAuth seat, API-key provider, or local runtime.
pub fn instantiate_model(req: ModelBuildRequest) -> Result<AnyLanguageModel> {
    let http = req.http.unwrap_or_else(crate::build_inference_http_client);
    let base_override = req.base_url.as_deref().filter(|s| !s.trim().is_empty());

    match (req.provider_id.as_str(), req.secret) {
        #[cfg(feature = "subscription-oauth")]
        (_, Some(CredentialSecret::Codex { access, account_id, .. })) => {
            let mut m = ChatGptCodexModel::new(&req.model_name, access, account_id)
                .with_http(http)
                .with_tools(req.tools);
            if let Some(u) = base_override {
                m = m.with_base_url(u);
            }
            Ok(AnyLanguageModel::new(m))
        }
        #[cfg(feature = "subscription-oauth")]
        (_, Some(CredentialSecret::AnthropicSetupToken { token })) => {
            let mut m = AnthropicOAuthModel::new(&req.model_name, token)
                .with_http(http)
                .with_tools(req.tools);
            if let Some(u) = base_override {
                m = m.with_base_url(u);
            }
            Ok(AnyLanguageModel::new(m))
        }
        #[cfg(feature = "subscription-oauth")]
        (
            _,
            Some(CredentialSecret::GeminiCli {
                access,
                project_id,
                ..
            }),
        ) => {
            let mut m = GoogleCodeAssistModel::new(&req.model_name, access, project_id)
                .with_http(http)
                .with_tools(req.tools);
            if let Some(u) = base_override {
                m = m.with_base_url(u);
            }
            Ok(AnyLanguageModel::new(m))
        }
        #[cfg(feature = "subscription-oauth")]
        (
            _,
            Some(CredentialSecret::GitHubCopilot {
                access,
                base_url,
                ..
            }),
        ) => {
            let url = base_override.unwrap_or(if base_url.trim().is_empty() {
                crate::copilot::DEFAULT_COPILOT_BASE_URL
            } else {
                &base_url
            });
            let m = GitHubCopilotModel::new(&req.model_name, access)
                .with_base_url(url)
                .with_http(http)
                .with_tools(req.tools);
            let meta = m.upstream_meta_handle();
            Ok(AnyLanguageModel::with_upstream_meta(m, meta))
        }
        #[cfg(feature = "subscription-oauth")]
        (_, Some(CredentialSecret::Xai { access, .. })) => build_aisdk_builtin_provider(
            "xai",
            &req.model_name,
            Some(&access),
            base_override,
            req.headers,
            req.body,
        ),
        #[cfg(feature = "subscription-oauth")]
        (
            _,
            Some(CredentialSecret::QwenPortal {
                access,
                resource_url,
                ..
            }),
        ) => {
            let url = base_override
                .or(resource_url.as_deref())
                .unwrap_or("https://portal.qwen.ai/v1");
            let m = GitHubCopilotModel::openai_compatible(&req.model_name, url, Some(access))
                .with_http(http)
                .with_tools(req.tools);
            let meta = m.upstream_meta_handle();
            Ok(AnyLanguageModel::with_upstream_meta(m, meta))
        }
        #[cfg(feature = "subscription-oauth")]
        (
            _,
            Some(CredentialSecret::MiniMaxPortal {
                access,
                resource_url,
                ..
            }),
        ) => {
            let url = base_override
                .or(resource_url.as_deref())
                .unwrap_or("https://api.minimax.io/anthropic");
            let m = AnthropicOAuthModel::new(&req.model_name, access)
                .with_base_url(url)
                .with_http(http)
                .with_tools(req.tools);
            Ok(AnyLanguageModel::new(m))
        }
        #[cfg(feature = "subscription-oauth")]
        (_, Some(CredentialSecret::Chutes { access, .. })) => build_aisdk_builtin_provider(
            "chutes",
            &req.model_name,
            Some(&access),
            base_override,
            req.headers,
            req.body,
        ),
        #[cfg(not(feature = "subscription-oauth"))]
        (
            _,
            Some(
                CredentialSecret::Codex { .. }
                | CredentialSecret::AnthropicSetupToken { .. }
                | CredentialSecret::GeminiCli { .. }
                | CredentialSecret::GitHubCopilot { .. }
                | CredentialSecret::Xai { .. }
                | CredentialSecret::QwenPortal { .. }
                | CredentialSecret::MiniMaxPortal { .. }
                | CredentialSecret::Chutes { .. },
            ),
        ) => Err(QuixigateError::Validation(
            "Consumer subscription OAuth seats are disabled in this build (compiled without `subscription-oauth` feature)".to_owned(),
        )),
        ("openrouter", secret_opt)
        | (_, secret_opt @ Some(CredentialSecret::OpenRouterOidc { .. })) => {
            let url = base_override.unwrap_or("https://openrouter.ai/api/v1");
            let key = secret_opt.and_then(|s| match s {
                CredentialSecret::ApiKey { key } => Some(key),
                CredentialSecret::OpenRouterOidc { access, .. } => Some(access),
                _ => None,
            });
            let mut default_body = req.body.unwrap_or_else(|| json!({}));
            if let Some(obj) = default_body.as_object_mut() {
                obj.entry("usage".to_owned())
                    .or_insert_with(|| json!({ "include": true }));
            }
            let mut m = GitHubCopilotModel::openai_compatible(&req.model_name, url, key)
                .with_http(http)
                .with_tools(req.tools)
                .with_default_body(default_body);
            if let Some(hdrs) = req.headers {
                m = m.with_default_headers(hdrs);
            }
            let meta = m.upstream_meta_handle();
            Ok(AnyLanguageModel::with_upstream_meta(m, meta))
        }
        (provider_id, secret_opt)
            if matches!(
                provider_id,
                "ollama"
                    | "lmstudio"
                    | "lm-studio"
                    | "llamacpp"
                    | "llama.cpp"
                    | "llama-cpp"
                    | "llama-server"
                    | "vllm"
                    | "mlx"
                    | "local"
            ) || matches!(secret_opt, Some(CredentialSecret::LocalNoauth)) =>
        {
            let default_url = match provider_id {
                "ollama" => "http://127.0.0.1:11434/v1",
                "lmstudio" | "lm-studio" => "http://127.0.0.1:1234/v1",
                "llamacpp" | "llama.cpp" | "llama-cpp" | "llama-server" => {
                    "http://127.0.0.1:8080/v1"
                }
                "mlx" => "http://127.0.0.1:8085/v1",
                "vllm" => "http://127.0.0.1:8000/v1",
                _ => "http://127.0.0.1:11434/v1",
            };
            let url = base_override.unwrap_or(default_url);
            let key = secret_opt.and_then(|s| match s {
                CredentialSecret::ApiKey { key } => Some(key),
                _ => None,
            });
            let mut m = GitHubCopilotModel::openai_compatible(&req.model_name, url, key)
                .with_http(http)
                .with_tools(req.tools);
            if let Some(hdrs) = req.headers {
                m = m.with_default_headers(hdrs);
            }
            if let Some(body) = req.body {
                m = m.with_default_body(body);
            }
            let meta = m.upstream_meta_handle();
            Ok(AnyLanguageModel::with_upstream_meta(m, meta))
        }
        (provider_id, secret_opt) => {
            let api_key = secret_opt.and_then(|s| match s {
                CredentialSecret::ApiKey { key } => Some(key),
                _ => None,
            });
            let normalized = match provider_id {
                "fireworks" => "fireworks-ai",
                "together" => "togetherai",
                "gemini" => "google",
                "bedrock" => "amazon-bedrock",
                "custom" | "openai-compatible" | "openaichatcompletions" => "openaicompatible",
                other => other,
            };
            build_aisdk_builtin_provider(
                normalized,
                &req.model_name,
                api_key.as_deref(),
                base_override,
                req.headers,
                req.body,
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

/// Executes an embeddings call against OpenAI-compatible (`/embeddings`) or Gemini (`embedContent`)
/// endpoints, returning both vectors and token counts for ledger accounting.
pub async fn embed_texts(
    http: &reqwest::Client,
    provider_id: &str,
    model_name: &str,
    api_key: Option<&str>,
    base_url: Option<&str>,
    inputs: &[String],
    dimensions: Option<usize>,
) -> Result<EmbeddingOutput> {
    if matches!(provider_id, "google" | "gemini") {
        let base = base_url
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
            .trim_end_matches('/');
        let clean_model = model_name.trim_start_matches("models/");
        let key = api_key.unwrap_or_default();
        let url = format!("{base}/models/{clean_model}:batchEmbedContents?key={key}");
        let requests: Vec<Value> = inputs
            .iter()
            .map(|text| {
                let mut item = json!({
                    "model": format!("models/{clean_model}"),
                    "content": { "parts": [{ "text": text }] }
                });
                if let Some(dim) = dimensions {
                    item["outputDimensionality"] = json!(dim);
                }
                item
            })
            .collect();
        let resp = http
            .post(&url)
            .json(&json!({ "requests": requests }))
            .send()
            .await
            .map_err(|e| {
                let (st, why) = crate::describe_reqwest_error(&e);
                QuixigateError::Upstream {
                    status: st.map(|s| s.as_u16()).unwrap_or(502),
                    provider: provider_id.to_owned(),
                    message: format!("Gemini embeddings failed: {why}"),
                }
            })?;
        let status = resp.status();
        let val: Value = resp.json().await.unwrap_or_default();
        if !status.is_success() {
            return Err(QuixigateError::Upstream {
                status: status.as_u16(),
                provider: provider_id.to_owned(),
                message: val["error"]["message"]
                    .as_str()
                    .unwrap_or("Gemini embeddings error")
                    .to_owned(),
            });
        }
        let mut vectors = Vec::with_capacity(inputs.len());
        for emb in val["embeddings"].as_array().into_iter().flatten() {
            let vec: Vec<f32> = emb["values"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            vectors.push(vec);
        }
        let approx_tokens: u64 = inputs
            .iter()
            .map(|s| ((s.len() as u64).div_ceil(4)).max(1))
            .sum();
        return Ok(EmbeddingOutput {
            vectors,
            prompt_tokens: approx_tokens,
            total_tokens: approx_tokens,
        });
    }

    let default_base = match provider_id {
        "openai" => "https://api.openai.com/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "mistral" => "https://api.mistral.ai/v1",
        "togetherai" | "together" => "https://api.together.xyz/v1",
        "fireworks-ai" | "fireworks" => "https://api.fireworks.ai/inference/v1",
        "ollama" => "http://127.0.0.1:11434/v1",
        "lmstudio" | "lm-studio" => "http://127.0.0.1:1234/v1",
        "llamacpp" | "llama.cpp" | "llama-cpp" | "llama-server" => "http://127.0.0.1:8080/v1",
        "mlx" => "http://127.0.0.1:8085/v1",
        "vllm" => "http://127.0.0.1:8000/v1",
        _ => "https://api.openai.com/v1",
    };
    let base = base_url
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(default_base)
        .trim_end_matches('/');
    let url = if base.ends_with("/embeddings") {
        base.to_owned()
    } else {
        format!("{base}/embeddings")
    };
    let mut payload = json!({
        "model": model_name,
        "input": inputs,
    });
    if let Some(dim) = dimensions {
        payload["dimensions"] = json!(dim);
    }

    let mut req = http.post(&url).header("Content-Type", "application/json");
    if provider_id.eq_ignore_ascii_case("openrouter") {
        req = req
            .header("HTTP-Referer", quixigate_core::brand::DEFAULT_OPENROUTER_APP_URL)
            .header("X-Title", quixigate_core::brand::DEFAULT_OPENROUTER_APP_TITLE);
    }
    if let Some(k) = api_key.filter(|s| !s.trim().is_empty()) {
        req = req.bearer_auth(k);
    }
    let resp = req
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            let (st, why) = crate::describe_reqwest_error(&e);
            QuixigateError::Upstream {
                status: st.map(|s| s.as_u16()).unwrap_or(502),
                provider: provider_id.to_owned(),
                message: format!("embeddings request failed: {why}"),
            }
        })?;
    let status = resp.status();
    let val: Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        return Err(QuixigateError::Upstream {
            status: status.as_u16(),
            provider: provider_id.to_owned(),
            message: val["error"]["message"]
                .as_str()
                .unwrap_or("embeddings error")
                .to_owned(),
        });
    }
    let mut vectors = Vec::with_capacity(inputs.len());
    for item in val["data"].as_array().into_iter().flatten() {
        let vec: Vec<f32> = item["embedding"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        vectors.push(vec);
    }
    let approx: u64 = inputs
        .iter()
        .map(|s| ((s.len() as u64).div_ceil(4)).max(1))
        .sum();
    let prompt_tokens = val["usage"]["prompt_tokens"].as_u64().unwrap_or(approx);
    let total_tokens = val["usage"]["total_tokens"]
        .as_u64()
        .unwrap_or(prompt_tokens);
    Ok(EmbeddingOutput {
        vectors,
        prompt_tokens,
        total_tokens,
    })
}
