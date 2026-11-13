//! Canonical model specifier parser supporting OpenRouter slugs, `@preset/<slug>`,
//! `~author/model-latest` dynamic aliases, bare model names, `@provider` / `@seat`
//! pinning, and `:online` / `:free` / `:nitro` / `:floor` routing modifiers.

use serde::{Deserialize, Serialize};

/// Parsed representation of a client-requested model identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Original raw string received from the client.
    pub raw: String,
    /// Normalized model identifier or alias (`"anthropic/claude-sonnet-4-5"`, `"@preset/code-reviewer"`, `"gpt-5.2"`).
    pub base_model: String,
    /// Optional explicit provider or credential pin (`@github-copilot`, `@openrouter`).
    pub pinned_target: Option<String>,
    /// Attach server-side web search (`:online`).
    pub online: bool,
    /// Restrict strictly to `$0` local models, free portals, OpenRouter `:free`, or flat-rate OAuth seats (`:free`).
    pub free_only: bool,
    /// Order candidate backends by lowest rolling TTFT / highest `tok/s` (`:nitro`).
    pub nitro: bool,
    /// Prefer lowest-price upstream provider sorting on OpenRouter (`:floor`).
    pub floor: bool,
}

impl ModelSpec {
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        let (before_pin, pinned_target) = match trimmed.rsplit_once('@') {
            Some((left, right))
                if !left.is_empty() && !right.is_empty() && !right.contains('/') =>
            {
                (left, Some(right.trim().to_owned()))
            }
            _ => (trimmed, None),
        };

        let mut base = before_pin;
        let mut online = false;
        let mut free_only = false;
        let mut nitro = false;
        let mut floor = false;

        loop {
            if let Some(stripped) = base.strip_suffix(":online") {
                online = true;
                base = stripped;
            } else if let Some(stripped) = base.strip_suffix(":free") {
                free_only = true;
                base = stripped;
            } else if let Some(stripped) = base.strip_suffix(":nitro") {
                nitro = true;
                base = stripped;
            } else if let Some(stripped) = base.strip_suffix(":floor") {
                floor = true;
                base = stripped;
            } else {
                break;
            }
        }

        Self {
            raw: trimmed.to_owned(),
            base_model: base.trim().to_owned(),
            pinned_target,
            online,
            free_only,
            nitro,
            floor,
        }
    }

    /// True if `base_model` is an OpenRouter `@preset/<slug>` target.
    pub fn is_openrouter_preset(&self) -> bool {
        self.base_model.starts_with("@preset/") && self.base_model.len() > "@preset/".len()
    }

    /// Splits `"author/model"` if a vendor prefix is present (stripping any leading `~` dynamic marker on the vendor).
    pub fn split_vendor(&self) -> (Option<&str>, &str) {
        if self.is_openrouter_preset() {
            return (Some("@preset"), &self.base_model["@preset/".len()..]);
        }
        match self.base_model.split_once('/') {
            Some((vendor, model)) if !vendor.is_empty() && !model.is_empty() => {
                let clean_vendor = vendor.trim_start_matches('~');
                if clean_vendor.is_empty() {
                    (None, model)
                } else {
                    (Some(clean_vendor), model)
                }
            }
            _ => (None, self.base_model.as_str()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openrouter_modifiers_and_provider_pin() {
        let spec = ModelSpec::parse("anthropic/claude-sonnet-4-5:online:nitro@github-copilot");
        assert_eq!(spec.base_model, "anthropic/claude-sonnet-4-5");
        assert_eq!(spec.pinned_target.as_deref(), Some("github-copilot"));
        assert!(spec.online);
        assert!(spec.nitro);
        assert!(!spec.floor);
        assert!(!spec.free_only);
        assert_eq!(
            spec.split_vendor(),
            (Some("anthropic"), "claude-sonnet-4-5")
        );
    }

    #[test]
    fn parses_openrouter_preset_and_dynamic_tilde_alias() {
        let preset = ModelSpec::parse("@preset/code-reviewer:online");
        assert!(preset.is_openrouter_preset());
        assert_eq!(preset.base_model, "@preset/code-reviewer");
        assert_eq!(preset.pinned_target, None);
        assert!(preset.online);
        assert_eq!(preset.split_vendor(), (Some("@preset"), "code-reviewer"));

        let pinned_preset = ModelSpec::parse("@preset/inbound-classifier@openrouter");
        assert!(pinned_preset.is_openrouter_preset());
        assert_eq!(pinned_preset.base_model, "@preset/inbound-classifier");
        assert_eq!(pinned_preset.pinned_target.as_deref(), Some("openrouter"));

        let dyn_alias = ModelSpec::parse("~anthropic/claude-opus-latest:floor");
        assert_eq!(dyn_alias.base_model, "~anthropic/claude-opus-latest");
        assert!(dyn_alias.floor);
        assert_eq!(
            dyn_alias.split_vendor(),
            (Some("anthropic"), "claude-opus-latest")
        );
    }

    #[test]
    fn preserves_ollama_colon_tags_while_stripping_known_modifiers() {
        let spec = ModelSpec::parse("ollama/qwen3-coder:30b:free");
        assert_eq!(spec.base_model, "ollama/qwen3-coder:30b");
        assert!(spec.free_only);
        assert!(!spec.online);
    }

    #[test]
    fn parses_bare_agent_model_names() {
        let spec = ModelSpec::parse("claude-sonnet-4-5-20250929");
        assert_eq!(spec.base_model, "claude-sonnet-4-5-20250929");
        assert_eq!(spec.split_vendor(), (None, "claude-sonnet-4-5-20250929"));
    }
}
