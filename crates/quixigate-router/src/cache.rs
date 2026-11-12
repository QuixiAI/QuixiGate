//! Deterministic BLAKE3 + bounded `moka` RAM + SQLite write-through response cache.

use std::time::Duration;

use moka::future::Cache;
use quixigate_db::Database;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedGeneration {
    pub resolved_model: String,
    pub provider_id: String,
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<CachedToolCall>,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone)]
pub struct ResponseCache {
    mem: Cache<String, CachedGeneration>,
    db: Database,
}

impl ResponseCache {
    pub fn new(db: Database) -> Self {
        let mem = Cache::builder()
            .max_capacity(2_048)
            .time_to_live(Duration::from_secs(3_600))
            .build();
        Self { mem, db }
    }

    pub fn compute_key(fingerprint_payload: &Value) -> String {
        let bytes = serde_json::to_vec(fingerprint_payload).unwrap_or_default();
        let hash = blake3::hash(&bytes);
        format!("ogc_{}", hash.to_hex())
    }

    pub async fn get(&self, key: &str) -> Option<CachedGeneration> {
        if let Some(hit) = self.mem.get(key).await {
            return Some(hit);
        }
        if let Ok(Some(json_str)) = self.db.get_cached_response(key)
            && let Ok(parsed) = serde_json::from_str::<CachedGeneration>(&json_str)
        {
            self.mem.insert(key.to_owned(), parsed.clone()).await;
            return Some(parsed);
        }
        None
    }

    pub async fn put(&self, key: &str, entry: &CachedGeneration, ttl_secs: u64) {
        self.mem.insert(key.to_owned(), entry.clone()).await;
        if let Ok(json_str) = serde_json::to_string(entry) {
            let _ = self
                .db
                .put_cached_response(key, &entry.resolved_model, &json_str, ttl_secs);
        }
    }

    pub fn clear_all(&self) -> quixigate_core::Result<usize> {
        self.mem.invalidate_all();
        self.db.clear_response_cache()
    }

    pub fn entry_counts(&self) -> (u64, usize) {
        let sqlite_count = self.db.count_cached_responses().unwrap_or(0);
        (self.mem.entry_count(), sqlite_count)
    }
}
