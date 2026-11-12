//! Lock-free in-memory RPM / TPM / max-concurrency rate limiters and
//! atomic pre-flight `NanoUsd` budget reservations.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use quixigate_core::{NanoUsd, QuixigateError, Result, now_ms};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const MINUTE_MS: i64 = 60_000;

#[derive(Debug)]
struct MinuteWindow {
    window_start_ms: i64,
    requests: u32,
    tokens: u64,
}

#[derive(Default)]
pub struct RuntimeLimiter {
    windows: DashMap<String, MinuteWindow>,
    semaphores: DashMap<(String, u32), Arc<Semaphore>>,
    in_flight_budget_nanos: DashMap<String, i64>,
    active_requests: AtomicU64,
}

pub struct ConcurrencyGuard {
    _permit: Option<OwnedSemaphorePermit>,
    limiter: Arc<RuntimeLimiter>,
}

impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        self.limiter.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl RuntimeLimiter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn active_requests(&self) -> u64 {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Checks RPM, estimated TPM, and max-concurrent-stream limits for `scope_key`.
    pub fn check_and_acquire(
        self: &Arc<Self>,
        scope_key: &str,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u64>,
        max_concurrent: Option<u32>,
        estimated_tokens: u64,
    ) -> Result<ConcurrencyGuard> {
        let permit = if let Some(max_conc) = max_concurrent.filter(|&c| c > 0) {
            let sem = self
                .semaphores
                .entry((scope_key.to_owned(), max_conc))
                .or_insert_with(|| Arc::new(Semaphore::new(max_conc as usize)))
                .clone();
            Some(sem.try_acquire_owned().map_err(|_| QuixigateError::RateLimited {
                message: format!("max concurrent requests ({max_conc}) exceeded"),
                retry_after_secs: Some(1),
            })?)
        } else {
            None
        };

        let now = now_ms();
        if rpm_limit.is_some() || tpm_limit.is_some() {
            let mut entry = self.windows.entry(scope_key.to_owned()).or_insert(MinuteWindow {
                window_start_ms: now,
                requests: 0,
                tokens: 0,
            });
            if now - entry.window_start_ms >= MINUTE_MS {
                entry.window_start_ms = now;
                entry.requests = 0;
                entry.tokens = 0;
            }
            let retry_after = (((entry.window_start_ms + MINUTE_MS - now).max(1000)) / 1000) as u64;
            if let Some(rpm) = rpm_limit
                && entry.requests >= rpm
            {
                return Err(QuixigateError::RateLimited {
                    message: format!("RPM limit ({rpm}/min) exceeded"),
                    retry_after_secs: Some(retry_after),
                });
            }
            if let Some(tpm) = tpm_limit
                && entry.tokens.saturating_add(estimated_tokens) > tpm
            {
                return Err(QuixigateError::RateLimited {
                    message: format!("TPM limit ({tpm} tokens/min) exceeded"),
                    retry_after_secs: Some(retry_after),
                });
            }
            entry.requests += 1;
            entry.tokens = entry.tokens.saturating_add(estimated_tokens);
        }

        self.active_requests.fetch_add(1, Ordering::Relaxed);
        Ok(ConcurrencyGuard {
            _permit: permit,
            limiter: Arc::clone(self),
        })
    }

    pub fn record_additional_tokens(&self, scope_key: &str, extra_tokens: u64) {
        if extra_tokens == 0 {
            return;
        }
        if let Some(mut entry) = self.windows.get_mut(scope_key) {
            entry.tokens = entry.tokens.saturating_add(extra_tokens);
        }
    }

    pub fn in_flight_nanos(&self, budget_scope: &str) -> i64 {
        self.in_flight_budget_nanos
            .get(budget_scope)
            .map_or(0, |v| *v)
    }

    pub fn add_in_flight(&self, budget_scope: &str, delta_nanos: i64) {
        let mut entry = self
            .in_flight_budget_nanos
            .entry(budget_scope.to_owned())
            .or_insert(0);
        *entry = (*entry + delta_nanos).max(0);
    }

    pub fn check_scope_budget(
        &self,
        scope_label: &str,
        budget_scope_key: &str,
        committed_spend: NanoUsd,
        limit: Option<NanoUsd>,
        reserve_nanos: NanoUsd,
    ) -> Result<()> {
        let Some(limit) = limit else {
            return Ok(());
        };
        let in_flight = NanoUsd(self.in_flight_nanos(budget_scope_key));
        let projected = committed_spend.saturating_add(in_flight).saturating_add(reserve_nanos);
        if projected > limit {
            return Err(QuixigateError::BudgetExceeded(format!(
                "{scope_label} budget limit {} reached (spent {}, reserved {})",
                limit.format_usd(),
                committed_spend.format_usd(),
                in_flight.format_usd(),
            )));
        }
        Ok(())
    }
}
