//! Lock-free in-memory RPM / TPM / max-concurrency rate limiters and
//! atomic pre-flight `NanoUsd` budget reservations.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

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
    windows: DashMap<Arc<str>, MinuteWindow>,
    semaphores: DashMap<(Arc<str>, u32), Arc<Semaphore>>,
    in_flight_budget_nanos: DashMap<Arc<str>, AtomicI64>,
    active_requests: AtomicU64,
}

pub struct ConcurrencyGuard {
    _permit: Option<OwnedSemaphorePermit>,
    limiter: Arc<RuntimeLimiter>,
}

impl Drop for ConcurrencyGuard {
    #[inline]
    fn drop(&mut self) {
        self.limiter.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl RuntimeLimiter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    #[inline]
    pub fn active_requests(&self) -> u64 {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Fast zero-allocation acquire when no RPM/TPM/concurrency limits are configured.
    #[inline]
    pub fn acquire_unbounded(self: &Arc<Self>) -> ConcurrencyGuard {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        ConcurrencyGuard {
            _permit: None,
            limiter: Arc::clone(self),
        }
    }

    /// Checks RPM, estimated TPM, and max-concurrent-stream limits using a pre-interned `Arc<str>`.
    pub fn check_and_acquire_arc(
        self: &Arc<Self>,
        scope_key: &Arc<str>,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u64>,
        max_concurrent: Option<u32>,
        estimated_tokens: u64,
    ) -> Result<ConcurrencyGuard> {
        if max_concurrent.is_none() && rpm_limit.is_none() && tpm_limit.is_none() {
            return Ok(self.acquire_unbounded());
        }

        let permit = if let Some(max_conc) = max_concurrent.filter(|&c| c > 0) {
            let sem =
                if let Some(existing) = self.semaphores.get(&(Arc::clone(scope_key), max_conc)) {
                    Arc::clone(existing.value())
                } else {
                    self.semaphores
                        .entry((Arc::clone(scope_key), max_conc))
                        .or_insert_with(|| Arc::new(Semaphore::new(max_conc as usize)))
                        .clone()
                };
            Some(
                sem.try_acquire_owned()
                    .map_err(|_| QuixigateError::RateLimited {
                        message: format!("max concurrent requests ({max_conc}) exceeded"),
                        retry_after_secs: Some(1),
                    })?,
            )
        } else {
            None
        };

        if rpm_limit.is_some() || tpm_limit.is_some() {
            let now = now_ms();
            let mut entry = self
                .windows
                .entry(Arc::clone(scope_key))
                .or_insert(MinuteWindow {
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

    /// Checks RPM, estimated TPM, and max-concurrent-stream limits for `scope_key`.
    pub fn check_and_acquire(
        self: &Arc<Self>,
        scope_key: &str,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u64>,
        max_concurrent: Option<u32>,
        estimated_tokens: u64,
    ) -> Result<ConcurrencyGuard> {
        if max_concurrent.is_none() && rpm_limit.is_none() && tpm_limit.is_none() {
            return Ok(self.acquire_unbounded());
        }
        let arc_key: Arc<str> = Arc::from(scope_key);
        self.check_and_acquire_arc(
            &arc_key,
            rpm_limit,
            tpm_limit,
            max_concurrent,
            estimated_tokens,
        )
    }

    pub fn record_additional_tokens(&self, scope_key: &str, extra_tokens: u64) {
        if extra_tokens == 0 {
            return;
        }
        if let Some(mut entry) = self.windows.get_mut(scope_key) {
            entry.tokens = entry.tokens.saturating_add(extra_tokens);
        }
    }

    #[inline]
    pub fn in_flight_nanos(&self, budget_scope: &str) -> i64 {
        self.in_flight_budget_nanos
            .get(budget_scope)
            .map_or(0, |v| v.load(Ordering::Relaxed).max(0))
    }

    #[inline]
    pub fn add_in_flight_arc(&self, budget_scope: &Arc<str>, delta_nanos: i64) {
        if delta_nanos == 0 {
            return;
        }
        if let Some(slot) = self.in_flight_budget_nanos.get(budget_scope.as_ref()) {
            let prev = slot.fetch_add(delta_nanos, Ordering::Relaxed);
            if prev + delta_nanos < 0 {
                slot.store(0, Ordering::Relaxed);
            }
            return;
        }
        if delta_nanos > 0 {
            let slot = self
                .in_flight_budget_nanos
                .entry(Arc::clone(budget_scope))
                .or_insert_with(|| AtomicI64::new(0));
            let prev = slot.fetch_add(delta_nanos, Ordering::Relaxed);
            if prev + delta_nanos < 0 {
                slot.store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn add_in_flight(&self, budget_scope: &str, delta_nanos: i64) {
        if delta_nanos == 0 {
            return;
        }
        if let Some(slot) = self.in_flight_budget_nanos.get(budget_scope) {
            let prev = slot.fetch_add(delta_nanos, Ordering::Relaxed);
            if prev + delta_nanos < 0 {
                slot.store(0, Ordering::Relaxed);
            }
            return;
        }
        if delta_nanos > 0 {
            let arc_key: Arc<str> = Arc::from(budget_scope);
            self.add_in_flight_arc(&arc_key, delta_nanos);
        }
    }

    #[inline]
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
        let projected = committed_spend
            .saturating_add(in_flight)
            .saturating_add(reserve_nanos);
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
