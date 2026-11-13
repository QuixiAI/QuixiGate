//! Exact integer nano-dollar (`1 USD = 1_000_000_000 nano-USD`) accounting.
//!
//! IEEE-754 floating-point numbers are never used for ledger storage or budget
//! enforcement. `NanoUsd(i64)` represents values down to `$0.000000001` with a
//! range exceeding `±$9.22 billion`.

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use serde::{Deserialize, Serialize};

use crate::error::{QuixigateError, Result};

/// One US dollar in nano-dollars (`10^9`).
pub const NANOS_PER_USD: i64 = 1_000_000_000;

/// Exact monetary value in nano-dollars (`1e-9` USD).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct NanoUsd(pub i64);

impl NanoUsd {
    pub const ZERO: Self = Self(0);

    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    pub const fn nanos(self) -> i64 {
        self.0
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.0.checked_add(rhs.0).map(Self)
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
    }

    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Parses a decimal USD string (e.g. `"25"`, `"25.50"`, `"$0.00000125"`)
    /// into exact `NanoUsd` using integer-only parsing.
    pub fn parse_usd(input: &str) -> Result<Self> {
        let trimmed = input.trim().trim_start_matches('$').trim();
        if trimmed.is_empty() {
            return Err(QuixigateError::Validation("empty USD amount".to_owned()));
        }
        let (negative, unsigned) = if let Some(rest) = trimmed.strip_prefix('-') {
            (true, rest)
        } else if let Some(rest) = trimmed.strip_prefix('+') {
            (false, rest)
        } else {
            (false, trimmed)
        };

        let (whole_str, frac_str) = match unsigned.split_once('.') {
            Some((w, f)) => (w, f),
            None => (unsigned, ""),
        };
        if whole_str.is_empty() && frac_str.is_empty() {
            return Err(QuixigateError::Validation(format!(
                "invalid USD amount: {input}"
            )));
        }
        if !whole_str.chars().all(|c| c.is_ascii_digit())
            || !frac_str.chars().all(|c| c.is_ascii_digit())
        {
            return Err(QuixigateError::Validation(format!(
                "invalid USD amount: {input}"
            )));
        }

        let whole: i128 = if whole_str.is_empty() {
            0
        } else {
            whole_str
                .parse()
                .map_err(|_| QuixigateError::Validation(format!("USD amount overflow: {input}")))?
        };

        // Take up to 9 fractional digits; round half-up on the 10th digit if present.
        let mut padded = String::with_capacity(9);
        for ch in frac_str.chars().take(9) {
            padded.push(ch);
        }
        while padded.len() < 9 {
            padded.push('0');
        }
        let mut frac: i128 = padded.parse().unwrap_or(0);
        if let Some(tenth) = frac_str.chars().nth(9)
            && tenth >= '5'
        {
            frac += 1;
        }

        let total = whole
            .checked_mul(NANOS_PER_USD as i128)
            .and_then(|v| v.checked_add(frac))
            .ok_or_else(|| QuixigateError::Validation(format!("USD amount overflow: {input}")))?;
        let signed = if negative { -total } else { total };
        let nanos = i64::try_from(signed)
            .map_err(|_| QuixigateError::Validation(format!("USD amount out of range: {input}")))?;
        Ok(Self(nanos))
    }

    /// Computes exact token cost given a token count and price per 1,000,000 tokens
    /// in nano-dollars (`nanos_per_million`), rounding up sub-nano fractions so
    /// paid usage never under-bills to zero.
    pub fn from_tokens_and_rate(tokens: u64, nanos_per_million: i64) -> Self {
        if tokens == 0 || nanos_per_million <= 0 {
            return Self::ZERO;
        }
        let product = (tokens as i128) * (nanos_per_million as i128);
        let rounded_up = (product + 999_999) / 1_000_000;
        Self(i64::try_from(rounded_up).unwrap_or(i64::MAX))
    }

    /// Formats as a clean decimal USD string without `$`, suitable for OpenRouter
    /// / OpenInference JSON fields (e.g. `"0.004200"`, `"25.50"`).
    pub fn to_decimal_string(self) -> String {
        let sign = if self.0 < 0 { "-" } else { "" };
        let abs = self.0.unsigned_abs();
        let whole = abs / (NANOS_PER_USD as u64);
        let rem = abs % (NANOS_PER_USD as u64);
        if rem == 0 {
            format!("{sign}{whole}.00")
        } else {
            let frac = format!("{rem:09}");
            let trimmed = frac.trim_end_matches('0');
            if trimmed.len() < 2 {
                format!("{sign}{whole}.{trimmed:0<2}")
            } else {
                format!("{sign}{whole}.{trimmed}")
            }
        }
    }

    /// Formats with a leading `$` sign (e.g. `"$12.50"`, `"-$0.05"`, `"$0.000412"`).
    pub fn format_usd(self) -> String {
        if self.0 < 0 {
            format!("-${}", Self(self.0.saturating_neg()).to_decimal_string())
        } else {
            format!("${}", self.to_decimal_string())
        }
    }
}

impl fmt::Display for NanoUsd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_usd())
    }
}

impl Add for NanoUsd {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.saturating_add(rhs)
    }
}

impl AddAssign for NanoUsd {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.saturating_add(rhs);
    }
}

impl Sub for NanoUsd {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.saturating_sub(rhs)
    }
}

impl SubAssign for NanoUsd {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.saturating_sub(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_decimal_parsing_and_formatting() {
        assert_eq!(NanoUsd::parse_usd("25").unwrap(), NanoUsd(25_000_000_000));
        assert_eq!(
            NanoUsd::parse_usd("$25.50").unwrap(),
            NanoUsd(25_500_000_000)
        );
        assert_eq!(NanoUsd::parse_usd("0.000000001").unwrap(), NanoUsd(1));
        assert_eq!(NanoUsd::parse_usd(".5").unwrap(), NanoUsd(500_000_000));
        assert_eq!(
            NanoUsd::parse_usd("-1.25").unwrap(),
            NanoUsd(-1_250_000_000)
        );
        assert_eq!(NanoUsd(25_500_000_000).format_usd(), "$25.50");
        assert_eq!(NanoUsd(4_210_000).format_usd(), "$0.00421");
        assert_eq!(NanoUsd(0).format_usd(), "$0.00");
        assert_eq!(NanoUsd(-50_000_000).format_usd(), "-$0.05");
    }

    #[test]
    fn token_cost_never_underflows_or_drifts() {
        // $3.00 / 1M input tokens => 3_000_000_000 nanos per 1M => 3,000 nanos per token.
        let rate = NanoUsd::parse_usd("3.00").unwrap().nanos();
        assert_eq!(
            NanoUsd::from_tokens_and_rate(1_000, rate),
            NanoUsd(3_000_000)
        );
        // Even 1 token at $0.05 / 1M = 50_000_000 nanos/1M = 50 nanos exact.
        let cheap = NanoUsd::parse_usd("0.05").unwrap().nanos();
        assert_eq!(NanoUsd::from_tokens_and_rate(1, cheap), NanoUsd(50));
        // Sub-nano fractional cost rounds up to 1 nano, never 0.
        assert_eq!(NanoUsd::from_tokens_and_rate(1, 1), NanoUsd(1));
    }
}
