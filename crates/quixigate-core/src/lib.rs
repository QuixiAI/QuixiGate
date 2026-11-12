//! Core domain types, exact `NanoUsd(i64)` accounting, model specifier parser,
//! and structured gateway errors for QuixiGate.

pub mod brand;
pub mod error;
pub mod model_spec;
pub mod money;
pub mod types;

pub use error::{QuixigateError, Result};
pub use model_spec::ModelSpec;
pub use money::{NANOS_PER_USD, NanoUsd};
pub use types::*;
