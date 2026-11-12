//! Polyglot ingress wire codecs translating OpenAI Chat, OpenAI Responses (SSE + WS),
//! Anthropic Messages, Google Gemini, Vercel AI UI Stream, and Embeddings to & from `aisdk`.

pub mod anthropic;
pub mod common;
pub mod gemini;
pub mod misc;
pub mod openai_chat;
pub mod openai_responses;
pub mod vercel_ui;
