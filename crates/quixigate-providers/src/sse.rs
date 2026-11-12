//! Incremental raw-byte Server-Sent Events (SSE) decoder.
//!
//! Handles arbitrary TCP chunk splits, `\r\n` / `\n` / `\r` line endings,
//! optional `event:` names (used by Anthropic `/v1/messages`), and enforces a
//! 16 MiB frame limit against malformed upstream streams.

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseFrame {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FrameTooLarge;

#[derive(Debug, Default)]
pub struct SseDecoder {
    line: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
    frame_bytes: usize,
    after_cr: bool,
}

impl SseDecoder {
    /// Feeds raw bytes and returns every complete `SseFrame` they finish.
    pub fn push_frames(&mut self, bytes: &[u8]) -> Result<Vec<SseFrame>, FrameTooLarge> {
        let mut frames = Vec::new();
        for &byte in bytes {
            let after_cr = std::mem::replace(&mut self.after_cr, byte == b'\r');
            match byte {
                b'\n' if after_cr => {}
                b'\n' | b'\r' => self.end_line(&mut frames),
                _ => {
                    self.frame_bytes += 1;
                    if self.frame_bytes > MAX_FRAME_BYTES {
                        return Err(FrameTooLarge);
                    }
                    self.line.push(byte);
                }
            }
        }
        Ok(frames)
    }

    /// Convenience wrapper returning only the `data` strings of completed events.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, FrameTooLarge> {
        Ok(self
            .push_frames(bytes)?
            .into_iter()
            .map(|f| f.data)
            .collect())
    }

    /// Flushes any trailing frame when the byte stream closes without a blank line.
    pub fn finish_frame(&mut self) -> Option<SseFrame> {
        let mut frames = Vec::new();
        if !self.line.is_empty() {
            self.end_line(&mut frames);
        }
        self.end_line(&mut frames);
        frames.pop()
    }

    pub fn finish(&mut self) -> Option<String> {
        self.finish_frame().map(|f| f.data)
    }

    fn end_line(&mut self, frames: &mut Vec<SseFrame>) {
        let line = std::mem::take(&mut self.line);
        if line.is_empty() {
            self.frame_bytes = 0;
            let event = self.event.take();
            if !self.data.is_empty() {
                frames.push(SseFrame {
                    event,
                    data: std::mem::take(&mut self.data).join("\n"),
                });
            }
            return;
        }
        let line = String::from_utf8_lossy(&line);
        if let Some(value) = line.strip_prefix("data:") {
            self.data
                .push(value.strip_prefix(' ').unwrap_or(value).to_owned());
        } else if line == "data" {
            self.data.push(String::new());
        } else if let Some(ev) = line.strip_prefix("event:") {
            self.event = Some(ev.strip_prefix(' ').unwrap_or(ev).trim().to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_across_arbitrary_byte_splits_and_preserves_event_name() {
        let raw = "event: content_block_delta\r\ndata: {\"text\":\"Hel\"}\r\n\r\ndata: {\"text\":\"lo\u{00e9}\"}\n\n";
        for split in 1..raw.len() {
            let (a, b) = raw.as_bytes().split_at(split);
            let mut dec = SseDecoder::default();
            let mut frames = dec.push_frames(a).unwrap();
            frames.extend(dec.push_frames(b).unwrap());
            assert_eq!(frames.len(), 2, "split at {split}");
            assert_eq!(frames[0].event.as_deref(), Some("content_block_delta"));
            assert_eq!(frames[0].data, "{\"text\":\"Hel\"}");
            assert_eq!(frames[1].event, None);
            assert_eq!(frames[1].data, "{\"text\":\"lo\u{00e9}\"}");
        }
    }
}
