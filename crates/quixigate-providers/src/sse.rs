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
    /// Feeds raw bytes and returns every complete `SseFrame` they finish using SIMD newline scanning.
    pub fn push_frames(&mut self, bytes: &[u8]) -> Result<Vec<SseFrame>, FrameTooLarge> {
        let mut frames = Vec::new();
        if bytes.is_empty() {
            return Ok(frames);
        }

        let mut pos = 0usize;
        if self.after_cr {
            self.after_cr = false;
            if bytes[0] == b'\n' {
                pos = 1;
            }
        }

        while pos < bytes.len() {
            let rest = &bytes[pos..];
            match memchr::memchr2(b'\n', b'\r', rest) {
                Some(idx) => {
                    let chunk = &rest[..idx];
                    self.frame_bytes += chunk.len();
                    if self.frame_bytes > MAX_FRAME_BYTES {
                        return Err(FrameTooLarge);
                    }
                    if self.line.is_empty() {
                        self.process_line_bytes(chunk, &mut frames);
                    } else {
                        self.line.extend_from_slice(chunk);
                        let line = std::mem::take(&mut self.line);
                        self.process_line_bytes(&line, &mut frames);
                    }

                    if rest[idx] == b'\r' {
                        if idx + 1 < rest.len() {
                            pos += if rest[idx + 1] == b'\n' {
                                idx + 2
                            } else {
                                idx + 1
                            };
                        } else {
                            self.after_cr = true;
                            pos = bytes.len();
                        }
                    } else {
                        pos += idx + 1;
                    }
                }
                None => {
                    self.frame_bytes += rest.len();
                    if self.frame_bytes > MAX_FRAME_BYTES {
                        return Err(FrameTooLarge);
                    }
                    self.line.extend_from_slice(rest);
                    break;
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
            let line = std::mem::take(&mut self.line);
            self.process_line_bytes(&line, &mut frames);
        }
        self.process_line_bytes(&[], &mut frames);
        frames.pop()
    }

    pub fn finish(&mut self) -> Option<String> {
        self.finish_frame().map(|f| f.data)
    }

    #[inline]
    fn process_line_bytes(&mut self, line: &[u8], frames: &mut Vec<SseFrame>) {
        if line.is_empty() {
            self.frame_bytes = 0;
            let event = self.event.take();
            if !self.data.is_empty() {
                let data = if self.data.len() == 1 {
                    self.data.pop().unwrap_or_default()
                } else {
                    std::mem::take(&mut self.data).join("\n")
                };
                frames.push(SseFrame { event, data });
            }
            return;
        }
        let line_str = match std::str::from_utf8(line) {
            Ok(s) => std::borrow::Cow::Borrowed(s),
            Err(_) => String::from_utf8_lossy(line),
        };
        if let Some(value) = line_str.strip_prefix("data:") {
            self.data
                .push(value.strip_prefix(' ').unwrap_or(value).to_owned());
        } else if line_str == "data" {
            self.data.push(String::new());
        } else if let Some(ev) = line_str.strip_prefix("event:") {
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
