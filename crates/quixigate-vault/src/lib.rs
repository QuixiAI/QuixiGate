//! AES-256-GCM cryptographic vault for QuixiGate upstream secrets.
//!
//! The vault master key is strictly decoupled from admin logins, session
//! cookies, and `sk-qx-...` virtual keys so rotating user/admin credentials
//! never bricks stored OAuth tokens or provider API keys.

use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;
use std::process::Command;

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use quixigate_core::{QuixigateError, Result};
use rand::RngCore;
use serde::Serialize;
use serde::de::DeserializeOwned;
use zeroize::Zeroizing;

const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
const VERSION_V1: u8 = 1;

#[derive(Clone)]
pub struct Vault {
    cipher: Aes256Gcm,
    source: VaultKeySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaultKeySource {
    EnvVar,
    MacosKeychain,
    KeyFile,
    InMemory,
}

impl VaultKeySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EnvVar => "env:QUIXIGATE_VAULT_KEY",
            Self::MacosKeychain => "macos_keychain",
            Self::KeyFile => "keyfile",
            Self::InMemory => "in_memory",
        }
    }
}

impl std::fmt::Debug for Vault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vault")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl Vault {
    /// Creates an ephemeral random vault (useful for unit and integration tests).
    pub fn ephemeral() -> Self {
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        rand::thread_rng().fill_bytes(&mut *key);
        Self::from_raw_key(&key, VaultKeySource::InMemory)
    }

    pub fn from_raw_key(key: &[u8; KEY_LEN], source: VaultKeySource) -> Self {
        let cipher = Aes256Gcm::new_from_slice(key).expect("32-byte AES-256 key");
        Self { cipher, source }
    }

    pub fn source(&self) -> VaultKeySource {
        self.source
    }

    /// Loads or initializes the vault master key in strict precedence order:
    /// 1. `QUIXIGATE_VAULT_KEY` environment variable (hex or base64 32-byte key).
    /// 2. macOS Keychain when `QUIXIGATE_VAULT_KEYCHAIN=1`.
    /// 3. Atomic `0600` `<data_dir>/vault.key` file (generated on first boot).
    pub fn load_or_init(data_dir: &Path) -> Result<Self> {
        if let Some(raw) = quixigate_core::brand::env_opt("QUIXIGATE_VAULT_KEY") {
            let key = decode_key(&raw)?;
            return Ok(Self::from_raw_key(&key, VaultKeySource::EnvVar));
        }

        if cfg!(target_os = "macos")
            && quixigate_core::brand::env_opt("QUIXIGATE_VAULT_KEYCHAIN")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            && let Some(key) = load_or_create_macos_keychain()?
        {
            return Ok(Self::from_raw_key(&key, VaultKeySource::MacosKeychain));
        }

        let key_path = data_dir.join("vault.key");
        if key_path.exists() {
            let text = fs::read_to_string(&key_path).map_err(|err| {
                QuixigateError::Vault(format!("cannot read {}: {err}", key_path.display()))
            })?;
            let key = decode_key(text.trim())?;
            return Ok(Self::from_raw_key(&key, VaultKeySource::KeyFile));
        }

        ensure_private_dir(data_dir)?;
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        rand::thread_rng().fill_bytes(&mut *key);
        let encoded = Zeroizing::new(hex::encode(*key));

        let tmp = key_path.with_extension("key.tmp");
        let _ = fs::remove_file(&tmp);
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        opts.mode(0o600);
        let mut file = opts.open(&tmp).map_err(|err| {
            QuixigateError::Vault(format!("cannot create {}: {err}", tmp.display()))
        })?;
        file.write_all(encoded.as_bytes()).map_err(|err| {
            QuixigateError::Vault(format!("cannot write {}: {err}", tmp.display()))
        })?;
        file.sync_all().map_err(|err| {
            QuixigateError::Vault(format!("cannot sync {}: {err}", tmp.display()))
        })?;
        fs::rename(&tmp, &key_path).map_err(|err| {
            QuixigateError::Vault(format!("cannot finalize {}: {err}", key_path.display()))
        })?;

        Ok(Self::from_raw_key(&key, VaultKeySource::KeyFile))
    }

    /// Encrypts `bytes` using AES-256-GCM with `aad` bound into the authentication tag.
    /// Wire envelope: `[VERSION_V1 (1B) | nonce (12B) | ciphertext + GCM tag]`.
    pub fn encrypt(&self, aad: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = self
            .cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| QuixigateError::Vault("AES-256-GCM encryption failed".to_owned()))?;

        let mut out = Vec::with_capacity(1 + NONCE_LEN + ciphertext.len());
        out.push(VERSION_V1);
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypts a vault envelope and verifies both the GCM tag and `aad`.
    pub fn decrypt(&self, aad: &str, envelope: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
        if envelope.len() < 1 + NONCE_LEN + 16 {
            return Err(QuixigateError::Vault(
                "encrypted envelope is truncated".to_owned(),
            ));
        }
        if envelope[0] != VERSION_V1 {
            return Err(QuixigateError::Vault(format!(
                "unsupported vault envelope version: {}",
                envelope[0]
            )));
        }
        let nonce = Nonce::from_slice(&envelope[1..1 + NONCE_LEN]);
        let ciphertext = &envelope[1 + NONCE_LEN..];
        let plaintext = self
            .cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| {
                QuixigateError::Vault(
                    "vault decryption failed (wrong vault key or tampered AAD/ciphertext)"
                        .to_owned(),
                )
            })?;
        Ok(Zeroizing::new(plaintext))
    }

    pub fn encrypt_json<T: Serialize>(&self, aad_id: &str, value: &T) -> Result<Vec<u8>> {
        let bytes = Zeroizing::new(
            serde_json::to_vec(value)
                .map_err(|err| QuixigateError::Vault(format!("secret serialize failed: {err}")))?,
        );
        self.encrypt(aad_id, &bytes)
    }

    pub fn decrypt_json<T: DeserializeOwned>(&self, aad_id: &str, envelope: &[u8]) -> Result<T> {
        let plain = self.decrypt(aad_id, envelope)?;
        serde_json::from_slice(&plain)
            .map_err(|err| QuixigateError::Vault(format!("secret deserialize failed: {err}")))
    }
}

fn decode_key(input: &str) -> Result<Zeroizing<[u8; KEY_LEN]>> {
    let bytes = if input.len() == 64 && input.chars().all(|c| c.is_ascii_hexdigit()) {
        hex::decode(input)
            .map_err(|err| QuixigateError::Vault(format!("invalid hex vault key: {err}")))?
    } else {
        STANDARD
            .decode(input)
            .or_else(|_| URL_SAFE_NO_PAD.decode(input.trim_end_matches('=')))
            .map_err(|err| {
                QuixigateError::Vault(format!(
                    "vault key must be 64 hex chars or 32-byte base64: {err}"
                ))
            })?
    };

    if bytes.len() != KEY_LEN {
        return Err(QuixigateError::Vault(format!(
            "vault key must decode to {KEY_LEN} bytes, got {}",
            bytes.len()
        )));
    }
    let mut arr = Zeroizing::new([0u8; KEY_LEN]);
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

fn ensure_private_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        builder.mode(0o700);
        builder.create(dir).map_err(|err| {
            QuixigateError::Vault(format!("cannot create {}: {err}", dir.display()))
        })?;
    }
    Ok(())
}

fn load_or_create_macos_keychain() -> Result<Option<Zeroizing<[u8; KEY_LEN]>>> {
    let find = Command::new("security")
        .args([
            "find-generic-password",
            "-s",
            "quixigate-vault",
            "-a",
            "master",
            "-w",
        ])
        .output();
    if let Ok(out) = find
        && out.status.success()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        if !text.trim().is_empty() {
            return decode_key(text.trim()).map(Some);
        }
    }

    let mut key = Zeroizing::new([0u8; KEY_LEN]);
    rand::thread_rng().fill_bytes(&mut *key);
    let hex_key = Zeroizing::new(hex::encode(*key));
    let add = Command::new("security")
        .args([
            "add-generic-password",
            "-U",
            "-s",
            "quixigate-vault",
            "-a",
            "master",
            "-w",
            &hex_key,
        ])
        .output();
    if let Ok(out) = add
        && out.status.success()
    {
        return Ok(Some(key));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use quixigate_core::CredentialSecret;

    use super::*;

    #[test]
    fn roundtrips_secret_and_rejects_aad_swap_or_tampering() {
        let vault = Vault::ephemeral();
        let secret = CredentialSecret::Codex {
            access: "acc_123".to_owned(),
            refresh: "ref_456".to_owned(),
            expires_ms: 1_900_000_000_000,
            account_id: "acct_abc".to_owned(),
        };
        let blob = vault.encrypt_json("cred_1", &secret).unwrap();

        // Wrong credential_id AAD fails authentication!
        assert!(
            vault
                .decrypt_json::<CredentialSecret>("cred_2", &blob)
                .is_err()
        );

        // Bit flip in ciphertext fails authentication!
        let mut tampered = blob.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        assert!(
            vault
                .decrypt_json::<CredentialSecret>("cred_1", &tampered)
                .is_err()
        );

        // Matching AAD succeeds.
        let decoded: CredentialSecret = vault.decrypt_json("cred_1", &blob).unwrap();
        match decoded {
            CredentialSecret::Codex {
                access, account_id, ..
            } => {
                assert_eq!(access, "acc_123");
                assert_eq!(account_id, "acct_abc");
            }
            _ => panic!("wrong secret variant"),
        }
    }

    #[test]
    fn creates_0600_keyfile_and_reloads_identically() {
        let dir = tempfile::tempdir().unwrap();
        let v1 = Vault::load_or_init(dir.path()).unwrap();
        assert_eq!(v1.source(), VaultKeySource::KeyFile);
        let blob = v1.encrypt("row_a", b"super-secret").unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(dir.path().join("vault.key"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }

        let v2 = Vault::load_or_init(dir.path()).unwrap();
        let plain = v2.decrypt("row_a", &blob).unwrap();
        assert_eq!(&plain[..], b"super-secret");
    }
}
