#!/bin/sh
# QuixiGate installer (https://github.com/QuixiAI/QuixiGate)
# Usage: curl -fsSL https://raw.githubusercontent.com/QuixiAI/QuixiGate/main/install.sh | sh
set -eu

REPO="${QUIXIGATE_REPO:-QuixiAI/QuixiGate}"
INSTALL_DIR="${QUIXIGATE_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${QUIXIGATE_VERSION:-latest}"

log() {
  printf '==> %s\n' "$*"
}

err() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin) OS_TRIPLE="apple-darwin" ;;
  Linux)  OS_TRIPLE="unknown-linux-gnu" ;;
  *)      OS_TRIPLE="" ;;
esac

case "$ARCH" in
  arm64|aarch64) ARCH_TRIPLE="aarch64" ;;
  x86_64|amd64)  ARCH_TRIPLE="x86_64" ;;
  *)             ARCH_TRIPLE="" ;;
esac

mkdir -p "$INSTALL_DIR"

installed_prebuilt=0
if [ -n "$OS_TRIPLE" ] && [ -n "$ARCH_TRIPLE" ] && command -v curl >/dev/null 2>&1 && command -v tar >/dev/null 2>&1; then
  TARGET="${ARCH_TRIPLE}-${OS_TRIPLE}"
  ARCHIVE="quixigate-${TARGET}.tar.gz"
  if [ "$VERSION" = "latest" ]; then
    BASE_URL="https://github.com/${REPO}/releases/latest/download"
  else
    BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
  fi

  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

  if curl -fsSL "${BASE_URL}/${ARCHIVE}" -o "${TMP_DIR}/${ARCHIVE}" 2>/dev/null; then
    if curl -fsSL "${BASE_URL}/${ARCHIVE}.sha256" -o "${TMP_DIR}/${ARCHIVE}.sha256" 2>/dev/null; then
      EXPECTED="$(awk '{print $1}' "${TMP_DIR}/${ARCHIVE}.sha256")"
      if command -v shasum >/dev/null 2>&1; then
        ACTUAL="$(shasum -a 256 "${TMP_DIR}/${ARCHIVE}" | awk '{print $1}')"
      elif command -v sha256sum >/dev/null 2>&1; then
        ACTUAL="$(sha256sum "${TMP_DIR}/${ARCHIVE}" | awk '{print $1}')"
      else
        ACTUAL="$EXPECTED"
      fi
      [ "$EXPECTED" = "$ACTUAL" ] || err "SHA-256 checksum mismatch for ${ARCHIVE}"
    fi

    tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "$TMP_DIR"
    [ -f "${TMP_DIR}/quixigate" ] || err "Archive ${ARCHIVE} did not contain quixigate binary"
    chmod +x "${TMP_DIR}/quixigate"
    mv "${TMP_DIR}/quixigate" "${INSTALL_DIR}/quixigate"
    installed_prebuilt=1
    log "Installed prebuilt quixigate (${TARGET}) to ${INSTALL_DIR}/quixigate"
  fi
fi

if [ "$installed_prebuilt" -eq 0 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    err "No prebuilt release found for ${OS}/${ARCH} and 'cargo' is not installed. Install Rust via https://rustup.rs first."
  fi
  log "Building quixigate from source via cargo (https://github.com/${REPO})..."
  cargo install --git "https://github.com/${REPO}.git" --locked --root "${TMP_CARGO_ROOT:-$HOME/.cargo}" quixigate
  if [ "$INSTALL_DIR" != "${TMP_CARGO_ROOT:-$HOME/.cargo}/bin" ] && [ -f "${TMP_CARGO_ROOT:-$HOME/.cargo}/bin/quixigate" ]; then
    cp "${TMP_CARGO_ROOT:-$HOME/.cargo}/bin/quixigate" "${INSTALL_DIR}/quixigate"
    chmod +x "${INSTALL_DIR}/quixigate"
  fi
  log "Installed quixigate to ${INSTALL_DIR}/quixigate"
fi

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    printf '\nAdd %s to your PATH if needed:\n  export PATH="%s:$PATH"\n' "$INSTALL_DIR" "$INSTALL_DIR"
    ;;
esac

printf '\nQuickstart:\n  quixigate serve\n  quixigate login openrouter\n'
