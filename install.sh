#!/bin/sh
# QuixiGate installer (https://github.com/QuixiAI/QuixiGate)
# Usage: curl -fsSL https://raw.githubusercontent.com/QuixiAI/QuixiGate/main/install.sh | sh
set -eu

REPO="${QUIXIGATE_REPO:-QuixiAI/QuixiGate}"
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

# Resolve production install target:
# 1. Explicit QUIXIGATE_INSTALL_DIR override if set
# 2. /usr/local/bin (standard macOS & Linux FHS path — directly if writable/root, or via sudo when available)
# 3. $HOME/.local/bin fallback for unprivileged headless environments without sudo
USE_SUDO=0
if [ -n "${QUIXIGATE_INSTALL_DIR:-}" ]; then
  INSTALL_DIR="$QUIXIGATE_INSTALL_DIR"
elif [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
  INSTALL_DIR="/usr/local/bin"
elif [ "$(id -u)" -eq 0 ]; then
  INSTALL_DIR="/usr/local/bin"
elif command -v sudo >/dev/null 2>&1 && { sudo -n true 2>/dev/null || sh -c ': </dev/tty' 2>/dev/null; }; then
  INSTALL_DIR="/usr/local/bin"
  USE_SUDO=1
else
  INSTALL_DIR="$HOME/.local/bin"
fi

install_binary() {
  src="$1"
  chmod 0755 "$src"
  if [ "$USE_SUDO" -eq 1 ]; then
    log "Installing quixigate to ${INSTALL_DIR}/quixigate (sudo required)..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo install -m 0755 "$src" "${INSTALL_DIR}/quixigate"
  else
    mkdir -p "$INSTALL_DIR"
    cp "$src" "${INSTALL_DIR}/quixigate.tmp.$$"
    chmod 0755 "${INSTALL_DIR}/quixigate.tmp.$$"
    mv "${INSTALL_DIR}/quixigate.tmp.$$" "${INSTALL_DIR}/quixigate"
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

installed_prebuilt=0
if [ -n "$OS_TRIPLE" ] && [ -n "$ARCH_TRIPLE" ] && command -v curl >/dev/null 2>&1 && command -v tar >/dev/null 2>&1; then
  TARGET="${ARCH_TRIPLE}-${OS_TRIPLE}"
  ARCHIVE="quixigate-${TARGET}.tar.gz"
  if [ "$VERSION" = "latest" ]; then
    BASE_URL="https://github.com/${REPO}/releases/latest/download"
  else
    BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
  fi

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
    install_binary "${TMP_DIR}/quixigate"
    installed_prebuilt=1
    log "Installed prebuilt quixigate (${TARGET}) to ${INSTALL_DIR}/quixigate"
  fi
fi

if [ "$installed_prebuilt" -eq 0 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    err "No prebuilt release found for ${OS}/${ARCH} and 'cargo' is not installed. Install Rust via https://rustup.rs first."
  fi
  log "Building quixigate from source via cargo (https://github.com/${REPO})..."
  CARGO_STAGE="${TMP_CARGO_ROOT:-$TMP_DIR/cargo-root}"
  cargo install --git "https://github.com/${REPO}.git" --locked --root "$CARGO_STAGE" quixigate
  [ -f "${CARGO_STAGE}/bin/quixigate" ] || err "cargo install did not produce quixigate binary"
  install_binary "${CARGO_STAGE}/bin/quixigate"
  log "Installed quixigate to ${INSTALL_DIR}/quixigate"
fi

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    printf '\nAdd %s to your PATH if needed:\n  export PATH="%s:$PATH"\n' "$INSTALL_DIR" "$INSTALL_DIR"
    ;;
esac

printf '\nQuickstart:\n  quixigate serve\n  quixigate login openrouter\n'
