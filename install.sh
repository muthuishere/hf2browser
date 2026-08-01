#!/bin/sh
# hf2browser installer — picks the right binary, checks it, puts it on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/muthuishere/hf2browser/main/install.sh | sh
#
# Environment:
#   HF2BROWSER_VERSION   tag to install (default: the latest release)
#   HF2BROWSER_BIN_DIR   where to install (default: ~/.local/bin)
set -eu

REPO="muthuishere/hf2browser"
VERSION="${HF2BROWSER_VERSION:-latest}"
BIN_DIR="${HF2BROWSER_BIN_DIR:-$HOME/.local/bin}"

die() { printf '\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }
say() { printf '%s\n' "$1"; }

# --- what are we running on ----------------------------------------------
case "$(uname -s)" in
  Darwin) os=darwin ;;
  Linux)  os=linux ;;
  *) die "unsupported OS $(uname -s) — Windows users, see the README for the PowerShell one-liner" ;;
esac
case "$(uname -m)" in
  arm64|aarch64) arch=arm64 ;;
  x86_64|amd64)  arch=amd64 ;;
  *) die "unsupported architecture $(uname -m)" ;;
esac
# Only linux/amd64+arm64 and darwin/amd64+arm64 are published.
asset="hf2browser-${os}-${arch}"

# --- how do we fetch ------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
else
  die "need curl or wget"
fi

if [ "$VERSION" = latest ]; then
  base="https://github.com/$REPO/releases/latest/download"
else
  base="https://github.com/$REPO/releases/download/$VERSION"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

say "downloading $asset ($VERSION)…"
fetch "$base/$asset" "$tmp/$asset" || die "download failed — is $VERSION a published release?"

# --- verify, when we can --------------------------------------------------
# A binary you pipe from the internet deserves at least a checksum check.
if fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" 2>/dev/null; then
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$tmp/$asset" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    got="$(shasum -a 256 "$tmp/$asset" | cut -d' ' -f1)"
  else
    got=""
  fi
  want="$(grep " $asset\$" "$tmp/SHA256SUMS" | cut -d' ' -f1 || true)"
  if [ -n "$got" ] && [ -n "$want" ]; then
    [ "$got" = "$want" ] || die "checksum mismatch for $asset — refusing to install"
    say "checksum ok"
  fi
fi

# --- install --------------------------------------------------------------
mkdir -p "$BIN_DIR"
chmod +x "$tmp/$asset"
mv "$tmp/$asset" "$BIN_DIR/hf2browser"
say "installed $BIN_DIR/hf2browser"

case ":${PATH}:" in
  *":$BIN_DIR:"*) ;;
  *) say ""
     say "$BIN_DIR is not on your PATH. Add it:"
     say "    export PATH=\"\$PATH:$BIN_DIR\"" ;;
esac

say ""
say "next:  hf2browser serve      # search → convert → chat, all local"
say "       hf2browser init       # write an editable hf2browser.json"
