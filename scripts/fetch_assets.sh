#!/usr/bin/env bash
# Download the demo assets (model checkpoint + 42 demo DICOMs) from the GitHub release
# and verify every file against assets/SHA256SUMS. Safe to re-run.
# For education only, not clinical use.
set -euo pipefail

RELEASE_URL="https://github.com/arch-pulkit/AIML-02-ai-imaging-diagnosis/releases/download/assets-v1"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v sha256sum >/dev/null 2>&1; then
  verify() { sha256sum -c --quiet assets/SHA256SUMS; }
else
  verify() { shasum -a 256 -c --quiet assets/SHA256SUMS; }
fi
count="$(wc -l < assets/SHA256SUMS | tr -d ' ')"

if verify >/dev/null 2>&1; then
  echo "assets already present and verified ($count files)"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "downloading from $RELEASE_URL"
curl -fL --retry 3 -o "$tmp/densenet121_best.pt" "$RELEASE_URL/densenet121_best.pt"
curl -fL --retry 3 -o "$tmp/demo_dicom.tar.gz" "$RELEASE_URL/demo_dicom.tar.gz"

mkdir -p models data/demo
mv "$tmp/densenet121_best.pt" models/densenet121_best.pt
rm -rf data/demo/dicom
tar -xzf "$tmp/demo_dicom.tar.gz" -C data/demo

verify
echo "OK: $count files verified against assets/SHA256SUMS"
