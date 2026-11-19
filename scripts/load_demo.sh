#!/usr/bin/env bash
# Upload the demo DICOMs (data/demo/dicom/) to Orthanc using the login from .env.
# Safe to re-run: Orthanc answers "AlreadyStored" for files it already has.
# Run scripts/fetch_assets.sh and docker compose up -d first.
# For education only, not clinical use.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -f .env ] || { echo "missing .env: cp .env.example .env" >&2; exit 1; }
set -a
. ./.env
set +a
ORTHANC="http://127.0.0.1:8042"

shopt -s nullglob
files=(data/demo/dicom/*.dcm)
[ "${#files[@]}" -gt 0 ] || { echo "no DICOMs in data/demo/dicom: run scripts/fetch_assets.sh" >&2; exit 1; }

for i in $(seq 1 30); do
  code="$(curl -sS -m 2 -o /dev/null -w '%{http_code}' -u "$ORTHANC_USER:$ORTHANC_PASSWORD" "$ORTHANC/system" 2>/dev/null || true)"
  case "$code" in
    200) break ;;
    401) echo "Orthanc rejected the login from .env (HTTP 401)" >&2; exit 1 ;;
  esac
  [ "$i" -eq 30 ] && { echo "Orthanc not reachable at $ORTHANC (HTTP '$code'); is docker compose up?" >&2; exit 1; }
  sleep 2
done

ok=0
already=0
for f in "${files[@]}"; do
  status="$(curl -fsS -m 30 -u "$ORTHANC_USER:$ORTHANC_PASSWORD" -H 'Content-Type: application/dicom' \
    --data-binary "@$f" "$ORTHANC/instances" | tr -d '\n' | sed -n 's/.*"Status" *: *"\([A-Za-z]*\)".*/\1/p')"
  case "$status" in
    Success) ok=$((ok + 1)) ;;
    AlreadyStored) already=$((already + 1)) ;;
    *) echo "unexpected Orthanc status '$status' for $f" >&2; exit 1 ;;
  esac
done
echo "uploaded ${#files[@]} files to $ORTHANC: Success $ok, AlreadyStored $already"
