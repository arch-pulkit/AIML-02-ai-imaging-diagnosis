"""Minimal Orthanc REST client (stdlib only). Settings come from environment variables."""

import base64
import json
import os
import urllib.request

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://localhost:8042")
ORTHANC_USER = os.environ.get("ORTHANC_USER", "demo")
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", "demo")
TIMEOUT = 30


def _request(
    path: str, data: bytes | None = None, content_type: str | None = None
) -> bytes:
    token = base64.b64encode(f"{ORTHANC_USER}:{ORTHANC_PASSWORD}".encode()).decode()
    headers = {"Authorization": f"Basic {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    method = "GET" if data is None else "POST"
    req = urllib.request.Request(
        f"{ORTHANC_URL}{path}", data=data, headers=headers, method=method
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # raises on HTTP errors
        return resp.read()


def get_json(path: str):
    return json.loads(_request(path))


def get_bytes(path: str) -> bytes:
    return _request(path)


def upload_dicom(data: bytes) -> dict:
    return json.loads(_request("/instances", data, "application/dicom"))
