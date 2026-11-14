"""
Poll Orthanc for new instances, run inference + Grad-CAM, POST one result each.

For education only, not clinical use.

    python -m src.serve.worker          # poll forever
    python -m src.serve.worker --once   # process what is pending, then exit

Delivery is at-least-once: the cursor is saved only after a result is POSTed, so a
crash can re-send a result. The results API must therefore be idempotent.
"""

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.nih import PROJECT_ROOT
from src.explain.gradcam import load_model
from src.serve.inference import (
    analyze,
    dicom_to_image,
    load_thresholds,
    offline_size,
    preprocess,
)
from src.serve.orthanc import ORTHANC_URL, get_bytes, get_json
from src.train.train import CHECKPOINT

RESULTS_URL = os.environ.get("RESULTS_URL", "http://localhost:8090/api/results")
STATE_FILE = Path(
    os.environ.get(
        "WORKER_STATE", PROJECT_ROOT / "data" / "interim" / "worker" / "cursor.json"
    )
)
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "2"))
MAX_BACKOFF = 60
CHANGES_LIMIT = 100

log = logging.getLogger("worker")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def png_b64(cam: np.ndarray) -> str:
    """
    Heatmap in [0, 1] -> 8-bit grayscale PNG -> base64 text.
    """
    buf = io.BytesIO()
    Image.fromarray(np.round(cam * 255).astype(np.uint8)).save(buf, format="PNG")

    return base64.b64encode(buf.getvalue()).decode()


def load_cursor() -> int:
    if not STATE_FILE.exists():
        return 0

    return int(json.loads(STATE_FILE.read_text())["last_seq"])


def save_cursor(seq: int) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_seq": seq}))
    tmp.replace(STATE_FILE)  # atomic: never a half-written cursor


def post_result(payload: dict) -> None:
    req = urllib.request.Request(
        RESULTS_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30):  # raises on 4xx/5xx
        pass


class Worker:
    def __init__(self) -> None:
        self.model = load_model("cpu")
        self.thresholds = load_thresholds()
        self.size = offline_size()
        self.model_info = {
            "name": "densenet121",
            "checkpoint_sha256": sha256_file(CHECKPOINT),
        }

    def build_payload(self, instance_id: str, data: bytes) -> dict:
        payload = {
            "orthanc_instance_id": instance_id,
            "model": self.model_info,
            "processed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        try:
            gray, meta = dicom_to_image(data)
            result = analyze(self.model, preprocess(gray, self.size), self.thresholds)
        except (
            Exception
        ) as e:  # bad input for this instance: report it, don't block the queue
            log.exception("analysis failed for %s", instance_id)
            return {**payload, "status": "error", "error": f"{type(e).__name__}: {e}"}
        first_cam = next(iter(result.heatmaps.values()))

        return {
            **payload,
            **meta,
            "status": "ok",
            "logits": result.logits,
            "thresholds": self.thresholds,
            "flagged": result.flagged,
            "heatmap_size": list(first_cam.shape),
            "heatmaps_png_b64": {k: png_b64(v) for k, v in result.heatmaps.items()},
        }

    def handle(self, change: dict) -> None:
        instance_id = change["ID"]
        try:
            data = get_bytes(f"/instances/{instance_id}/file")
        except urllib.error.HTTPError as e:
            if e.code == 404:  # deleted after the change was logged
                log.warning(
                    "seq=%d instance %s is gone, skipping", change["Seq"], instance_id
                )
                return
            raise
        payload = self.build_payload(instance_id, data)
        post_result(payload)
        log.info(
            "seq=%d %s %s flagged=%d",
            change["Seq"],
            payload.get("image_index", "?"),
            payload["status"],
            len(payload.get("flagged", [])),
        )

    def run_once(self) -> int:
        """
        Process every pending change; return the number of NewInstance changes handled.
        """
        handled = 0
        cursor = load_cursor()
        while True:
            page = get_json(f"/changes?since={cursor}&limit={CHANGES_LIMIT}")
            for change in page["Changes"]:
                if change["ChangeType"] == "NewInstance":
                    self.handle(change)
                    handled += 1
                cursor = change["Seq"]
                save_cursor(cursor)  # only after the result was POSTed
            if page["Done"]:
                return handled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="exit when caught up")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    worker = Worker()
    log.info(
        "worker up: orthanc=%s results=%s cursor=%d checkpoint=%s",
        ORTHANC_URL,
        RESULTS_URL,
        load_cursor(),
        worker.model_info["checkpoint_sha256"][:12],
    )
    delay = POLL_SECONDS
    while True:
        try:
            t0 = time.perf_counter()
            n = worker.run_once()
            if n:
                log.info(
                    "processed %d new instance(s) in %.1f s",
                    n,
                    time.perf_counter() - t0,
                )
            delay = POLL_SECONDS
            if args.once:
                log.info("caught up at seq %d", load_cursor())
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            delay = min(delay * 2, MAX_BACKOFF)
            log.warning("transient error, retrying in %.0f s: %s", delay, e)
        time.sleep(delay)


if __name__ == "__main__":
    main()
