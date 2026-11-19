# Results API

Go service in `services/api/` (standard library `net/http` + SQLite). It stores the worker's results, serves them
to the viewer, records reviewer decisions and keeps the tamper-evident audit log. Base URL in the demo:
`http://127.0.0.1:8080` (the viewer reaches it as `/api/…` through nginx on port 3000).

- **Format:** JSON (`Content-Type: application/json`) unless stated. Errors are always `{"error": "<message>"}`
  with a 4xx/5xx status; the viewer shows the message verbatim.
- **Times:** RFC 3339 in UTC with milliseconds, e.g. `2026-11-19T03:48:27.064Z`.
- **IDs:** a result is identified by its Orthanc instance id, e.g. `a52f321f-18861d68-f4118abf-c0ad37cb-eaff259b`.
- **No authentication.** All ports are bound to `127.0.0.1`; the reviewer name is self-declared.

| Method and path | Purpose | Used by |
|---|---|---|
| `GET /healthz` | Health check | Docker health check |
| `POST /api/results` | Store one analysis result | Worker |
| `GET /api/results` | List all results (summaries) | Worklist |
| `GET /api/results/{id}` | One result with labels and decisions | Detail page |
| `GET /api/results/{id}/heatmaps/{label}` | Grad-CAM heatmap PNG | Overlay |
| `GET /api/results/{id}/preview` | X-ray preview PNG (proxied from Orthanc) | Detail page |
| `POST /api/results/{id}/decisions` | Record approve / reject | Decision form |
| `GET /api/audit` | Page through the audit log | Audit page |
| `GET /api/audit/verify` | Recompute the whole hash chain | Audit page badge |

## Health

`GET /healthz` → `200 {"schema_version":1,"sqlite":"3.54.1","status":"ok"}`, or
`503 {"status":"db_unavailable","error":"…"}` when the database cannot be reached. The container health check runs
the same binary with `-healthcheck`, which calls this endpoint.

## Results

### Summary object (worklist)

| Field | Type | Notes |
|---|---|---|
| `orthanc_instance_id` | string | Primary key |
| `status` | `"ok"` \| `"error"` | `error` = the worker could not analyse the image |
| `error` | string \| null | Worker's error message (error rows only) |
| `image_index` | string \| null | NIH file name including extension, e.g. `00000003_000.png` |
| `patient_id` | string \| null | NIH patient id |
| `flagged` | string[] \| null | Labels with `logit >= threshold`; null for error rows |
| `processed_at` | time | When the worker analysed the image |
| `received_at` | time | When the API stored the result |
| `latest_decision` | object \| null | `{decision, comment, reviewer, decided_at}` of the newest decision |

`GET /api/results` → `200 {"count": N, "results": [summary, …]}`, ordered by `received_at`, then id. Not paginated
(the demo has 42 results).

### Detail object

`GET /api/results/{id}` → `200` with the summary fields above plus:

- `model`: `{"name": "densenet121", "checkpoint_sha256": "<64 hex>"}`
- `labels`: all 14 labels sorted by logit, highest first:
  `{"label": "Hernia", "logit": 10.1446, "threshold": -3.8065, "flagged": true, "heatmap_url": "/api/results/<id>/heatmaps/Hernia"}`.
  `heatmap_url` is null for labels without a heatmap.
- `decisions`: every decision for this result, oldest first, each `{decision, comment, reviewer, decided_at}`.

Error rows have `labels: []` and `decisions: []`. Unknown id → `404 {"error":"result not found"}`.

### Images

- `GET /api/results/{id}/heatmaps/{label}` → `200 image/png`, 224 × 224 8-bit grayscale (the viewer colours it),
  `Cache-Control: private, max-age=31536000, immutable` (results are write-once). Missing → `404 heatmap not found`.
- `GET /api/results/{id}/preview` → `200 image/png`, rendered by Orthanc, `Cache-Control: private, max-age=3600`.
  Errors: `404 result not found` · `404 image not found in orthanc` · `502 orthanc unreachable` ·
  `502 orthanc returned status N` · `502 orthanc returned content type …` · `503 preview not configured`.
  Both image endpoints send `X-Content-Type-Options: nosniff`.

### Storing a result (worker only)

`POST /api/results` with one JSON object. The body is limited to 10 MiB (`413` above that); unknown fields, trailing
data and invalid values are rejected with `400`.

| Field | Required | Notes |
|---|---|---|
| `orthanc_instance_id` | always | |
| `model` | always | `name` and a 64-hex `checkpoint_sha256` |
| `processed_at` | always | RFC 3339 time |
| `status` | always | `"ok"` or `"error"` |
| `error` | error only | Non-empty message; an error result must not carry logits, thresholds, flags or heatmaps |
| `meta`, `logits`, `thresholds`, `flagged`, `heatmap_size`, `heatmaps_png_b64` | ok only | See below |

Checks for an ok result: `logits` and `thresholds` have the same labels; `flagged` must equal exactly the labels with
`logit >= threshold` (may be empty); at least one heatmap; every flagged label has a heatmap and every heatmap label
is a logits label; each heatmap is a base64 PNG, 8-bit grayscale, matching `heatmap_size` `[height, width]`.

Responses: `{"orthanc_instance_id": "…", "outcome": "created" | "replaced" | "duplicate"}` with `201 Created`
for `created` and `200 OK` for the other two (the worker retries until it gets a 2xx, so repeats are normal); or `400`
with the specific reason, `413`, or `500 could not store result`.

| Stored | Incoming | Outcome | Effect |
|---|---|---|---|
| nothing | ok or error | `created` | stored; audit `result_created` |
| error | ok | `replaced` | error row replaced; audit `result_replaced` |
| ok | ok or error | `duplicate` | nothing stored; audit `result_duplicate` with `{incoming_status, stored_status}` |
| error | error | `duplicate` | the first error is kept |

So an ok result is **write-once**: a repeat is not compared with the stored one, it is simply not stored.

## Decisions

`POST /api/results/{id}/decisions` with exactly `{"decision": "approve" | "reject", "reviewer": "…", "comment": "…"}`.

→ `201 {"id", "orthanc_instance_id", "decision", "comment", "reviewer", "decided_at"}`. The decision and its audit row
are written in one transaction.

Errors (the body is checked before the id is looked up):

| Status | Message |
|---|---|
| 400 | `body must contain a single JSON object` (also unknown fields) |
| 400 | `decision must be approve or reject, got "…"` |
| 400 | `reviewer is required` · `reviewer name worker is reserved for the inference worker` · `reviewer is longer than 100 characters` |
| 400 | `a reject needs a comment` · `comment is longer than 2000 characters` |
| 413 | body larger than 8 KiB |
| 404 | `result not found` |
| 409 | `decision requires an ok result` (error rows cannot be reviewed) |

There is no idempotency key: two POSTs record two decisions (the viewer locks its form after a save).

## Audit log

Every stored result and every decision appends one row. Rows cannot be updated or deleted (database triggers).

| Field | Notes |
|---|---|
| `seq` | 1, 2, 3, … without gaps |
| `at` | time |
| `actor` | `worker` or the reviewer name |
| `action` | `result_created` · `result_replaced` · `result_duplicate` · `decision` |
| `orthanc_instance_id` | string \| null |
| `details` | Object, depends on the action, e.g. `result_created`: `{"checkpoint_sha256", "flagged", "heatmaps", "status"}` (counts); `decision`: `{"comment", "decision", "decision_id"}` |
| `prev_hash` | `hash` of the previous row (seq 1 links to a fixed genesis value) |
| `hash` | SHA-256 over this row's fields and `prev_hash` |

### Paging

`GET /api/audit`

| Parameter | Default | Notes |
|---|---|---|
| `order` | `asc` | `asc` (oldest first) or `desc` (newest first) |
| `after` | 0 | asc only: rows with `seq > after` |
| `before` | none | desc only: rows with `seq < before` |
| `limit` | 100 | 1 … 1000 |

- asc → `{"entries": […], "next_after": SEQ | null}`; desc → `{"entries": […], "next_before": SEQ | null}`.
  Pass the returned value as `after` / `before` for the next page; `null` means there are no more rows.
- Example: `?order=desc&limit=2&before=41` → seq `[40, 39]`, `next_before: 39`; `?order=desc&before=3&limit=5` →
  `[2, 1]`, `next_before: null`.
- `400` messages: `order must be "asc" or "desc"` · `before needs order=desc` · `after needs order=asc` ·
  `before must be a non-negative integer` (same for `after`) · `limit must be between 1 and 1000`.

### Verification

`GET /api/audit/verify` recomputes every hash from seq 1.

- Intact: `200 {"last_hash": "<64 hex>", "ok": true, "rows": N}`.
- Broken: `500 {"error": "…", "ok": false, "rows": N}`, where the error is `seq N: hash mismatch (row was changed)`
  or `seq N: expected M (gap or reorder)`.
- Removing rows from the **end** keeps the chain valid. To detect that, keep a copy of `last_hash` and `rows`
  elsewhere (the audit page shows them) and compare later.

## Examples

```bash
curl -s http://127.0.0.1:8080/api/results | python3 -m json.tool | head -30
curl -s 'http://127.0.0.1:8080/api/audit?order=desc&limit=5'
curl -s http://127.0.0.1:8080/api/audit/verify
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"decision":"reject","reviewer":"Ana","comment":"heat outside the lung"}' \
  http://127.0.0.1:8080/api/results/<orthanc_instance_id>/decisions
```
