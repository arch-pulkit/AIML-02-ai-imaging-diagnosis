-- SQLite schema for the results API. Applied at every startup (idempotent: IF NOT EXISTS).
-- Times are UTC ISO-8601 strings, e.g. "2026-11-15T00:00:00Z".
-- Rules live here, not only in Go: ok results are write-once; heatmaps, decisions and the
-- audit log are append-only. Admin reset = delete the database file.

CREATE TABLE IF NOT EXISTS results (
    orthanc_instance_id TEXT PRIMARY KEY,
    status              TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error               TEXT,
    model_name          TEXT,
    checkpoint_sha256   TEXT,
    processed_at        TEXT NOT NULL,          -- worker clock
    received_at         TEXT NOT NULL,          -- API clock
    image_index         TEXT,
    patient_id          TEXT,
    study_uid           TEXT,
    series_uid          TEXT,
    sop_uid             TEXT,
    logits_json         TEXT CHECK (logits_json IS NULL OR json_valid(logits_json)),
    thresholds_json     TEXT CHECK (thresholds_json IS NULL OR json_valid(thresholds_json)),
    flagged_json        TEXT CHECK (flagged_json IS NULL OR json_valid(flagged_json)),
    CHECK ((status = 'error') = (error IS NOT NULL)),
    -- IS (not =): json_type(NULL) = 'object' is NULL, and a NULL CHECK passes silently
    CHECK (status = 'error' OR (
        json_type(logits_json) IS 'object'
        AND json_type(thresholds_json) IS 'object'
        AND json_type(flagged_json) IS 'array'
        AND model_name IS NOT NULL
        AND checkpoint_sha256 IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS heatmaps (
    orthanc_instance_id TEXT NOT NULL REFERENCES results (orthanc_instance_id),
    label               TEXT NOT NULL,
    png                 BLOB NOT NULL CHECK (substr(png, 1, 8) = X'89504E470D0A1A0A'),
    PRIMARY KEY (orthanc_instance_id, label)
) STRICT;

CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY,
    orthanc_instance_id TEXT NOT NULL REFERENCES results (orthanc_instance_id),
    decision            TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
    comment             TEXT NOT NULL DEFAULT '',
    reviewer            TEXT NOT NULL CHECK (length(trim(reviewer)) > 0),
    created_at          TEXT NOT NULL,
    CHECK (decision = 'approve' OR length(trim(comment)) > 0)
) STRICT;

CREATE INDEX IF NOT EXISTS decisions_by_instance ON decisions (orthanc_instance_id, id);

-- Every new entry links to the previous hash, making silent edits easy to detect.
CREATE TABLE IF NOT EXISTS audit_log (
    seq                 INTEGER PRIMARY KEY,
    at                  TEXT NOT NULL,
    actor               TEXT NOT NULL,          -- 'worker' or reviewer name
    action              TEXT NOT NULL CHECK (action IN
                            ('result_created', 'result_replaced', 'result_duplicate', 'decision')),
    orthanc_instance_id TEXT,
    details_json        TEXT NOT NULL CHECK (json_valid(details_json)),
    prev_hash           TEXT NOT NULL CHECK (length(prev_hash) = 64),
    hash                TEXT NOT NULL UNIQUE CHECK (length(hash) = 64)
) STRICT;

CREATE INDEX IF NOT EXISTS audit_by_instance ON audit_log (orthanc_instance_id, seq);

-- results: an error row may be replaced by an ok row; an ok row never changes
CREATE TRIGGER IF NOT EXISTS results_ok_write_once BEFORE UPDATE ON results
WHEN OLD.status = 'ok'
BEGIN SELECT RAISE(ABORT, 'ok results are write-once'); END;

CREATE TRIGGER IF NOT EXISTS results_no_delete BEFORE DELETE ON results
BEGIN SELECT RAISE(ABORT, 'results cannot be deleted'); END;

-- heatmaps: only for ok results, append-only
CREATE TRIGGER IF NOT EXISTS heatmaps_require_ok BEFORE INSERT ON heatmaps
WHEN (SELECT status FROM results WHERE orthanc_instance_id = NEW.orthanc_instance_id) IS NOT 'ok'
BEGIN SELECT RAISE(ABORT, 'heatmaps require an ok result'); END;

CREATE TRIGGER IF NOT EXISTS heatmaps_no_update BEFORE UPDATE ON heatmaps
BEGIN SELECT RAISE(ABORT, 'heatmaps are append-only'); END;

CREATE TRIGGER IF NOT EXISTS heatmaps_no_delete BEFORE DELETE ON heatmaps
BEGIN SELECT RAISE(ABORT, 'heatmaps are append-only'); END;

-- decisions: only on ok results, append-only (current decision = latest row)
CREATE TRIGGER IF NOT EXISTS decisions_require_ok BEFORE INSERT ON decisions
WHEN (SELECT status FROM results WHERE orthanc_instance_id = NEW.orthanc_instance_id) IS NOT 'ok'
BEGIN SELECT RAISE(ABORT, 'decision requires an ok result'); END;

CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;

CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;

-- audit log: append-only
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

PRAGMA user_version = 1;