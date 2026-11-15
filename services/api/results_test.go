package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func grayPNG(t *testing.T, w, h int) string {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewGray(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// validPayload is a small ok payload; Nodule sits exactly on its threshold (logit >= thr -> flagged).
func validPayload(id, heatmap string) map[string]any {
	return map[string]any{
		"orthanc_instance_id": id,
		"model":               map[string]any{"name": "densenet121", "checkpoint_sha256": strings.Repeat("ab", 32)},
		"processed_at":        "2026-11-14T05:12:01+00:00",
		"status":              "ok",
		"image_index":         "00000001_000.png",
		"patient_id":          "1",
		"study_uid":           "1.2.3",
		"series_uid":          "1.2.3.4",
		"sop_uid":             "1.2.3.4.5",
		"logits":              map[string]any{"Effusion": 1.5, "Hernia": -2.0, "Nodule": 0.25},
		"thresholds":          map[string]any{"Effusion": 0.1, "Hernia": 0.3, "Nodule": 0.25},
		"flagged":             []any{"Effusion", "Nodule"},
		"heatmap_size":        []any{224, 224},
		"heatmaps_png_b64":    map[string]any{"Effusion": heatmap, "Nodule": heatmap},
	}
}

func errorPayload(id string) map[string]any {
	return map[string]any{
		"orthanc_instance_id": id,
		"model":               map[string]any{"name": "densenet121", "checkpoint_sha256": strings.Repeat("ab", 32)},
		"processed_at":        "2026-11-14T05:12:01+00:00",
		"status":              "error",
		"error":               "ValueError: only 8-bit MONOCHROME2 is supported",
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func postResult(t *testing.T, h http.Handler, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/results", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func countRows(t *testing.T, db *sql.DB, table string) int {
	t.Helper()
	var n int
	if err := db.QueryRow("SELECT count(*) FROM " + table).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestPostResultOutcomes(t *testing.T) {
	db := newTestDB(t)
	mux := newMux(db, "test")
	heatmap := grayPNG(t, 224, 224)

	steps := []struct {
		name    string
		payload map[string]any
		code    int
		outcome string
	}{
		{"new ok A", validPayload("A", heatmap), http.StatusCreated, "created"},
		{"retry ok A", validPayload("A", heatmap), http.StatusOK, "duplicate"},
		{"new error B", errorPayload("B"), http.StatusCreated, "created"},
		{"retry error B", errorPayload("B"), http.StatusOK, "duplicate"},
		{"ok replaces error B", validPayload("B", heatmap), http.StatusOK, "replaced"},
		{"late error for ok A", errorPayload("A"), http.StatusOK, "duplicate"},
	}
	for _, s := range steps {
		rec := postResult(t, mux, mustJSON(t, s.payload))
		var resp struct {
			Outcome string `json:"outcome"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("%s: response %q: %v", s.name, rec.Body.String(), err)
		}
		if rec.Code != s.code || resp.Outcome != s.outcome {
			t.Fatalf("%s: got %d %q, want %d %q (body %s)", s.name, rec.Code, resp.Outcome, s.code, s.outcome, rec.Body.String())
		}
	}

	for table, want := range map[string]int{"results": 2, "heatmaps": 4, "decisions": 0, "audit_log": 6} {
		if got := countRows(t, db, table); got != want {
			t.Errorf("%s rows = %d, want %d", table, got, want)
		}
	}
	var processedAt, status string
	if err := db.QueryRow("SELECT processed_at, status FROM results WHERE orthanc_instance_id = 'B'").Scan(&processedAt, &status); err != nil {
		t.Fatal(err)
	}
	if processedAt != "2026-11-14T05:12:01.000Z" || status != "ok" {
		t.Errorf("B: processed_at %q, status %q; want 2026-11-14T05:12:01.000Z, ok", processedAt, status)
	}
	var actions string
	if err := db.QueryRow("SELECT group_concat(action, ',') FROM (SELECT action FROM audit_log ORDER BY seq)").Scan(&actions); err != nil {
		t.Fatal(err)
	}
	wantActions := "result_created,result_duplicate,result_created,result_duplicate,result_replaced,result_duplicate"
	if actions != wantActions {
		t.Errorf("audit actions = %s, want %s", actions, wantActions)
	}
	if n, err := verifyAudit(context.Background(), db); err != nil || n != 6 {
		t.Errorf("verifyAudit = %d, %v; want 6, nil", n, err)
	}
}

func TestPostResultRejects(t *testing.T) {
	db := newTestDB(t)
	mux := newMux(db, "test")
	heatmap := grayPNG(t, 224, 224)
	small := grayPNG(t, 100, 100)
	toError := func(m map[string]any) {
		clear(m)
		maps.Copy(m, errorPayload("E"))
	}
	sub := func(m map[string]any, key string) map[string]any { return m[key].(map[string]any) }

	cases := []struct {
		name    string
		mutate  func(m map[string]any)
		raw     string // if set, sent instead of the payload
		suffix  string // appended to the JSON body
		wantErr string
	}{
		{name: "missing instance id", mutate: func(m map[string]any) { delete(m, "orthanc_instance_id") }, wantErr: "orthanc_instance_id is required"},
		{name: "bad checkpoint sha", mutate: func(m map[string]any) { sub(m, "model")["checkpoint_sha256"] = "xyz" }, wantErr: "checkpoint_sha256"},
		{name: "bad processed_at", mutate: func(m map[string]any) { m["processed_at"] = "yesterday" }, wantErr: "processed_at"},
		{name: "unknown status", mutate: func(m map[string]any) { m["status"] = "maybe" }, wantErr: "status ="},
		{name: "ok with error message", mutate: func(m map[string]any) { m["error"] = "boom" }, wantErr: "must not carry an error"},
		{name: "ok without sop_uid", mutate: func(m map[string]any) { delete(m, "sop_uid") }, wantErr: "sop_uid is required"},
		{name: "threshold labels differ", mutate: func(m map[string]any) { delete(sub(m, "thresholds"), "Hernia") }, wantErr: "same labels"},
		{name: "flagged disagrees with logits", mutate: func(m map[string]any) { m["flagged"] = []any{"Effusion"} }, wantErr: "logit >= threshold"},
		{name: "flagged label without heatmap", mutate: func(m map[string]any) { delete(sub(m, "heatmaps_png_b64"), "Nodule") }, wantErr: "has no heatmap"},
		{name: "heatmap label not in logits", mutate: func(m map[string]any) { sub(m, "heatmaps_png_b64")["Unicorn"] = heatmap }, wantErr: "not a logits label"},
		{name: "heatmap not base64", mutate: func(m map[string]any) { sub(m, "heatmaps_png_b64")["Effusion"] = "%%%" }, wantErr: "base64"},
		{name: "heatmap not PNG", mutate: func(m map[string]any) {
			sub(m, "heatmaps_png_b64")["Effusion"] = base64.StdEncoding.EncodeToString([]byte("hello"))
		}, wantErr: "not a PNG"},
		{name: "heatmap wrong size", mutate: func(m map[string]any) { sub(m, "heatmaps_png_b64")["Effusion"] = small }, wantErr: "heatmap_size says"},
		{name: "heatmap_size wrong length", mutate: func(m map[string]any) { m["heatmap_size"] = []any{224} }, wantErr: "heatmap_size must be"},
		{name: "unknown field", mutate: func(m map[string]any) { m["extra"] = 1 }, wantErr: "unknown field"},
		{name: "string logit", mutate: func(m map[string]any) { sub(m, "logits")["Effusion"] = "1.5" }, wantErr: "cannot unmarshal"},
		{name: "error without message", mutate: func(m map[string]any) { toError(m); delete(m, "error") }, wantErr: "non-empty error message"},
		{name: "error with logits", mutate: func(m map[string]any) {
			toError(m)
			m["logits"] = map[string]any{"Effusion": 1.0}
		}, wantErr: "must not carry logits"},
		{name: "not JSON", raw: "{", wantErr: "decode JSON"},
		{name: "trailing data", suffix: "{}", wantErr: "trailing data"},
	}
	if len(cases) != 20 {
		t.Fatalf("len(cases) = %d, want 20", len(cases))
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := validPayload("R", heatmap)
			if c.mutate != nil {
				c.mutate(m)
			}
			body := append(mustJSON(t, m), c.suffix...)
			if c.raw != "" {
				body = []byte(c.raw)
			}
			rec := postResult(t, mux, body)
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), c.wantErr) {
				t.Fatalf("got %d %s, want 400 containing %q", rec.Code, rec.Body.String(), c.wantErr)
			}
		})
	}
	// Rejected requests must leave no trace.
	for _, table := range []string{"results", "heatmaps", "audit_log"} {
		if got := countRows(t, db, table); got != 0 {
			t.Errorf("%s rows = %d after rejects, want 0", table, got)
		}
	}
}

func TestPostResultTooLarge(t *testing.T) {
	db := newTestDB(t)
	// Spaces are valid JSON whitespace, so the decoder keeps reading until the size limit trips.
	rec := postResult(t, newMux(db, "test"), bytes.Repeat([]byte(" "), maxResultBytes+1))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("got %d %s, want 413", rec.Code, rec.Body.String())
	}
}

func TestAuditDetectsTampering(t *testing.T) {
	db := newTestDB(t)
	mux := newMux(db, "test")
	heatmap := grayPNG(t, 224, 224)
	for _, id := range []string{"A", "B"} {
		if rec := postResult(t, mux, mustJSON(t, validPayload(id, heatmap))); rec.Code != http.StatusCreated {
			t.Fatalf("POST %s: %d %s", id, rec.Code, rec.Body.String())
		}
	}
	ctx := context.Background()
	if n, err := verifyAudit(ctx, db); err != nil || n != 2 {
		t.Fatalf("before tampering: verifyAudit = %d, %v; want 2, nil", n, err)
	}
	// Simulate someone editing the database file directly: bypass the append-only trigger.
	if _, err := db.Exec("DROP TRIGGER audit_log_no_update"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("UPDATE audit_log SET actor = 'mallory' WHERE seq = 1"); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyAudit(ctx, db); err == nil || !strings.Contains(err.Error(), "seq 1: hash mismatch") {
		t.Fatalf("after tampering: err = %v, want seq 1 hash mismatch", err)
	}
}

// TestRealWorkerPayloads replays the payloads the Docker worker sent to the stub (real code path).
func TestRealWorkerPayloads(t *testing.T) {
	files, err := filepath.Glob("../../data/interim/stub_api_docker/*.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Skip("no stub payloads in data/interim/stub_api_docker (local scratch)")
	}
	db := newTestDB(t)
	mux := newMux(db, "test")
	for _, want := range []int{http.StatusCreated, http.StatusOK} { // second pass = worker retries
		for _, f := range files {
			body, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			if rec := postResult(t, mux, body); rec.Code != want {
				t.Fatalf("%s: got %d %s, want %d", filepath.Base(f), rec.Code, rec.Body.String(), want)
			}
		}
	}

	// Cross-check: every stored heatmap is byte-identical to what the worker sent.
	heatmaps := 0
	for _, f := range files {
		body, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var p struct {
			ID       string            `json:"orthanc_instance_id"`
			Heatmaps map[string]string `json:"heatmaps_png_b64"`
		}
		if err := json.Unmarshal(body, &p); err != nil {
			t.Fatal(err)
		}
		for label, b64 := range p.Heatmaps {
			raw, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				t.Fatal(err)
			}
			var stored []byte
			if err := db.QueryRow("SELECT png FROM heatmaps WHERE orthanc_instance_id = ? AND label = ?", p.ID, label).Scan(&stored); err != nil {
				t.Fatalf("%s %s: %v", p.ID, label, err)
			}
			if !bytes.Equal(stored, raw) {
				t.Errorf("%s %s: stored heatmap differs from payload", p.ID, label)
			}
			heatmaps++
		}
	}
	if got := countRows(t, db, "results"); got != len(files) {
		t.Errorf("results = %d, want %d", got, len(files))
	}
	if got := countRows(t, db, "heatmaps"); got != heatmaps {
		t.Errorf("heatmaps = %d, want %d", got, heatmaps)
	}
	n, err := verifyAudit(context.Background(), db)
	if err != nil || n != 2*len(files) {
		t.Errorf("verifyAudit = %d, %v; want %d, nil", n, err, 2*len(files))
	}
	t.Logf("%d payloads, %d heatmaps, %d audit rows, chain verified", len(files), heatmaps, n)
}
