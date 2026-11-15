package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"image/png"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"regexp"
	"slices"
	"time"
)

// maxResultBytes caps a POST body; a real payload is about 40 KB.
const maxResultBytes = 10 << 20

// timeFormat is used for every timestamp the API stores: UTC, millisecond precision, "Z" suffix.
const timeFormat = "2006-01-02T15:04:05.000Z07:00"

var sha256Hex = regexp.MustCompile(`^[0-9a-f]{64}$`)

type modelInfo struct {
	Name             string `json:"name"`
	CheckpointSHA256 string `json:"checkpoint_sha256"`
}

// resultPayload mirrors the worker's POST body (src/serve/worker.py build_payload).
type resultPayload struct {
	OrthancInstanceID string             `json:"orthanc_instance_id"`
	Model             *modelInfo         `json:"model"`
	ProcessedAt       string             `json:"processed_at"`
	Status            string             `json:"status"`
	Error             *string            `json:"error"`
	ImageIndex        *string            `json:"image_index"`
	PatientID         *string            `json:"patient_id"`
	StudyUID          *string            `json:"study_uid"`
	SeriesUID         *string            `json:"series_uid"`
	SOPUID            *string            `json:"sop_uid"`
	Logits            map[string]float64 `json:"logits"`
	Thresholds        map[string]float64 `json:"thresholds"`
	Flagged           []string           `json:"flagged"`
	HeatmapSize       []int              `json:"heatmap_size"`
	HeatmapsPNGB64    map[string]string  `json:"heatmaps_png_b64"`
}

// validResult is a payload that passed validation, in the form the database stores.
type validResult struct {
	p           *resultPayload
	processedAt string
	heatmaps    map[string][]byte
}

type namedText struct {
	name  string
	value *string
}

func sortedKeys[V any](m map[string]V) []string {
	return slices.Sorted(maps.Keys(m))
}

func textOrNil(s *string) any {
	if s == nil {
		return nil
	}
	return *s
}

func decodeResult(r io.Reader) (*resultPayload, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var p resultPayload
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("trailing data after the JSON object")
	}
	return &p, nil
}

func validateResult(p *resultPayload) (*validResult, error) {
	if p.OrthancInstanceID == "" {
		return nil, errors.New("orthanc_instance_id is required")
	}
	if p.Model == nil || p.Model.Name == "" || !sha256Hex.MatchString(p.Model.CheckpointSHA256) {
		return nil, errors.New("model.name and a 64-hex model.checkpoint_sha256 are required")
	}
	t, err := time.Parse(time.RFC3339, p.ProcessedAt)
	if err != nil {
		return nil, fmt.Errorf("processed_at: %w", err)
	}
	v := &validResult{p: p, processedAt: t.UTC().Format(timeFormat)}

	switch p.Status {
	case "error":
		if p.Error == nil || *p.Error == "" {
			return nil, errors.New("status error needs a non-empty error message")
		}
		if p.Logits != nil || p.Thresholds != nil || p.Flagged != nil || p.HeatmapsPNGB64 != nil {
			return nil, errors.New("status error must not carry logits, thresholds, flagged or heatmaps")
		}
		return v, nil
	case "ok":
	default:
		return nil, fmt.Errorf("status = %q, want \"ok\" or \"error\"", p.Status)
	}

	if p.Error != nil {
		return nil, errors.New("status ok must not carry an error")
	}
	for _, f := range []namedText{
		{"image_index", p.ImageIndex}, {"patient_id", p.PatientID},
		{"study_uid", p.StudyUID}, {"series_uid", p.SeriesUID}, {"sop_uid", p.SOPUID},
	} {
		if f.value == nil || *f.value == "" {
			return nil, fmt.Errorf("%s is required for status ok", f.name)
		}
	}
	if len(p.Logits) == 0 {
		return nil, errors.New("logits are required for status ok")
	}
	if !slices.Equal(sortedKeys(p.Logits), sortedKeys(p.Thresholds)) {
		return nil, errors.New("logits and thresholds must have the same labels")
	}
	if p.Flagged == nil {
		return nil, errors.New("flagged is required for status ok (may be empty)")
	}
	// Cross-check: flagged must be exactly the labels with logit >= threshold (the worker's rule).
	var want []string
	for _, label := range sortedKeys(p.Logits) {
		if p.Logits[label] >= p.Thresholds[label] {
			want = append(want, label)
		}
	}
	got := slices.Sorted(slices.Values(p.Flagged))
	if !slices.Equal(got, want) {
		return nil, fmt.Errorf("flagged = %v, but logit >= threshold gives %v", got, want)
	}

	if len(p.HeatmapSize) != 2 || p.HeatmapSize[0] <= 0 || p.HeatmapSize[1] <= 0 {
		return nil, errors.New("heatmap_size must be [height, width] with positive values")
	}
	if len(p.HeatmapsPNGB64) == 0 {
		return nil, errors.New("status ok needs at least one heatmap")
	}
	for _, label := range p.Flagged {
		if _, ok := p.HeatmapsPNGB64[label]; !ok {
			return nil, fmt.Errorf("flagged label %q has no heatmap", label)
		}
	}
	v.heatmaps = make(map[string][]byte, len(p.HeatmapsPNGB64))
	for label, b64 := range p.HeatmapsPNGB64 {
		if _, ok := p.Logits[label]; !ok {
			return nil, fmt.Errorf("heatmap label %q is not a logits label", label)
		}
		raw, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			return nil, fmt.Errorf("heatmap %q: base64: %w", label, err)
		}
		cfg, err := png.DecodeConfig(bytes.NewReader(raw))
		if err != nil {
			return nil, fmt.Errorf("heatmap %q: not a PNG: %w", label, err)
		}
		if cfg.Height != p.HeatmapSize[0] || cfg.Width != p.HeatmapSize[1] {
			return nil, fmt.Errorf("heatmap %q is %dx%d (w x h), heatmap_size says %v (h, w)",
				label, cfg.Width, cfg.Height, p.HeatmapSize)
		}
		if cfg.ColorModel != color.GrayModel {
			return nil, fmt.Errorf("heatmap %q is not 8-bit grayscale", label)
		}
		v.heatmaps[label] = raw
	}
	return v, nil
}

const insertResultSQL = `INSERT INTO results (orthanc_instance_id, status, error, model_name, checkpoint_sha256,
	processed_at, received_at, image_index, patient_id, study_uid, series_uid, sop_uid,
	logits_json, thresholds_json, flagged_json)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

// replaceResultSQL turns an error row into an ok row (the only update the write-once trigger allows).
// Same columns as insertResultSQL minus the id, which moves to the WHERE clause.
const replaceResultSQL = `UPDATE results SET status = ?, error = ?, model_name = ?, checkpoint_sha256 = ?,
	processed_at = ?, received_at = ?, image_index = ?, patient_id = ?, study_uid = ?, series_uid = ?,
	sop_uid = ?, logits_json = ?, thresholds_json = ?, flagged_json = ?
	WHERE orthanc_instance_id = ?`

// resultArgs returns the values in insertResultSQL column order (id first).
func resultArgs(v *validResult, receivedAt string) ([]any, error) {
	p := v.p
	args := []any{
		p.OrthancInstanceID, p.Status, textOrNil(p.Error), p.Model.Name, p.Model.CheckpointSHA256,
		v.processedAt, receivedAt, textOrNil(p.ImageIndex), textOrNil(p.PatientID),
		textOrNil(p.StudyUID), textOrNil(p.SeriesUID), textOrNil(p.SOPUID), nil, nil, nil,
	}
	if p.Status == "ok" {
		for i, x := range []any{p.Logits, p.Thresholds, p.Flagged} {
			b, err := json.Marshal(x)
			if err != nil {
				return nil, fmt.Errorf("encode result JSON: %w", err)
			}
			args[12+i] = string(b)
		}
	}
	return args, nil
}

// ingestResult stores a validated result idempotently: "created", "replaced" (error -> ok) or "duplicate".
func ingestResult(ctx context.Context, db *sql.DB, v *validResult, now time.Time) (string, error) {
	p := v.p
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return "", fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback() // no-op after Commit. Inside the tx use only tx: one connection.

	var stored string
	err = tx.QueryRowContext(ctx, "SELECT status FROM results WHERE orthanc_instance_id = ?",
		p.OrthancInstanceID).Scan(&stored)
	var outcome string
	switch {
	case errors.Is(err, sql.ErrNoRows):
		outcome = "created"
	case err != nil:
		return "", fmt.Errorf("read existing result: %w", err)
	case stored == "error" && p.Status == "ok":
		outcome = "replaced"
	default:
		outcome = "duplicate"
	}

	at := now.UTC().Format(timeFormat)
	details := map[string]any{"incoming_status": p.Status, "stored_status": stored}
	if outcome != "duplicate" {
		args, err := resultArgs(v, at)
		if err != nil {
			return "", err
		}
		query := insertResultSQL
		if outcome == "replaced" {
			query, args = replaceResultSQL, slices.Concat(args[1:], args[:1])
		}
		if _, err := tx.ExecContext(ctx, query, args...); err != nil {
			return "", fmt.Errorf("%s result: %w", outcome, err)
		}
		for _, label := range sortedKeys(v.heatmaps) {
			if _, err := tx.ExecContext(ctx, "INSERT INTO heatmaps (orthanc_instance_id, label, png) VALUES (?, ?, ?)",
				p.OrthancInstanceID, label, v.heatmaps[label]); err != nil {
				return "", fmt.Errorf("insert heatmap %q: %w", label, err)
			}
		}
		details = map[string]any{"status": p.Status, "checkpoint_sha256": p.Model.CheckpointSHA256,
			"flagged": len(p.Flagged), "heatmaps": len(v.heatmaps)}
	}
	if err := appendAudit(ctx, tx, at, "worker", "result_"+outcome, p.OrthancInstanceID, details); err != nil {
		return "", err
	}
	if err := tx.Commit(); err != nil {
		return "", fmt.Errorf("commit: %w", err)
	}
	return outcome, nil
}

// handlePostResult: 201 created; 200 replaced or duplicate (worker retries must succeed);
// 400 invalid; 413 too large; 500 storage error (the worker treats it as transient and retries).
func handlePostResult(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxResultBytes)
		p, err := decodeResult(r.Body)
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{"error": err.Error()})
				return
			}
			slog.Warn("result rejected", "err", err)
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		v, err := validateResult(p)
		if err != nil {
			slog.Warn("result rejected", "instance", p.OrthancInstanceID, "err", err)
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		outcome, err := ingestResult(r.Context(), db, v, time.Now())
		if err != nil {
			slog.Error("result not stored", "instance", p.OrthancInstanceID, "err", err)
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "could not store result"})
			return
		}
		slog.Info("result", "instance", p.OrthancInstanceID, "status", p.Status, "outcome", outcome)
		code := http.StatusOK
		if outcome == "created" {
			code = http.StatusCreated
		}
		writeJSON(w, code, map[string]any{"orthanc_instance_id": p.OrthancInstanceID, "outcome": outcome})
	}
}
