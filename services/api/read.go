package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"sort"
	"strconv"
)

// decisionView is one reviewer decision as the API returns it.
type decisionView struct {
	Decision  string `json:"decision"`
	Comment   string `json:"comment"`
	Reviewer  string `json:"reviewer"`
	DecidedAt string `json:"decided_at"`
}

// resultSummary is one worklist entry. Pointer fields are null for error rows.
type resultSummary struct {
	ID             string        `json:"orthanc_instance_id"`
	Status         string        `json:"status"`
	Error          *string       `json:"error"`
	ImageIndex     *string       `json:"image_index"`
	PatientID      *string       `json:"patient_id"`
	ProcessedAt    string        `json:"processed_at"`
	ReceivedAt     string        `json:"received_at"`
	Flagged        []string      `json:"flagged"`
	LatestDecision *decisionView `json:"latest_decision"`
}

type modelView struct {
	Name             *string `json:"name"`
	CheckpointSHA256 *string `json:"checkpoint_sha256"`
}

type labelView struct {
	Label      string  `json:"label"`
	Logit      float64 `json:"logit"`
	Threshold  float64 `json:"threshold"`
	Flagged    bool    `json:"flagged"`
	HeatmapURL *string `json:"heatmap_url"`
}

// resultDetail is the full view of one result; resultSummary's fields are inlined.
type resultDetail struct {
	resultSummary
	Model     modelView      `json:"model"`
	Labels    []labelView    `json:"labels"`
	Decisions []decisionView `json:"decisions"`
}

// summarySelect joins each result to its latest decision: decisions are
// append-only, so the latest one has the highest id.
const summarySelect = `SELECT r.orthanc_instance_id, r.status, r.error, r.image_index, r.patient_id,
       r.processed_at, r.received_at, r.flagged_json,
       d.decision, d.comment, d.reviewer, d.created_at
FROM results r
LEFT JOIN decisions d ON d.id = (
    SELECT max(id) FROM decisions WHERE orthanc_instance_id = r.orthanc_instance_id)`

// rowScanner is satisfied by both *sql.Row and *sql.Rows.
type rowScanner interface {
	Scan(dest ...any) error
}

func scanSummary(row rowScanner) (*resultSummary, error) {
	var s resultSummary
	var flagged, decision, comment, reviewer, decidedAt sql.NullString
	err := row.Scan(&s.ID, &s.Status, &s.Error, &s.ImageIndex, &s.PatientID,
		&s.ProcessedAt, &s.ReceivedAt, &flagged,
		&decision, &comment, &reviewer, &decidedAt)
	if err != nil {
		return nil, err
	}
	if flagged.Valid {
		if err := json.Unmarshal([]byte(flagged.String), &s.Flagged); err != nil {
			return nil, err
		}
	}
	if decision.Valid {
		s.LatestDecision = &decisionView{
			Decision:  decision.String,
			Comment:   comment.String,
			Reviewer:  reviewer.String,
			DecidedAt: decidedAt.String,
		}
	}
	return &s, nil
}

func serverError(w http.ResponseWriter, what string, err error) {
	slog.Error(what, "err", err)
	writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "internal error"})
}

func handleListResults(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rows, err := db.QueryContext(r.Context(), summarySelect+"\nORDER BY r.received_at, r.orthanc_instance_id")
		if err != nil {
			serverError(w, "list results", err)
			return
		}
		defer rows.Close()
		list := []*resultSummary{}
		for rows.Next() {
			s, err := scanSummary(rows)
			if err != nil {
				serverError(w, "scan result", err)
				return
			}
			list = append(list, s)
		}
		if err := rows.Err(); err != nil {
			serverError(w, "list results", err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"count": len(list), "results": list})
	}
}

// loadDetail reads one result inside a read transaction, so the summary, labels
// and decision history come from one snapshot. Every query here uses tx: with one
// open connection, a query through db would wait forever for the one tx holds.
func loadDetail(ctx context.Context, db *sql.DB, id string) (*resultDetail, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	s, err := scanSummary(tx.QueryRowContext(ctx, summarySelect+"\nWHERE r.orthanc_instance_id = ?", id))
	if err != nil {
		return nil, err
	}
	d := &resultDetail{resultSummary: *s, Labels: []labelView{}, Decisions: []decisionView{}}

	var logits, thresholds sql.NullString
	err = tx.QueryRowContext(ctx,
		`SELECT model_name, checkpoint_sha256, logits_json, thresholds_json
		 FROM results WHERE orthanc_instance_id = ?`, id).
		Scan(&d.Model.Name, &d.Model.CheckpointSHA256, &logits, &thresholds)
	if err != nil {
		return nil, err
	}

	if logits.Valid {
		heatmaps, err := heatmapLabels(ctx, tx, id)
		if err != nil {
			return nil, err
		}
		d.Labels, err = buildLabels(id, logits.String, thresholds.String, d.Flagged, heatmaps)
		if err != nil {
			return nil, err
		}
	}
	d.Decisions, err = decisionsFor(ctx, tx, id)
	if err != nil {
		return nil, err
	}
	return d, nil
}

func heatmapLabels(ctx context.Context, tx *sql.Tx, id string) (map[string]bool, error) {
	rows, err := tx.QueryContext(ctx, `SELECT label FROM heatmaps WHERE orthanc_instance_id = ?`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	labels := map[string]bool{}
	for rows.Next() {
		var label string
		if err := rows.Scan(&label); err != nil {
			return nil, err
		}
		labels[label] = true
	}
	return labels, rows.Err()
}

func decisionsFor(ctx context.Context, tx *sql.Tx, id string) ([]decisionView, error) {
	rows, err := tx.QueryContext(ctx,
		`SELECT decision, comment, reviewer, created_at FROM decisions
		 WHERE orthanc_instance_id = ? ORDER BY id`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []decisionView{}
	for rows.Next() {
		var d decisionView
		if err := rows.Scan(&d.Decision, &d.Comment, &d.Reviewer, &d.DecidedAt); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// buildLabels returns one entry per label, sorted by logit (highest first).
func buildLabels(id, logitsJSON, thresholdsJSON string, flagged []string, heatmaps map[string]bool) ([]labelView, error) {
	var logits, thresholds map[string]float64
	if err := json.Unmarshal([]byte(logitsJSON), &logits); err != nil {
		return nil, err
	}
	if err := json.Unmarshal([]byte(thresholdsJSON), &thresholds); err != nil {
		return nil, err
	}
	isFlagged := make(map[string]bool, len(flagged))
	for _, l := range flagged {
		isFlagged[l] = true
	}
	labels := make([]labelView, 0, len(logits))
	for name, logit := range logits {
		lv := labelView{Label: name, Logit: logit, Threshold: thresholds[name], Flagged: isFlagged[name]}
		if heatmaps[name] {
			u := "/api/results/" + url.PathEscape(id) + "/heatmaps/" + url.PathEscape(name)
			lv.HeatmapURL = &u
		}
		labels = append(labels, lv)
	}
	sort.Slice(labels, func(i, j int) bool {
		if labels[i].Logit != labels[j].Logit {
			return labels[i].Logit > labels[j].Logit
		}
		return labels[i].Label < labels[j].Label
	})
	return labels, nil
}

func handleGetResult(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		d, err := loadDetail(r.Context(), db, r.PathValue("id"))
		if errors.Is(err, sql.ErrNoRows) {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "result not found"})
			return
		}
		if err != nil {
			serverError(w, "get result", err)
			return
		}
		writeJSON(w, http.StatusOK, d)
	}
}

func handleGetHeatmap(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var png []byte
		err := db.QueryRowContext(r.Context(),
			`SELECT png FROM heatmaps WHERE orthanc_instance_id = ? AND label = ?`,
			r.PathValue("id"), r.PathValue("label")).Scan(&png)
		if errors.Is(err, sql.ErrNoRows) {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "heatmap not found"})
			return
		}
		if err != nil {
			serverError(w, "get heatmap", err)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", strconv.Itoa(len(png)))
		w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Write(png)
	}
}
