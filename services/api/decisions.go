package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	maxDecisionBytes = 8 << 10
	maxReviewerRunes = 100
	maxCommentRunes  = 2000
)

// decisionRequest is the body of POST /api/results/{id}/decisions.
type decisionRequest struct {
	Decision string `json:"decision"`
	Comment  string `json:"comment"`
	Reviewer string `json:"reviewer"`
}

type decisionResponse struct {
	ID         int64  `json:"id"`
	InstanceID string `json:"orthanc_instance_id"`
	decisionView
}

var (
	errResultNotFound = errors.New("result not found")
	errResultNotOK    = errors.New("decision requires an ok result")
)

// decodeDecision reads exactly one JSON object; unknown fields and trailing data are errors.
func decodeDecision(r io.Reader) (*decisionRequest, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var req decisionRequest
	if err := dec.Decode(&req); err != nil {
		return nil, err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("body must contain a single JSON object")
	}
	return &req, nil
}

// validate trims the text fields and applies the decisions table's rules,
// so clients get a clear 400 instead of a constraint error.
func (d *decisionRequest) validate() error {
	d.Reviewer = strings.TrimSpace(d.Reviewer)
	d.Comment = strings.TrimSpace(d.Comment)
	switch {
	case d.Decision != "approve" && d.Decision != "reject":
		return fmt.Errorf("decision must be approve or reject, got %q", d.Decision)
	case d.Reviewer == "":
		return errors.New("reviewer is required")
	case strings.EqualFold(d.Reviewer, "worker"):
		return errors.New("reviewer name worker is reserved for the inference worker")
	case utf8.RuneCountInString(d.Reviewer) > maxReviewerRunes:
		return fmt.Errorf("reviewer is longer than %d characters", maxReviewerRunes)
	case d.Decision == "reject" && d.Comment == "":
		return errors.New("a reject needs a comment")
	case utf8.RuneCountInString(d.Comment) > maxCommentRunes:
		return fmt.Errorf("comment is longer than %d characters", maxCommentRunes)
	}
	return nil
}

// storeDecision checks the result, then inserts the decision and its audit row in one transaction.
func storeDecision(ctx context.Context, db *sql.DB, id string, d *decisionRequest, now time.Time) (*decisionResponse, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	var status string
	err = tx.QueryRowContext(ctx, `SELECT status FROM results WHERE orthanc_instance_id = ?`, id).Scan(&status)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errResultNotFound
	}
	if err != nil {
		return nil, err
	}
	if status != "ok" {
		return nil, errResultNotOK
	}

	at := now.UTC().Format(timeFormat)
	res, err := tx.ExecContext(ctx,
		`INSERT INTO decisions (orthanc_instance_id, decision, comment, reviewer, created_at) VALUES (?, ?, ?, ?, ?)`,
		id, d.Decision, d.Comment, d.Reviewer, at)
	if err != nil {
		return nil, fmt.Errorf("insert decision: %w", err)
	}
	decisionID, err := res.LastInsertId()
	if err != nil {
		return nil, err
	}
	details := map[string]any{"decision_id": decisionID, "decision": d.Decision, "comment": d.Comment}
	if err := appendAudit(ctx, tx, at, d.Reviewer, "decision", id, details); err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return &decisionResponse{
		ID:           decisionID,
		InstanceID:   id,
		decisionView: decisionView{Decision: d.Decision, Comment: d.Comment, Reviewer: d.Reviewer, DecidedAt: at},
	}, nil
}

func handlePostDecision(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		d, err := decodeDecision(http.MaxBytesReader(w, r.Body, maxDecisionBytes))
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{"error": err.Error()})
				return
			}
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		if err := d.validate(); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		resp, err := storeDecision(r.Context(), db, r.PathValue("id"), d, time.Now())
		switch {
		case errors.Is(err, errResultNotFound):
			writeJSON(w, http.StatusNotFound, map[string]any{"error": err.Error()})
		case errors.Is(err, errResultNotOK):
			writeJSON(w, http.StatusConflict, map[string]any{"error": err.Error()})
		case err != nil:
			serverError(w, "store decision", err)
		default:
			slog.Info("decision", "instance", resp.InstanceID, "decision", resp.Decision, "reviewer", resp.Reviewer)
			writeJSON(w, http.StatusCreated, resp)
		}
	}
}
