package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
)

const (
	defaultAuditLimit = 100
	maxAuditLimit     = 1000
)

// auditEntry is one audit_log row as the API returns it.
type auditEntry struct {
	Seq        int64           `json:"seq"`
	At         string          `json:"at"`
	Actor      string          `json:"actor"`
	Action     string          `json:"action"`
	InstanceID *string         `json:"orthanc_instance_id"`
	Details    json.RawMessage `json:"details"`
	PrevHash   string          `json:"prev_hash"`
	Hash       string          `json:"hash"`
}

// queryInt parses an optional non-negative integer query parameter.
func queryInt(r *http.Request, name string, def int64) (int64, error) {
	s := r.URL.Query().Get(name)
	if s == "" {
		return def, nil
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("%s must be a non-negative integer", name)
	}
	return n, nil
}

func handleListAudit(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		after, err := queryInt(r, "after", 0)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		limit, err := queryInt(r, "limit", defaultAuditLimit)
		if err == nil && (limit < 1 || limit > maxAuditLimit) {
			err = fmt.Errorf("limit must be between 1 and %d", maxAuditLimit)
		}
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		rows, err := db.QueryContext(r.Context(),
			`SELECT seq, at, actor, action, orthanc_instance_id, details_json, prev_hash, hash
			 FROM audit_log WHERE seq > ? ORDER BY seq LIMIT ?`, after, limit)
		if err != nil {
			serverError(w, "list audit", err)
			return
		}
		defer rows.Close()
		entries := []auditEntry{}
		for rows.Next() {
			var e auditEntry
			var details string
			if err := rows.Scan(&e.Seq, &e.At, &e.Actor, &e.Action, &e.InstanceID, &details, &e.PrevHash, &e.Hash); err != nil {
				serverError(w, "scan audit", err)
				return
			}
			e.Details = json.RawMessage(details)
			entries = append(entries, e)
		}
		if err := rows.Err(); err != nil {
			serverError(w, "list audit", err)
			return
		}
		resp := map[string]any{"entries": entries, "next_after": nil}
		if int64(len(entries)) == limit {
			resp["next_after"] = entries[len(entries)-1].Seq
		}
		writeJSON(w, http.StatusOK, resp)
	}
}

func handleVerifyAudit(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		n, err := verifyAudit(r.Context(), db)
		if err != nil {
			// A broken chain means the stored history cannot be trusted: answer 5xx so scripts fail loudly.
			slog.Error("audit verify failed", "rows", n, "err", err)
			writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "rows": n, "error": err.Error()})
			return
		}
		last := genesisHash
		if n > 0 {
			err := db.QueryRowContext(r.Context(), `SELECT hash FROM audit_log WHERE seq = ?`, n).Scan(&last)
			if err != nil {
				serverError(w, "audit last hash", err)
				return
			}
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "rows": n, "last_hash": last})
	}
}
