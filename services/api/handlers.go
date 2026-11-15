package main

import (
	"database/sql"
	"encoding/json"
	"log/slog"
	"net/http"
)

func newMux(db *sql.DB, sqliteVer string) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := db.PingContext(r.Context()); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "db_unavailable", "error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "sqlite": sqliteVer, "schema_version": schemaVersion})
	})
	mux.HandleFunc("POST /api/results", handlePostResult(db))
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false) // a JSON API, not HTML: keep <, > and & readable in messages
	if err := enc.Encode(v); err != nil {
		slog.Error("write response", "err", err)
	}
}
