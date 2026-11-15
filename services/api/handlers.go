// web api layer
// wires the SQLite pool into an HTTP router (ServeMux) and adds a helper to send JSON responses.
/*
net/http package codes:
http.NewServeMux 				- function
http.ServeMux 					- struct
http.ResponseWriter 			- Interface
http.Request 					- struct
http.StatusServiceUnavailable 	- int status code
http.StatusOK 					- int
*/
package main

import (
	"database/sql"
	"encoding/json"
	"log/slog"
	"net/http"
)

// ServeMux struct already exists in net/http library
func newMux(db *sql.DB, sqliteVer string) *http.ServeMux {
	// creating a new HTTP request router to map incoming HTTP routes to handler function.
	mux := http.NewServeMux()
	// registering this endpoint; http.Request is a pointer to Request struct, hence (*)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		//doing ping check to check if SQLite is still reachable
		if err := db.PingContext(r.Context()); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "db_unavailable", "error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "sqlite": sqliteVer, "schema_version": schemaVersion})
	})
	return mux
}

// http.ResponseWriter is an interface not a struct, hence no (*)
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("write response", "err", err)
	}
}
