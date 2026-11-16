package main

import (
	"database/sql"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// orthancClient reads from Orthanc's REST API. The credentials stay in the API process; the browser only ever calls /api/results/{id}/preview.
type orthancClient struct {
	baseURL  string
	user     string
	password string
	http     *http.Client
}

func newOrthancClient(baseURL, user, password string) *orthancClient {
	return &orthancClient{
		baseURL:  strings.TrimRight(baseURL, "/"),
		user:     user,
		password: password,
		http:     &http.Client{Timeout: 10 * time.Second},
	}
}

func handlePreview(db *sql.DB, oc *orthancClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		// Only ids we hold results for are forwarded: the API is not an open proxy into Orthanc.
		var one int
		err := db.QueryRowContext(r.Context(), `SELECT 1 FROM results WHERE orthanc_instance_id = ?`, id).Scan(&one)
		if errors.Is(err, sql.ErrNoRows) {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "result not found"})
			return
		}
		if err != nil {
			serverError(w, "preview lookup", err)
			return
		}
		if oc == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "preview not configured"})
			return
		}

		req, err := http.NewRequestWithContext(r.Context(), http.MethodGet,
			oc.baseURL+"/instances/"+url.PathEscape(id)+"/preview", nil)
		if err != nil {
			serverError(w, "preview request", err)
			return
		}
		if oc.user != "" {
			req.SetBasicAuth(oc.user, oc.password)
		}
		resp, err := oc.http.Do(req)
		if err != nil {
			slog.Error("orthanc preview", "instance", id, "err", err)
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": "orthanc unreachable"})
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusNotFound {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "image not found in orthanc"})
			return
		}
		if resp.StatusCode != http.StatusOK {
			slog.Error("orthanc preview", "instance", id, "status", resp.StatusCode)
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": fmt.Sprintf("orthanc returned status %d", resp.StatusCode)})
			return
		}
		if ct := resp.Header.Get("Content-Type"); ct != "image/png" {
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": "orthanc returned content type " + ct})
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "private, max-age=3600")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if _, err := io.Copy(w, resp.Body); err != nil {
			slog.Error("copy preview", "instance", id, "err", err)
		}
	}
}
