package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthz(t *testing.T) {
	db := newTestDB(t)
	version, err := sqliteVersion(db)
	if err != nil {
		t.Fatal(err)
	}
	mux := newMux(db, version, nil)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /healthz = %d, want 200", rec.Code)
	}
	var body struct {
		Status        string `json:"status"`
		SQLite        string `json:"sqlite"`
		SchemaVersion int    `json:"schema_version"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Status != "ok" || body.SQLite != version || body.SchemaVersion != schemaVersion {
		t.Fatalf("body = %+v", body)
	}

	// Negative controls: wrong method, unknown path.
	controls := []struct {
		method string
		path   string
		want   int
	}{
		{http.MethodPost, "/healthz", http.StatusMethodNotAllowed},
		{http.MethodGet, "/nope", http.StatusNotFound},
	}
	for _, c := range controls {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(c.method, c.path, nil))
		if rec.Code != c.want {
			t.Errorf("%s %s = %d, want %d", c.method, c.path, rec.Code, c.want)
		}
	}
}
