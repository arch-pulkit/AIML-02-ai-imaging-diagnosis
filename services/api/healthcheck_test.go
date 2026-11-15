package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCheckHealth(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ok.Close()
	if err := checkHealth(ok.URL); err != nil {
		t.Fatalf("healthy server: want nil, got %v", err)
	}

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer bad.Close()
	err := checkHealth(bad.URL)
	if err == nil || !strings.Contains(err.Error(), "status 503") {
		t.Fatalf("503 server: want error containing status 503, got %v", err)
	}

	gone := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	goneURL := gone.URL
	gone.Close()
	err = checkHealth(goneURL)
	if err == nil || !strings.Contains(err.Error(), "connection refused") {
		t.Fatalf("closed server: want connection refused, got %v", err)
	}

	if code := runHealthcheck(nil); code != 2 {
		t.Fatalf("no URL: want exit 2, got %d", code)
	}
}
