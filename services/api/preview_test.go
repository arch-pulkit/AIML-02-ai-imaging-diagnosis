package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

var fakePreviewPNG = []byte("\x89PNG\r\n\x1a\nfake preview")

// fakeOrthanc serves the ok row's preview with the given content type, answers
// 404 for every other instance, and demands demo/demo basic auth.
func fakeOrthanc(t *testing.T, contentType string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, pass, ok := r.BasicAuth()
		if !ok || user != "demo" || pass != "demo" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.URL.Path != "/instances/"+readOKID+"/preview" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", contentType)
		w.Write(fakePreviewPNG)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func getPreview(t *testing.T, oc *orthancClient, id string) *httptest.ResponseRecorder {
	t.Helper()
	mux, _ := seedReadDB(t, oc)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/results/"+id+"/preview", nil))
	return rec
}

func TestPreviewProxy(t *testing.T) {
	rec := getPreview(t, newOrthancClient(fakeOrthanc(t, "image/png"), "demo", "demo"), readOKID)
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "image/png" {
		t.Fatalf("preview: %d %q, want 200 image/png; body %s", rec.Code, rec.Header().Get("Content-Type"), rec.Body.String())
	}
	if !bytes.Equal(rec.Body.Bytes(), fakePreviewPNG) {
		t.Fatalf("preview body %q, want the bytes Orthanc sent", rec.Body.Bytes())
	}
}

func TestPreviewErrors(t *testing.T) {
	okURL := fakeOrthanc(t, "image/png")
	htmlURL := fakeOrthanc(t, "text/html")
	down := httptest.NewServer(http.NotFoundHandler())
	downURL := down.URL
	down.Close()

	cases := []struct {
		name string
		oc   *orthancClient
		id   string
		code int
		want string
	}{
		{"unknown result", newOrthancClient(okURL, "demo", "demo"), "nope", 404, `{"error":"result not found"}`},
		{"not configured", nil, readOKID, 503, `{"error":"preview not configured"}`},
		{"missing in orthanc", newOrthancClient(okURL, "demo", "demo"), readErrID, 404, `{"error":"image not found in orthanc"}`},
		{"wrong password", newOrthancClient(okURL, "demo", "wrong"), readOKID, 502, `{"error":"orthanc returned status 401"}`},
		{"wrong content type", newOrthancClient(htmlURL, "demo", "demo"), readOKID, 502, `{"error":"orthanc returned content type text/html"}`},
		{"orthanc down", newOrthancClient(downURL, "demo", "demo"), readOKID, 502, `{"error":"orthanc unreachable"}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := getPreview(t, c.oc, c.id)
			if rec.Code != c.code || rec.Body.String() != c.want+"\n" {
				t.Fatalf("got %d %s, want %d %s", rec.Code, rec.Body.String(), c.code, c.want)
			}
		})
	}
}
