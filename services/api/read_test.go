package main

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"
)

// Ids chosen so that id order (ok first) is the opposite of arrival order (error first).
const (
	readOKID  = "11111111-11111111-11111111-11111111-11111111"
	readErrID = "22222222-22222222-22222222-22222222-22222222"
)

func getJSON(t *testing.T, h http.Handler, path string, wantStatus int, v any) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	if rec.Code != wantStatus {
		t.Fatalf("GET %s: status %d, want %d; body %s", path, rec.Code, wantStatus, rec.Body.String())
	}
	if err := json.Unmarshal(rec.Body.Bytes(), v); err != nil {
		t.Fatalf("GET %s: decode: %v; body %s", path, err, rec.Body.String())
	}
}

// seedReadDB stores an error row, then an ok row whose Hernia logit ties with
// Nodule, then two decisions on the ok row (approve, then reject).
// It returns the handler and the heatmap PNG bytes that were posted.
func seedReadDB(t *testing.T, oc *orthancClient) (http.Handler, []byte) {
	t.Helper()
	db := newTestDB(t)
	mux := newMux(db, "test", oc)
	if rec := postResult(t, mux, mustJSON(t, errorPayload(readErrID))); rec.Code != http.StatusCreated {
		t.Fatalf("post error row: %d %s", rec.Code, rec.Body.String())
	}
	time.Sleep(5 * time.Millisecond) // received_at has ms precision: make arrival order unambiguous
	heatmap := grayPNG(t, 224, 224)
	p := validPayload(readOKID, heatmap)
	p["logits"].(map[string]any)["Hernia"] = 0.25 // ties with Nodule, still below Hernia's 0.3 threshold
	if rec := postResult(t, mux, mustJSON(t, p)); rec.Code != http.StatusCreated {
		t.Fatalf("post ok row: %d %s", rec.Code, rec.Body.String())
	}
	_, err := db.Exec(`INSERT INTO decisions (orthanc_instance_id, decision, comment, reviewer, created_at) VALUES
		(?, 'approve', '', 'dr_a', '2026-11-16T08:00:00.000Z'),
		(?, 'reject', 'Nodule flag looks wrong', 'dr_b', '2026-11-16T08:05:00.000Z')`, readOKID, readOKID)
	if err != nil {
		t.Fatalf("insert decisions: %v", err)
	}
	png, err := base64.StdEncoding.DecodeString(heatmap)
	if err != nil {
		t.Fatal(err)
	}
	return mux, png
}

func TestListResultsEmpty(t *testing.T) {
	mux := newMux(newTestDB(t), "test", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/results", nil))
	if rec.Code != http.StatusOK || rec.Body.String() != "{\"count\":0,\"results\":[]}\n" {
		t.Fatalf("empty worklist: %d %q, want 200 with results []", rec.Code, rec.Body.String())
	}
}

func TestListResults(t *testing.T) {
	mux, _ := seedReadDB(t, nil)
	var got struct {
		Count   int             `json:"count"`
		Results []resultSummary `json:"results"`
	}
	getJSON(t, mux, "/api/results", http.StatusOK, &got)
	if got.Count != 2 || len(got.Results) != 2 {
		t.Fatalf("count %d, len %d, want 2", got.Count, len(got.Results))
	}
	e, o := got.Results[0], got.Results[1]
	if e.ID != readErrID || o.ID != readOKID {
		t.Fatalf("order %s, %s: want the error row first (it arrived first)", e.ID, o.ID)
	}
	if e.Status != "error" || e.Error == nil || e.ImageIndex != nil || e.Flagged != nil || e.LatestDecision != nil {
		t.Fatalf("error row: %+v, want error set, meta and flagged null, no decision", e)
	}
	if !slices.Equal(o.Flagged, []string{"Effusion", "Nodule"}) {
		t.Fatalf("flagged %v, want [Effusion Nodule]", o.Flagged)
	}
	if o.LatestDecision == nil || o.LatestDecision.Decision != "reject" || o.LatestDecision.Reviewer != "dr_b" {
		t.Fatalf("latest decision: %+v, want the reject by dr_b", o.LatestDecision)
	}
}

func TestGetResultDetail(t *testing.T) {
	mux, png := seedReadDB(t, nil)
	var d resultDetail
	getJSON(t, mux, "/api/results/"+readOKID, http.StatusOK, &d)

	var order []string
	for _, l := range d.Labels {
		order = append(order, l.Label)
	}
	if want := []string{"Effusion", "Hernia", "Nodule"}; !slices.Equal(order, want) {
		t.Fatalf("label order %v, want %v (logit desc, tie by name)", order, want)
	}
	hernia, nodule := d.Labels[1], d.Labels[2]
	if hernia.Flagged || hernia.HeatmapURL != nil {
		t.Fatalf("Hernia: %+v, want not flagged and no heatmap_url", hernia)
	}
	wantURL := "/api/results/" + readOKID + "/heatmaps/Nodule"
	if !nodule.Flagged || nodule.HeatmapURL == nil || *nodule.HeatmapURL != wantURL {
		t.Fatalf("Nodule: %+v, want flagged with heatmap_url %s", nodule, wantURL)
	}
	if len(d.Decisions) != 2 || d.Decisions[0].Decision != "approve" || d.Decisions[1].Decision != "reject" {
		t.Fatalf("decisions %+v, want approve then reject", d.Decisions)
	}
	if d.LatestDecision == nil || *d.LatestDecision != d.Decisions[1] {
		t.Fatalf("latest_decision %+v, want the last entry of decisions", d.LatestDecision)
	}
	if d.Model.Name == nil || *d.Model.Name != "densenet121" {
		t.Fatalf("model %+v, want densenet121", d.Model)
	}

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, wantURL, nil))
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "image/png" {
		t.Fatalf("heatmap: %d %q, want 200 image/png", rec.Code, rec.Header().Get("Content-Type"))
	}
	if !slices.Equal(rec.Body.Bytes(), png) {
		t.Fatalf("heatmap bytes differ from the posted PNG (%d vs %d bytes)", rec.Body.Len(), len(png))
	}
}

func TestGetResultErrorRow(t *testing.T) {
	mux, _ := seedReadDB(t, nil)
	var d resultDetail
	getJSON(t, mux, "/api/results/"+readErrID, http.StatusOK, &d)
	if d.Status != "error" || d.Error == nil || d.ImageIndex != nil {
		t.Fatalf("error row: %+v, want status error, error set, image_index null", d.resultSummary)
	}
	if d.Labels == nil || len(d.Labels) != 0 || d.Decisions == nil || len(d.Decisions) != 0 {
		t.Fatalf("labels %v, decisions %v: want both [] (not null)", d.Labels, d.Decisions)
	}
}

func TestReadNotFound(t *testing.T) {
	mux, _ := seedReadDB(t, nil)
	cases := []struct{ path, want string }{
		{"/api/results/nope", "result not found"},
		{"/api/results/" + readOKID + "/heatmaps/Hernia", "heatmap not found"}, // label exists, no heatmap stored
		{"/api/results/" + readErrID + "/heatmaps/Nodule", "heatmap not found"},
	}
	for _, c := range cases {
		var body map[string]string
		getJSON(t, mux, c.path, http.StatusNotFound, &body)
		if body["error"] != c.want {
			t.Fatalf("GET %s: error %q, want %q", c.path, body["error"], c.want)
		}
	}
}
