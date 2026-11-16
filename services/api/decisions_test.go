package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// seedDecisionDB stores an error row and an ok row through the real POST /api/results path.
func seedDecisionDB(t *testing.T) (*sql.DB, http.Handler) {
	t.Helper()
	db := newTestDB(t)
	mux := newMux(db, "test", nil)
	for _, p := range []map[string]any{errorPayload(readErrID), validPayload(readOKID, grayPNG(t, 224, 224))} {
		if rec := postResult(t, mux, mustJSON(t, p)); rec.Code != http.StatusCreated {
			t.Fatalf("seed: %d %s", rec.Code, rec.Body.String())
		}
	}
	return db, mux
}

func postDecision(t *testing.T, h http.Handler, id, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/results/"+id+"/decisions", strings.NewReader(body)))
	return rec
}

func TestPostDecision(t *testing.T) {
	db, mux := seedDecisionDB(t)

	rec := postDecision(t, mux, readOKID, `{"decision":"approve","reviewer":"  dr_a  "}`)
	if rec.Code != http.StatusCreated {
		t.Fatalf("approve: %d %s", rec.Code, rec.Body.String())
	}
	var got decisionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ID != 1 || got.InstanceID != readOKID || got.Reviewer != "dr_a" || got.Decision != "approve" {
		t.Fatalf("response %+v, want id 1, approve, reviewer trimmed to dr_a", got)
	}
	if _, err := time.Parse(timeFormat, got.DecidedAt); err != nil || !strings.HasSuffix(got.DecidedAt, "Z") {
		t.Fatalf("decided_at %q: want UTC in timeFormat (%v)", got.DecidedAt, err)
	}

	rec = postDecision(t, mux, readOKID, `{"decision":"reject","comment":"Nodule flag looks wrong","reviewer":"dr_b"}`)
	if rec.Code != http.StatusCreated {
		t.Fatalf("reject: %d %s", rec.Code, rec.Body.String())
	}

	var d resultDetail
	getJSON(t, mux, "/api/results/"+readOKID, http.StatusOK, &d)
	if len(d.Decisions) != 2 || d.LatestDecision == nil || d.LatestDecision.Reviewer != "dr_b" {
		t.Fatalf("decisions %+v, latest %+v: want 2, dr_b's reject last", d.Decisions, d.LatestDecision)
	}

	var actor, action, details string
	err := db.QueryRow(`SELECT actor, action, details_json FROM audit_log ORDER BY seq DESC LIMIT 1`).Scan(&actor, &action, &details)
	if err != nil {
		t.Fatal(err)
	}
	wantDetails := `{"comment":"Nodule flag looks wrong","decision":"reject","decision_id":2}`
	if actor != "dr_b" || action != "decision" || details != wantDetails {
		t.Fatalf("last audit row: %s %s %s", actor, action, details)
	}
	if n, err := verifyAudit(context.Background(), db); n != 4 || err != nil {
		t.Fatalf("verifyAudit: %d rows, %v; want 4 rows (2 results + 2 decisions), no error", n, err)
	}
}

func TestPostDecisionRejects(t *testing.T) {
	db, mux := seedDecisionDB(t)
	cases := []struct {
		name, id, body string
		code           int
		want           string
	}{
		{"unknown result", "nope", `{"decision":"approve","reviewer":"dr_a"}`, 404, "result not found"},
		{"error row", readErrID, `{"decision":"approve","reviewer":"dr_a"}`, 409, "decision requires an ok result"},
		{"bad decision", readOKID, `{"decision":"maybe","reviewer":"dr_a"}`, 400, "decision must be approve or reject"},
		{"no reviewer", readOKID, `{"decision":"approve"}`, 400, "reviewer is required"},
		{"blank reviewer", readOKID, `{"decision":"approve","reviewer":"   "}`, 400, "reviewer is required"},
		{"reserved reviewer", readOKID, `{"decision":"approve","reviewer":"Worker"}`, 400, "reserved"},
		{"long reviewer", readOKID, `{"decision":"approve","reviewer":"` + strings.Repeat("x", 101) + `"}`, 400, "longer than 100"},
		{"reject without comment", readOKID, `{"decision":"reject","reviewer":"dr_a"}`, 400, "a reject needs a comment"},
		{"reject blank comment", readOKID, `{"decision":"reject","comment":"  ","reviewer":"dr_a"}`, 400, "a reject needs a comment"},
		{"unknown field", readOKID, `{"decision":"approve","reviewer":"dr_a","approved":true}`, 400, `unknown field "approved"`},
		{"trailing data", readOKID, `{"decision":"approve","reviewer":"dr_a"} {}`, 400, "single JSON object"},
		{"not json", readOKID, `approve`, 400, "invalid character"},
		{"too large", readOKID, `{"decision":"approve","reviewer":"` + strings.Repeat("x", maxDecisionBytes) + `"}`, 413, "request body too large"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := postDecision(t, mux, c.id, c.body)
			var body map[string]string
			_ = json.Unmarshal(rec.Body.Bytes(), &body)
			if rec.Code != c.code || !strings.Contains(body["error"], c.want) {
				t.Fatalf("got %d %s, want %d containing %q", rec.Code, rec.Body.String(), c.code, c.want)
			}
		})
	}
	if n := countRows(t, db, "decisions"); n != 0 {
		t.Fatalf("decisions: %d rows after rejected requests, want 0", n)
	}
	if n := countRows(t, db, "audit_log"); n != 2 {
		t.Fatalf("audit_log: %d rows, want only the 2 result rows", n)
	}
}

func TestAuditEndpoints(t *testing.T) {
	db, mux := seedDecisionDB(t)
	if rec := postDecision(t, mux, readOKID, `{"decision":"approve","reviewer":"dr_a"}`); rec.Code != http.StatusCreated {
		t.Fatalf("approve: %d %s", rec.Code, rec.Body.String())
	}

	type page struct {
		Entries   []auditEntry `json:"entries"`
		NextAfter *int64       `json:"next_after"`
	}
	var p1, p2 page
	getJSON(t, mux, "/api/audit?limit=2", http.StatusOK, &p1)
	if len(p1.Entries) != 2 || p1.Entries[0].Seq != 1 || p1.NextAfter == nil || *p1.NextAfter != 2 {
		t.Fatalf("page 1: %d entries, next_after %v; want seq 1-2 and next_after 2", len(p1.Entries), p1.NextAfter)
	}
	getJSON(t, mux, "/api/audit?after=2&limit=2", http.StatusOK, &p2)
	if len(p2.Entries) != 1 || p2.Entries[0].Action != "decision" || p2.Entries[0].Actor != "dr_a" || p2.NextAfter != nil {
		t.Fatalf("page 2: %+v; want only dr_a's decision and next_after null", p2)
	}
	if p2.Entries[0].PrevHash != p1.Entries[1].Hash {
		t.Fatalf("seq 3 prev_hash %s, want seq 2's hash %s", p2.Entries[0].PrevHash, p1.Entries[1].Hash)
	}

	for _, q := range []string{"limit=0", "limit=1001", "limit=abc", "after=-1"} {
		var body map[string]string
		getJSON(t, mux, "/api/audit?"+q, http.StatusBadRequest, &body)
		if body["error"] == "" {
			t.Fatalf("?%s: empty error message", q)
		}
	}

	type verifyResp struct {
		OK       bool   `json:"ok"`
		Rows     int    `json:"rows"`
		LastHash string `json:"last_hash"`
		Error    string `json:"error"`
	}
	var good verifyResp
	getJSON(t, mux, "/api/audit/verify", http.StatusOK, &good)
	if !good.OK || good.Rows != 3 || good.LastHash != p2.Entries[0].Hash {
		t.Fatalf("verify: %+v, want ok, 3 rows, last_hash of seq 3", good)
	}

	// A forged row with a made-up hash: the triggers allow INSERT, the chain must not.
	_, err := db.Exec(`INSERT INTO audit_log (seq, at, actor, action, orthanc_instance_id, details_json, prev_hash, hash)
		VALUES (4, '2026-11-16T09:00:00.000Z', 'mallory', 'decision', ?, '{}', ?, ?)`,
		readOKID, good.LastHash, strings.Repeat("f", 64))
	if err != nil {
		t.Fatalf("insert forged row: %v", err)
	}
	var bad verifyResp
	getJSON(t, mux, "/api/audit/verify", http.StatusInternalServerError, &bad)
	if bad.OK || !strings.Contains(bad.Error, "seq 4: hash mismatch") {
		t.Fatalf("verify after forgery: %+v, want ok false with seq 4: hash mismatch", bad)
	}
}
