package main

import (
	"database/sql"
	"maps"
	"path/filepath"
	"strings"
	"testing"
)

func newTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := openDB(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("openDB: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestSchemaIsIdempotent(t *testing.T) {
	db := newTestDB(t) // first apply happens in openDB
	if _, err := db.Exec(schemaSQL); err != nil {
		t.Fatalf("second apply: %v", err)
	}
	rows, err := db.Query("SELECT type, count(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' GROUP BY type")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	got := map[string]int{}
	for rows.Next() {
		var typ string
		var n int
		if err := rows.Scan(&typ, &n); err != nil {
			t.Fatal(err)
		}
		got[typ] = n
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"index": 2, "table": 4, "trigger": 10}
	if !maps.Equal(got, want) {
		t.Fatalf("objects = %v, want %v", got, want)
	}
}

func TestCheckDBNeedsForeignKeys(t *testing.T) {
	// Negative control: without the DSN pragmas, checkDB must refuse the database.
	db, err := sql.Open("sqlite", "file:"+filepath.Join(t.TempDir(), "plain.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	err = checkDB(db)
	if err == nil || !strings.Contains(err.Error(), "foreign_keys = 0, want 1") {
		t.Fatalf("checkDB without pragmas: err = %v, want a foreign_keys error", err)
	}
}

func TestSchemaRules(t *testing.T) {
	db := newTestDB(t)
	const now = "2026-11-15T00:00:00Z"
	const check = "CHECK constraint failed"
	png := []byte("\x89PNG\r\n\x1a\nrest")
	h0 := strings.Repeat("0", 64)
	insOK := "INSERT INTO results (orthanc_instance_id, status, model_name, checkpoint_sha256, processed_at, " +
		"received_at, logits_json, thresholds_json, flagged_json) " +
		"VALUES (?, 'ok', 'densenet121', 'abc', ?, ?, ?, ?, ?)"
	insErr := "INSERT INTO results (orthanc_instance_id, status, error, processed_at, received_at) " +
		"VALUES (?, 'error', ?, ?, ?)"
	insDec := "INSERT INTO decisions (orthanc_instance_id, decision, comment, reviewer, created_at) " +
		"VALUES (?, ?, ?, 'dr_demo', ?)"
	insAud := "INSERT INTO audit_log (at, actor, action, orthanc_instance_id, details_json, prev_hash, hash) " +
		"VALUES (?, 'worker', ?, 'A', '{}', ?, ?)"

	// Cases share one database and run in order. wantErr "" means the statement must succeed.
	cases := []struct {
		name    string
		sql     string
		args    []any
		wantErr string
		wantN   int64
	}{
		{"ok result A", insOK, []any{"A", now, now, `{"x": 1.0}`, `{"x": 0.0}`, `["x"]`}, "", 1},
		{"duplicate plain INSERT", insOK, []any{"A", now, now, "{}", "{}", "[]"}, "UNIQUE constraint failed", 0},
		{"duplicate ON CONFLICT DO NOTHING", insOK + " ON CONFLICT (orthanc_instance_id) DO NOTHING",
			[]any{"A", now, now, "{}", "{}", "[]"}, "", 0},
		{"update ok A", "UPDATE results SET flagged_json = '[]' WHERE orthanc_instance_id = 'A'", nil,
			"ok results are write-once", 0},
		{"delete A", "DELETE FROM results WHERE orthanc_instance_id = 'A'", nil, "results cannot be deleted", 0},
		{"ok without logits", "INSERT INTO results (orthanc_instance_id, status, model_name, checkpoint_sha256, " +
			"processed_at, received_at) VALUES ('N', 'ok', 'm', 'abc', ?, ?)", []any{now, now}, check, 0},
		{"malformed JSON", insOK, []any{"M", now, now, "{bad", "{}", "[]"}, check, 0},
		{"logits array not object", insOK, []any{"M", now, now, "[1]", "{}", "[]"}, check, 0},
		{"error result B", insErr, []any{"B", "boom", now, now}, "", 1},
		{"error without message", insErr, []any{"E", nil, now, now}, check, 0},
		{"heatmap on error B", "INSERT INTO heatmaps VALUES ('B', 'x', ?)", []any{png},
			"heatmaps require an ok result", 0},
		{"replace error B -> ok", "UPDATE results SET status = 'ok', error = NULL, model_name = 'densenet121', " +
			"checkpoint_sha256 = 'abc', logits_json = '{}', thresholds_json = '{}', flagged_json = '[]' " +
			"WHERE orthanc_instance_id = 'B'", nil, "", 1},
		{"update B after ok", "UPDATE results SET flagged_json = '[]' WHERE orthanc_instance_id = 'B'", nil,
			"ok results are write-once", 0},
		{"heatmap PNG on A", "INSERT INTO heatmaps VALUES ('A', 'x', ?)", []any{png}, "", 1},
		{"heatmap not PNG", "INSERT INTO heatmaps VALUES ('A', 'y', ?)", []any{[]byte("GIF89a..")}, check, 0},
		{"heatmap unknown instance", "INSERT INTO heatmaps VALUES ('Z', 'x', ?)", []any{png},
			"heatmaps require an ok result", 0},
		{"heatmap overwrite", "UPDATE heatmaps SET png = ? WHERE orthanc_instance_id = 'A'", []any{png},
			"heatmaps are append-only", 0},
		{"error result C", insErr, []any{"C", "boom", now, now}, "", 1},
		{"decision on error C", insDec, []any{"C", "approve", "", now}, "decision requires an ok result", 0},
		{"approve A", insDec, []any{"A", "approve", "", now}, "", 1},
		{"reject A without comment", insDec, []any{"A", "reject", "  ", now}, check, 0},
		{"reject A with comment", insDec, []any{"A", "reject", "box misses the effusion", now}, "", 1},
		{"decision update", "UPDATE decisions SET decision = 'approve'", nil, "decisions are append-only", 0},
		{"decision delete", "DELETE FROM decisions", nil, "decisions are append-only", 0},
		{"audit append", insAud, []any{now, "result_created", h0, strings.Repeat("a", 64)}, "", 1},
		{"audit unknown action", insAud, []any{now, "hack", strings.Repeat("a", 64), strings.Repeat("b", 64)},
			check, 0},
		{"audit update", "UPDATE audit_log SET actor = 'x'", nil, "audit_log is append-only", 0},
		{"audit delete", "DELETE FROM audit_log", nil, "audit_log is append-only", 0},
	}
	if len(cases) != 28 {
		t.Fatalf("len(cases) = %d, want 28 (same cases as the Python check)", len(cases))
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res, err := db.Exec(c.sql, c.args...)
			if c.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), c.wantErr) {
					t.Fatalf("err = %v, want error containing %q", err, c.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			n, err := res.RowsAffected()
			if err != nil {
				t.Fatal(err)
			}
			if n != c.wantN {
				t.Fatalf("rows affected = %d, want %d", n, c.wantN)
			}
		})
	}

	for table, want := range map[string]int{"results": 3, "heatmaps": 1, "decisions": 2, "audit_log": 1} {
		var got int
		if err := db.QueryRow("SELECT count(*) FROM " + table).Scan(&got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s rows = %d, want %d", table, got, want)
		}
	}
}
