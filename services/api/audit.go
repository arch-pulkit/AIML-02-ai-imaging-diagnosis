package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// genesisHash is the prev_hash of the first audit row.
var genesisHash = strings.Repeat("0", 64)

// auditRecord is what each audit hash covers: the row's content plus the previous row's hash.
// Field order is fixed by the struct, so the JSON (and the hash) is deterministic.
type auditRecord struct {
	Seq        int64           `json:"seq"`
	At         string          `json:"at"`
	Actor      string          `json:"actor"`
	Action     string          `json:"action"`
	InstanceID string          `json:"orthanc_instance_id"`
	Details    json.RawMessage `json:"details"`
	PrevHash   string          `json:"prev_hash"`
}

func (r auditRecord) hash() (string, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return "", fmt.Errorf("encode audit record: %w", err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// appendAudit adds one row to the hash-chained audit log, inside the caller's transaction.
func appendAudit(ctx context.Context, tx *sql.Tx, at, actor, action, instanceID string, details any) error {
	detailsJSON, err := json.Marshal(details)
	if err != nil {
		return fmt.Errorf("encode audit details: %w", err)
	}
	rec := auditRecord{At: at, Actor: actor, Action: action, InstanceID: instanceID, Details: detailsJSON}
	var lastSeq int64
	var lastHash string
	err = tx.QueryRowContext(ctx, "SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1").Scan(&lastSeq, &lastHash)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		rec.Seq, rec.PrevHash = 1, genesisHash
	case err != nil:
		return fmt.Errorf("read audit tail: %w", err)
	default:
		rec.Seq, rec.PrevHash = lastSeq+1, lastHash
	}
	h, err := rec.hash()
	if err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx,
		"INSERT INTO audit_log (seq, at, actor, action, orthanc_instance_id, details_json, prev_hash, hash) "+
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
		rec.Seq, rec.At, rec.Actor, rec.Action, rec.InstanceID, string(rec.Details), rec.PrevHash, h)
	if err != nil {
		return fmt.Errorf("append audit: %w", err)
	}
	return nil
}

// verifyAudit recomputes the whole chain and reports the first row that does not match.
// It detects changed or reordered rows; removing rows from the END needs an outside copy of the last hash.
func verifyAudit(ctx context.Context, db *sql.DB) (int, error) {
	rows, err := db.QueryContext(ctx,
		"SELECT seq, at, actor, action, orthanc_instance_id, details_json, prev_hash, hash FROM audit_log ORDER BY seq")
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	prev := genesisHash
	n := 0
	for rows.Next() {
		var rec auditRecord
		var instance sql.NullString
		var details, stored string
		if err := rows.Scan(&rec.Seq, &rec.At, &rec.Actor, &rec.Action, &instance, &details, &rec.PrevHash, &stored); err != nil {
			return n, err
		}
		rec.InstanceID = instance.String
		rec.Details = json.RawMessage(details)
		n++
		if rec.Seq != int64(n) {
			return n, fmt.Errorf("seq %d: expected %d (gap or reorder)", rec.Seq, n)
		}
		if rec.PrevHash != prev {
			return n, fmt.Errorf("seq %d: prev_hash does not match the previous row", rec.Seq)
		}
		h, err := rec.hash()
		if err != nil {
			return n, err
		}
		if h != stored {
			return n, fmt.Errorf("seq %d: hash mismatch (row was changed)", rec.Seq)
		}
		prev = stored
	}
	return n, rows.Err()
}
