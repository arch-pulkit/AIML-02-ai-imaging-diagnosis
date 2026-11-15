package main

import (
	"database/sql"
	_ "embed"
	"fmt"

	_ "modernc.org/sqlite" // registers the "sqlite" driver (pure Go, no cgo)
)

//go:embed schema.sql
var schemaSQL string

// schemaVersion must match "PRAGMA user_version" at the end of schema.sql.
const schemaVersion = 1

// Per-connection settings; with a single connection they apply to every query.
const dsnPragmas = "?_pragma=foreign_keys(1)&_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"

// openDB opens or creates the SQLite file, applies the schema (idempotent) and verifies the settings.
func openDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", "file:"+path+dsnPragmas)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	// One connection = one writer, so our own goroutines never hit "database is locked".
	// Caveat: never query through db while holding a tx from db; it would wait forever.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schemaSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("apply schema to %s: %w", path, err)
	}
	if err := checkDB(db); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

// checkDB fails loudly if a setting the schema relies on is not active.
func checkDB(db *sql.DB) error {
	var foreignKeys, userVersion int
	var journalMode string
	if err := db.QueryRow("PRAGMA foreign_keys").Scan(&foreignKeys); err != nil {
		return fmt.Errorf("read foreign_keys: %w", err)
	}
	if foreignKeys != 1 {
		return fmt.Errorf("foreign_keys = %d, want 1 (check the DSN pragmas)", foreignKeys)
	}
	if err := db.QueryRow("PRAGMA journal_mode").Scan(&journalMode); err != nil {
		return fmt.Errorf("read journal_mode: %w", err)
	}
	if journalMode != "wal" {
		return fmt.Errorf("journal_mode = %q, want \"wal\"", journalMode)
	}
	if err := db.QueryRow("PRAGMA user_version").Scan(&userVersion); err != nil {
		return fmt.Errorf("read user_version: %w", err)
	}
	if userVersion != schemaVersion {
		return fmt.Errorf("user_version = %d, want %d (schema.sql and schemaVersion disagree)", userVersion, schemaVersion)
	}
	return nil
}

func sqliteVersion(db *sql.DB) (string, error) {
	var v string
	err := db.QueryRow("SELECT sqlite_version()").Scan(&v)
	return v, err
}
