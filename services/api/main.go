// Command api is the results API: the only writer to SQLite, fed by the worker, read by the viewer.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "-healthcheck" {
		os.Exit(runHealthcheck(os.Args[2:]))
	}
	if err := run(); err != nil {
		slog.Error("api stopped", "err", err)
		os.Exit(1)
	}
}

func run() error {
	dbPath := os.Getenv("API_DB")
	if dbPath == "" {
		return errors.New("API_DB is not set (path to the SQLite file)")
	}
	addr := os.Getenv("API_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	orthancURL := os.Getenv("ORTHANC_URL")
	if orthancURL == "" {
		return errors.New("ORTHANC_URL is not set (Orthanc REST base URL, for image previews)")
	}
	orthanc := newOrthancClient(orthancURL, os.Getenv("ORTHANC_USER"), os.Getenv("ORTHANC_PASSWORD"))
	slog.Info("orthanc client", "url", orthancURL)

	db, err := openDB(dbPath)
	if err != nil {
		return err
	}
	defer db.Close()
	version, err := sqliteVersion(db)
	if err != nil {
		return err
	}
	slog.Info("database ready", "path", dbPath, "sqlite", version, "schema_version", schemaVersion)

	srv := &http.Server{
		Addr:              addr,
		Handler:           newMux(db, version, orthanc),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
	}

	// SIGTERM (docker stop) or Ctrl-C: stop accepting, finish in-flight requests, then exit 0.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errc := make(chan error, 1)
	go func() {
		slog.Info("listening", "addr", addr)
		errc <- srv.ListenAndServe()
	}()
	select {
	case err := <-errc:
		return err // e.g. port already in use
	case <-ctx.Done():
	}
	slog.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}
