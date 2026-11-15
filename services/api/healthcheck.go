package main

import (
	"fmt"
	"net/http"
	"os"
	"time"
)

// check health GETs url adn returns nil only for HTTP 200
func checkHealth(url string) error {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("healthcheck: %s returned status %d", url, resp.StatusCode)
	}
	return nil
}

// implements api-healthcheck URL and returns exit code
func runHealthcheck(args []string) int {
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "usage: api -healthcheck URL")
		return 2
	}
	if err := checkHealth(args[0]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}
