package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultsWithNoConfigFile(t *testing.T) {
	t.Chdir(t.TempDir())
	t.Setenv("PORT", "")
	t.Setenv("HF2BROWSER_WORK_DIR", "")
	t.Setenv("HF2BROWSER_MODELS_DIR", "")

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Path != "" {
		t.Errorf("unexpectedly loaded %s", cfg.Path)
	}
	if cfg.Dtype != "q4" {
		t.Errorf("dtype = %q, want q4", cfg.Dtype)
	}
	if !cfg.OpenBrowser {
		t.Error("open_browser should default on")
	}
	// Unset means "resolve against wherever the pipeline runs from", so a
	// checkout keeps its own models/ and an installed binary uses the work dir.
	if cfg.ModelsDir != "" {
		t.Errorf("models dir = %q, want empty until resolved", cfg.ModelsDir)
	}
	if want := filepath.Join(cfg.WorkDir, "models"); cfg.ModelsUnder(cfg.WorkDir) != want {
		t.Errorf("ModelsUnder = %s, want %s", cfg.ModelsUnder(cfg.WorkDir), want)
	}
	if want := filepath.Join("/checkout", "models"); cfg.ModelsUnder("/checkout") != want {
		t.Errorf("ModelsUnder(checkout) = %s, want %s", cfg.ModelsUnder("/checkout"), want)
	}
}

func TestFileIsReadFromTheWorkingDirectory(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	t.Setenv("PORT", "")
	path := filepath.Join(dir, Name)
	if err := os.WriteFile(path, []byte(`{"port":9000,"dtype":"q8","open_browser":false}`), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Path != path {
		t.Errorf("loaded %q, want %q", cfg.Path, path)
	}
	if cfg.Port != 9000 || cfg.Dtype != "q8" || cfg.OpenBrowser {
		t.Errorf("config not applied: %+v", cfg)
	}
}

// Environment beats the file: that is how a container or CI run overrides it.
func TestEnvironmentOverridesFile(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile(filepath.Join(dir, Name), []byte(`{"port":9000}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PORT", "9500")

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != 9500 {
		t.Errorf("port = %d, want 9500 from $PORT", cfg.Port)
	}
}

func TestTildeExpands(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile(filepath.Join(dir, Name), []byte(`{"models_dir":"~/hf-models"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory")
	}

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(home, "hf-models"); cfg.ModelsDir != want {
		t.Errorf("models dir = %s, want %s", cfg.ModelsDir, want)
	}
}

// A path the user typed must fail loudly; the implicit locations are optional.
func TestExplicitMissingConfigIsAnError(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("expected an error for a missing --config path")
	}
}

func TestBadJSONIsReported(t *testing.T) {
	path := filepath.Join(t.TempDir(), Name)
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("expected a parse error")
	}
}

// `hf2browser init` must produce a file Load() accepts.
func TestWriteRoundTrips(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	path := filepath.Join(dir, Name)
	if err := Default().Write(path); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("init wrote a config Load() rejects: %v", err)
	}
	if cfg.Dtype != "q4" {
		t.Errorf("dtype = %q", cfg.Dtype)
	}
	// The token must never be written to a file, only read from the environment.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"token", "TOKEN"} {
		if strings.Contains(string(data), forbidden) {
			t.Errorf("generated config mentions %q — secrets belong in the environment", forbidden)
		}
	}
}
