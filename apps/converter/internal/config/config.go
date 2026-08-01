// Package config is the one file a user edits to run hf2browser their way.
//
// Precedence is flags > environment > config file > defaults, so a config file
// is always optional and never overrides something you said more specifically.
//
// Deliberately absent: HF_TOKEN. A token is a secret and belongs in the
// environment, not in a file that gets copied around and committed by accident.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Name is the file looked for next to the binary and in the working directory.
const Name = "hf2browser.json"

type Config struct {
	// Port to serve on. 0 = pick the first free port from 8917.
	Port int `json:"port"`
	// OpenBrowser opens the UI on `serve`.
	OpenBrowser bool `json:"open_browser"`
	// WorkDir holds the unpacked pipeline, the verifier and (by default) the
	// converted models. Defaults to ~/.hf2browser.
	WorkDir string `json:"work_dir"`
	// ModelsDir is where converted models land. Empty means "next to the
	// pipeline" — the checkout's models/ in a checkout, <work_dir>/models for
	// an installed binary — which is why it is resolved by the caller.
	ModelsDir string `json:"models_dir"`
	// Dtype is the quantization produced by default. q4 is the smallest
	// variant that runs in a browser.
	Dtype string `json:"dtype"`
	// HFEndpoint is a Hugging Face mirror ($HF_ENDPOINT).
	HFEndpoint string `json:"hf_endpoint"`
	// HFTimeout is the Hub API timeout in seconds ($HF_TIMEOUT).
	HFTimeout int `json:"hf_timeout"`

	// Path records where this config was loaded from ("" = defaults only).
	Path string `json:"-"`
}

func Default() Config {
	return Config{
		Port:        0,
		OpenBrowser: true,
		WorkDir:     "",
		ModelsDir:   "",
		Dtype:       "q4",
		HFTimeout:   30,
	}
}

// Load reads the first config file found, applies environment overrides, and
// fills in defaults. An explicit path that does not exist is an error; the
// implicit locations are simply skipped.
func Load(explicit string) (Config, error) {
	cfg := Default()

	candidates := []string{}
	if explicit != "" {
		candidates = append(candidates, explicit)
	} else {
		if cwd, err := os.Getwd(); err == nil {
			candidates = append(candidates, filepath.Join(cwd, Name))
		}
		if exe, err := os.Executable(); err == nil {
			candidates = append(candidates, filepath.Join(filepath.Dir(exe), Name))
		}
		if home, err := os.UserHomeDir(); err == nil {
			candidates = append(candidates, filepath.Join(home, ".hf2browser", Name))
		}
	}

	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err != nil {
			if explicit != "" {
				return cfg, fmt.Errorf("config %s: %w", path, err)
			}
			continue
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			return cfg, fmt.Errorf("config %s: %w", path, err)
		}
		cfg.Path = path
		break
	}

	// Environment wins over the file: it is how CI and containers configure a run.
	if v := os.Getenv("PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Port = n
		}
	}
	if v := os.Getenv("HF_ENDPOINT"); v != "" {
		cfg.HFEndpoint = v
	}
	if v := os.Getenv("HF_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.HFTimeout = n
		}
	}
	if v := os.Getenv("HF2BROWSER_WORK_DIR"); v != "" {
		cfg.WorkDir = v
	}
	if v := os.Getenv("HF2BROWSER_MODELS_DIR"); v != "" {
		cfg.ModelsDir = v
	}

	if cfg.Dtype == "" {
		cfg.Dtype = "q4"
	}
	if cfg.HFTimeout <= 0 {
		cfg.HFTimeout = 30
	}
	var err error
	if cfg.WorkDir, err = resolveDir(cfg.WorkDir, ".hf2browser"); err != nil {
		return cfg, err
	}
	if cfg.ModelsDir != "" {
		if cfg.ModelsDir, err = expand(cfg.ModelsDir); err != nil {
			return cfg, err
		}
		cfg.ModelsDir, _ = filepath.Abs(cfg.ModelsDir)
	}
	return cfg, nil
}

// ModelsUnder returns the configured models directory, defaulting to models/
// inside the directory the pipeline runs from. Keeping the default relative to
// the runtime root means a checkout keeps using its own models/ folder while an
// installed binary collects them under the work directory.
func (c Config) ModelsUnder(root string) string {
	if c.ModelsDir != "" {
		return c.ModelsDir
	}
	return filepath.Join(root, "models")
}

// Write saves a config file, so `hf2browser init` produces something editable
// rather than making people guess the field names.
func (c Config) Write(path string) error {
	c.Path = ""
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func resolveDir(dir, fallbackUnderHome string) (string, error) {
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("cannot determine home directory: %w", err)
		}
		return filepath.Join(home, fallbackUnderHome), nil
	}
	expanded, err := expand(dir)
	if err != nil {
		return "", err
	}
	return filepath.Abs(expanded)
}

// expand resolves a leading ~ so config files stay portable between machines.
func expand(path string) (string, error) {
	if path != "~" && !strings.HasPrefix(path, "~/") {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(path, "~"), "/")), nil
}
