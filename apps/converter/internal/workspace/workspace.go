// Package workspace decides where hf2browser does its work.
//
// Two modes, chosen automatically:
//
//   - Source checkout — if the pipeline tree sits next to the binary or the
//     working directory, it is used in place, so contributors edit a file and
//     rerun without an unpack step in between.
//   - Installed binary — otherwise the embedded copies of the pipeline and the
//     verifier are unpacked into the work directory (~/.hf2browser by default).
//     Nothing else is needed: a downloaded binary is the whole product.
package workspace

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	hf2browser "github.com/muthuishere/hf2browser"
)

// marker identifies a directory that can run the pipeline.
const marker = "pytools/tjs_scripts"

// stamp records which build unpacked the current contents, so an upgraded
// binary refreshes them and an unchanged one does no work at all.
const stamp = ".assets-version"

// Prepare returns a directory the pipeline can run in, unpacking the embedded
// assets first if this is not a source checkout.
func Prepare(workDir string) (string, error) {
	if dir := findSource(); dir != "" {
		return dir, nil
	}
	if workDir == "" {
		return "", fmt.Errorf("no work directory configured")
	}
	if err := unpack(workDir); err != nil {
		return "", fmt.Errorf("unpacking runtime files into %s: %w", workDir, err)
	}
	return workDir, nil
}

// findSource looks for a checkout, from the binary's directory and the cwd.
func findSource() string {
	starts := []string{}
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	if cwd, err := os.Getwd(); err == nil {
		starts = append(starts, cwd)
	}
	for _, start := range starts {
		dir := start
		for {
			if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(marker))); err == nil {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return ""
}

// sources are the embedded trees, unpacked with their paths preserved.
func sources() []fs.FS {
	return []fs.FS{hf2browser.PyTools, hf2browser.Verify, hf2browser.Demo}
}

// unpack writes the embedded assets into dir unless they are already current.
func unpack(dir string) error {
	want, err := fingerprint()
	if err != nil {
		return err
	}
	if got, err := os.ReadFile(filepath.Join(dir, stamp)); err == nil && string(got) == want {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, src := range sources() {
		if err := copyTree(src, dir); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(dir, stamp), []byte(want), 0o644)
}

func copyTree(src fs.FS, dir string) error {
	return fs.WalkDir(src, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || path == "." {
			return err
		}
		target := filepath.Join(dir, filepath.FromSlash(path))
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := fs.ReadFile(src, path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
}

// fingerprint hashes every embedded file's path and contents. Contents, not a
// build timestamp: rebuilding without changing anything must not force a
// needless re-unpack.
func fingerprint() (string, error) {
	h := sha256.New()
	for _, src := range sources() {
		var paths []string
		err := fs.WalkDir(src, ".", func(path string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				paths = append(paths, path)
			}
			return err
		})
		if err != nil {
			return "", err
		}
		sort.Strings(paths)
		for _, p := range paths {
			f, err := src.Open(p)
			if err != nil {
				return "", err
			}
			io.WriteString(h, p+"\x00")
			_, err = io.Copy(h, f)
			f.Close()
			if err != nil {
				return "", err
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:32], nil
}

// Describe is what `serve` prints so the layout is never a mystery.
func Describe(root, models string) string {
	mode := "installed"
	if IsSource(root) {
		mode = "source checkout"
	}
	return fmt.Sprintf("%s (%s)\nmodels:    %s", root, mode, models)
}

// IsSource reports whether dir is a checkout rather than an unpacked work dir.
func IsSource(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, stamp)); err == nil {
		return false
	}
	_, err := os.Stat(filepath.Join(dir, strings.ReplaceAll(marker, "/", string(filepath.Separator))))
	return err == nil
}
