package workspace

import (
	"os"
	"path/filepath"
	"testing"
)

// A downloaded binary carries everything it needs; unpacking must produce a
// tree the pipeline, the verifier and the demo page can all run from.
func TestUnpackProducesARunnableTree(t *testing.T) {
	dir := t.TempDir()
	if err := unpack(dir); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"pytools/__init__.py",
		"pytools/tjs_scripts/convert.py",
		"pytools/tjs_scripts/requirements.txt",
		"pytools/tjs_scripts/requirements-modern.txt",
		"verify/verify.mjs",
		"verify/package.json",
	} {
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(want))); err != nil {
			t.Errorf("missing %s: %v", want, err)
		}
	}
	// The marker findSource() looks for must exist, or an unpacked dir would
	// be mistaken for an unusable one.
	if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(marker))); err != nil {
		t.Errorf("unpacked tree is not recognisable as a workspace: %v", err)
	}
}

// Unpacking on every run would clobber a user's edits and waste startup time.
func TestUnpackIsSkippedWhenCurrent(t *testing.T) {
	dir := t.TempDir()
	if err := unpack(dir); err != nil {
		t.Fatal(err)
	}
	canary := filepath.Join(dir, "verify", "verify.mjs")
	if err := os.WriteFile(canary, []byte("// edited"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := unpack(dir); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(canary)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "// edited" {
		t.Error("unpack overwrote an up-to-date workspace")
	}
}

// A changed binary must refresh the tree, otherwise an upgrade silently keeps
// running the old pipeline.
func TestUnpackRefreshesOnVersionChange(t *testing.T) {
	dir := t.TempDir()
	if err := unpack(dir); err != nil {
		t.Fatal(err)
	}
	canary := filepath.Join(dir, "verify", "verify.mjs")
	if err := os.WriteFile(canary, []byte("// stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, stamp), []byte("different-build"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := unpack(dir); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(canary)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) == "// stale" {
		t.Error("unpack did not refresh after a version change")
	}
}

// The fingerprint is content-based, so rebuilding without edits is a no-op.
func TestFingerprintIsStable(t *testing.T) {
	a, err := fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	b, err := fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	if a != b || a == "" {
		t.Fatalf("fingerprint unstable: %q vs %q", a, b)
	}
}

// In a checkout the source tree wins, so contributors edit files in place.
func TestPrepareUsesTheCheckout(t *testing.T) {
	root, err := Prepare("")
	if err != nil {
		t.Fatalf("Prepare in a checkout should not need a work dir: %v", err)
	}
	if !IsSource(root) {
		t.Errorf("%s not detected as a source checkout", root)
	}
}

// An unpacked work dir must not be mistaken for a checkout — it is disposable.
func TestUnpackedDirIsNotASource(t *testing.T) {
	dir := t.TempDir()
	if err := unpack(dir); err != nil {
		t.Fatal(err)
	}
	if IsSource(dir) {
		t.Error("unpacked work dir reported as a source checkout")
	}
}
