package server

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeModel lays out a fake converted model on disk.
func writeModel(t *testing.T, root, id string, files map[string]string) {
	t.Helper()
	for name, content := range files {
		p := filepath.Join(root, "models", filepath.FromSlash(id), filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func getZip(t *testing.T, srv *Server, target string) *zip.Reader {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, target, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/zip" {
		t.Fatalf("content-type = %q", ct)
	}
	body := rec.Body.Bytes()
	zr, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		t.Fatalf("not a readable zip: %v", err)
	}
	return zr
}

// modelManifest mirrors what browser-llm-nexus's importModel() expects.
type modelManifest struct {
	Kind    string   `json:"kind"`
	ModelID string   `json:"modelId"`
	Dtypes  []string `json:"dtypes"`
	Files   []struct {
		File string `json:"file"`
		URL  string `json:"url"`
		Path string `json:"path"`
	} `json:"files"`
}

func readManifest(t *testing.T, zr *zip.Reader) modelManifest {
	t.Helper()
	for _, f := range zr.File {
		if f.Name != "manifest.json" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		defer rc.Close()
		var m modelManifest
		if err := json.NewDecoder(rc).Decode(&m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	t.Fatal("archive has no manifest.json")
	return modelManifest{}
}

func newServer(t *testing.T) *Server {
	t.Helper()
	return New(t.TempDir(), nil)
}

func TestModelZipLayout(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{
		"config.json":               `{"model_type":"qwen3"}`,
		"tokenizer.json":            `{"v":1}`,
		"onnx/model_q4.onnx":        "Q4WEIGHTS",
		"onnx/model_quantized.onnx": "Q8WEIGHTS",
	})

	zr := getZip(t, srv, "/api/model.zip?model=acme/tiny")
	m := readManifest(t, zr)

	if m.Kind != "model" || m.ModelID != "acme/tiny" {
		t.Fatalf("manifest = %+v", m)
	}
	if len(m.Files) != 4 {
		t.Fatalf("expected 4 files, got %d", len(m.Files))
	}
	// Every manifest entry must point at a real zip entry, and URLs must be
	// the ones the browser will actually request.
	inZip := map[string]bool{}
	for _, f := range zr.File {
		inZip[f.Name] = true
	}
	for _, f := range m.Files {
		if !inZip[f.File] {
			t.Errorf("manifest references missing entry %s", f.File)
		}
		want := "http://example.com/models/acme/tiny/" + f.Path
		if f.URL != want {
			t.Errorf("url = %s, want %s", f.URL, want)
		}
	}
}

func TestModelZipContentsRoundTrip(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{
		"config.json":        `{"model_type":"qwen3"}`,
		"onnx/model_q4.onnx": "Q4WEIGHTS",
	})

	zr := getZip(t, srv, "/api/model.zip?model=acme/tiny")
	m := readManifest(t, zr)

	byName := map[string]*zip.File{}
	for _, f := range zr.File {
		byName[f.Name] = f
	}
	for _, entry := range m.Files {
		rc, err := byName[entry.File].Open()
		if err != nil {
			t.Fatal(err)
		}
		got, _ := io.ReadAll(rc)
		rc.Close()
		want := map[string]string{
			"config.json":        `{"model_type":"qwen3"}`,
			"onnx/model_q4.onnx": "Q4WEIGHTS",
		}[entry.Path]
		if string(got) != want {
			t.Errorf("%s = %q, want %q", entry.Path, got, want)
		}
	}
}

func TestModelZipDtypeFilter(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{
		"config.json":          `{}`,
		"onnx/model_q4.onnx":   "Q4",
		"onnx/model_fp16.onnx": "FP16",
	})

	m := readManifest(t, getZip(t, srv, "/api/model.zip?model=acme/tiny&dtype=q4"))
	if len(m.Dtypes) != 1 || m.Dtypes[0] != "q4" {
		t.Fatalf("dtypes = %v, want [q4]", m.Dtypes)
	}
	for _, f := range m.Files {
		if f.Path == "onnx/model_fp16.onnx" {
			t.Error("fp16 should have been filtered out")
		}
	}
}

func TestModelZipSkipsMissingFiles(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{"config.json": `{}`})

	m := readManifest(t, getZip(t, srv, "/api/model.zip?model=acme/tiny"))
	if len(m.Files) != 1 || m.Files[0].Path != "config.json" {
		t.Fatalf("files = %+v", m.Files)
	}
}

func TestModelZipRejectsBadInput(t *testing.T) {
	srv := newServer(t)
	for _, tc := range []struct {
		target string
		want   int
	}{
		{"/api/model.zip", http.StatusBadRequest},
		{"/api/model.zip?model=../../etc", http.StatusBadRequest},
		{"/api/model.zip?model=acme/missing", http.StatusNotFound},
	} {
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tc.target, nil))
		if rec.Code != tc.want {
			t.Errorf("%s: status = %d, want %d", tc.target, rec.Code, tc.want)
		}
	}
}

func TestLocalModelsListing(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{"onnx/model_q4.onnx": "Q4"})
	writeModel(t, srv.Root, "acme/big", map[string]string{"onnx/model_fp16.onnx": "FP16"})

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var got []struct {
		ID    string `json:"id"`
		Dtype string `json:"dtype"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 models, got %+v", got)
	}
	dtypes := map[string]string{}
	for _, m := range got {
		dtypes[m.ID] = m.Dtype
	}
	if dtypes["acme/tiny"] != "q4" || dtypes["acme/big"] != "fp16" {
		t.Fatalf("dtypes = %+v", dtypes)
	}
}

// get fetches a target and returns the recorder, failing on an unexpected status.
func get(t *testing.T, srv *Server, target string, want int) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, target, nil))
	if rec.Code != want {
		t.Fatalf("%s: status = %d (want %d), body = %s", target, rec.Code, want, rec.Body.String())
	}
	return rec
}

// The generated page is the "run it anywhere" artifact: it must carry a real
// model id, an absolute archive URL back to this server, and a pinned runtime —
// no placeholder may survive substitution.
func TestStandalonePage(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{"onnx/model_q4.onnx": "Q4"})

	rec := get(t, srv, "/api/standalone.html?model=acme/tiny&dtype=q4", http.StatusOK)
	body := rec.Body.String()

	if strings.Contains(body, "__") && strings.Contains(body, "__MODEL_ID__") {
		t.Error("template placeholder left in the generated page")
	}
	for _, placeholder := range []string{"__MODEL_ID__", "__ARCHIVE_URL__", "__NEXUS_VERSION__", "__DTYPE_OPTION__"} {
		if strings.Contains(body, placeholder) {
			t.Errorf("%s not substituted", placeholder)
		}
	}
	want := "http://example.com/api/model.zip?model=acme%2Ftiny&dtype=q4"
	if !strings.Contains(body, want) {
		t.Errorf("archive url %q not found in page", want)
	}
	if !strings.Contains(body, "dtype: 'q4'") {
		t.Error("requested dtype not passed to the loader")
	}
	if !strings.Contains(body, "cdn.jsdelivr.net/npm/browser-llm-nexus@") {
		t.Error("page does not load browser-llm-nexus from a CDN")
	}
	if !strings.Contains(body, "<title>acme/tiny") {
		t.Error("model id not in the page title")
	}
	// Downloading is the point: it must arrive as a file, named after the model.
	if cd := rec.Header().Get("Content-Disposition"); cd != `attachment; filename="acme_tiny-chat.html"` {
		t.Errorf("content-disposition = %q", cd)
	}
}

// ?inline=1 previews the same page in a tab instead of downloading it.
func TestStandaloneInlinePreview(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{"onnx/model_q4.onnx": "Q4"})

	rec := get(t, srv, "/api/standalone.html?model=acme/tiny&inline=1", http.StatusOK)
	if cd := rec.Header().Get("Content-Disposition"); cd != "" {
		t.Errorf("inline preview should not be an attachment, got %q", cd)
	}
	// With no dtype the loader must be left on auto, not given an empty value.
	if body := rec.Body.String(); strings.Contains(body, "dtype: ''") {
		t.Error("empty dtype emitted")
	}
}

func TestStandaloneRejectsBadInput(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{"onnx/model_q4.onnx": "Q4"})
	for _, tc := range []struct {
		target string
		want   int
	}{
		{"/api/standalone.html", http.StatusBadRequest},
		{"/api/standalone.html?model=../../etc", http.StatusBadRequest},
		{"/api/standalone.html?model=acme/tiny&dtype=q4'+alert(1)", http.StatusBadRequest},
		{"/api/standalone.html?model=acme/tiny&dtype=int8", http.StatusBadRequest},
		{"/api/standalone.html?model=acme/notconverted", http.StatusNotFound},
	} {
		get(t, srv, tc.target, tc.want)
	}
}

// The whole point of the artifacts is that they run from somewhere else, so the
// read-only model endpoints must be fetchable cross-origin.
func TestModelArtifactsAreCrossOriginReadable(t *testing.T) {
	srv := newServer(t)
	writeModel(t, srv.Root, "acme/tiny", map[string]string{
		"config.json":        `{"model_type":"qwen3"}`,
		"onnx/model_q4.onnx": "Q4",
	})
	for _, target := range []string{
		"/api/model.zip?model=acme/tiny",
		"/models/acme/tiny/config.json",
	} {
		rec := get(t, srv, target, http.StatusOK)
		if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
			t.Errorf("%s: Access-Control-Allow-Origin = %q, want *", target, got)
		}
	}
}
