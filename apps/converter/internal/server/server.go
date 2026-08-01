// Package server exposes the search/convert pipeline as a local web UI.
package server

import (
	"archive/zip"
	_ "embed"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/muthuishere/hf2browser/internal/hf"
	"github.com/muthuishere/hf2browser/internal/pipeline"
)

//go:embed ui.html
var uiHTML []byte

//go:embed standalone.html
var standaloneHTML []byte

// nexusFallbackVersion is used when the installed package can't be read; the
// generated page pins a version so it keeps working after a breaking release.
const nexusFallbackVersion = "0.4.2"

// Server wires the HF client and conversion pipeline to HTTP handlers.
type Server struct {
	Root string
	HF   *hf.Client

	mu   sync.Mutex // one conversion at a time
	busy string     // model currently converting, "" if idle
}

func New(root string, client *hf.Client) *Server {
	return &Server{Root: root, HF: client}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Write(uiHTML)
	})
	mux.HandleFunc("GET /api/search", s.handleSearch)
	mux.HandleFunc("GET /api/models", s.handleLocalModels)
	mux.HandleFunc("GET /api/tools", s.handleTools)
	mux.HandleFunc("GET /api/model.zip", s.handleModelZip)
	mux.HandleFunc("GET /api/standalone.html", s.handleStandalone)
	mux.HandleFunc("GET /api/convert", s.handleConvert) // SSE
	// Model files are the artifact people take away — a page served from
	// anywhere else must be able to fetch them, so they are readable cross-origin.
	mux.Handle("GET /models/", cors(http.StripPrefix("/models/", http.FileServer(http.Dir(s.Root+"/models")))))
	mux.Handle("GET /demo/", http.StripPrefix("/demo/", http.FileServer(http.Dir(s.Root+"/demo"))))
	// browser-llm-nexus ships as an npm package; serve it from verify/node_modules
	// so the demo page loads the same build the CPU verifier uses.
	nexusDist := filepath.Join(s.Root, "verify", "node_modules", "browser-llm-nexus", "dist")
	mux.Handle("GET /nexus/", http.StripPrefix("/nexus/", http.FileServer(http.Dir(nexusDist))))
	return mux
}

// cors marks a response as a public static artifact. Only the read-only model
// endpoints use it — search and convert stay same-origin.
func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		h.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	w.WriteHeader(code)
	writeJSON(w, map[string]string{"error": err.Error()})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	opts := hf.SearchOptions{
		Query:    q.Get("q"),
		Pipeline: q.Get("pipeline"),
		Sort:     q.Get("sort"),
		Limit:    20,
	}
	if opts.Pipeline == "" {
		opts.Pipeline = "text-generation"
	}
	if tags := q.Get("tags"); tags != "" {
		opts.Tags = strings.Split(tags, ",")
	}
	results, err := s.HF.Search(opts)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	type row struct {
		hf.ModelInfo
		Converted bool `json:"converted"`
	}
	rows := make([]row, len(results))
	for i, m := range results {
		rows[i] = row{ModelInfo: m, Converted: s.isConverted(m.ID)}
	}
	writeJSON(w, rows)
}

// handleLocalModels lists converted models on disk with their best dtype.
func (s *Server) handleLocalModels(w http.ResponseWriter, r *http.Request) {
	type localModel struct {
		ID     string `json:"id"`
		Dtype  string `json:"dtype"`
		SizeMB int64  `json:"size_mb"`
	}
	// preference order mirrors the demo: smallest usable first
	dtypeFiles := [][2]string{{"q4", "model_q4.onnx"}, {"q8", "model_quantized.onnx"}, {"fp16", "model_fp16.onnx"}, {"fp32", "model.onnx"}}
	out := []localModel{}
	root := filepath.Join(s.Root, "models")
	orgs, _ := os.ReadDir(root)
	for _, org := range orgs {
		if !org.IsDir() {
			continue
		}
		names, _ := os.ReadDir(filepath.Join(root, org.Name()))
		for _, name := range names {
			if !name.IsDir() {
				continue
			}
			for _, df := range dtypeFiles {
				p := filepath.Join(root, org.Name(), name.Name(), "onnx", df[1])
				if info, err := os.Stat(p); err == nil {
					out = append(out, localModel{
						ID:     org.Name() + "/" + name.Name(),
						Dtype:  df[0],
						SizeMB: info.Size() / 1e6,
					})
					break
				}
			}
		}
	}
	writeJSON(w, out)
}

// isConverted reports whether any browser-runnable onnx variant exists locally.
func (s *Server) isConverted(modelID string) bool {
	if strings.Contains(modelID, "..") {
		return false
	}
	for _, f := range []string{"model_q4.onnx", "model_quantized.onnx", "model_fp16.onnx", "model.onnx"} {
		if _, err := os.Stat(filepath.Join(s.Root, "models", modelID, "onnx", f)); err == nil {
			return true
		}
	}
	return false
}

// handleModelZip serves a converted model as one portable archive, in the
// layout browser-llm-nexus's importModel() reads: manifest.json + files/N.bin.
// Hand the URL (or the downloaded file) to NexusChat.fromArchive and the model
// loads with no further network calls.
//
//	GET /api/model.zip?model=<id>[&dtype=q4]
func (s *Server) handleModelZip(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" || strings.Contains(model, "..") {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("valid model required"))
		return
	}
	dtype := r.URL.Query().Get("dtype")
	modelDir := filepath.Join(s.Root, "models", model)
	if _, err := os.Stat(modelDir); err != nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("%s is not converted", model))
		return
	}

	type manifestFile struct {
		File string `json:"file"`
		URL  string `json:"url"`
		Path string `json:"path"`
	}
	type manifest struct {
		Kind      string         `json:"kind"`
		ModelID   string         `json:"modelId"`
		CreatedAt string         `json:"createdAt"`
		Dtypes    []string       `json:"dtypes"`
		Files     []manifestFile `json:"files"`
	}

	dtypeOf := map[string]string{
		"onnx/model_q4.onnx":        "q4",
		"onnx/model_quantized.onnx": "q8",
		"onnx/model_fp16.onnx":      "fp16",
		"onnx/model.onnx":           "fp32",
		"onnx/model.onnx_data":      "fp32",
	}

	// The origin the browser will request these from, so cached URLs match.
	root := fmt.Sprintf("%s/models/%s/", origin(r), model)

	var paths []string
	for _, f := range []string{
		"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
		"special_tokens_map.json", "preprocessor_config.json", "vocab.json", "merges.txt",
		"added_tokens.json", "chat_template.jinja",
	} {
		paths = append(paths, f)
	}
	for p, d := range dtypeOf {
		if dtype == "" || dtype == d {
			paths = append(paths, p)
		}
	}
	sort.Strings(paths)

	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Access-Control-Allow-Origin", "*") // a page hosted elsewhere must be able to fetch it
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=%q", strings.ReplaceAll(model, "/", "_")+".zip"))

	zw := zip.NewWriter(w)
	defer zw.Close()

	mf := manifest{Kind: "model", ModelID: model, CreatedAt: time.Now().UTC().Format(time.RFC3339)}
	seen := map[string]bool{}
	for _, p := range paths {
		full := filepath.Join(modelDir, filepath.FromSlash(p))
		data, err := os.ReadFile(full)
		if err != nil || len(data) == 0 {
			continue
		}
		name := fmt.Sprintf("files/%d.bin", len(mf.Files))
		// Stored, not deflated: weights are dense, compressing them buys nothing.
		fw, err := zw.CreateRaw(&zip.FileHeader{Name: name, Method: zip.Store,
			CRC32: crc32.ChecksumIEEE(data), CompressedSize64: uint64(len(data)), UncompressedSize64: uint64(len(data))})
		if err != nil {
			return
		}
		if _, err := fw.Write(data); err != nil {
			return
		}
		mf.Files = append(mf.Files, manifestFile{File: name, URL: root + p, Path: p})
		if d := dtypeOf[p]; d != "" && !seen[d] {
			seen[d] = true
			mf.Dtypes = append(mf.Dtypes, d)
		}
	}

	body, _ := json.Marshal(mf)
	if fw, err := zw.Create("manifest.json"); err == nil {
		fw.Write(body)
	}
}

// origin is the scheme+host the browser reached this server on, so URLs we bake
// into archives and generated pages point back at the same place.
func origin(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + r.Host
}

// nexusVersion reports the browser-llm-nexus version installed for the CPU
// verifier, so a generated page pins the exact runtime this repo tested with.
func (s *Server) nexusVersion() string {
	data, err := os.ReadFile(filepath.Join(s.Root, "verify", "node_modules", "browser-llm-nexus", "package.json"))
	if err != nil {
		return nexusFallbackVersion
	}
	var pkg struct {
		Version string `json:"version"`
	}
	if json.Unmarshal(data, &pkg) != nil || pkg.Version == "" {
		return nexusFallbackVersion
	}
	return pkg.Version
}

// handleStandalone generates a single, self-contained HTML chat page for a
// converted model: browser-llm-nexus from a CDN, the model from this server's
// /api/model.zip (or a file the visitor picks). No build step, no framework,
// nothing of ours at runtime — drop it on any static host.
//
//	GET /api/standalone.html?model=<id>[&dtype=q4][&inline=1]
func (s *Server) handleStandalone(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" || strings.Contains(model, "..") {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("valid model required"))
		return
	}
	if !s.isConverted(model) {
		writeErr(w, http.StatusNotFound, fmt.Errorf("%s is not converted", model))
		return
	}

	archive := origin(r) + "/api/model.zip?model=" + url.QueryEscape(model)
	dtypeOption := ""
	if dtype := r.URL.Query().Get("dtype"); dtype != "" {
		// Allowlist, not escaping: this value is interpolated into the page's
		// JavaScript, so only the four names the runtime knows may get through.
		switch dtype {
		case "q4", "q8", "fp16", "fp32":
		default:
			writeErr(w, http.StatusBadRequest, fmt.Errorf("unknown dtype %q", dtype))
			return
		}
		archive += "&dtype=" + url.QueryEscape(dtype)
		dtypeOption = "dtype: '" + dtype + "'"
	}

	page := strings.NewReplacer(
		"__MODEL_ID__", model,
		"__ARCHIVE_URL__", archive,
		"__DTYPE_OPTION__", dtypeOption,
		"__NEXUS_VERSION__", s.nexusVersion(),
	).Replace(string(standaloneHTML))

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	if r.URL.Query().Get("inline") != "1" {
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q",
			strings.ReplaceAll(model, "/", "_")+"-chat.html"))
	}
	fmt.Fprint(w, page)
}

func (s *Server) handleTools(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("model required"))
		return
	}
	tools, err := s.HF.SupportsToolCalling(model)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, map[string]bool{"tools": tools})
}

// sseWriter streams log lines as SSE "log" events.
type sseWriter struct {
	w  http.ResponseWriter
	fl http.Flusher
}

func (s *sseWriter) event(name, data string) {
	for _, line := range strings.Split(strings.TrimRight(data, "\n"), "\n") {
		fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", name, line)
	}
	s.fl.Flush()
}

func (s *sseWriter) Write(p []byte) (int, error) {
	s.event("log", string(p))
	return len(p), nil
}

// handleConvert runs download+convert+verify, streaming output over SSE.
// GET /api/convert?model=<id>&modes=q8,q4,fp16&force=1
func (s *Server) handleConvert(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" || strings.Contains(model, "..") {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("valid model required"))
		return
	}
	modes := r.URL.Query().Get("modes")
	if modes == "" {
		modes = "q4" // smallest variant that runs in the browser
	}
	force := r.URL.Query().Get("force") == "1"
	task := r.URL.Query().Get("task") // e.g. feature-extraction for embedding models

	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	if !s.mu.TryLock() {
		writeErr(w, http.StatusConflict, fmt.Errorf("a conversion is already running (%s)", s.busy))
		return
	}
	s.busy = model
	defer func() { s.busy = ""; s.mu.Unlock() }()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	sse := &sseWriter{w: w, fl: fl}

	isChat := task == "" || strings.HasPrefix(task, "text-generation")
	if isChat {
		sse.event("status", "checking "+model)
		tools, err := s.HF.SupportsToolCalling(model)
		if err != nil {
			sse.event("error", "check failed: "+err.Error())
			return
		}
		if !tools && !force {
			sse.event("error", model+" has no tool-calling chat template (convert with force to override)")
			return
		}
		if !tools {
			sse.event("log", "warning: no tool-calling support, converting anyway (force)")
		}
	}

	var extra []string
	if task != "" {
		extra = append(extra, "--task", task)
	}
	sse.event("status", "converting (downloads the model on first run)")
	if err := pipeline.Convert(s.Root, sse, model, strings.Split(modes, ","), extra); err != nil {
		sse.event("error", "conversion failed: "+err.Error())
		return
	}
	verifyTask := "text-generation"
	if !isChat {
		verifyTask = "feature-extraction"
	}
	sse.event("status", "verifying on CPU")
	if err := pipeline.Verify(s.Root, sse, model, verifyTask, modes); err != nil {
		sse.event("error", "verification failed: "+err.Error())
		return
	}
	sse.event("done", model)
}
