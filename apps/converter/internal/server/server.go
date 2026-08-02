// Package server exposes the search/convert pipeline as a local web UI.
package server

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/muthuishere/hf2browser/internal/archive"
	"github.com/muthuishere/hf2browser/internal/hf"
	"github.com/muthuishere/hf2browser/internal/pipeline"
)

//go:embed ui.html
var uiHTML []byte

//go:embed standalone.html
var standaloneHTML []byte

// nexusFallbackVersion is used when the installed package can't be read; the
// generated page pins a version so it keeps working after a breaking release.
const nexusFallbackVersion = "0.6.0"

// Server wires the HF client and conversion pipeline to HTTP handlers.
type Server struct {
	// Root is where the pipeline and verifier live (a checkout, or the
	// unpacked work directory). Models is where converted models land — the
	// two are separate so `models_dir` can point anywhere, e.g. a big disk.
	Root   string
	Models string
	HF     *hf.Client

	mu   sync.Mutex // one conversion at a time
	busy string     // model currently converting, "" if idle
}

func New(root, models string, client *hf.Client) *Server {
	return &Server{Root: root, Models: models, HF: client}
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
	// Preflights for the two endpoints a page on another origin may fetch.
	mux.HandleFunc("OPTIONS /api/model.zip", handlePreflight)
	mux.HandleFunc("OPTIONS /models/", handlePreflight)
	// Model files are the artifact people take away — a page served from
	// anywhere else must be able to fetch them, so they are readable cross-origin.
	mux.Handle("GET /models/", cors(http.StripPrefix("/models/", http.FileServer(http.Dir(s.Models)))))
	return mux
}

// cors marks a response as a public static artifact. Only the read-only model
// endpoints use it — search and convert stay same-origin.
//
// Access-Control-Allow-Private-Network is the opt-in for Private Network
// Access: a page on https://example.github.io fetching http://localhost is a
// public→loopback request, which Chrome blocks by default. Without this header
// a hosted page (the browser-llm-nexus demo, say) cannot load a model from a
// converter running on the user's own machine — which is the whole hand-off.
func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		h.ServeHTTP(w, r)
	})
}

func setCORS(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Private-Network", "true")
	h.Set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
	h.Set("Access-Control-Allow-Headers", "*")
}

// handlePreflight answers the CORS/PNA preflight for the model endpoints.
func handlePreflight(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	w.Header().Set("Access-Control-Max-Age", "600")
	w.WriteHeader(http.StatusNoContent)
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
	root := s.Models
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
		if _, err := os.Stat(filepath.Join(s.Models, modelID, "onnx", f)); err == nil {
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
	modelDir := filepath.Join(s.Models, model)
	if _, err := os.Stat(modelDir); err != nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("%s is not converted", model))
		return
	}

	// The origin the browser will request these from, so cached URLs match.
	root := fmt.Sprintf("%s/models/%s/", origin(r), model)

	w.Header().Set("Content-Type", "application/zip")
	setCORS(w) // a page hosted elsewhere must be able to fetch it
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=%q", strings.ReplaceAll(model, "/", "_")+".zip"))

	// Same writer the CLI's `export` uses — one implementation of the format.
	archive.Write(w, modelDir, model, root, dtype, time.Now())
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
	if err := pipeline.Convert(s.Root, s.Models, sse, model, strings.Split(modes, ","), extra); err != nil {
		sse.event("error", "conversion failed: "+err.Error())
		return
	}
	verifyTask := "text-generation"
	if !isChat {
		verifyTask = "feature-extraction"
	}
	sse.event("status", "verifying on CPU")
	if err := pipeline.Verify(s.Root, s.Models, sse, model, verifyTask, modes); err != nil {
		sse.event("error", "verification failed: "+err.Error())
		return
	}
	sse.event("done", model)
}
