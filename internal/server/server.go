// Package server exposes the search/convert pipeline as a local web UI.
package server

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/muthuishere/hf2browser/internal/hf"
	"github.com/muthuishere/hf2browser/internal/pipeline"
)

//go:embed ui.html
var uiHTML []byte

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
	mux.HandleFunc("GET /api/convert", s.handleConvert) // SSE
	mux.Handle("GET /models/", http.StripPrefix("/models/", http.FileServer(http.Dir(s.Root+"/models"))))
	mux.Handle("GET /demo/", http.StripPrefix("/demo/", http.FileServer(http.Dir(s.Root+"/demo"))))
	mux.Handle("GET /lib/", http.StripPrefix("/lib/", http.FileServer(http.Dir(s.Root+"/lib"))))
	return mux
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
