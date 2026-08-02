// Package archive packs a converted model into the one portable file
// browser-llm-nexus loads: manifest.json + files/N.bin.
//
// This is deliberately the ONLY place that format is written. The CLI's
// `export` and the server's /api/model.zip are the same bytes by construction
// — a format that drifts between the two is a file that loads from one path
// and not the other, which is the worst kind of bug to debug from a browser.
package archive

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// File is one entry in the manifest. `url` is where the file would have been
// fetched from, so a restored cache matches what a live load would have keyed.
type File struct {
	File string `json:"file"`
	URL  string `json:"url"`
	Path string `json:"path"`
}

// Manifest is browser-llm-nexus's ModelManifest, field for field.
type Manifest struct {
	Kind      string   `json:"kind"`
	ModelID   string   `json:"modelId"`
	CreatedAt string   `json:"createdAt"`
	Dtypes    []string `json:"dtypes"`
	Files     []File   `json:"files"`
}

// DtypeOf maps a weights path to the quantization it holds.
var DtypeOf = map[string]string{
	"onnx/model_q4.onnx":        "q4",
	"onnx/model_quantized.onnx": "q8",
	"onnx/model_fp16.onnx":      "fp16",
	"onnx/model.onnx":           "fp32",
	"onnx/model.onnx_data":      "fp32",
}

// configFiles are the non-weight files every converted model needs. Missing
// ones are skipped, not an error — models differ (merges.txt is BPE-only).
var configFiles = []string{
	"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
	"special_tokens_map.json", "preprocessor_config.json", "vocab.json", "merges.txt",
	"added_tokens.json", "chat_template.jinja",
}

// Paths lists what to pack for a model. An empty dtype means every variant
// present — which is the right default, because which quantization can call a
// tool is model-specific, and an archive pinned to one leaves
// NexusChat.loadForTools() nothing to fall back to.
func Paths(dtype string) []string {
	paths := append([]string{}, configFiles...)
	for p, d := range DtypeOf {
		if dtype == "" || dtype == d {
			paths = append(paths, p)
		}
	}
	sort.Strings(paths)
	return paths
}

// Write packs modelDir into w. root is the URL prefix the files would be
// served from, baked into the manifest so a restored cache matches a live load.
// Returns the manifest actually written, so a caller can report what went in.
func Write(w io.Writer, modelDir, modelID, root, dtype string, now time.Time) (Manifest, error) {
	zw := zip.NewWriter(w)
	mf := Manifest{Kind: "model", ModelID: modelID, CreatedAt: now.UTC().Format(time.RFC3339)}
	seen := map[string]bool{}

	for _, p := range Paths(dtype) {
		data, err := os.ReadFile(filepath.Join(modelDir, filepath.FromSlash(p)))
		if err != nil || len(data) == 0 {
			continue
		}
		name := fmt.Sprintf("files/%d.bin", len(mf.Files))
		// Stored, not deflated: weights are dense, compressing them buys nothing
		// and costs real time on a multi-gigabyte fp16 export.
		fw, err := zw.CreateRaw(&zip.FileHeader{Name: name, Method: zip.Store,
			CRC32: crc32.ChecksumIEEE(data), CompressedSize64: uint64(len(data)), UncompressedSize64: uint64(len(data))})
		if err != nil {
			return mf, err
		}
		if _, err := fw.Write(data); err != nil {
			return mf, err
		}
		mf.Files = append(mf.Files, File{File: name, URL: root + p, Path: p})
		if d := DtypeOf[p]; d != "" && !seen[d] {
			seen[d] = true
			mf.Dtypes = append(mf.Dtypes, d)
		}
	}
	if len(mf.Files) == 0 {
		return mf, fmt.Errorf("nothing to pack in %s — is it converted?", modelDir)
	}

	body, err := json.Marshal(mf)
	if err != nil {
		return mf, err
	}
	fw, err := zw.Create("manifest.json")
	if err != nil {
		return mf, err
	}
	if _, err := fw.Write(body); err != nil {
		return mf, err
	}
	return mf, zw.Close()
}
