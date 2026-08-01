// Package pipeline drives the ONNX conversion and CPU verification steps.
package pipeline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Root is the repo root (directory containing vendor/tjs_scripts).
func Root() (string, error) {
	exe, err := os.Executable()
	if err == nil {
		if dir := findRoot(filepath.Dir(exe)); dir != "" {
			return dir, nil
		}
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if dir := findRoot(cwd); dir != "" {
		return dir, nil
	}
	return "", fmt.Errorf("cannot locate repo root (vendor/tjs_scripts) from executable or cwd")
}

func findRoot(start string) string {
	dir := start
	for {
		if _, err := os.Stat(filepath.Join(dir, "pytools", "tjs_scripts")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func run(dir string, out io.Writer, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	// hf_transfer = HF's Rust download engine; the biggest time slice for
	// large models is the weight download, and this saturates bandwidth.
	cmd.Env = append(os.Environ(), "HF_HUB_ENABLE_HF_TRANSFER=1")
	cmd.Stdout = out
	cmd.Stderr = out
	fmt.Fprintln(out, "+", name, args)
	return cmd.Run()
}

// Convert exports the model to ONNX with quantized variants under models/,
// streaming subprocess output to out. modes are quantization modes
// (q8, q4, fp16, ...); extra is passed through to the underlying converter.
// HF_TOKEN / HF_ENDPOINT from the environment are inherited by the exporter.
//
// It first runs the pinned (battle-tested) dependency set; if that fails
// because the architecture is newer than the pinned transformers knows,
// it automatically retries with requirements-modern.txt.
func Convert(root string, out io.Writer, modelID string, modes []string, extra []string) error {
	buildArgs := func(reqs string) []string {
		args := []string{
			"run", "--python", "3.11",
			"--with-requirements", filepath.Join("pytools", "tjs_scripts", reqs),
			"python", "-m", "pytools.tjs_scripts.convert",
			"--model_id", modelID,
			"--quantize",
			"--output_parent_dir", filepath.Join(root, "models"),
		}
		if len(modes) > 0 {
			args = append(args, "--modes")
			args = append(args, modes...)
		}
		return append(args, extra...)
	}

	var buf bytes.Buffer
	err := run(root, io.MultiWriter(out, &buf), "uv", buildArgs("requirements.txt")...)
	if err != nil && strings.Contains(buf.String(), "does not recognize this architecture") {
		fmt.Fprintln(out, "\narchitecture too new for the pinned toolchain — retrying with requirements-modern.txt")
		// no_post_process/skip_validation: newer fp32 graphs routinely exceed
		// protobuf's 2GiB serialize limit in optimum's merge/validate steps.
		args := append(buildArgs("requirements-modern.txt"), "--no_post_process", "--skip_validation")
		err = run(root, out, "uv", args...)
	}
	if err == nil {
		modelDir := filepath.Join(root, "models", modelID)
		if e := inlineChatTemplate(modelDir, out); e != nil {
			fmt.Fprintf(out, "warning: could not inline chat template: %v\n", e)
		}
		pruneVariants(modelDir, modes, out)
	}
	return err
}

// variantFiles maps quantization mode -> onnx filename(s) in the output.
var variantFiles = map[string][]string{
	"q4":   {"model_q4.onnx"},
	"q8":   {"model_quantized.onnx", "model_q8.onnx"},
	"int8": {"model_int8.onnx"},
	"fp16": {"model_fp16.onnx"},
	"fp32": {"model.onnx", "model.onnx_data"},
}

// pruneVariants deletes onnx variants that weren't requested — most
// importantly the fp32 export (model.onnx + external data), which is only an
// intermediate and can be 4-30x the size of the quantized files.
func pruneVariants(modelDir string, requested []string, out io.Writer) {
	keep := map[string]bool{}
	for _, m := range requested {
		for _, f := range variantFiles[strings.TrimSpace(m)] {
			keep[f] = true
		}
	}
	onnxDir := filepath.Join(modelDir, "onnx")
	entries, err := os.ReadDir(onnxDir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if keep[e.Name()] {
			continue
		}
		p := filepath.Join(onnxDir, e.Name())
		if info, err := os.Stat(p); err == nil {
			if err := os.Remove(p); err == nil {
				fmt.Fprintf(out, "pruned %s (%.0f MB, not in requested modes)\n", e.Name(), float64(info.Size())/1e6)
			}
		}
	}
}

// inlineChatTemplate copies a standalone chat_template.jinja into
// tokenizer_config.json's chat_template key. Newer transformers versions save
// the template as a separate file, which Transformers.js does not read.
func inlineChatTemplate(modelDir string, out io.Writer) error {
	tpl, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja"))
	if err != nil {
		return nil // no separate template file — nothing to do
	}
	cfgPath := filepath.Join(modelDir, "tokenizer_config.json")
	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return err
	}
	if _, ok := cfg["chat_template"]; ok {
		return nil // already inline
	}
	cfg["chat_template"] = string(tpl)
	updated, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(cfgPath, updated, 0o644); err != nil {
		return err
	}
	fmt.Fprintln(out, "inlined chat_template.jinja into tokenizer_config.json (Transformers.js compatibility)")
	return nil
}

// Verify loads the converted model with Transformers.js on CPU, checks it
// generates, and (for text-generation) tests tool calling per dtype variant.
func Verify(root string, out io.Writer, modelID, task, dtypes string) error {
	verifyDir := filepath.Join(root, "verify")
	if _, err := os.Stat(filepath.Join(verifyDir, "node_modules")); err != nil {
		if err := run(verifyDir, out, "npm", "install", "--silent"); err != nil {
			return fmt.Errorf("npm install: %w", err)
		}
	}
	return run(root, out, "node", filepath.Join("verify", "verify.mjs"), modelID, "--task", task, "--dtypes", dtypes)
}
