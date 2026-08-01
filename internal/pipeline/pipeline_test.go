package pipeline

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestInlineChatTemplate(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "chat_template.jinja"), "{{ tools }}")
	writeFile(t, filepath.Join(dir, "tokenizer_config.json"), `{"model_max_length": 4096}`)

	var out bytes.Buffer
	if err := inlineChatTemplate(dir, &out); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg["chat_template"] != "{{ tools }}" {
		t.Fatalf("chat_template = %v", cfg["chat_template"])
	}
	if cfg["model_max_length"] == nil {
		t.Fatal("existing keys must be preserved")
	}
}

func TestInlineChatTemplateNoJinja(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "tokenizer_config.json"), `{"chat_template": "existing"}`)
	var out bytes.Buffer
	if err := inlineChatTemplate(dir, &out); err != nil {
		t.Fatal(err)
	}
}

func TestInlineChatTemplateAlreadyInline(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "chat_template.jinja"), "new")
	writeFile(t, filepath.Join(dir, "tokenizer_config.json"), `{"chat_template": "existing"}`)
	var out bytes.Buffer
	if err := inlineChatTemplate(dir, &out); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	var cfg map[string]any
	json.Unmarshal(raw, &cfg)
	if cfg["chat_template"] != "existing" {
		t.Fatal("inline template must not be overwritten")
	}
}

func TestPruneVariants(t *testing.T) {
	dir := t.TempDir()
	onnx := filepath.Join(dir, "onnx")
	for _, f := range []string{"model.onnx", "model.onnx_data", "model_q4.onnx", "model_quantized.onnx", "model_fp16.onnx"} {
		writeFile(t, filepath.Join(onnx, f), "x")
	}
	var out bytes.Buffer
	pruneVariants(dir, []string{"q4"}, &out)

	if _, err := os.Stat(filepath.Join(onnx, "model_q4.onnx")); err != nil {
		t.Fatal("requested q4 must be kept")
	}
	for _, f := range []string{"model.onnx", "model.onnx_data", "model_quantized.onnx", "model_fp16.onnx"} {
		if _, err := os.Stat(filepath.Join(onnx, f)); err == nil {
			t.Fatalf("%s should have been pruned", f)
		}
	}
}

func TestPruneVariantsKeepsMultiple(t *testing.T) {
	dir := t.TempDir()
	onnx := filepath.Join(dir, "onnx")
	for _, f := range []string{"model.onnx", "model_q4.onnx", "model_fp16.onnx"} {
		writeFile(t, filepath.Join(onnx, f), "x")
	}
	var out bytes.Buffer
	pruneVariants(dir, []string{"q4", "fp16"}, &out)
	for _, f := range []string{"model_q4.onnx", "model_fp16.onnx"} {
		if _, err := os.Stat(filepath.Join(onnx, f)); err != nil {
			t.Fatalf("%s must be kept", f)
		}
	}
	if _, err := os.Stat(filepath.Join(onnx, "model.onnx")); err == nil {
		t.Fatal("fp32 should have been pruned")
	}
}
