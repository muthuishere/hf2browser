package hf

import "testing"

func TestChatTemplatesString(t *testing.T) {
	cfg := &TokenizerConfig{ChatTemplate: "{% if tools %}...{% endif %}"}
	got := chatTemplates(cfg)
	if len(got) != 1 || got[0] != "{% if tools %}...{% endif %}" {
		t.Fatalf("got %v", got)
	}
}

func TestChatTemplatesNamedList(t *testing.T) {
	cfg := &TokenizerConfig{ChatTemplate: []any{
		map[string]any{"name": "default", "template": "plain"},
		map[string]any{"name": "tool_use", "template": "{{ tools }}"},
	}}
	got := chatTemplates(cfg)
	if len(got) != 2 || got[1] != "{{ tools }}" {
		t.Fatalf("got %v", got)
	}
}

func TestChatTemplatesMissing(t *testing.T) {
	if got := chatTemplates(&TokenizerConfig{}); got != nil {
		t.Fatalf("expected nil, got %v", got)
	}
}

func TestNewFromEnvDefaults(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "")
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")
	t.Setenv("HF_TIMEOUT", "")
	c := NewFromEnv()
	if c.Endpoint != "https://huggingface.co" {
		t.Fatalf("endpoint = %q", c.Endpoint)
	}
	if c.Token != "" {
		t.Fatal("token should be empty")
	}
}

func TestNewFromEnvOverrides(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "https://mirror.example/")
	t.Setenv("HF_TIMEOUT", "5")
	c := NewFromEnv()
	if c.Endpoint != "https://mirror.example" {
		t.Fatalf("endpoint = %q (trailing slash should be trimmed)", c.Endpoint)
	}
	if c.HTTP.Timeout.Seconds() != 5 {
		t.Fatalf("timeout = %v", c.HTTP.Timeout)
	}
}
