// Package hf inspects and searches Hugging Face Hub models.
package hf

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

// Client talks to a Hugging Face Hub endpoint (or a mirror).
// Token is sent as a bearer header but never logged or printed.
type Client struct {
	Endpoint string
	Token    string
	HTTP     *http.Client
}

// NewFromEnv builds a client from HF_ENDPOINT, HF_TOKEN (or
// HUGGING_FACE_HUB_TOKEN) and HF_TIMEOUT (seconds), with sane defaults.
func NewFromEnv() *Client {
	endpoint := os.Getenv("HF_ENDPOINT")
	if endpoint == "" {
		endpoint = "https://huggingface.co"
	}
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	timeout := 30 * time.Second
	if s := os.Getenv("HF_TIMEOUT"); s != "" {
		if secs, err := strconv.Atoi(s); err == nil && secs > 0 {
			timeout = time.Duration(secs) * time.Second
		}
	}
	return &Client{
		Endpoint: strings.TrimRight(endpoint, "/"),
		Token:    token,
		HTTP:     &http.Client{Timeout: timeout},
	}
}

func (c *Client) get(path string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, c.Endpoint+path, nil)
	if err != nil {
		return nil, err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	return c.HTTP.Do(req)
}

func (c *Client) getJSON(path string, v any) error {
	resp, err := c.get(path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
		return json.NewDecoder(resp.Body).Decode(v)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("GET %s: %s (gated/private model? set HF_TOKEN)", path, resp.Status)
	case http.StatusNotFound:
		return fmt.Errorf("not found: %s", path)
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("GET %s: %s: %s", path, resp.Status, strings.TrimSpace(string(body)))
	}
}

// TokenizerConfig is the subset of tokenizer_config.json we care about.
type TokenizerConfig struct {
	ChatTemplate any `json:"chat_template"` // string, or []{name,template}
}

// ModelInfo is the subset of the HF model API response we care about.
type ModelInfo struct {
	ID           string   `json:"id"`
	PipelineTag  string   `json:"pipeline_tag"`
	Tags         []string `json:"tags"`
	Downloads    int64    `json:"downloads"`
	Likes        int64    `json:"likes"`
	Gated        any      `json:"gated"`
	SafetensorsD *struct {
		Total int64 `json:"total"`
	} `json:"safetensors"`
}

// Info fetches model metadata.
func (c *Client) Info(modelID string) (*ModelInfo, error) {
	var info ModelInfo
	if err := c.getJSON("/api/models/"+modelID, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

// SearchOptions filter a Hub model search.
type SearchOptions struct {
	Query    string
	Tags     []string // e.g. conversational, text-generation-inference
	Pipeline string   // e.g. text-generation
	Sort     string   // downloads | likes | lastModified | trendingScore
	Limit    int
}

// Search queries the Hub model index.
func (c *Client) Search(opts SearchOptions) ([]ModelInfo, error) {
	q := url.Values{}
	if opts.Query != "" {
		q.Set("search", opts.Query)
	}
	for _, t := range opts.Tags {
		q.Add("filter", t)
	}
	if opts.Pipeline != "" {
		q.Set("pipeline_tag", opts.Pipeline)
	}
	sort := opts.Sort
	if sort == "" {
		sort = "downloads"
	}
	q.Set("sort", sort)
	q.Set("direction", "-1")
	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	q.Set("limit", strconv.Itoa(limit))
	for _, f := range []string{"safetensors", "downloads", "likes", "pipeline_tag", "gated"} {
		q.Add("expand[]", f)
	}
	var out []ModelInfo
	if err := c.getJSON("/api/models?"+q.Encode(), &out); err != nil {
		return nil, err
	}
	return out, nil
}

// chatTemplates flattens the chat_template field (string or named-template list).
func chatTemplates(cfg *TokenizerConfig) []string {
	switch t := cfg.ChatTemplate.(type) {
	case string:
		return []string{t}
	case []any:
		var out []string
		for _, e := range t {
			if m, ok := e.(map[string]any); ok {
				if s, ok := m["template"].(string); ok {
					out = append(out, s)
				}
			}
		}
		return out
	}
	return nil
}

// SupportsToolCalling reports whether the model's chat template handles tools.
// A template that references the `tools` variable can format tool definitions
// and tool-call turns — the standard HF signal for tool-calling support.
func (c *Client) SupportsToolCalling(modelID string) (bool, error) {
	var cfg TokenizerConfig
	if err := c.getJSON(fmt.Sprintf("/%s/resolve/main/tokenizer_config.json", modelID), &cfg); err != nil {
		return false, err
	}
	for _, tpl := range chatTemplates(&cfg) {
		if strings.Contains(tpl, "tools") {
			return true, nil
		}
	}
	// Some repos ship the template in a separate chat_template.jinja / .json.
	for _, name := range []string{"chat_template.jinja", "chat_template.json"} {
		resp, err := c.get(fmt.Sprintf("/%s/resolve/main/%s", modelID, name))
		if err != nil {
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK && strings.Contains(string(body), "tools") {
			return true, nil
		}
	}
	return false, nil
}
