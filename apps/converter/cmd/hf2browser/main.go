// hf2browser — convert Hugging Face LLMs to browser-ready CPU (ONNX/WASM) models.
//
//	hf2browser search  <query> [flags]       search the HF Hub
//	hf2browser check   <model-id>            inspect: tool-calling, size, task
//	hf2browser convert <model-id> [flags]    check + convert + CPU-verify
//	hf2browser verify  <model-id> [flags]    CPU-verify an already converted model
//	hf2browser serve   [flags]               web UI: search → convert → chat
//
// Environment: HF_TOKEN (gated/private models), HF_ENDPOINT (hub mirror),
// HF_TIMEOUT (seconds, API calls).
package main

import (
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/muthuishere/hf2browser/internal/hf"
	"github.com/muthuishere/hf2browser/internal/pipeline"
	"github.com/muthuishere/hf2browser/internal/server"
)

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  hf2browser search  <query> [--tags a,b] [--pipeline text-generation] [--sort downloads] [--limit 20] [--check-tools] [--tools-only]
  hf2browser check   <model-id>
  hf2browser convert <model-id> [--modes q8,q4,fp16] [--task auto] [--force] [--skip-verify]
  hf2browser verify  <model-id> [--task text-generation] [--dtypes q4,q8,fp16]
  hf2browser serve   [--port 8917]   (auto-picks a free port; $PORT respected)

env: HF_TOKEN (gated models), HF_ENDPOINT (hub mirror), HF_TIMEOUT (seconds)
convert refuses models whose chat template lacks tool-calling support (override with --force).`)
	os.Exit(1)
}

// listenAuto binds the requested port, or scans 8917..8937 for a free one
// (0 = no preference). Returns the listener and the port actually bound.
func listenAuto(preferred int) (net.Listener, int, error) {
	candidates := []int{}
	if preferred != 0 {
		candidates = append(candidates, preferred)
	} else {
		for p := 8917; p <= 8937; p++ {
			candidates = append(candidates, p)
		}
	}
	for _, p := range candidates {
		ln, err := net.Listen("tcp", fmt.Sprintf(":%d", p))
		if err == nil {
			return ln, p, nil
		}
	}
	if preferred != 0 {
		return nil, 0, fmt.Errorf("port %d is busy", preferred)
	}
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		return nil, 0, err
	}
	return ln, ln.Addr().(*net.TCPAddr).Port, nil
}

// openBrowser opens url in the default browser (best-effort).
func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", a...)
	os.Exit(1)
}

func check(client *hf.Client, modelID string) (toolCalling bool) {
	info, err := client.Info(modelID)
	if err != nil {
		fatal("model lookup failed: %v", err)
	}
	tools, err := client.SupportsToolCalling(modelID)
	if err != nil {
		fatal("tokenizer_config lookup failed: %v", err)
	}
	fmt.Printf("model:         %s\n", info.ID)
	fmt.Printf("pipeline tag:  %s\n", info.PipelineTag)
	if info.SafetensorsD != nil {
		fmt.Printf("parameters:    %.1fM\n", float64(info.SafetensorsD.Total)/1e6)
	}
	fmt.Printf("tool calling:  %v\n", tools)
	if gated, ok := info.Gated.(string); ok && gated != "" {
		fmt.Printf("gated:         %s (you may need HF_TOKEN)\n", gated)
	}
	return tools
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	client := hf.NewFromEnv()
	root, err := pipeline.Root()
	if err != nil {
		fatal("%v", err)
	}

	switch cmd {
	case "search":
		fs := flag.NewFlagSet("search", flag.ExitOnError)
		tags := fs.String("tags", "", "filter tags (comma-separated)")
		pipe := fs.String("pipeline", "text-generation", "pipeline tag filter")
		sort := fs.String("sort", "downloads", "sort: downloads|likes|lastModified|trendingScore")
		limit := fs.Int("limit", 20, "max results")
		checkTools := fs.Bool("check-tools", false, "also check tool-calling support per result (slower)")
		toolsOnly := fs.Bool("tools-only", false, "show only models with tool-calling support (implies --check-tools)")
		var query string
		if len(os.Args) > 2 && !strings.HasPrefix(os.Args[2], "-") {
			query = os.Args[2]
			fs.Parse(os.Args[3:])
		} else {
			fs.Parse(os.Args[2:])
		}
		opts := hf.SearchOptions{Query: query, Pipeline: *pipe, Sort: *sort, Limit: *limit}
		if *tags != "" {
			opts.Tags = strings.Split(*tags, ",")
		}
		results, err := client.Search(opts)
		if err != nil {
			fatal("search failed: %v", err)
		}
		for _, m := range results {
			params := "     —"
			if m.SafetensorsD != nil && m.SafetensorsD.Total > 0 {
				if t := m.SafetensorsD.Total; t >= 1e9 {
					params = fmt.Sprintf("%5.1fB", float64(t)/1e9)
				} else {
					params = fmt.Sprintf("%5.0fM", float64(t)/1e6)
				}
			}
			line := fmt.Sprintf("%-55s %s params %10d downloads", m.ID, params, m.Downloads)
			if *checkTools || *toolsOnly {
				tools, err := client.SupportsToolCalling(m.ID)
				if *toolsOnly && (err != nil || !tools) {
					continue
				}
				if err == nil {
					if tools {
						line += "  [tool calling]"
					} else {
						line += "  [no tools]"
					}
				}
			}
			fmt.Println(line)
		}

	case "check":
		if len(os.Args) < 3 {
			usage()
		}
		check(client, os.Args[2])

	case "convert":
		if len(os.Args) < 3 {
			usage()
		}
		modelID := os.Args[2]
		fs := flag.NewFlagSet("convert", flag.ExitOnError)
		modes := fs.String("modes", "q4", "quantization modes (comma-separated); q4 = smallest browser variant")
		task := fs.String("task", "", "optimum task override (default: auto)")
		force := fs.Bool("force", false, "convert even without tool-calling support")
		skipVerify := fs.Bool("skip-verify", false, "skip the CPU generation test")
		fs.Parse(os.Args[3:])

		if !check(client, modelID) {
			if !*force {
				fatal("%s has no tool-calling chat template; pass --force to convert anyway", modelID)
			}
			fmt.Fprintln(os.Stderr, "warning: no tool-calling support, converting anyway (--force)")
		}

		var extra []string
		if *task != "" {
			extra = append(extra, "--task", *task)
		}
		fmt.Println("\n== converting to ONNX (this downloads the model on first run) ==")
		if err := pipeline.Convert(root, os.Stdout, modelID, strings.Split(*modes, ","), extra); err != nil {
			fatal("conversion failed: %v", err)
		}
		if !*skipVerify {
			fmt.Println("\n== verifying on CPU (generation + tool calling per dtype) ==")
			if err := pipeline.Verify(root, os.Stdout, modelID, "text-generation", *modes); err != nil {
				fatal("verification failed: %v", err)
			}
		}
		fmt.Printf("\ndone: models/%s (serve it and load with Transformers.js)\n", modelID)

	case "verify":
		if len(os.Args) < 3 {
			usage()
		}
		modelID := os.Args[2]
		fs := flag.NewFlagSet("verify", flag.ExitOnError)
		task := fs.String("task", "text-generation", "pipeline task")
		dtypes := fs.String("dtypes", "q4,q8,fp16", "dtype variants to test (comma-separated)")
		fs.Parse(os.Args[3:])
		if err := pipeline.Verify(root, os.Stdout, modelID, *task, *dtypes); err != nil {
			fatal("verification failed: %v", err)
		}

	case "serve":
		fs := flag.NewFlagSet("serve", flag.ExitOnError)
		port := fs.Int("port", 0, "port (default: $PORT, else first free port from 8917)")
		noOpen := fs.Bool("no-open", false, "do not open the browser automatically")
		fs.Parse(os.Args[2:])
		if *port == 0 {
			if p, err := strconv.Atoi(os.Getenv("PORT")); err == nil {
				*port = p
			}
		}
		ln, actualPort, err := listenAuto(*port)
		if err != nil {
			fatal("%v", err)
		}
		srv := server.New(root, client)
		url := fmt.Sprintf("http://localhost:%d/", actualPort)
		fmt.Printf("UI:     %s\nmodels: %smodels/\n", url, url)
		if !*noOpen {
			openBrowser(url)
		}
		if client.Token != "" {
			fmt.Println("HF_TOKEN detected — gated models available")
		}
		if err := http.Serve(ln, srv.Handler()); err != nil {
			fatal("%v", err)
		}

	default:
		usage()
	}
}
