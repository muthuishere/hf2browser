# hf2browser

[![release](https://img.shields.io/github/v/release/muthuishere/hf2browser?color=0a66c2)](https://github.com/muthuishere/hf2browser/releases/latest)
[![ci](https://github.com/muthuishere/hf2browser/actions/workflows/ci.yml/badge.svg)](https://github.com/muthuishere/hf2browser/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**Any Hugging Face LLM → running in the browser on plain CPU, with tool calling.**

🔗 **[Try it in your browser](https://muthuishere.github.io/hf2browser/try/)** — no install; a converted
model streams from the Hugging Face Hub into the page and calls JavaScript tools you can edit live.

One command gives you: HF Hub search with tool-calling detection → download → ONNX
conversion + q4 quantization → CPU verification (including a real tool-call test) →
a browser chat where the model calls **your JavaScript tools** — all local, no cloud,
no server-side inference. WebGPU is used when the browser has it and plain CPU/WASM
when it does not, so nothing here *requires* a GPU.

```bash
./hf2browser serve     # or `task up` from a checkout — build, serve, open the browser
```

Then take it with you: every converted model can be downloaded as one portable
`model.zip` **and** as a **single self-contained `chat.html`** — no build step, no
framework, no server of ours. Drop that one file on any static host and it runs.

The converted models are then run by
**[browser-llm-nexus](https://www.npmjs.com/package/browser-llm-nexus)**
[![npm](https://img.shields.io/npm/v/browser-llm-nexus?color=cb3837&logo=npm)](https://www.npmjs.com/package/browser-llm-nexus)
— the browser runtime this project pairs with (`npm install browser-llm-nexus`). It is a
separate, standalone package: tool calling, embeddings, RAG and offline bundles, on GPU or
CPU with the same API. This repo produces models for it; it does not depend on this repo.

## What's inside

| piece | language | job |
|---|---|---|
| `cmd/hf2browser`, `internal/` | Go | single-binary CLI + web UI/API server, orchestrates everything |
| `embed.go` | Go | compiles the pipeline, verifier and demo page into the binary, so a download is the whole product |
| `pytools/tjs_scripts/` | Python (auto-managed by `uv`) | the ONNX export + quantization pipeline (build-time only) |
| [`browser-llm-nexus`](https://github.com/muthuishere/browser-llm-nexus) | TypeScript (npm) | the browser runtime this produces models for: tool calling, embeddings, RAG, offline bundles, metrics |
| `verify/` | JavaScript (Node) | behavioral CPU verification: does it generate? does it *actually* emit tool calls? |
| `demo/` | JavaScript | chat page (GPU or CPU) with a live JS tool editor |
| `internal/server/standalone.html` | HTML | the generated single-file chat page handed to users |
| `site/` | HTML | the GitHub Pages landing page and its live in-browser demo |

Why three languages: Go is the product (fast single binary), Python is build-time
tooling only — the `optimum`/PyTorch export stack is the only thing on earth that can
trace HF architectures to ONNX (the architectures are *defined* in Python; a Go/cgo
port would mean reimplementing every model family by hand, to speed up a step that is
already >95% native C++). JS is the deployment target. Conversion speed is dominated
by download (accelerated with `hf_transfer`, HF's Rust engine) and native export math.

## Quick start

**Run the binary** — nothing to build, nothing to clone:

```bash
# grab the one for your platform from
# https://github.com/muthuishere/hf2browser/releases/latest — then
chmod +x hf2browser && ./hf2browser serve
```

Binaries are published for macOS (arm64/amd64), Linux (amd64/arm64) and Windows,
with a `SHA256SUMS` alongside them.

The conversion pipeline, the CPU verifier and the chat page are compiled into the
executable; it unpacks them into `~/.hf2browser` the first time it runs.

**Or build from source:**

```bash
git clone https://github.com/muthuishere/hf2browser && cd hf2browser
task up
```

In a checkout the binary uses the checkout — its `models/`, its `pytools/` — so
editing a file and rerunning is all there is to the dev loop.

Needed only for *converting*: [`uv`](https://docs.astral.sh/uv/) (drives the Python
export toolchain) and Node 18+ (the CPU verification step). Serving and chatting
need neither. Building from source additionally needs Go 1.22+ and
[`task`](https://taskfile.dev).

### Configuration

Optional. `hf2browser init` writes a `hf2browser.json` next to you:

```json
{
  "port": 0,                 // 0 = first free port from 8917
  "open_browser": true,
  "work_dir": "",            // "" = ~/.hf2browser
  "models_dir": "",          // "" = models/ beside the pipeline; point it at a big disk
  "dtype": "q4",             // default quantization
  "hf_endpoint": "",         // Hugging Face mirror
  "hf_timeout": 30
}
```

It is read from the working directory, then next to the binary, then
`~/.hf2browser/`; `--config <path>` overrides all three. Precedence is
**flags > environment > config file > defaults**, so nothing in here can
override something you said more specifically.

`HF_TOKEN` is deliberately *not* a config field — a token is a secret and stays in
the environment, never in a file that gets copied around. `HF_ENDPOINT`,
`HF_TIMEOUT` and `PORT` are also read from the environment.

`hf2browser where` prints which config, work directory and models directory are in
effect, so none of it has to be guessed.

In the UI that opens:

1. **Search** — live results with params, estimated q8/q4 download size, downloads,
   and a **tool-calling badge** per model (detected from the chat template). Filter to
   tool-calling only, cap by size, click column headers to sort.
2. **Convert** — streams the whole pipeline log live (SSE). Default output is **q4
   only** — the smallest browser variant; larger intermediates are pruned automatically.
3. **Chat** — converted models show a Chat button. The chat page auto-loads the best
   dtype and has a **live JS tool editor**: write tools in real JavaScript, hit Apply,
   and the model calls them.
4. **Take it anywhere** — every converted model gets a row with `model.zip` (the weights),
   `chat.html` (a self-contained page), and `code` (a snippet for your own app). See below.

### The CLI

Every command is a subcommand of the binary — there is no second vocabulary of
build tasks wrapping them:

```bash
hf2browser search "qwen instruct" --tools-only
hf2browser check   Qwen/Qwen3-0.6B
hf2browser convert Qwen/Qwen3-0.6B           # gate → export → q4 → CPU verify
hf2browser verify  Qwen/Qwen3-0.6B --dtypes q4
hf2browser serve                             # the UI
hf2browser init | where                      # config
```

The Taskfile has four entries — `up`, `build`, `test`, `dist` — and exists only to
build and launch.

## Running the converted models

Converted models are plain Transformers.js folders — load them with anything. The companion
runtime is **[browser-llm-nexus](https://www.npmjs.com/package/browser-llm-nexus)** on npm
(`npm install browser-llm-nexus`) — zero dependencies, TypeScript types included, GPU or CPU
with the same API, plus tool calling, embeddings, RAG and offline knowledge bundles.
This repo's own demo page and CPU verifier both run on it.

```ts
import { NexusChat } from 'browser-llm-nexus';

// the source is always explicit — this server's /models/ folder here
const nexus = await NexusChat.load({ base: '/models/', id: 'Qwen/Qwen3-0.6B' });

nexus.tool('get_weather', 'Get current weather for a city',
  { city: 'string' },                                     // shorthand JSON schema
  async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json());

nexus.on('token', t => render(t));                        // hooks, not callbacks
nexus.on('toolCall', (call, result) => console.log(call, result));

const answer = await nexus.chat('What is the weather in Chennai?');
console.log(nexus.metrics.summary());   // tokens/sec, tool_calls_ok, load_ms …
// → model emits the tool call → your handler runs → result fed back →
//   final grounded answer. Multi-round, multi-tool.
```

- Parses every common tool-call format (Qwen/Hermes `<tool_call>`, Mistral
  `[TOOL_CALLS]`, Llama bare JSON, fenced JSON, string-encoded arguments).
- Handles reasoning models (`enable_thinking: false`, `<think>` stripping).
- Injects a system prompt so small models call tools instead of hallucinating.
- `nexus.evalTools(code)` evaluates user-written JS tool definitions at runtime —
  the decorator pattern as a function (native `@decorator` syntax isn't in browsers yet;
  the same `tool(...)` call will work as one when it lands).

## Measured findings you should know

**Quantization vs tool calling.** On small models, aggressive quantization can destroy
tool-calling while leaving chat intact. Measured with identical greedy prompts:

| model | q4 | q8 | fp16 |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | tools ✗ (identical failure in the official onnx-community artifact) | tools ✗ | tools ✓ |
| Qwen3-0.6B | **tools ✓** | — | broken graph (known float16 converter gap) |

The verify step measures this per model instead of guessing, and recommends the
smallest dtype that passes. This is also why the chat pages pin the dtype the server
reports rather than probing for one: probing prefers fp16 on WebGPU, and a broken
fp16 graph fails inside ONNX Runtime as a bare numeric abort with no usable message. Rule of thumb: prefer newer model generations; Qwen3-0.6B
q4 (~1 GB) is the current sweet spot for browser tool calling.

**One binary, two modes.** Run it inside a checkout and it uses the checkout —
`pytools/`, `verify/`, `demo/`, `models/` — so editing a file and rerunning is the
whole dev loop. Run it anywhere else and it unpacks its embedded copies into
`~/.hf2browser` (refreshed automatically when the binary changes, left alone when it
has not). `hf2browser where` says which mode you are in.

**Architecture coverage.** The pinned toolchain (transformers 4.49 era) covers Llama,
Qwen2, Gemma, Phi, Mistral, SmolLM and ~100 more. Newer architectures (Qwen3, …)
auto-retry with a modern toolchain (`optimum-onnx`, `--no_post_process
--skip_validation` for >2 GiB fp32 graphs). Chat templates shipped as separate
`chat_template.jinja` files are inlined into `tokenizer_config.json` for
Transformers.js.

## Take it anywhere

Converting is only half the job — the point is to *ship* what came out. The server
exposes each converted model as plain static artifacts, and the UI's **Take it anywhere**
panel links all three per model:

| endpoint | what you get |
|---|---|
| `GET /api/model.zip?model=<id>&dtype=q4` | the weights as one portable archive (`manifest.json` + `files/N.bin`) |
| `GET /api/standalone.html?model=<id>&dtype=q4` | **a single HTML file** — a complete chat page with tool calling |
| `GET /models/<id>/…` | the raw Transformers.js folder, if you'd rather serve it yourself |

Model endpoints send `Access-Control-Allow-Origin: *`, so a page hosted somewhere else
can fetch them.

### The single HTML file

`chat.html` is one file with no build step, no framework and no bundler: it loads
[browser-llm-nexus](https://www.npmjs.com/package/browser-llm-nexus) from a CDN (pinned to
the version this repo tested with) and the model from whichever source you point it at.
Streaming, a live JS tool editor, the full tool-call loop, and tokens/sec are all in there.

Put it on GitHub Pages, S3, an intranet share, a USB stick — it never talks back to
hf2browser at runtime. One line decides where the weights come from:

```js
const SOURCE = { archive: 'https://your-host/model.zip' };      // as generated
// const SOURCE = { archive: fileTheUserPicked };               // the page's file picker
// const SOURCE = { base: './models/', id: 'Qwen/Qwen3-0.6B' }; // folder next to the page
// const SOURCE = { hub: 'onnx-community/Qwen3-0.6B-ONNX' };    // Hugging Face
```

Serve it over http(s) rather than opening it as `file://` — model weights live in the
Cache API, which browsers don't expose to `file://` origins. Any static server works
(`python3 -m http.server`).

Verified end to end: the generated page for `Qwen/Qwen3-0.6B` q4, served from a *different*
origin, picked WebGPU on its own, ran the full tool loop (`get_weather({city:"Chennai"})` →
handler → grounded answer), and loaded the same model again from a zip picked off disk.

## Offline knowledge systems

The archive restores into the browser cache, so everything after that is offline.
Compose it with a vector store and you have a full knowledge bundle
(`NexusKnowledge.exportZip({ includeModels: true })` → one zip, runs air-gapped).
One engine stack for embeddings *and* chat, GPU when present, CPU when not.

## The landing page

[muthuishere.github.io/hf2browser](https://muthuishere.github.io/hf2browser/) is a static
site built from `site/` — a short pitch, the download link, and **[a live demo](https://muthuishere.github.io/hf2browser/try/)**
that runs a converted model in the visitor's own browser, straight from the Hub, with the tool
loop working. It is not a hosted service: converting needs local compute and disk, so the page
exists to *show* the result and hand you the binary, not to do the work for you.

## Layout

```
apps/converter/              the converter app
  cmd/hf2browser/              Go CLI entrypoint
  internal/hf|pipeline|server  HF API, orchestration, web UI + JSON/SSE API
  pytools/tjs_scripts/         vendored conversion pipeline (Python, build-time)
  verify/                      Node CPU verification (generation + tool calls)
  demo/                        browser chat with live JS tool editor
  models/                      converted output (gitignored)
site/                        GitHub Pages: landing page + try/ live demo
```

The browser runtime lives in its own repo:
[muthuishere/browser-llm-nexus](https://github.com/muthuishere/browser-llm-nexus)
(installed here as an npm dependency of `apps/converter/verify`, and served to the
demo page at `/nexus/`).

## License

Vendored conversion scripts under `pytools/tjs_scripts/` are from
[huggingface/transformers.js](https://github.com/huggingface/transformers.js)
(Apache-2.0), with local patches (NumPy 2 compatibility, non-fatal checker,
`--no_post_process` flag). Everything else: MIT.
