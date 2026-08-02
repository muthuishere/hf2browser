# hf2browser

[![release](https://img.shields.io/github/v/release/muthuishere/hf2browser?color=0a66c2)](https://github.com/muthuishere/hf2browser/releases/latest)
[![ci](https://github.com/muthuishere/hf2browser/actions/workflows/ci.yml/badge.svg)](https://github.com/muthuishere/hf2browser/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**Run a Hugging Face model in the browser.**

Models on the Hub ship PyTorch weights. A browser can't load those — it needs ONNX,
quantized small enough to download into a tab. `hf2browser` does that conversion and hands
you the result as a folder to serve or a single zip to carry.

No cloud, no GPU required, no server-side inference. The model runs in the tab.

---

## Four commands

**1 — install.** One line, nothing to clone or build:

```bash
curl -fsSL https://raw.githubusercontent.com/muthuishere/hf2browser/main/install.sh | sh
```

<details>
<summary>Windows · direct download · from source</summary>

```powershell
irm https://raw.githubusercontent.com/muthuishere/hf2browser/main/install.ps1 | iex
```

Or take a binary straight from the [latest release](https://github.com/muthuishere/hf2browser/releases/latest)
— every asset has a `SHA256SUMS` beside it:

```bash
curl -fsSL -o hf2browser https://github.com/muthuishere/hf2browser/releases/latest/download/hf2browser-darwin-arm64
chmod +x hf2browser
```

From a checkout (needs Go 1.22+ and [`task`](https://taskfile.dev)):

```bash
git clone https://github.com/muthuishere/hf2browser && cd hf2browser
task serve
```

Inside a checkout the binary uses the checkout — its `models/`, its `pytools/` — so editing
a file and rerunning is the whole dev loop. Anywhere else it unpacks its embedded copies
into `~/.hf2browser`. `hf2browser where` tells you which mode you're in.
</details>

**2 — convert a model:**

```bash
hf2browser convert onnx-community/Qwen3-0.6B-ONNX --modes q4,q8
```

**3 — serve it:**

```bash
hf2browser serve
```

A page opens where you can search the Hub, convert, and download what you converted.

**4 — run it.** Each converted model gives you a `model.zip` — that one file is the whole
model. Drop it into the [live demo](https://muthuishere.github.io/browser-llm-nexus/demo/)
under *Archive file*, or load it in your own page:

```ts
import { NexusChat } from 'browser-llm-nexus';

const chat = await NexusChat.loadForTools({ archive: fileTheUserPicked });

chat.tool('get_weather', 'Get current weather for a city', { city: 'string' },
  async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json());

console.log(await chat.chat('What is the weather in Chennai?'));
```

The model is now running locally, in the tab, calling your JavaScript.

---

## Why not just use the existing tools?

Converting a model to ONNX is not new. What is missing everywhere else is **finding out
whether the thing you converted actually works** before you ship it to users.

**Pre-converted repos** (`onnx-community/*`, `Xenova/*`) are genuinely the fastest path when
your model is already there. A few hundred are. If yours isn't, or you fine-tuned it, you're
on your own.

**`optimum-cli export onnx`** is the underlying export and it's excellent — it's what runs
under the hood here. You pick the task, quantize separately, assemble the Transformers.js
layout, and set up Python yourself.

**transformers.js `convert.py`** is the canonical script for exactly this layout. Clone the
repo, build a Python env, run it per dtype. No verification of the result.

**WebLLM / MLC** gives faster inference than anything here, but a different compile
toolchain and model format — and it **requires WebGPU**, so there's no CPU fallback for the
locked-down laptop.

**llama.cpp / GGUF in WASM** has enormous model coverage in a different format entirely;
chat templates and tool calling are yours to wire up.

Three things this does that the others don't:

**It checks before you download.** `hf2browser check` reads the chat template and tells you
whether the model can do tool calling at all — before you spend 2 GB finding out.

**It checks after it converts.** The verify step runs the converted model on CPU and asks
it to make a real tool call. A model that exports cleanly and then can't call a tool is a
silent failure everywhere else; here it's a failed conversion.

**It hands over something portable.** A `model.zip` and a self-contained `chat.html`, not
just a folder of tensors — so "it works on my machine" turns into a file you can send
someone.

Honest scope: if `onnx-community` already has your model, use that — this exists for the
long tail, for your own fine-tunes, and for when you need a specific set of quantizations.

## Two things worth knowing before converting

**Convert more than one quantization.** `--modes` defaults to `q4` alone and *deletes* the
variants you didn't ask for. Which quantization can actually call a tool is
model-specific and does not transfer — Qwen2.5-0.5B works at q4 and is poor at q8, while
Qwen3-0.6B is fine at both and broken at fp16. Pass `--modes q4,q8,fp16` and let the
runtime pick on the user's machine. The cost is size: fp16 is roughly 4× q4.

**A tool-calling template is not the same as tool-calling ability.** SmolLM2-360M has the
template and fails every test. The search UI badges models actually
[measured working](https://muthuishere.github.io/browser-llm-nexus/verified-models/) — a
list kept with the runtime, including the combinations that fail.

## What you get per model

- **`models/<id>/`** — plain Transformers.js layout, serve from any static host
  (`GET /models/<id>/…`)
- **`model.zip`** — the whole model in one file; works offline, off a USB stick
  (`GET /api/model.zip?model=<id>`)
- **`chat.html`** — a self-contained chat page, no build step, no framework
  (`GET /api/standalone.html?model=<id>`)

Model endpoints send `Access-Control-Allow-Origin: *`, so a page hosted elsewhere can fetch
them. Nothing at runtime talks back to this server.

Serve `chat.html` over http(s) rather than opening it as `file://` — weights live in the
Cache API, which browsers don't expose to `file://` origins. `python3 -m http.server` is
enough. One line in it decides where the weights come from:

```js
const SOURCE = { archive: 'https://your-host/model.zip' };      // as generated
// const SOURCE = { archive: fileTheUserPicked };               // the page's file picker
// const SOURCE = { base: './models/', id: 'Qwen/Qwen3-0.6B' }; // folder next to the page
// const SOURCE = { hub: 'onnx-community/Qwen3-0.6B-ONNX' };    // Hugging Face
```

## The CLI

Every command is a subcommand of the binary — there's no second vocabulary of build tasks:

```bash
hf2browser search "qwen instruct" --tools-only   # find candidates, tool-calling badge each
hf2browser check   Qwen/Qwen3-0.6B               # template, size, task — before downloading
hf2browser convert Qwen/Qwen3-0.6B --modes q4,q8 # gate → export → quantize → CPU verify
hf2browser verify  Qwen/Qwen3-0.6B --dtypes q4   # re-run the behavioural check
hf2browser serve                                 # the UI
hf2browser init | where                          # config, and where things live
```

Converting needs [`uv`](https://docs.astral.sh/uv/) (drives the Python export toolchain)
and Node 18+ (the CPU verification step). Serving and running need neither.

<details>
<summary>Configuration</summary>

Optional. `hf2browser init` writes a `hf2browser.json`:

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

Read from the working directory, then next to the binary, then `~/.hf2browser/`;
`--config <path>` overrides all three. Precedence is **flags > environment > config file >
defaults**, so nothing here can override something you said more specifically.

`HF_TOKEN` is deliberately *not* a config field — a token is a secret and stays in the
environment, never in a file that gets copied around. `HF_ENDPOINT`, `HF_TIMEOUT` and
`PORT` are also read from the environment.
</details>

<details>
<summary>Serving to a hosted page (localhost + local network permission)</summary>

When a hosted (https) page fetches your converter on `localhost`, the browser treats it as
a public→loopback request and asks for local-network permission — allow it once and the
demo loads straight from your machine. If it's declined, or the browser has no such
prompt, download `model.zip` and use the demo's *Archive file* tab; the result is
identical. The server also sends `Access-Control-Allow-Private-Network` for the older
preflight-based opt-in, though in testing the browser permission was the deciding factor.
</details>

<details>
<summary>Architecture coverage</summary>

The pinned toolchain (transformers 4.49 era) covers Llama, Qwen2, Gemma, Phi, Mistral,
SmolLM and ~100 more. Newer architectures (Qwen3, …) auto-retry with a modern toolchain
(`optimum-onnx`, `--no_post_process --skip_validation` for >2 GiB fp32 graphs). Chat
templates shipped as separate `chat_template.jinja` files are inlined into
`tokenizer_config.json` for Transformers.js.
</details>

## How it's built

- **`cmd/hf2browser`, `internal/`** — Go: single-binary CLI plus the web UI/API server
- **`pytools/tjs_scripts/`** — Python (auto-managed by `uv`): ONNX export and quantization,
  build-time only
- **`verify/`** — JavaScript on Node: does it generate? does it *actually* emit tool calls?
- **`internal/server/standalone.html`** — the generated single-file chat page

Python is build-time tooling only: the `optimum`/PyTorch export stack is the only thing
that can trace HF architectures to ONNX, because those architectures are *defined* in
Python. A Go port would mean reimplementing every model family by hand to speed up a step
that is already >95% native C++. Conversion time is dominated by download (accelerated
with `hf_transfer`, HF's Rust engine) and native export math.

```
apps/converter/              the converter app
  cmd/hf2browser/              Go CLI entrypoint
  internal/hf|pipeline|server  HF API, orchestration, web UI + JSON/SSE API
  pytools/tjs_scripts/         vendored conversion pipeline (Python, build-time)
  verify/                      Node CPU verification (generation + tool calls)
  models/                      converted output (gitignored)
site/                        GitHub Pages landing page
```

## Running the models

Converted models are plain Transformers.js folders — load them with anything. The companion
runtime is **[browser-llm-nexus](https://www.npmjs.com/package/browser-llm-nexus)**
[![npm](https://img.shields.io/npm/v/browser-llm-nexus?color=cb3837&logo=npm)](https://www.npmjs.com/package/browser-llm-nexus)
(`npm install browser-llm-nexus`) — zero dependencies, TypeScript types, WebGPU or CPU with
the same API, plus tool calling, embeddings, RAG and offline knowledge bundles. It parses
every common tool-call format, handles reasoning models, and verifies that the model it
loaded can really call a tool.

It's a separate, standalone package: this repo produces models for it and does not depend
on it at runtime. Running models is deliberately not this tool's job — the demo already
does chat, tool calling and copy-paste code properly, and duplicating it here would mean
two of everything.

The archive restores into the browser cache, so everything after the first load is offline.
Compose it with a vector store and you get a full knowledge bundle
(`NexusKnowledge.exportZip({ includeModels: true })` → one zip, runs air-gapped).

## License

Vendored conversion scripts under `pytools/tjs_scripts/` are from
[huggingface/transformers.js](https://github.com/huggingface/transformers.js)
(Apache-2.0), with local patches (NumPy 2 compatibility, non-fatal checker,
`--no_post_process` flag). Everything else: MIT.
