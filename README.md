# hf2browser

**Any Hugging Face LLM → running in the browser on plain CPU, with tool calling.**

One command gives you: HF Hub search with tool-calling detection → download → ONNX
conversion + q4 quantization → CPU verification (including a real tool-call test) →
a browser chat where the model calls **your JavaScript tools** — all local, no WebGPU,
no cloud, no server-side inference.

```bash
task up          # build + serve, opens the browser, do everything from there
```

## What's inside

| piece | language | job |
|---|---|---|
| `cmd/hf2browser`, `internal/` | Go | single-binary CLI + web UI/API server, orchestrates everything |
| `pytools/tjs_scripts/` | Python (auto-managed by `uv`) | the ONNX export + quantization pipeline (build-time only) |
| `libs/browser-llm-nexus` | JavaScript | browser tool-calling runtime: register JS tools, agentic parse→execute→answer loop |
| `verify/` | JavaScript (Node) | behavioral CPU verification: does it generate? does it *actually* emit tool calls? |
| `demo/` | JavaScript | chat page on the WASM backend with a live JS tool editor |

Why three languages: Go is the product (fast single binary), Python is build-time
tooling only — the `optimum`/PyTorch export stack is the only thing on earth that can
trace HF architectures to ONNX (the architectures are *defined* in Python; a Go/cgo
port would mean reimplementing every model family by hand, to speed up a step that is
already >95% native C++). JS is the deployment target. Conversion speed is dominated
by download (accelerated with `hf_transfer`, HF's Rust engine) and native export math.

## Quick start

```bash
git clone https://github.com/muthuishere/hf2browser && cd hf2browser
task up
```

Requirements: Go 1.22+, [`uv`](https://docs.astral.sh/uv/), Node 18+,
[`task`](https://taskfile.dev). Everything else self-provisions on first run.

In the UI that opens:

1. **Search** — live results with params, estimated q8/q4 download size, downloads,
   and a **tool-calling badge** per model (detected from the chat template). Filter to
   tool-calling only, cap by size, click column headers to sort.
2. **Convert** — streams the whole pipeline log live (SSE). Default output is **q4
   only** — the smallest browser variant; larger intermediates are pruned automatically.
3. **Chat** — converted models show a Chat button. The chat page auto-loads the best
   dtype and has a **live JS tool editor**: write tools in real JavaScript, hit Apply,
   and the model calls them.

### CLI equivalents

```bash
task search  -- "qwen instruct" --tools-only
task check   -- Qwen/Qwen3-0.6B
task convert -- Qwen/Qwen3-0.6B              # gate → export → q4 → CPU verify
task verify  -- Qwen/Qwen3-0.6B --dtypes q4
task test                                     # Go + JS test suites
```

Environment (picked up automatically): `HF_TOKEN` (gated models, never printed),
`HF_ENDPOINT` (hub mirror), `HF_TIMEOUT` (seconds), `PORT` (serve port; otherwise
auto-picks a free one from 8917).

## browser-llm-nexus — the browser library

```js
import { NexusChat } from '/libs/browser-llm-nexus';

const nexus = await NexusChat.load('Qwen/Qwen3-0.6B');   // auto-picks best dtype

nexus.tool('get_weather', 'Get current weather for a city',
  { city: 'string' },                                     // shorthand JSON schema
  async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json());

const answer = await nexus.chat('What is the weather in Chennai?', {
  onToken: t => process.stdout.write(t),                  // streaming
  onToolCall: (call, result) => console.log(call, result),
});
// → model emits the tool call → toolnexus runs your handler → feeds the result
//   back → returns the final grounded answer. Multi-round, multi-tool.
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
smallest dtype that passes. Rule of thumb: prefer newer model generations; Qwen3-0.6B
q4 (~1 GB) is the current sweet spot for browser tool calling.

**Architecture coverage.** The pinned toolchain (transformers 4.49 era) covers Llama,
Qwen2, Gemma, Phi, Mistral, SmolLM and ~100 more. Newer architectures (Qwen3, …)
auto-retry with a modern toolchain (`optimum-onnx`, `--no_post_process
--skip_validation` for >2 GiB fp32 graphs). Chat templates shipped as separate
`chat_template.jinja` files are inlined into `tokenizer_config.json` for
Transformers.js.

## Integrating with offline knowledge systems

Built as the model-supply side for
[`offline-llm-knowledge-system`](../offline-llm-knowledge-system) (WebLLM/WebGPU today).
Its Cache-API round-trip (`embed-cache/` ↔ `transformers-cache`) works unchanged for
these models: add a `'transformersjs'` engine at the `useInference.ts` seam, capture
the chat model's cache into a `chat-cache/` section on export, restore on import.
Result: one engine stack (Transformers.js/WASM) for embeddings *and* chat, tool calling
included, zero WebGPU requirement.

## Layout

```
apps/converter/              the converter app
  cmd/hf2browser/              Go CLI entrypoint
  internal/hf|pipeline|server  HF API, orchestration, web UI + JSON/SSE API
  pytools/tjs_scripts/         vendored conversion pipeline (Python, build-time)
  verify/                      Node CPU verification (generation + tool calls)
  demo/                        browser chat with live JS tool editor
  models/                      converted output (gitignored)
libs/browser-llm-nexus/      the TypeScript browser library (npm-ready)
  src/chat|embed|rag|bundle|metrics|toolcalls|hooks
site/                        GitHub Pages landing (ships the built library)
```

## License

Vendored conversion scripts under `pytools/tjs_scripts/` are from
[huggingface/transformers.js](https://github.com/huggingface/transformers.js)
(Apache-2.0), with local patches (NumPy 2 compatibility, non-fatal checker,
`--no_post_process` flag). Everything else: MIT.
