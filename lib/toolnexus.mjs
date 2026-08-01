// toolnexus — tiny tool-calling runtime for converted browser LLMs.
//
// Wraps Transformers.js (WASM/CPU) with an agentic tool loop:
// register tools as JSON schema + handler, chat, and toolnexus parses the
// model's tool calls, runs your handlers, feeds results back, and returns
// the final answer.
//
//   import { ToolNexus } from '/lib/toolnexus.mjs';
//   const nexus = await ToolNexus.load('Qwen/Qwen3-0.6B');
//   nexus.tool('get_weather', 'Get current weather for a city',
//     { city: { type: 'string' } },
//     async ({ city }) => ({ city, temperature_c: 31, condition: 'humid' }));
//   const answer = await nexus.chat('What is the weather in Chennai?',
//     { onToken: (t) => process(t), onToolCall: (c, r) => console.log(c, r) });
//
// Works against any model folder produced by llm-browser-converter.

const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const DTYPE_FILES = { q4: 'model_q4.onnx', q8: 'model_quantized.onnx', fp16: 'model_fp16.onnx', fp32: 'model.onnx' };
const DTYPE_ORDER = ['q4', 'q8', 'fp16', 'fp32'];

/** Parse tool calls from raw LLM output into [{name, arguments}].
 *  Handles Qwen/Hermes <tool_call>, Mistral [TOOL_CALLS], bare/fenced JSON. */
export function parseToolCalls(text) {
  const tryJSON = (s) => { try { return JSON.parse(s); } catch { return null; } };
  const calls = [];
  const push = (o) => {
    if (!o || typeof o !== 'object') return;
    const name = o.name || o.function?.name;
    let args = o.arguments ?? o.parameters ?? o.function?.arguments ?? {};
    if (typeof args === 'string') args = tryJSON(args) ?? args;
    if (name) calls.push({ name, arguments: args });
  };
  for (const m of text.matchAll(/<tool_call>\s*([\s\S]*?)\s*(?:<\/tool_call>|$)/g)) push(tryJSON(m[1]));
  if (!calls.length) {
    const m = text.match(/\[TOOL_CALLS\]\s*(\[[\s\S]*?\])/);
    if (m) for (const c of tryJSON(m[1]) ?? []) push(c);
  }
  if (!calls.length) {
    const body = (text.match(/```(?:json)?\s*([\s\S]*?)```/)?.[1] ?? text).trim();
    const parsed = tryJSON(body) ?? tryJSON(body.match(/\{[\s\S]*\}/)?.[0] ?? '');
    Array.isArray(parsed) ? parsed.forEach(push) : push(parsed);
  }
  return calls;
}

/** Strip <think>…</think> traces reasoning models prepend. */
const stripThinking = (t) => t.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();

export class ToolNexus {
  constructor(generator, dtype, TextStreamer) {
    this.generator = generator;
    this.dtype = dtype;
    this.TextStreamer = TextStreamer;
    this.tools = new Map();   // name -> {schema, handler}
    this.messages = [];       // running conversation
    this.maxRounds = 4;
  }

  /** Load a converted model. opts: {modelsUrl='/models/', dtype='auto', transformers} */
  static async load(modelId, opts = {}) {
    const tjs = opts.transformers ?? await import(/* @vite-ignore */ CDN);
    const { pipeline, env } = tjs;
    const base = opts.modelsUrl ?? '/models/';
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = new URL(base, typeof location !== 'undefined' ? location.href : 'file:///').href;

    let dtype = opts.dtype;
    if (!dtype || dtype === 'auto') {
      for (const d of DTYPE_ORDER) {
        try {
          const res = await fetch(`${env.localModelPath}${modelId}/onnx/${DTYPE_FILES[d]}`, { method: 'HEAD' });
          if (res.ok) { dtype = d; break; }
        } catch {}
      }
      if (!dtype) throw new Error(`no converted dtype found for ${modelId} under ${env.localModelPath}`);
    }
    const generator = await pipeline('text-generation', modelId, {
      dtype, device: opts.device ?? 'wasm',
      progress_callback: opts.onProgress,
    });
    return new ToolNexus(generator, dtype, tjs.TextStreamer);
  }

  /** Register a tool: JSON-schema properties + async handler(arguments).
   *  Properties accept shorthand: { city: 'string' } ≡ { city: { type: 'string' } }. */
  tool(name, description, properties, handler, { required } = {}) {
    const props = Object.fromEntries(Object.entries(properties ?? {})
      .map(([k, v]) => [k, typeof v === 'string' ? { type: v } : v]));
    this.tools.set(name, {
      schema: {
        type: 'function',
        function: {
          name, description,
          parameters: { type: 'object', properties: props, required: required ?? Object.keys(props) },
        },
      },
      handler,
    });
    return this;
  }

  /**
   * Evaluate user-written JS that defines tools (decorator-pattern API).
   * The code runs with `tool(name, description, schema, handler)` in scope:
   *
   *   tool('get_weather', 'Current weather for a city', { city: 'string' },
   *     async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json());
   *
   * Replaces all previously registered tools. Returns the registered names.
   */
  async evalTools(code) {
    this.tools.clear();
    const register = (...args) => this.tool(...args);
    const fn = new Function('tool', `'use strict';\nreturn (async () => {\n${code}\n})();`);
    await fn(register);
    return [...this.tools.keys()];
  }

  get toolSchemas() { return [...this.tools.values()].map(t => t.schema); }

  async #generate(opts) {
    const tok = this.generator.tokenizer;
    const prompt = tok.apply_chat_template(this.messages, {
      tools: this.tools.size ? this.toolSchemas : undefined,
      tokenize: false, add_generation_prompt: true, enable_thinking: false,
    });
    let streamer;
    if (opts.onToken && this.TextStreamer) {
      streamer = new this.TextStreamer(tok, { skip_prompt: true, callback_function: opts.onToken });
    }
    const out = await this.generator(prompt, {
      max_new_tokens: opts.maxNewTokens ?? 256, do_sample: false,
      return_full_text: false, streamer,
    });
    return out[0].generated_text;
  }

  /**
   * Chat with the automatic tool loop.
   * opts: {onToken(t), onToolCall(call, result), maxNewTokens}
   * Returns the model's final text answer.
   */
  async chat(userText, opts = {}) {
    // Small models happily hallucinate instead of calling tools — anchor them.
    if (this.tools.size && !this.messages.some(m => m.role === 'system')) {
      this.messages.unshift({
        role: 'system',
        content: this.systemPrompt ??
          'You have access to tools. When a question relates to a tool, you MUST call the tool instead of guessing or inventing data. Answer from tool results.',
      });
    }
    this.messages.push({ role: 'user', content: userText });
    for (let round = 0; round < this.maxRounds; round++) {
      const raw = await this.#generate(opts);
      const calls = parseToolCalls(raw).filter(c => this.tools.has(c.name));
      if (!calls.length) {
        const answer = stripThinking(raw);
        this.messages.push({ role: 'assistant', content: answer });
        return answer;
      }
      this.messages.push({ role: 'assistant', content: raw });
      for (const call of calls) {
        let result;
        try {
          result = await this.tools.get(call.name).handler(call.arguments ?? {});
        } catch (e) {
          result = { error: String(e.message ?? e) };
        }
        opts.onToolCall?.(call, result);
        this.messages.push({ role: 'tool', name: call.name, content: JSON.stringify(result) });
      }
    }
    const answer = 'tool loop exceeded ' + this.maxRounds + ' rounds';
    this.messages.push({ role: 'assistant', content: answer });
    return answer;
  }

  reset() { this.messages = []; }
  dispose() { return this.generator.dispose(); }
}
