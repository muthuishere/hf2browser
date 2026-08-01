// Verify a converted model runs on CPU via Transformers.js (same runtime the browser uses).
// For text-generation models, also tests TOOL CALLING per dtype and recommends the
// smallest dtype that still emits a tool call.
//
// usage: node verify/verify.mjs <model-id> [--task text-generation|feature-extraction] [--dtypes q4,q8,fp16]
import { pipeline, env } from '@huggingface/transformers';
import { parseToolCalls } from '../../../libs/browser-llm-nexus/dist/toolcalls.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
env.localModelPath = path.join(ROOT, 'models');
env.allowRemoteModels = false;

const args = process.argv.slice(2);
const modelId = args[0];
if (!modelId) {
  console.error('usage: node verify/verify.mjs <model-id> [--task t] [--dtypes q4,q8,fp16]');
  process.exit(1);
}
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i !== -1 ? args[i + 1] : dflt;
};
const task = opt('task', 'text-generation');

const DTYPE_FILES = { q4: 'model_q4.onnx', q8: 'model_quantized.onnx', fp16: 'model_fp16.onnx', fp32: 'model.onnx' };
const onnxDir = path.join(ROOT, 'models', modelId, 'onnx');
const available = (d) => fs.existsSync(path.join(onnxDir, DTYPE_FILES[d]));
const dtypes = opt('dtypes', 'q4,q8,fp16').split(',').filter(available);
if (!dtypes.length) {
  console.error(`FAIL: no requested dtype variants found in ${onnxDir}`);
  process.exit(1);
}

const TOOLS = [{
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get current weather for a city',
    parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
  },
}];

if (task === 'feature-extraction') {
  const dtype = dtypes[0];
  const pipe = await pipeline(task, modelId, { dtype, device: 'cpu' });
  const out = await pipe('Hello world', { pooling: 'mean', normalize: true });
  console.log(`[${dtype}] embedding dims: ${out.dims.join('x')}`);
  if (!out.data.length) process.exit(1);
  console.log('\nPASS: model works on CPU');
  process.exit(0);
}

const results = [];
for (const dtype of dtypes) {
  console.log(`\n== dtype ${dtype} ==`);
  const t0 = Date.now();
  let gen;
  try {
    gen = await pipeline('text-generation', modelId, { dtype, device: 'cpu' });
  } catch (e) {
    console.log(`LOAD FAILED: ${String(e.message || e).split('\n')[0].slice(0, 160)}`);
    results.push({ dtype, generates: false, toolCall: false, loadError: true });
    continue;
  }
  console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const chat = await gen([{ role: 'user', content: 'What is the capital of France? Answer in one sentence.' }],
    { max_new_tokens: 40, do_sample: false });
  const answer = chat[0].generated_text.at(-1).content.trim();
  const generates = answer.length > 0;
  console.log(`generation: ${generates ? 'ok' : 'EMPTY'} — "${answer.slice(0, 80)}"`);

  let toolCall = false;
  try {
    // enable_thinking:false — reasoning models (Qwen3 etc.) otherwise spend the
    // token budget on their thinking trace before emitting the tool call.
    const prompt = gen.tokenizer.apply_chat_template(
      [{ role: 'user', content: 'Get me the current weather in Chennai using the available tools.' }],
      { tools: TOOLS, tokenize: false, add_generation_prompt: true, enable_thinking: false });
    const out = await gen(prompt, { max_new_tokens: 128, do_sample: false, return_full_text: false });
    const calls = parseToolCalls(out[0].generated_text);
    // valid = parses to a structured object, right function, required arg present
    const call = calls.find(c => c.name === 'get_weather' && c.arguments && typeof c.arguments === 'object' && 'city' in c.arguments);
    toolCall = Boolean(call);
    console.log(`tool call:  ${toolCall
      ? 'ok — parsed ' + JSON.stringify(call)
      : calls.length
        ? 'emitted but invalid against schema: ' + JSON.stringify(calls)
        : 'NOT emitted (answered in prose)'}`);
  } catch (e) {
    console.log(`tool call:  template does not accept tools (${e.message.slice(0, 80)})`);
  }
  results.push({ dtype, generates, toolCall });
  await gen.dispose();
}

console.log('\n== summary ==');
for (const r of results) {
  const size = (fs.statSync(path.join(onnxDir, DTYPE_FILES[r.dtype])).size / 1e6).toFixed(0);
  const gen = r.loadError ? 'BROKEN (does not load)' : `generate:${r.generates ? 'pass' : 'FAIL'}  tools:${r.toolCall ? 'pass' : 'fail'}`;
  console.log(`${r.dtype.padEnd(5)} ${size.padStart(5)} MB  ${gen}`);
}
const best = results.find(r => r.generates && r.toolCall);
if (best) {
  console.log(`\nrecommended dtype for tool calling: ${best.dtype}`);
} else if (results.some(r => r.generates)) {
  console.log('\nWARNING: model generates but no tested dtype emitted a tool call.');
  console.log('For small models tool calling often needs fp16 (add --modes fp16) or a larger model (>=1.5B).');
}
if (!results.some(r => r.generates)) {
  console.error('\nFAIL: no dtype produced output');
  process.exit(1);
}
console.log('\nPASS: model works on CPU');
