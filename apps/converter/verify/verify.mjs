// Verify a converted model on CPU — through browser-llm-nexus, the same library
// the browser uses. If this passes, the model works in a page.
//
// usage: node verify/verify.mjs <model-id> [--task text-generation|feature-extraction] [--dtypes q4,q8,fp16]
import { NexusChat, NexusEmbedder } from 'browser-llm-nexus';
import * as transformers from '@huggingface/transformers';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const args = process.argv.slice(2);
const modelId = args[0];
if (!modelId) {
  console.error('usage: node verify/verify.mjs <model-id> [--task t] [--dtypes q4,q8,fp16] [--models dir]');
  process.exit(1);
}
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i !== -1 ? args[i + 1] : dflt;
};
const task = opt('task', 'text-generation');
// Models may live outside this tree (see hf2browser.json's models_dir).
const MODELS = path.resolve(opt('models', path.join(ROOT, 'models')));

const DTYPE_FILES = { q4: 'model_q4.onnx', q8: 'model_quantized.onnx', fp16: 'model_fp16.onnx', fp32: 'model.onnx' };
const onnxDir = path.join(MODELS, modelId, 'onnx');
const available = (d) => fs.existsSync(path.join(onnxDir, DTYPE_FILES[d]));
const dtypes = opt('dtypes', 'q4,q8,fp16').split(',').filter(available);
if (!dtypes.length) {
  console.error(`FAIL: no requested dtype variants found in ${onnxDir}`);
  process.exit(1);
}

// Explicit source: this repo's models/ folder. Nothing is assumed.
const source = { base: MODELS, id: modelId };
const runtime = { transformers, device: 'cpu' };

if (task === 'feature-extraction') {
  const embedder = await NexusEmbedder.load(source, { ...runtime, dtype: dtypes[0] });
  const [vector] = await embedder.embedBatch(['Hello world']);
  console.log(`[${dtypes[0]}] embedding dims: ${vector.length}`);
  if (!vector.length) process.exit(1);
  console.log('\nPASS: model works on CPU');
  process.exit(0);
}

const results = [];
for (const dtype of dtypes) {
  console.log(`\n== dtype ${dtype} ==`);
  const t0 = Date.now();
  let chat;
  try {
    chat = await NexusChat.load(source, { ...runtime, dtype });
  } catch (e) {
    console.log(`LOAD FAILED: ${String(e.message || e).split('\n')[0].slice(0, 160)}`);
    results.push({ dtype, generates: false, toolCall: false, loadError: true });
    continue;
  }
  console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const answer = (await chat.chat('What is the capital of France? Answer in one sentence.', { maxNewTokens: 40 })).trim();
  const generates = answer.length > 0;
  console.log(`generation: ${generates ? 'ok' : 'EMPTY'} — "${answer.slice(0, 80)}"`);

  // Tool calling through the library's own loop: register a tool, ask for it,
  // and see whether the handler actually ran with the right argument.
  let called = null;
  chat.reset();
  chat.tool('get_weather', 'Get current weather for a city', { city: 'string' }, async (a) => {
    called = a;
    return { city: a.city, temperature_c: 31, condition: 'humid' };
  });
  await chat.chat('Get me the current weather in Chennai using the available tools.', { maxNewTokens: 128 });
  const toolCall = Boolean(called && typeof called.city === 'string');
  console.log(`tool call:  ${toolCall ? `ok — handler ran with ${JSON.stringify(called)}` : 'NOT emitted (answered in prose)'}`);

  results.push({ dtype, generates, toolCall, metrics: chat.metrics.summary() });
  await chat.dispose();
}

console.log('\n== summary ==');
for (const r of results) {
  const size = (fs.statSync(path.join(onnxDir, DTYPE_FILES[r.dtype])).size / 1e6).toFixed(0);
  const status = r.loadError
    ? 'BROKEN (does not load)'
    : `generate:${r.generates ? 'pass' : 'FAIL'}  tools:${r.toolCall ? 'pass' : 'fail'}` +
      (r.metrics?.tokens_per_second ? `  ${r.metrics.tokens_per_second} tok/s` : '');
  console.log(`${r.dtype.padEnd(5)} ${size.padStart(5)} MB  ${status}`);
}
const best = results.find((r) => r.generates && r.toolCall);
if (best) {
  console.log(`\nrecommended dtype for tool calling: ${best.dtype}`);
} else if (results.some((r) => r.generates)) {
  console.log('\nWARNING: model generates but no tested dtype emitted a tool call.');
  console.log('For small models tool calling often needs fp16 (add --modes fp16) or a larger model (>=1.5B).');
}
if (!results.some((r) => r.generates)) {
  console.error('\nFAIL: no dtype produced output');
  process.exit(1);
}
console.log('\nPASS: model works on CPU');
