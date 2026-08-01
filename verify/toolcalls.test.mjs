// node --test verify/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseToolCalls } from './toolcalls.mjs';

const CALL = { name: 'get_weather', arguments: { city: 'Chennai' } };

test('qwen/hermes <tool_call> format', () => {
  const out = parseToolCalls('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Chennai"}}\n</tool_call>');
  assert.deepEqual(out, [CALL]);
});

test('unclosed <tool_call> (generation cut off)', () => {
  const out = parseToolCalls('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Chennai"}}');
  assert.deepEqual(out, [CALL]);
});

test('mistral [TOOL_CALLS] format', () => {
  const out = parseToolCalls('[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Chennai"}}]');
  assert.deepEqual(out, [CALL]);
});

test('llama bare JSON with parameters key', () => {
  const out = parseToolCalls('{"name": "get_weather", "parameters": {"city": "Chennai"}}');
  assert.deepEqual(out, [CALL]);
});

test('markdown-fenced JSON with string-encoded arguments', () => {
  const out = parseToolCalls('```json\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Chennai\\"}"}\n```');
  assert.deepEqual(out, [CALL]);
});

test('openai-style nested function object', () => {
  const out = parseToolCalls('{"function": {"name": "get_weather", "arguments": {"city": "Chennai"}}}');
  assert.deepEqual(out, [CALL]);
});

test('prose yields no calls', () => {
  assert.deepEqual(parseToolCalls('The capital of France is Paris.'), []);
});

test('multiple tool calls', () => {
  const out = parseToolCalls(
    '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>');
  assert.equal(out.length, 2);
  assert.equal(out[1].name, 'b');
});

test('malformed JSON inside tool_call is skipped', () => {
  assert.deepEqual(parseToolCalls('<tool_call>{not json}</tool_call>'), []);
});
