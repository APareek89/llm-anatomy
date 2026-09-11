import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { scalarBytes, matrixShape, bf16ToFloat32, computeStats, type ModelManifest } from '../src/data/hf.ts';
const root = resolve(import.meta.dirname, '..');
const m: ModelManifest = JSON.parse(await readFile(resolve(root, 'public/data/manifest.json'), 'utf8'));

test('every tensor matches its original pinned shard header and index exactly', async () => {
  const names = Object.keys(m.index.weight_map).sort(); assert.deepEqual(Object.keys(m.tensors).sort(), names);
  const headers = new Map<string, any>();
  for (const shard of Object.keys(m.shards)) headers.set(shard, JSON.parse(await readFile(resolve(root, 'public/data', `${shard}.header.json`), 'utf8')));
  for (const name of names) {
    const t = m.tensors[name], shard = m.index.weight_map[name], saved = headers.get(shard), source = saved.header[name];
    assert.ok(source, name); assert.equal(t.shard, shard, name); assert.deepEqual(t.shape, source.shape, name); assert.equal(t.dtype, source.dtype, name);
    assert.deepEqual(t.dataOffsets, source.data_offsets, name); assert.deepEqual(t.absoluteOffsets, source.data_offsets.map((o: number) => o + saved.headerLength + 8), name);
    assert.equal(t.params, source.shape.reduce((a: number, b: number) => a * b, 1), name); assert.equal(t.bytes, t.params * scalarBytes(t.dtype), name);
  }
});

test('parameter split is exhaustive and byte count equals safetensors index metadata', () => {
  assert.equal(m.totalBytes, m.index.metadata.total_size);
  assert.equal(m.totalParams, Object.values(m.tensors).reduce((n, t) => n + t.params, 0));
  assert.equal(m.totalBytes, Object.values(m.components).reduce((n, c) => n + c.bytes, 0));
  assert.equal(m.totalParams, Object.values(m.components).reduce((n, c) => n + c.params, 0));
  assert.equal(Object.keys(m.tensors).length, Object.values(m.components).reduce((n, c) => n + c.count, 0));
  assert.equal(m.components.other, undefined, 'Every tensor must have a verified component classification.');
});

test('all scene layer descriptors use tensor shapes and pattern from the real config', () => {
  const c = m.config.text_config; assert.equal(m.layers.length, c.num_hidden_layers); assert.equal(c.layer_types.length, c.num_hidden_layers);
  const encountered = new Set<string>();
  for (const layer of m.layers) {
    assert.equal(layer.kind, c.layer_types[layer.index]); assert.ok(layer.names.length > 0);
    assert.equal(layer.params, layer.names.reduce((n, name) => n + m.tensors[name].params, 0));
    for (const name of layer.names) { assert.equal(m.tensors[name].layer, layer.index); assert.ok(!encountered.has(name)); encountered.add(name); }
    const prefix = `model.language_model.layers.${layer.index}`;
    assert.deepEqual(m.tensors[`${prefix}.mlp.gate_proj.weight`].shape, [c.intermediate_size, c.hidden_size]);
    assert.deepEqual(m.tensors[`${prefix}.mlp.up_proj.weight`].shape, [c.intermediate_size, c.hidden_size]);
    assert.deepEqual(m.tensors[`${prefix}.mlp.down_proj.weight`].shape, [c.hidden_size, c.intermediate_size]);
    if (layer.kind === 'full_attention') {
      assert.deepEqual(m.tensors[`${prefix}.self_attn.q_proj.weight`].shape, [2 * c.num_attention_heads * c.head_dim, c.hidden_size]);
      assert.deepEqual(m.tensors[`${prefix}.self_attn.k_proj.weight`].shape, [c.num_key_value_heads * c.head_dim, c.hidden_size]);
      assert.deepEqual(m.tensors[`${prefix}.self_attn.v_proj.weight`].shape, [c.num_key_value_heads * c.head_dim, c.hidden_size]);
    } else {
      const projected = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim;
      assert.deepEqual(m.tensors[`${prefix}.linear_attn.in_proj_qkv.weight`].shape, [projected, c.hidden_size]);
    }
  }
  assert.equal(encountered.size, Object.values(m.tensors).filter(t => t.layer !== undefined).length);
});

test('downloaded Float32 files are in bounds and retain their checked binary hashes', async () => {
  assert.ok(m.tiles.length > 0, 'Run npm run prefetch to prepare real default tiles.');
  for (const tile of m.tiles) {
    const [rows, cols] = matrixShape(m.tensors[tile.name]); assert.ok(tile.row >= 0 && tile.row + tile.rows <= rows); assert.ok(tile.col >= 0 && tile.col + tile.cols <= cols);
    const bytes = await readFile(resolve(root, 'public', tile.path)); assert.equal(bytes.length, tile.rows * tile.cols * 4); assert.equal(bytes.length, tile.bytes);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), tile.sha256, tile.path);
    assert.equal(tile.stats.count, tile.rows * tile.cols); assert.equal(tile.stats.scope, 'loaded values only');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); for (let i = 0; i < bytes.length; i += 4) assert.ok(Number.isFinite(view.getFloat32(i, true)), `${tile.path}: nonfinite value`);
  }
});

test('default prefetch covers every layer 0/3 tensor, full Q/K head bands, embedding rows and LM rows', () => {
  for (const layer of [0, 3]) for (const name of m.layers[layer].names) assert.ok(m.tiles.some(t => t.name === name && t.row === 0 && t.col === 0), name);
  const c = m.config.text_config;
  assert.ok(m.tiles.some(t => t.name.endsWith('layers.3.self_attn.q_proj.weight') && t.rows === 2 * c.head_dim && t.cols === c.hidden_size));
  assert.ok(m.tiles.some(t => t.name.endsWith('layers.3.self_attn.k_proj.weight') && t.rows === c.head_dim && t.cols === c.hidden_size));
  const embeddings = m.tiles.filter(t => t.name === 'model.language_model.embed_tokens.weight'); assert.equal(embeddings.length, m.defaultTokenIds.length + 64);
  for (const id of m.defaultTokenIds) { assert.ok(embeddings.some(t => t.row === id && t.rows === 1 && t.cols === c.hidden_size)); assert.ok(m.tiles.some(t => t.name === 'lm_head.weight' && t.row === id && t.rows === 1)); }
});

test('the network ledger reconciles and remains below the explicit 150 MB budget', async () => {
  const ledger = JSON.parse(await readFile(resolve(root, 'public/data/download-ledger.json'), 'utf8'));
  assert.equal(ledger.requests.reduce((n: number, r: { bytes: number }) => n + r.bytes, 0), ledger.downloadedBytes);
  assert.equal(ledger.downloadedBytes, m.budget.downloadedBytes); assert.equal(m.budget.limitBytes, 150_000_000); assert.ok(ledger.downloadedBytes <= 150_000_000);
});

test('BF16 decoding respects the sign, exponent and mantissa with exact known bit patterns', () => {
  assert.deepEqual([...bf16ToFloat32(Uint8Array.from([0, 0, 128, 63, 128, 191, 0, 64]))], [0, 1, -1, 2]);
  const stats = computeStats(new Float32Array([-2, 0, 2])); assert.equal(stats.mean, 0); assert.equal(stats.min, -2); assert.equal(stats.max, 2); assert.equal(stats.nearZeroFraction, 1 / 3);
});
