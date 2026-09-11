import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { loadModelData, loadTile, loadPrefetchedTile, downloadStatus, validateRangeHeaders, DOWNLOAD_BUDGET, REVISION, type ModelManifest, type TensorMeta } from '../src/data/hf.ts';

test('Range validation rejects absent, wrong, encoded, malformed and oversize responses', () => {
  assert.doesNotThrow(() => validateRangeHeaders(new Headers({ 'Content-Range': 'bytes 10-19/100', 'Content-Length': '10' }), 10, 10));
  for (const headers of [{}, { 'Content-Range': 'bytes 10-19/*' }, { 'Content-Range': 'bytes 10-19/19' }, { 'Content-Range': 'bytes 10-20/100' }, { 'Content-Range': 'bytes 10-19/100 extra' }, { 'Content-Range': 'bytes 10-19/100', 'Content-Length': '11' }, { 'Content-Range': 'bytes 10-19/100', 'Content-Encoding': 'gzip' }]) {
    assert.throws(() => validateRangeHeaders(new Headers(headers as Record<string, string>), 10, 10));
  }
});

test('exact sparse rows are verified locally; malformed counters, coordinates and concurrent overspend fail closed', async () => {
  const root = resolve(import.meta.dirname, '..');
  const real: ModelManifest = JSON.parse(await readFile(resolve(root, 'public/data/manifest.json'), 'utf8'));
  const originalFetch = globalThis.fetch;
  const storageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const usage = new Map<string, string>();
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: (k: string) => usage.get(k) ?? null, setItem: (k: string, v: string) => usage.set(k, v) } });
  const byteLength = 256 * 256 * 2;
  const makeProbe = (name: string): TensorMeta => ({ name, shape: [256, 256], dtype: 'BF16', shard: `${name}.safetensors`, dataOffsets: [0, byteLength], absoluteOffsets: [128, 128 + byteLength], params: 65536, bytes: byteLength, component: 'test-fixture' });
  const fixture: ModelManifest = structuredClone(real);
  fixture.tensors['probe.a'] = makeProbe('probe.a'); fixture.tensors['probe.b'] = makeProbe('probe.b');
  fixture.budget.downloadedBytes = DOWNLOAD_BUDGET - byteLength;
  let remoteCalls = 0, localCalls = 0, serveBadHash = false, failRemote = true;
  let started!: () => void, release!: () => void;
  let startedPromise = new Promise<void>(r => { started = r; });
  let gate = new Promise<void>(r => { release = r; });
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input);
    if (url.endsWith('/data/manifest.json')) return new Response(JSON.stringify(fixture), { headers: { 'Content-Type': 'application/json' } });
    if (url.startsWith('/weights/')) {
      localCalls++; const b = await readFile(resolve(root, 'public', url.slice(1))); const copied = new Uint8Array(b);
      if (serveBadHash) copied[0] ^= 1;
      return new Response(copied);
    }
    if (url.startsWith('https://huggingface.co/')) {
      remoteCalls++; if (failRemote) return new Response(null, { status: 200 });
      started(); await gate;
      const raw = new Uint8Array(byteLength); for (let i = 0; i < raw.length; i += 2) { raw[i] = 128; raw[i + 1] = 63; }
      return new Response(raw, { status: 206, headers: { 'Content-Range': `bytes 128-${128 + byteLength - 1}/${128 + byteLength}`, 'Content-Length': String(byteLength) } });
    }
    throw new Error(`Unexpected network access in isolated test: ${url}`);
  }) as typeof fetch;
  try {
    await loadModelData();
    const f = fixture.tiles.find(t => t.name.endsWith('embed_tokens.weight') && t.row === fixture.defaultTokenIds[0])!;
    const tile = await loadPrefetchedTile(f);
    assert.equal(tile.row, fixture.defaultTokenIds[0]); assert.equal(tile.rows, 1); assert.equal(tile.cols, fixture.config.text_config.hidden_size); assert.equal(tile.source, 'real'); assert.equal(tile.origin, 'prefetched');
    assert.equal(createHash('sha256').update(new Uint8Array(tile.values.buffer)).digest('hex'), f.sha256);
    assert.equal(remoteCalls, 0); assert.equal(localCalls, 1);
    serveBadHash = true; await assert.rejects(loadPrefetchedTile(f), /checksum mismatch/); serveBadHash = false;
    await assert.rejects(loadTile('probe.a', NaN, 0), /finite integers/);
    usage.set(`qwen-range-bytes:${REVISION}`, 'broken'); await assert.rejects(loadTile('probe.a'), /ledger is invalid/); usage.clear();
    await assert.rejects(loadTile('probe.a'), /ignored HTTP Range/);
    assert.equal((await downloadStatus()).remaining, byteLength, 'Failed headers must release reserved bytes.');
    failRemote = false;
    const first = loadTile('probe.a'); await startedPromise;
    const denied = assert.rejects(loadTile('probe.b'), /budget would exceed/);
    release(); await denied;
    assert.equal(remoteCalls, 2, 'Only the aborted probe and the accepted concurrent request may reach fetch.');
    const downloaded = await first; assert.equal(downloaded.values[0], 1); assert.equal(downloaded.values.length, 65536);
    assert.equal((await downloadStatus()).remaining, 0); assert.equal((await downloadStatus()).used, DOWNLOAD_BUDGET);
  } finally {
    release?.(); globalThis.fetch = originalFetch;
    if (storageDescriptor) Object.defineProperty(globalThis, 'localStorage', storageDescriptor); else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});
