/** Node fallback and reproducible offline default bundle. Refuses full-shard responses. */
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { openSync, closeSync, writeFileSync, readFileSync, unlinkSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { REPO, REVISION, DOWNLOAD_BUDGET, computeStats, scalarBytes, decodeValues, matrixShape, validateRangeHeaders, type ModelManifest, type TensorMeta, type TileFile, type Stats } from '../src/data/hf.ts';
import { RealTokenizer } from '../src/data/tokenizer.ts';
const root = resolve(import.meta.dirname, '..'), data = resolve(root, 'public/data'), weights = resolve(root, 'public/weights');
await mkdir(data, { recursive: true }); await mkdir(weights, { recursive: true });
// A single ledger must never be read and overwritten by two concurrent prefetch processes.
const lockPath = resolve(data, '.prefetch.lock');
try {
  const lock = openSync(lockPath, 'wx'); writeFileSync(lock, String(process.pid)); closeSync(lock);
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
  let pid = 0; try { pid = Number(readFileSync(lockPath, 'utf8')); } catch { /* Treat uncertain ownership as busy. */ }
  let alive = true; if (Number.isSafeInteger(pid) && pid > 0) { try { process.kill(pid, 0); } catch (e) { if ((e as NodeJS.ErrnoException).code === 'ESRCH') alive = false; } }
  if (!alive) { unlinkSync(lockPath); throw new Error('Removed a stale prefetch lock. Re-run the same command; no files were downloaded.'); }
  throw new Error('Another prefetch process owns the download ledger. Wait for it to finish.');
}
process.on('exit', () => { try { if (readFileSync(lockPath, 'utf8') === String(process.pid)) unlinkSync(lockPath); } catch { /* Already released. */ } });
const base = `https://huggingface.co/${REPO}/resolve/${REVISION}/`;
const sha = (b: Uint8Array | string) => createHash('sha256').update(b).digest('hex');
const exists = async (p: string) => !!await stat(p).catch(() => null);
interface Ledger { revision: string; limitBytes: number; downloadedBytes: number; importedBrowserBytes?: number; requests: { resource: string; bytes: number; time: string; range?: string }[] }
const ledgerPath = resolve(data, 'download-ledger.json');
if (!await exists(ledgerPath) && await exists(resolve(data, 'manifest.json'))) throw new Error('Download ledger is missing from an existing bundle. Restore it before fetching additional data.');
let ledger: Ledger = await exists(ledgerPath) ? JSON.parse(await readFile(ledgerPath, 'utf8')) : { revision: REVISION, limitBytes: DOWNLOAD_BUDGET, downloadedBytes: 0, requests: [] };
if (!Number.isSafeInteger(ledger.downloadedBytes) || ledger.downloadedBytes < 0 || ledger.downloadedBytes > DOWNLOAD_BUDGET || !Array.isArray(ledger.requests) || ledger.requests.some(r => !Number.isSafeInteger(r.bytes) || r.bytes < 0) || ledger.requests.reduce((n, r) => n + r.bytes, 0) !== ledger.downloadedBytes) throw new Error('Download ledger is malformed or does not reconcile. Refusing further downloads.');
if (ledger.importedBrowserBytes !== undefined && (!Number.isSafeInteger(ledger.importedBrowserBytes) || ledger.importedBrowserBytes < 0 || ledger.importedBrowserBytes > ledger.downloadedBytes)) throw new Error('Imported browser ledger is malformed.');
const browserOption = process.argv.indexOf('--browser-bytes');
if (browserOption !== -1) {
  const reported = Number(process.argv[browserOption + 1]); if (!Number.isSafeInteger(reported) || reported < 0) throw new Error('--browser-bytes requires a nonnegative integer from the browser download counter.');
  const additional = Math.max(0, reported - (ledger.importedBrowserBytes ?? 0));
  if (ledger.downloadedBytes + additional > DOWNLOAD_BUDGET) throw new Error('Reported browser usage already exhausts the shared download budget.');
  if (additional) ledger.requests.push({ resource: 'Browser response bytes imported from runtime ledger', bytes: additional, time: new Date().toISOString() });
  ledger.downloadedBytes += additional; ledger.importedBrowserBytes = Math.max(reported, ledger.importedBrowserBytes ?? 0);
}
if (ledger.revision !== REVISION) throw new Error('Download ledger belongs to a different model revision.');
await writeFile(ledgerPath, JSON.stringify(ledger, null, 2));
async function fetchBounded(resource: string, start?: number, length?: number, maximum = 25_000_000): Promise<Uint8Array> {
  const requested = length ?? maximum;
  if (!Number.isSafeInteger(requested) || requested < 1 || (start !== undefined && (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(length) || length! < 1))) throw new Error('Invalid requested byte interval.'); if (ledger.downloadedBytes + requested > DOWNLOAD_BUDGET) throw new Error(`150 MB download budget exceeded before ${resource}; ask the user before continuing.`);
  const range = start !== undefined ? `bytes=${start}-${start + length! - 1}` : undefined;
  const url = `${base}${resource}${range ? `?range=${start}-${start! + length! - 1}` : ''}`;
  const response = await fetch(url, { headers: range ? { Range: range } : {}, signal: AbortSignal.timeout(90000) });
  if (range && response.status !== 206) { await response.body?.cancel(); throw new Error(`Range ignored (${response.status}) for ${resource}: cancelled without downloading shard.`); }
  if (!response.ok || !response.body) { await response.body?.cancel(); throw new Error(`HTTP ${response.status} fetching ${resource}`); }
  if (range) { try { validateRangeHeaders(response.headers, start!, length!); } catch (error) { await response.body.cancel(); throw error; } }
  const contentLength = Number(response.headers.get('content-length') || 0); if (!Number.isSafeInteger(contentLength) || contentLength < 0 || contentLength > requested) { await response.body.cancel(); throw new Error('Response exceeds expected byte budget.'); }
  const reader = response.body.getReader(), parts: Uint8Array[] = []; let got = 0;
  try { while (true) { const part = await reader.read(); if (part.done) break; got += part.value.length; ledger.downloadedBytes += part.value.length; if (got > requested || ledger.downloadedBytes > DOWNLOAD_BUDGET) { await reader.cancel(); throw new Error('Streaming byte budget exceeded; response cancelled.'); } parts.push(part.value); } }
  finally { ledger.requests.push({ resource, bytes: got, time: new Date().toISOString(), ...(range ? { range } : {}) }); await writeFile(ledgerPath, JSON.stringify(ledger, null, 2)); reader.releaseLock(); }
  if (length !== undefined && got !== length) throw new Error(`Truncated range ${resource}: ${got} != ${length}`);
  const result = new Uint8Array(got); let cursor = 0; for (const part of parts) { result.set(part, cursor); cursor += part.length; } return result;
}
async function smallFile(name: string, max = 25_000_000): Promise<any> { const path = resolve(data, name); if (!await exists(path)) await writeFile(path, await fetchBounded(name, undefined, undefined, max)); return JSON.parse(await readFile(path, 'utf8')); }
const config = await smallFile('config.json', 100_000), index = await smallFile('model.safetensors.index.json', 500_000);
const manifestPath = resolve(data, 'manifest.json');
const old: ModelManifest | undefined = await exists(manifestPath) ? JSON.parse(await readFile(manifestPath, 'utf8')) : undefined;
const tensors: Record<string, TensorMeta> = {}, shards: ModelManifest['shards'] = {};
const componentOf = (name: string) => name.startsWith('model.visual.') ? 'vision' : name.startsWith('mtp.') ? 'mtp' : name.includes('embed_tokens') ? 'embedding' : name === 'lm_head.weight' ? 'lmHead' : name.includes('.mlp.') ? 'ffn' : name.includes('.linear_attn.') ? 'deltaNet' : name.includes('.self_attn.') ? 'attention' : name.includes('norm') ? 'norms' : 'other';
for (const shard of [...new Set<string>(Object.values(index.weight_map))]) {
  const path = resolve(data, `${shard}.header.json`); let header: Record<string, any>; let headerLength: number; let headerSHA256: string;
  if (await exists(path)) { const cached = JSON.parse(await readFile(path, 'utf8')); ({ header, headerLength, headerSHA256 } = cached); }
  else {
    const prefix = await fetchBounded(shard, 0, 8); headerLength = Number(new DataView(prefix.buffer).getBigUint64(0, true));
    if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > 10_000_000) throw new Error(`Invalid safetensors header length ${headerLength}`);
    const bytes = await fetchBounded(shard, 8, headerLength); header = JSON.parse(new TextDecoder().decode(bytes)); headerSHA256 = sha(bytes);
    await writeFile(path, JSON.stringify({ headerLength, headerSHA256, header }, null, 2));
  }
  let count = 0;
  for (const [name, value] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (index.weight_map[name] !== shard) throw new Error(`Index/header shard mismatch: ${name}`);
    const shape: number[] = value.shape, params = shape.reduce((a, b) => a * b, 1), bytes = value.data_offsets[1] - value.data_offsets[0];
    if (bytes !== params * scalarBytes(value.dtype)) throw new Error(`Shape/dtype/bytes disagree: ${name}`);
    const layerMatch = name.match(/^model\.language_model\.layers\.(\d+)\./);
    tensors[name] = { name, shape, dtype: value.dtype, shard, dataOffsets: value.data_offsets, absoluteOffsets: [8 + headerLength + value.data_offsets[0], 8 + headerLength + value.data_offsets[1]], params, bytes, component: componentOf(name), ...(layerMatch ? { layer: Number(layerMatch[1]) } : {}) }; count++;
  }
  shards[shard] = { headerLength, tensorCount: count, headerSHA256 }; process.stdout.write(`Header ${shard}: ${count} tensors\n`);
}
if (Object.keys(tensors).length !== Object.keys(index.weight_map).length) throw new Error('Not every weight_map entry has a tensor header.');
const components: ModelManifest['components'] = {};
for (const tensor of Object.values(tensors)) { const c = components[tensor.component] ??= { params: 0, bytes: 0, count: 0 }; c.params += tensor.params; c.bytes += tensor.bytes; c.count++; }
const totalParams = Object.values(tensors).reduce((s, t) => s + t.params, 0), totalBytes = Object.values(tensors).reduce((s, t) => s + t.bytes, 0);
if (totalBytes !== index.metadata.total_size) throw new Error('Index total_size differs from sum of exact header byte counts.');
const layers = config.text_config.layer_types.map((kind: string, layer: number) => { const values = Object.values(tensors).filter(t => t.layer === layer); return { index: layer, kind, names: values.map(t => t.name), params: values.reduce((s, t) => s + t.params, 0), bytes: values.reduce((s, t) => s + t.bytes, 0) }; });
const tiles: TileFile[] = old?.revision === REVISION ? old.tiles : [];
const manifest: ModelManifest = { repo: REPO, revision: REVISION, fetchedAt: old?.fetchedAt ?? new Date().toISOString(), config, index, tensors, layers, components, totalParams, totalBytes, shards, tiles, tensorStats: old?.tensorStats ?? {}, defaultTokenIds: old?.defaultTokenIds ?? [], budget: { limitBytes: DOWNLOAD_BUDGET, downloadedBytes: ledger.downloadedBytes, remainingBytes: DOWNLOAD_BUDGET - ledger.downloadedBytes, accounting: 'Actual response body bytes for metadata, tokenizer, shard headers, all requested row-band bytes including unused columns; browser on-demand bytes are added separately.' }, range: old?.range ?? { node: true, browserCORS: 'Range headers expose Access-Control-Allow-Origin: *. Browser verification pending.', checkedAt: new Date().toISOString() } };
async function persist(): Promise<void> { manifest.budget.downloadedBytes = ledger.downloadedBytes; manifest.budget.remainingBytes = DOWNLOAD_BUDGET - ledger.downloadedBytes; await writeFile(manifestPath, JSON.stringify(manifest, null, 2)); }
await persist();
if (process.argv.includes('--metadata')) { process.stdout.write(`Metadata ready: ${totalParams} parameters, ${totalBytes} weight bytes.\n`); process.exit(0); }
async function tile(name: string, row: number, col: number, rows: number, cols: number, purpose: string): Promise<void> {
  const t = tensors[name]; if (!t) throw new Error(`Unknown requested tensor ${name}`);
  if (![row, col, rows, cols].every(Number.isSafeInteger) || rows < 1 || cols < 1) throw new Error('Tile coordinates and dimensions must be finite positive integers.');
  const [height, width] = matrixShape(t); if (row + rows > height || col + cols > width || row < 0 || col < 0) throw new Error(`Out-of-bounds tile ${name}`);
  const id = sha(`${REVISION}:${name}:${row}:${col}:${rows}:${cols}`).slice(0, 24), path = `weights/${id}.f32`, absolute = resolve(root, 'public', path);
  const previous = tiles.find(x => x.id === id); if (previous && await exists(absolute)) { const saved = await readFile(absolute); if (saved.length !== previous.bytes || sha(saved) !== previous.sha256) throw new Error(`Cached tile corrupted ${path}`); return; }
  const size = scalarBytes(t.dtype), start = t.absoluteOffsets[0] + (row * width + col) * size, sourceBytes = ((rows - 1) * width + cols) * size;
  const buffer = await fetchBounded(t.shard, start, sourceBytes); const values = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) values.set(decodeValues(buffer.subarray(r * width * size, r * width * size + cols * size), t.dtype), r * cols);
  const output = new Uint8Array(values.buffer); await writeFile(absolute, output);
  tiles.push({ id, name, row, col, rows, cols, path, bytes: output.length, sourceBytes, stats: computeStats(values), purpose, sha256: sha(output) });
  await persist(); process.stdout.write(`Tile ${name} [${row}:${row + rows}, ${col}:${col + cols}] · ${(ledger.downloadedBytes / 1e6).toFixed(2)} MB received\n`);
}
const single = process.argv.indexOf('--tile');
if (single !== -1) {
  const name = process.argv[single + 1], t = tensors[name]; if (!t) throw new Error(`Unknown tensor ${name}`);
  const row = Number(process.argv[single + 2] ?? 0), col = Number(process.argv[single + 3] ?? 0), [height, width] = matrixShape(t);
  await tile(name, row, col, Math.min(256, height - row), Math.min(256, width - col), 'Requested on-demand fallback tile');
} else {
  const tokenizerData = await smallFile('tokenizer.json', 30_000_000); const tokenizer = new RealTokenizer(tokenizerData);
  const tokenIds = [...new Set(tokenizer.encode('the cat sat on the mat'))]; manifest.defaultTokenIds = tokenIds;
  await writeFile(resolve(data, 'tokenizer-verification.json'), JSON.stringify({ source: '[real]', revision: REVISION, preTokenizer: tokenizerData.pre_tokenizer, cases: ['the cat sat on the mat', 'the cat sat on', 'the dog sat on the rug', 'Hello, world!', 'नमस्ते मुंबई', "I'm learning AI.", '<|im_start|>user\nHi!<|im_end|>'].map(text => ({ text, ids: tokenizer.encode(text), roundtrip: tokenizer.decode(tokenizer.encode(text)) })) }, null, 2));
  // Complete real head 0 in gated Q projection: Q rows 0:256, output-gate rows 256:512.
  for (const layer of [0, 3]) for (const name of layers[layer].names as string[]) {
    const t = tensors[name], [height, width] = matrixShape(t);
    if (layer === 3 && name.endsWith('self_attn.q_proj.weight')) await tile(name, 0, 0, 2 * config.text_config.head_dim, width, 'Full gated head 0: Q then gate, per official Qwen3_5Attention');
    else if (layer === 3 && name.endsWith('self_attn.k_proj.weight')) await tile(name, 0, 0, config.text_config.head_dim, width, 'Full key head 0; shared by six Q heads');
    else await tile(name, 0, 0, Math.min(256, height), Math.min(256, width), t.shape.length === 3 ? 'Top-left short-convolution slice; last dimensions flattened' : 'Default top-left tile (256 × 256 maximum)');
  }
  const embedding = Object.keys(tensors).find(n => n.endsWith('embed_tokens.weight') && n.startsWith('model.language_model.'));
  if (!embedding || !tensors['lm_head.weight']) throw new Error('Embedding or LM-head name is ambiguous or absent.');
  let seed = 20260911; const randomIds = new Set<number>();
  while (randomIds.size < 64) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; const id = seed % config.text_config.vocab_size; if (!tokenIds.includes(id)) randomIds.add(id); }
  for (const id of [...tokenIds, ...randomIds]) await tile(embedding, id, 0, 1, config.text_config.hidden_size, tokenIds.includes(id) ? 'Full embedding row used by default real-tokenizer sentence' : 'Full embedding row selected with seed 20260911');
  for (const id of tokenIds) await tile('lm_head.weight', id, 0, 1, config.text_config.hidden_size, 'Full LM-head row for default sentence token');
}
// Pool each loaded coordinate once, so overlapping tiles cannot bias the reported summaries.
for (const name of new Set(tiles.map(t => t.name))) {
  const values: number[] = [], seen = new Set<number>(), width = matrixShape(tensors[name])[1];
  for (const f of tiles.filter(t => t.name === name)) { const b = await readFile(resolve(root, 'public', f.path)); const view = new DataView(b.buffer, b.byteOffset, b.byteLength); for (let r = 0; r < f.rows; r++) for (let c = 0; c < f.cols; c++) { const coordinate = (f.row + r) * width + f.col + c; if (!seen.has(coordinate)) { seen.add(coordinate); values.push(view.getFloat32((r * f.cols + c) * 4, true)); } } }
  manifest.tensorStats[name] = computeStats(values);
}
await persist();
process.stdout.write(`Complete. ${tiles.length} real tiles; ${(ledger.downloadedBytes / 1e6).toFixed(3)} / 150 MB received, ${(tiles.reduce((s, t) => s + t.bytes, 0) / 1e6).toFixed(3)} MB Float32 cached.\n`);
