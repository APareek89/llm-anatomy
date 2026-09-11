/** Pinned safetensors metadata and honest, budgeted real-value tile access. Storage is [output, input]. */
export const REPO = 'Qwen/Qwen3.8-27B';
export const REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0';
export const DOWNLOAD_BUDGET = 150_000_000;
export const TILE_SIZE = 256;
export interface ModelConfig {
  model_type: string;
  text_config: { hidden_size: number; num_hidden_layers: number; intermediate_size: number; vocab_size: number; num_attention_heads: number; num_key_value_heads: number; head_dim: number; layer_types: string[]; linear_num_key_heads: number; linear_num_value_heads: number; linear_key_head_dim: number; linear_value_head_dim: number; partial_rotary_factor: number; max_position_embeddings: number; [key: string]: unknown };
  vision_config: { depth: number; [key: string]: unknown }; [key: string]: unknown;
}
export interface TensorMeta { name: string; shape: number[]; dtype: string; shard: string; dataOffsets: [number, number]; absoluteOffsets: [number, number]; params: number; bytes: number; component: string; layer?: number }
export interface ComponentMeta { params: number; bytes: number; count: number }
export interface LayerMeta { index: number; kind: string; names: string[]; params: number; bytes: number }
export interface Stats { count: number; mean: number; std: number; min: number; max: number; nearZeroFraction: number; p01: number; p99: number; scope: 'loaded values only' }
export interface TileFile { id: string; name: string; row: number; col: number; rows: number; cols: number; path: string; bytes: number; sourceBytes: number; stats: Stats; purpose: string; sha256: string }
export interface ModelManifest {
  repo: string; revision: string; fetchedAt: string; config: ModelConfig;
  index: { metadata: { total_size: number }; weight_map: Record<string, string> };
  tensors: Record<string, TensorMeta>; layers: LayerMeta[]; components: Record<string, ComponentMeta>;
  totalParams: number; totalBytes: number; shards: Record<string, { headerLength: number; tensorCount: number; headerSHA256: string }>;
  tiles: TileFile[]; tensorStats: Record<string, Stats>; defaultTokenIds: number[];
  budget: { limitBytes: number; downloadedBytes: number; remainingBytes: number; accounting: string }; range: { node: boolean; browserCORS: string; checkedAt: string };
}
export interface WeightTile { name: string; row: number; col: number; rows: number; cols: number; values: Float32Array; source: 'real'; stats: Stats; origin: 'prefetched' | 'indexeddb' | 'huggingface-range' }
export function scalarBytes(dtype: string): number { const n = ({ F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 } as Record<string, number>)[dtype]; if (!n) throw new Error(`Unsupported dtype ${dtype}`); return n; }
export function bf16ToFloat32(bytes: Uint8Array): Float32Array { if (bytes.byteLength % 2) throw new Error('Odd BF16 byte length'); const out = new Float32Array(bytes.byteLength / 2), bits = new Uint32Array(out.buffer), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); for (let i = 0; i < out.length; i++) bits[i] = view.getUint16(i * 2, true) << 16; return out; }
export function decodeValues(bytes: Uint8Array, dtype: string): Float32Array {
  if (dtype === 'BF16') return bf16ToFloat32(bytes);
  if (dtype === 'F32') { const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true)); }
  throw new Error(`Real tile decoder does not support ${dtype}; no values were invented.`);
}
export function computeStats(values: Float32Array | number[]): Stats {
  const sorted = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b), n = sorted.length;
  if (!n) throw new Error('Cannot calculate stats from an empty tile.');
  let mean = 0, m2 = 0, near = 0; sorted.forEach((v, i) => { const d = v - mean; mean += d / (i + 1); m2 += d * (v - mean); if (Math.abs(v) < 1e-3) near++; });
  return { count: n, mean, std: Math.sqrt(m2 / n), min: sorted[0], max: sorted[n - 1], nearZeroFraction: near / n, p01: sorted[Math.floor((n - 1) * .01)], p99: sorted[Math.floor((n - 1) * .99)], scope: 'loaded values only' };
}
export function matrixShape(t: TensorMeta): [number, number] { return t.shape.length === 1 ? [t.shape[0], 1] : [t.shape[0], t.shape.slice(1).reduce((a, b) => a * b, 1)]; }
let manifestPromise: Promise<ModelManifest> | undefined;
export function loadModelData(): Promise<ModelManifest> { return manifestPromise ??= fetch(`${import.meta.env.BASE_URL}data/manifest.json`).then(r => { if (!r.ok) throw new Error('Pinned metadata unavailable. Run npm run prefetch.'); return r.json(); }); }
export async function listTensorTiles(name: string): Promise<TileFile[]> { return (await loadModelData()).tiles.filter(t => t.name === name); }
const keyOf = (name: string, row: number, col: number, rows: number, cols: number) => `${REVISION}:${name}:${row}:${col}:${rows}:${cols}`;
function db(): Promise<IDBDatabase> { return new Promise((resolve, reject) => { const r = indexedDB.open('qwen-real-tiles-v1', 1); r.onupgradeneeded = () => { if (!r.result.objectStoreNames.contains('tiles')) r.result.createObjectStore('tiles'); }; r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); }); }
async function cached(key: string): Promise<WeightTile | undefined> { try { const d = await db(); return await new Promise((resolve, reject) => { const r = d.transaction('tiles').objectStore('tiles').get(key); r.onsuccess = () => { d.close(); resolve(r.result); }; r.onerror = () => { d.close(); reject(r.error); }; }); } catch { return undefined; } }
async function save(key: string, tile: WeightTile): Promise<void> { try { const d = await db(); await new Promise<void>((resolve, reject) => { const tx = d.transaction('tiles', 'readwrite'); tx.objectStore('tiles').put(tile, key); tx.oncomplete = () => { d.close(); resolve(); }; tx.onerror = () => { d.close(); reject(tx.error); }; }); } catch { /* Private browsing may disable IndexedDB; values remain valid. */ } }
let runtimeBytes = 0;
let reservedBytes = 0;
function usedBytes(): number { try { return Math.max(runtimeBytes, Number(localStorage.getItem(`qwen-range-bytes:${REVISION}`) || 0)); } catch { return runtimeBytes; } }
function charge(bytes: number): void { runtimeBytes = usedBytes() + bytes; try { localStorage.setItem(`qwen-range-bytes:${REVISION}`, String(runtimeBytes)); } catch { /* Session-only budget remains enforced. */ } }
export async function downloadStatus(): Promise<{ used: number; limit: number; remaining: number }> { const m = await loadModelData(), used = m.budget.downloadedBytes + usedBytes(); return { used, limit: DOWNLOAD_BUDGET, remaining: Math.max(0, DOWNLOAD_BUDGET - used - reservedBytes) }; }
const pending = new Map<string, Promise<WeightTile>>();
export async function loadTile(name: string, row = 0, col = 0, onProgress?: (fraction: number) => void): Promise<WeightTile> {
  const m = await loadModelData(), t = m.tensors[name]; if (!t) throw new Error(`Unknown real tensor ${name}`);
  const [height, width] = matrixShape(t); row = Math.floor(row / TILE_SIZE) * TILE_SIZE; col = Math.floor(col / TILE_SIZE) * TILE_SIZE;
  if (row < 0 || col < 0 || row >= height || col >= width) throw new Error('Tile outside tensor shape.');
  const rows = Math.min(TILE_SIZE, height - row), cols = Math.min(TILE_SIZE, width - col), key = keyOf(name, row, col, rows, cols);
  const hit = await cached(key); if (hit) { onProgress?.(1); return { ...hit, origin: 'indexeddb' }; }
  const existing = pending.get(key); if (existing) return existing;
  const task = (async () => {
    const local = m.tiles.find(f => f.name === name && f.row <= row && f.col <= col && f.row + f.rows >= row + rows && f.col + f.cols >= col + cols);
    let values: Float32Array; let origin: WeightTile['origin'];
    if (local) {
      const response = await fetch(`${import.meta.env.BASE_URL}${local.path}`); if (!response.ok) throw new Error(`Prefetched tile missing: ${local.path}`);
      const all = new Float32Array(await response.arrayBuffer()); if (all.byteLength !== local.bytes) throw new Error('Prefetched tile byte count mismatch.');
      values = new Float32Array(rows * cols); for (let r = 0; r < rows; r++) values.set(all.subarray((row - local.row + r) * local.cols + col - local.col, (row - local.row + r) * local.cols + col - local.col + cols), r * cols);
      origin = 'prefetched'; onProgress?.(1);
    } else {
      // One row-band range is much faster than 256 separate HTTP requests. Gap bytes are budgeted.
      const size = scalarBytes(t.dtype), start = t.absoluteOffsets[0] + (row * width + col) * size, length = ((rows - 1) * width + cols) * size;
      const remaining = DOWNLOAD_BUDGET - m.budget.downloadedBytes - usedBytes() - reservedBytes;
      if (length > remaining) throw new Error(`Download budget would exceed 150 MB. ${Math.ceil(length / 1e6)} MB needed; ${Math.floor(remaining / 1e6)} MB remains.`);
      reservedBytes += length; let unconsumed = length;
      try {
      let response: Response;
      try { response = await fetch(`https://huggingface.co/${REPO}/resolve/${REVISION}/${t.shard}?tile=${start}-${start + length - 1}`, { headers: { Range: `bytes=${start}-${start + length - 1}` } }); }
      catch { throw new Error(`Browser Range/CORS unavailable. Prefetch this tile with: npm run prefetch -- --tile '${name}' ${row} ${col}`); }
      if (response.status !== 206) { await response.body?.cancel(); throw new Error('Server ignored HTTP Range. Full shard download cancelled.'); }
      const contentRange = response.headers.get('Content-Range'); if (!contentRange || !contentRange.startsWith(`bytes ${start}-${start + length - 1}/`)) { await response.body?.cancel(); throw new Error('Range response does not match tensor offsets.'); }
      if (!response.body) throw new Error('Range response has no body.');
      const reader = response.body.getReader(), buffer = new Uint8Array(length); let got = 0;
      try { while (true) { const part = await reader.read(); if (part.done) break; charge(part.value.byteLength); const consumed = Math.min(unconsumed, part.value.byteLength); unconsumed -= consumed; reservedBytes -= consumed; if (got + part.value.byteLength > length) { await reader.cancel(); throw new Error('Oversize range response cancelled.'); } buffer.set(part.value, got); got += part.value.byteLength; onProgress?.(got / length); } } finally { reader.releaseLock(); }
      if (got !== length) throw new Error('Truncated real weight range.');
      values = new Float32Array(rows * cols); for (let r = 0; r < rows; r++) values.set(decodeValues(buffer.subarray(r * width * size, r * width * size + cols * size), t.dtype), r * cols);
      origin = 'huggingface-range';
      } finally { reservedBytes -= unconsumed; }
    }
    const tile: WeightTile = { name, row, col, rows, cols, values, source: 'real', stats: computeStats(values), origin }; await save(key, tile); return tile;
  })(); pending.set(key, task); try { return await task; } finally { pending.delete(key); }
}
