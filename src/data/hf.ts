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
export interface TensorMeta { name: string; shape: number[]; dtype: string; shard: string; dataOffsets: [number, number]; absoluteOffsets: [number, number]; params: number; bytes: number; component: string; layer?: number; tiedEmbedding?: boolean; embeddingScale?: number }
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
export interface WeightTile { name: string; row: number; col: number; rows: number; cols: number; values: Float32Array; source: 'real' | 'micro'; stats: Stats; origin: 'prefetched' | 'indexeddb' | 'huggingface-range' | 'micro-worker' }
export function scalarBytes(dtype: string): number { const n = ({ F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 } as Record<string, number>)[dtype]; if (!n) throw new Error(`Unsupported dtype ${dtype}`); return n; }
export function bf16ToFloat32(bytes: Uint8Array): Float32Array { if (bytes.byteLength % 2) throw new Error('Odd BF16 byte length'); const out = new Float32Array(bytes.byteLength / 2), bits = new Uint32Array(out.buffer), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); for (let i = 0; i < out.length; i++) bits[i] = view.getUint16(i * 2, true) << 16; return out; }
export function decodeValues(bytes: Uint8Array, dtype: string): Float32Array {
  if (dtype === 'BF16') return bf16ToFloat32(bytes);
  if (dtype === 'F32') { if (bytes.byteLength % 4) throw new Error('Misaligned Float32 byte length'); const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true)); }
  throw new Error(`Real tile decoder does not support ${dtype}; no values were invented.`);
}
export function computeStats(values: Float32Array | number[]): Stats {
  const sorted = Array.from(values), n = sorted.length;
  if (!n) throw new Error('Cannot calculate stats from an empty tile.');
  if (!sorted.every(Number.isFinite)) throw new Error('Cannot hide nonfinite values in summary statistics.');
  sorted.sort((a, b) => a - b);
  let mean = 0, m2 = 0, near = 0; sorted.forEach((v, i) => { const d = v - mean; mean += d / (i + 1); m2 += d * (v - mean); if (Math.abs(v) < 1e-3) near++; });
  return { count: n, mean, std: Math.sqrt(m2 / n), min: sorted[0], max: sorted[n - 1], nearZeroFraction: near / n, p01: sorted[Math.floor((n - 1) * .01)], p99: sorted[Math.floor((n - 1) * .99)], scope: 'loaded values only' };
}
export function matrixShape(t: TensorMeta): [number, number] { return t.shape.length === 1 ? [t.shape[0], 1] : [t.shape[0], t.shape.slice(1).reduce((a, b) => a * b, 1)]; }
const assetBase = () => import.meta.env?.BASE_URL ?? '/';
let manifestPromise: Promise<ModelManifest> | undefined;
export function loadModelData(): Promise<ModelManifest> { return manifestPromise ??= fetch(`${assetBase()}data/manifest.json`).then(r => { if (!r.ok) throw new Error('Pinned metadata unavailable. Run npm run prefetch.'); return r.json(); }).then((m: ModelManifest) => { if (m.repo !== REPO || m.revision !== REVISION || !Number.isSafeInteger(m.budget.downloadedBytes) || m.budget.downloadedBytes < 0 || m.budget.downloadedBytes > DOWNLOAD_BUDGET) throw new Error('Invalid pinned metadata or download ledger.'); return m; }); }
export async function listTensorTiles(name: string): Promise<TileFile[]> { return (await loadModelData()).tiles.filter(t => t.name === name); }
const keyOf = (name: string, row: number, col: number, rows: number, cols: number) => `${REVISION}:${name}:${row}:${col}:${rows}:${cols}`;
function db(): Promise<IDBDatabase> { return new Promise((resolve, reject) => { const r = indexedDB.open('qwen-real-tiles-v1', 1); r.onupgradeneeded = () => { if (!r.result.objectStoreNames.contains('tiles')) r.result.createObjectStore('tiles'); }; r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); }); }
async function cached(key: string): Promise<WeightTile | undefined> { try { const d = await db(); return await new Promise((resolve, reject) => { const r = d.transaction('tiles').objectStore('tiles').get(key); r.onsuccess = () => { d.close(); resolve(r.result); }; r.onerror = () => { d.close(); reject(r.error); }; }); } catch { return undefined; } }
async function save(key: string, tile: WeightTile): Promise<void> { try { const d = await db(); await new Promise<void>((resolve, reject) => { const tx = d.transaction('tiles', 'readwrite'); tx.objectStore('tiles').put(tile, key); tx.oncomplete = () => { d.close(); resolve(); }; tx.onerror = () => { d.close(); reject(tx.error); }; }); } catch { /* Private browsing may disable IndexedDB; values remain valid. */ } }
let runtimeBytes = 0;
let reservedBytes = 0;
function usedBytes(): number {
  let stored: string | null = null; try { stored = localStorage.getItem(`qwen-range-bytes:${REVISION}`); } catch { return runtimeBytes; }
  const value = stored === null ? 0 : Number(stored);
  if (!Number.isSafeInteger(value) || value < 0) throw new Error('Browser download ledger is invalid. Budgeted downloads are paused to prevent overspending.');
  return Math.max(runtimeBytes, value);
}
function charge(bytes: number): void { runtimeBytes = usedBytes() + bytes; try { localStorage.setItem(`qwen-range-bytes:${REVISION}`, String(runtimeBytes)); } catch { /* Session-only budget remains enforced. */ } }
export async function downloadStatus(): Promise<{ used: number; limit: number; remaining: number }> { const m = await loadModelData(), used = m.budget.downloadedBytes + usedBytes(); return { used, limit: DOWNLOAD_BUDGET, remaining: Math.max(0, DOWNLOAD_BUDGET - used - reservedBytes) }; }
const pending = new Map<string, Promise<WeightTile>>();
export async function loadTile(name: string, row = 0, col = 0, onProgress?: (fraction: number) => void): Promise<WeightTile> {
  const m = await loadModelData(), t = m.tensors[name]; if (!t) throw new Error(`Unknown real tensor ${name}`);
  if (!Number.isSafeInteger(row) || !Number.isSafeInteger(col)) throw new Error('Tile row and column must be finite integers.');
  const [height, width] = matrixShape(t); row = Math.floor(row / TILE_SIZE) * TILE_SIZE; col = Math.floor(col / TILE_SIZE) * TILE_SIZE;
  if (row < 0 || col < 0 || row >= height || col >= width) throw new Error('Tile outside tensor shape.');
  const rows = Math.min(TILE_SIZE, height - row), cols = Math.min(TILE_SIZE, width - col), key = keyOf(name, row, col, rows, cols);
  const hit = await cached(key); if (hit) { onProgress?.(1); return { ...hit, origin: 'indexeddb' }; }
  const existing = pending.get(key); if (existing) return existing;
  const task = (async () => {
    const local = m.tiles.find(f => f.name === name && f.row <= row && f.col <= col && f.row + f.rows >= row + rows && f.col + f.cols >= col + cols);
    let values: Float32Array; let origin: WeightTile['origin'];
    if (local) {
      const all = (await loadPrefetchedTile(local)).values;
      values = new Float32Array(rows * cols); for (let r = 0; r < rows; r++) values.set(all.subarray((row - local.row + r) * local.cols + col - local.col, (row - local.row + r) * local.cols + col - local.col + cols), r * cols);
      origin = 'prefetched'; onProgress?.(1);
    } else {
      // Web Locks coordinate the shared usage counter across browser tabs; reservations cover
      // independent requests in runtimes without that API. The check happens inside the lock.
      const rangeDownload = async (): Promise<Float32Array> => {
        const size = scalarBytes(t.dtype), start = t.absoluteOffsets[0] + (row * width + col) * size, length = ((rows - 1) * width + cols) * size;
        const remaining = DOWNLOAD_BUDGET - m.budget.downloadedBytes - usedBytes() - reservedBytes;
        if (length > remaining) throw new Error(`Download budget would exceed 150 MB. ${Math.ceil(length / 1e6)} MB needed; ${Math.floor(remaining / 1e6)} MB remains.`);
        reservedBytes += length; let unconsumed = length;
        try {
          let response: Response;
          try { response = await fetch(`https://huggingface.co/${REPO}/resolve/${REVISION}/${t.shard}?tile=${start}-${start + length - 1}`, { headers: { Range: `bytes=${start}-${start + length - 1}` }, signal: AbortSignal.timeout(90000) }); }
          catch { throw new Error(`Browser Range/CORS unavailable. Prefetch this tile with: npm run prefetch -- --tile '${name}' ${row} ${col} --browser-bytes ${usedBytes()}`); }
          if (response.status !== 206) { await response.body?.cancel(); throw new Error('Server ignored HTTP Range. Full shard download cancelled.'); }
          try { validateRangeHeaders(response.headers, start, length); } catch (error) { await response.body?.cancel(); throw error; }
          if (!response.body) throw new Error('Range response has no body.');
          const reader = response.body.getReader(), buffer = new Uint8Array(length); let got = 0;
          try {
            while (true) {
              const part = await reader.read(); if (part.done) break;
              charge(part.value.byteLength); const consumed = Math.min(unconsumed, part.value.byteLength); unconsumed -= consumed; reservedBytes -= consumed;
              if (got + part.value.byteLength > length) { await reader.cancel(); throw new Error('Oversize range response cancelled.'); }
              buffer.set(part.value, got); got += part.value.byteLength; onProgress?.(got / length);
            }
          } finally { reader.releaseLock(); }
          if (got !== length) throw new Error('Truncated real weight range.');
          const decoded = new Float32Array(rows * cols); for (let r = 0; r < rows; r++) decoded.set(decodeValues(buffer.subarray(r * width * size, r * width * size + cols * size), t.dtype), r * cols);
          if (!decoded.every(Number.isFinite)) throw new Error('Downloaded tile contains nonfinite values.');
          return decoded;
        } finally { reservedBytes -= unconsumed; }
      };
      values = globalThis.navigator?.locks ? await navigator.locks.request(`qwen-download-budget:${REVISION}`, rangeDownload) : await rangeDownload();
      origin = 'huggingface-range';
    }
    const tile: WeightTile = { name, row, col, rows, cols, values, source: 'real', stats: computeStats(values), origin }; await save(key, tile); return tile;
  })(); pending.set(key, task); try { return await task; } finally { pending.delete(key); }
}

export function validateRangeHeaders(headers: Headers, start: number, length: number): void {
  const raw = headers.get('Content-Range'), match = raw?.match(/^bytes (\d+)-(\d+)\/(\d+)$/);
  if (!match) throw new Error('Missing or malformed Content-Range; tensor bytes cannot be verified.');
  const first = Number(match[1]), last = Number(match[2]), total = Number(match[3]);
  if (![first, last, total, start, length].every(Number.isSafeInteger) || first !== start || last !== start + length - 1 || total <= last || length < 1) throw new Error('Range response does not match tensor offsets.');
  const rawLength = headers.get('Content-Length');
  if (rawLength !== null && (!/^\d+$/.test(rawLength) || Number(rawLength) !== length)) throw new Error('Content-Length does not match the requested tensor range.');
  const encoding = headers.get('Content-Encoding'); if (encoding && encoding !== 'identity') throw new Error('Encoded tensor range cannot be interpreted as original safetensors offsets.');
}

const prefetchedPending = new Map<string, Promise<WeightTile>>();
/** Fetch one exact manifest block, including a sparse single-token embedding row, without grid rounding. */
export async function loadPrefetchedTile(file: TileFile): Promise<WeightTile> {
  const m = await loadModelData(), stored = m.tiles.find(t => t.id === file.id);
  if (!stored || stored.name !== file.name || stored.path !== file.path || stored.sha256 !== file.sha256) throw new Error('Prefetched block is absent from the pinned manifest.');
  const t = m.tensors[stored.name], [height, width] = matrixShape(t);
  if (![stored.row, stored.col, stored.rows, stored.cols].every(Number.isSafeInteger) || stored.row < 0 || stored.col < 0 || stored.rows < 1 || stored.cols < 1 || stored.row + stored.rows > height || stored.col + stored.cols > width || stored.bytes !== stored.rows * stored.cols * 4) throw new Error('Malformed prefetched tile dimensions.');
  const key = `bundle:${stored.id}:${stored.sha256}`;
  const hit = await cached(key); if (hit) return { ...hit, origin: 'indexeddb' };
  const pendingTile = prefetchedPending.get(key); if (pendingTile) return pendingTile;
  const task = (async () => {
    const response = await fetch(`${assetBase()}${stored.path}`); if (!response.ok) throw new Error(`Prefetched tile missing: ${stored.path}`);
    const buffer = await response.arrayBuffer(); if (buffer.byteLength !== stored.bytes) throw new Error('Prefetched tile byte count mismatch.');
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))].map(b => b.toString(16).padStart(2, '0')).join('');
    if (digest !== stored.sha256) throw new Error('Prefetched tile checksum mismatch. No values will be displayed.');
    const view = new DataView(buffer), values = Float32Array.from({ length: stored.rows * stored.cols }, (_, i) => view.getFloat32(i * 4, true));
    if (!values.every(Number.isFinite)) throw new Error('Prefetched tile contains nonfinite values.');
    const tile: WeightTile = { name: stored.name, row: stored.row, col: stored.col, rows: stored.rows, cols: stored.cols, values, source: 'real', stats: stored.stats, origin: 'prefetched' };
    await save(key, tile); return tile;
  })();
  prefetchedPending.set(key, task); try { return await task; } finally { prefetchedPending.delete(key); }
}
