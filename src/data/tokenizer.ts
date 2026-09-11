/** Dependency-free byte-level BPE, driven by the pinned Qwen tokenizer.json. */
export interface TokenPiece { id: number; token: string; text: string }
interface TokenizerJSON {
  added_tokens: { id: number; content: string; special: boolean }[];
  normalizer: { type: string } | null;
  pre_tokenizer: { type: string; pretokenizers?: { type: string; pattern?: { Regex: string }; behavior?: string }[] };
  model: { type: string; vocab: Record<string, number>; merges: (string | [string, string])[]; byte_fallback?: boolean };
}
const byteChars = (() => {
  const bytes = [...Array.from({ length: 94 }, (_, i) => i + 33), ...Array.from({ length: 12 }, (_, i) => i + 161), ...Array.from({ length: 82 }, (_, i) => i + 174)];
  const chars = [...bytes]; let n = 0;
  for (let b = 0; b < 256; b++) if (!bytes.includes(b)) { bytes.push(b); chars.push(256 + n++); }
  return new Map(bytes.map((b, i) => [b, String.fromCodePoint(chars[i])]));
})();
const charBytes = new Map([...byteChars].map(([b, c]) => [c, b]));
export class RealTokenizer {
  private vocab: Record<string, number>;
  private inverse = new Map<number, string>();
  private ranks = new Map<string, number>();
  private special = new Map<string, number>();
  private split: RegExp;
  private nfc: boolean;
  private specialPattern: RegExp | null;
  private cache = new Map<string, string[]>();
  constructor(data: TokenizerJSON) {
    if (data.model.type !== 'BPE' || (data.normalizer !== null && data.normalizer.type !== 'NFC')) throw new Error('Unsupported tokenizer configuration: expected Qwen byte-level BPE with NFC normalization.');
    this.nfc = data.normalizer?.type === 'NFC';
    this.vocab = data.model.vocab;
    for (const [piece, id] of Object.entries(this.vocab)) this.inverse.set(id, piece);
    data.model.merges.forEach((pair, i) => { const parts = Array.isArray(pair) ? pair : pair.split(' '); this.ranks.set(parts.join('\u0000'), i); });
    for (const t of data.added_tokens) { this.special.set(t.content, t.id); this.inverse.set(t.id, t.content); }
    const pattern = data.pre_tokenizer.pretokenizers?.find(p => p.type === 'Split')?.pattern?.Regex;
    if (!pattern || !data.pre_tokenizer.pretokenizers?.some(p => p.type === 'ByteLevel')) throw new Error('Unsupported Qwen pre-tokenizer configuration.');
    // Rust regex supports scoped (?i:…); JS lacks it. Expand only its ASCII contraction group.
    const jsPattern = pattern.replace(/\(\?i:([^)]*)\)/g, (_, value: string) => `(?:${value.replace(/[a-z]/g, c => `[${c}${c.toUpperCase()}]`)})`);
    this.split = new RegExp(jsPattern, 'gu');
    const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    this.specialPattern = this.special.size ? new RegExp(`(${[...this.special.keys()].sort((a, b) => b.length - a.length).map(escape).join('|')})`, 'g') : null;
  }
  private bpe(piece: string): string[] {
    const cached = this.cache.get(piece); if (cached) return cached;
    let parts = Array.from(piece);
    while (parts.length > 1) {
      let best = Infinity, index = -1;
      for (let i = 0; i < parts.length - 1; i++) { const rank = this.ranks.get(`${parts[i]}\u0000${parts[i + 1]}`); if (rank !== undefined && rank < best) { best = rank; index = i; } }
      if (index < 0) break;
      // Hugging Face BPE merges all occurrences of the lowest ranked adjacent pair.
      const left = parts[index], right = parts[index + 1]; const next: string[] = [];
      for (let i = 0; i < parts.length; i++) { if (parts[i] === left && parts[i + 1] === right) { next.push(left + right); i++; } else next.push(parts[i]); }
      parts = next;
    }
    if (this.cache.size < 10000) this.cache.set(piece, parts);
    return parts;
  }
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const segment of this.specialPattern ? text.split(this.specialPattern) : [text]) {
      if (!segment) continue;
      const special = this.special.get(segment); if (special !== undefined) { ids.push(special); continue; }
      this.split.lastIndex = 0;
      const normalized = this.nfc ? segment.normalize('NFC') : segment;
      const matches = [...normalized.matchAll(this.split)];
      if (matches.map(m => m[0]).join('') !== normalized) throw new Error('Tokenizer regex did not cover the input.');
      for (const match of matches) {
        const encoded = [...new TextEncoder().encode(match[0])].map(b => byteChars.get(b)!).join('');
        for (const piece of this.bpe(encoded)) { const id = this.vocab[piece]; if (id === undefined) throw new Error(`Token absent from pinned vocabulary: ${piece}`); ids.push(id); }
      }
    }
    return ids;
  }
  decode(ids: number[]): string {
    let out = ''; let bytes: number[] = [];
    const flush = () => { if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes = []; } };
    for (const id of ids) { const piece = this.inverse.get(id); if (piece === undefined) throw new Error(`Unknown token ID ${id}`); if (this.special.has(piece)) { flush(); out += piece; } else for (const char of piece) { const b = charBytes.get(char); if (b === undefined) throw new Error('Invalid byte-level token'); bytes.push(b); } }
    flush(); return out;
  }
  tokenPieces(text: string): TokenPiece[] { return this.encode(text).map(id => ({ id, token: this.inverse.get(id)!, text: this.decode([id]) })); }
}
let tokenizerPromise: Promise<RealTokenizer> | undefined;
export function loadTokenizer(): Promise<RealTokenizer> {
  return tokenizerPromise ??= fetch(`${import.meta.env.BASE_URL}data/tokenizer.json`).then(r => { if (!r.ok) throw new Error('Real tokenizer unavailable. Run npm run prefetch.'); return r.json(); }).then(data => new RealTokenizer(data));
}
