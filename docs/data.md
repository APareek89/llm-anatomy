# Real Qwen data and provenance

The explorer reads `Qwen/Qwen3.8-27B` at the pinned Hugging Face revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The model exists. Its configuration identifies the architecture as `qwen3_5`, with 64 language layers and a real pattern of three DeltaNet layers followed by one full-attention layer. Configuration, index, every shard header, tokenizer and downloaded weight files are included locally for reproducibility.

`public/data/manifest.json` contains all 1,199 named tensors. Shapes, dtypes and byte offsets come directly from the JSON headers of all 18 safetensors shards. The safetensors index provides names and shard membership; it does **not** itself contain shapes. Tests cross-check both sources. Header lengths and SHA-256 checksums are retained in the manifest and per-shard JSON files.

The exact stored parameter count is 27,781,427,952, occupying 55,562,855,904 bytes in BF16 (55.563 decimal GB, about 51.747 GiB). This includes vision, multi-token prediction, untied embeddings and LM head. Parameter counts and memory estimates are `[config]`; sampled parameter values and statistics are `[real]`. Counts are based on the downloaded metadata, including every stored parameter. They do not estimate runtime activation or KV-cache memory.

## Downloads and budget

`npm run prefetch` is a reproducible Node fallback and creates an offline default bundle in `public/weights`. It uses strict HTTP Range requests; it aborts a server response before reading a shard body unless the response is `206` and its `Content-Range` matches the requested interval. It reads the first eight bytes for the header length, then the JSON header, never the full 55 GB checkpoint. `download-ledger.json` records actual received response-body bytes and `manifest.json` records the total. The 150,000,000-byte limit includes configuration, index, headers, real tokenizer, and unused columns inside fetched row bands. The script fails before a request that could cross that limit.

Each ordinary tile has at most 256 rows × 256 columns. Remote retrieval requests a contiguous row band spanning the tile; unused input columns between rows are downloaded, counted against the budget, then discarded. This uses fewer HTTP requests. Downloaded BF16 samples are converted exactly to Float32 and saved as little-endian `.f32` files. Every saved file has a SHA-256 digest and checked dimensions. Browser reads verify that digest before displaying values; Node reruns verify existing files before reusing them. `loadPrefetchedTile` reads an exact saved block, allowing sparse embedding rows to be displayed without rounding the token ID to a 256-row grid. Missing files and unsupported dtypes produce an error; random stand-in values are never substituted.

Defaults include every tensor in layer 0 and layer 3 (including normalization vectors, DeltaNet scalar parameters and a convolution slice), all six FFN matrices, one complete head-0 Q-plus-gate band, one complete head-0 K band, full embedding rows for the real token IDs in `the cat sat on the mat`, 64 different seeded random embedding rows, and LM-head rows for those same sentence IDs. The random seed chooses row indices only; the **values** are downloaded from the actual checkpoint.

At runtime the browser first reads bundled files or IndexedDB. Other tiles use the pinned Hugging Face Range URL. Runtime download accounting persists in localStorage, rejects invalid counters, and reserves pending request bytes so simultaneous tile requests cannot overspend the remaining limit. Web Locks serialize downloads across browser tabs where supported. Requests stop when the cumulative saved-build-plus-browser budget would exceed 150 MB. The browser reports a fallback command if CORS fails:

```sh
npm run prefetch -- --tile 'model.language_model.layers.40.mlp.gate_proj.weight' 0 0
```

The generated browser fallback command includes `--browser-bytes N`, carrying its current response-byte count into the Node ledger. Use that generated command after browser downloads: the script imports additional usage once and checks the shared 150 MB limit before requesting data. Imported browser usage may also remain in localStorage, causing a conservative double count in the browser. Bundled default files are reused without additional Hugging Face traffic. Preserve both accounting records while using the budget; clearing them loses usage history. Concurrent Node processes cannot modify the same ledger: a process lock protects it, and corrupted ledgers stop new downloads.

## Correct reading of dimensions

For stored linear weights, a row is an **output** feature and a column is an **input** feature. The multiplication is `output[row] = sum(input[col] × weight[row,col])`. A `[5120,17408]` down-projection therefore reduces 17,408 FFN features into the 5,120-dimensional residual stream.

Layer 3's Q projection has shape `[12288,5120]`: each of 24 attention heads occupies 512 consecutive rows. Inside each head, the first 256 rows form Q and the next 256 form its output gate. The downloaded head-0 band includes all 512 rows. The K and V projections each have 1,024 rows: four heads of dimension 256. Six Q heads share each KV head. These interpretations follow `Qwen3_5Attention.forward` in the Transformers reference implementation.

DeltaNet's `in_proj_qkv.weight` has 10,240 output rows: 16 × 128 Q dimensions, 16 × 128 K dimensions, and 48 × 128 V dimensions. Its `[10240,1,4]` short-convolution tensor retains that exact original shape in metadata. Only the display/storage helper flattens its last two axes to show a two-dimensional 256 × 4 slice.

The manifest's per-tensor summary statistics pool **unique coordinates from loaded tiles only**. They are not whole-model or whole-tensor estimates. “Near zero” means absolute value below 0.001. Heatmap percentiles use the 1st and 99th percentiles of the loaded tile values.

## Real tokenizer

`public/data/tokenizer.json` is the actual pinned Qwen tokenizer, about 12.8 MB. The dependency-free implementation performs its configured NFC Unicode normalization, exact configured Unicode pre-tokenization, canonical byte-level encoding, ranked byte-pair merges, and special-token handling. It is independently checked against the installed Hugging Face `tokenizers` implementation on multilingual, whitespace, code, contractions, composed/decomposed Unicode and special-token samples. This is a genuine tokenizer, not the Micro-Qwen word vocabulary.

The real IDs for `the cat sat on the mat` are `[1719, 7993, 7338, 383, 279, 5344]`. Notice that initial `the` and space-prefixed ` the` have different IDs. NFC normalization can change the byte spelling of equivalent Unicode text; decoding preserves the normalized form. Individual token display pieces can contain replacement glyphs when one token carries only part of a multibyte character; decoding the full sequence recombines bytes correctly.
