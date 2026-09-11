# Acceptance record

Verified on 11 September 2026 on an Apple M4, macOS arm64, Node 24.8.0, using the Codex in-app browser and the local Vite development and production servers. Browser checks used 1280 × 720 and 1440 × 900 viewports. These are observed results on this machine, not guarantees for every browser or device.

## Numerical foundation

- [x] Config, index, all 18 safetensors headers and sampled weights share revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- [x] All 1,199 tensor names, shapes, dtypes and byte ranges reconcile with the original headers and index. The exhaustive component split totals 27,781,427,952 parameters / 55,562,855,904 BF16 bytes.
- [x] The 101 bundled real blocks pass their binary checksums. BF16 decoding passes known sign/exponent/mantissa cases. No missing weight values are synthesized.
- [x] The initial network ledger reconciles at 65,019,823 bytes against the 150,000,000-byte cap. Browser Range testing added approximately 2.61 MB, displayed as 67.6 / 150 MB. Malformed counters, coordinates, transport responses and concurrent overspending are rejected by tests.
- [x] Full-model finite differences test 250 parameter coordinates across every tensor in a tiny four-layer hybrid configuration. Maximum normalized error: **2.292530854 × 10⁻⁷**. Separate checks cover grouped KV heads, nonzero rotary positions and multi-head recurrent state carry.
- [x] Recorded differentiable training inputs and outputs have actual gradients. Attention score/probability adjoints agree with the fused graph; displayed Delta state input gradients agree with independent local finite differences. Discrete IDs and inference without a loss correctly have no gradients.
- [x] Same-seed runs yield identical weights, batches, losses and SGD updates. Checkpoint restoration preserves logits and subsequent training progression.
- [x] The actual real tokenizer matches Hugging Face `tokenizers` 0.20.3 on multilingual, Unicode, whitespace, code and special-token fixtures.
- [x] **`npm test`: 19 passed, 0 failed.** The final run took 7.95 seconds, including 7.33 seconds for the 300-step training loop.

## Browser verification

- [x] The model opens with 64 language layers, the three-DeltaNet/one-attention pattern, shape-derived tensor blocks, parameter shares, vision tower and MTP head. Search, layer/tensor selection, breadcrumbs and Back/Escape were exercised.
- [x] Layer 3 `q_proj` loads its authentic 256 × 256 sample. Head controls use the real 512-row Q-plus-gate band per head. Head 1 loaded previously uncached rows 512–767, columns 0–255 directly from Hugging Face, with `[real]` / `huggingface-range` provenance and no CORS errors. Its observed mean was −2.399 × 10⁻⁵, standard deviation 0.01462, minimum −0.07275 and maximum 0.10400.
- [x] Missing regions show an unloaded state; fetch controls show progress and then real values. The default tiles work from the bundled Node-prefetched files; browser HTTP Range retrieval works for additional regions.
- [x] Continuous zoom reaches numeric instanced cells. Q-projection `w[125,136] = −0.011962890625` was cross-checked against the saved binary. Row/column hover highlights the corresponding strip; dense rows mean output features and columns mean input features. Embedding, normalization and convolution cards use their own correct axis meanings.
- [x] Panning from Q cell (124,250) across the tile edge loaded columns 256–511 from IndexedDB without additional network usage. Numeric zoom was preserved, the previous tile remained dimmed, and clicking new cell (123,258) showed −0.0038452148. The browser reported 60 FPS / 320 cells after the transition.
- [x] Inference controls were exercised: play/pause, forward/back, scrub, layer jumps, 0.10× speed, temperature, camera follow/detach and appending a predicted word. Real Qwen IDs appear alongside the distinct micro word IDs. The standalone tokenizer accepts multilingual text.
- [x] The calculation lens displays actual recorded inputs, outputs, shapes, formulas and expanded arithmetic. Delta state updates and a numerical causal attention-score matrix are visible in both the scene and lens. Future attention positions are masked.
- [x] One animated training step exposes the batch, forward computations, per-example loss, reverse parameter gradients and actual SGD updates. LM-head gradients appear before lower layers; the embedding is last. Before/gradient/after views and the largest changed coordinates show true recorded values, with clipping included in the update formula.
- [x] Reset to seed 42, batch 4, learning rate 0.05, then train 300 in the browser: **8.26 seconds**. Corpus loss fell from **3.340265 to 0.562799**, accuracy reached **77.48%**, and probes predicted `the` (98.45%), `mat` (92.68%) and `rug` (88.51%). The shipped starter and Node benchmark reproduce these numbers.
- [x] Optional WebGPU projection verification ran: maximum absolute difference from the Float64 CPU reference **5.827 × 10⁻⁷**, with **2.00 ms** dispatch/readback. This is a selected projection check, not full-graph GPU training.
- [x] All **12 guided-tour stops** completed, including the training and gradient stops; Finish returned control normally. Glossary and source views were exercised.
- [x] `npm run build` passes TypeScript and produces a working static `dist/`. The production preview loads its model/tokenizer workers and real samples. No runtime warnings or errors appeared in the inspected browser logs. Vite reports the expected large-entry-chunk advisory because the entry includes three.js.
- [x] WebGL2 reported steady **60 FPS** in the inspected model, layer and tile views and at a numeric-cell view with **640 instances**. Camera movement briefly showed approximately **55 FPS**. The hard cell cap is **25,600**; sustained slow frames reduce rendering quality. The full cap and other hardware were not stress-tested, so a universal 30-FPS floor is not claimed.

## Honest implementation boundaries

The full 27B model is explored through its metadata and sampled real parameters. All live activations, predictions, gradients and training belong to the separately labelled 278,932-parameter Micro-Qwen. The browser does not run full-Qwen inference, vision or MTP computation.

Playback advances through recorded operations with numerical matrices and a schematic flow ribbon. It does not animate every scalar multiply, build attention scores one cell at a time, or continuously interpolate every changing weight. First-sentence operations are fully expanded; remaining batch sentences have true loss/probability summaries and contribute to the aggregate gradients and updates. Preview matrices show head 0; full recorded arrays remain inspectable.

Adjacent real 256-grid regions load at individual-cell zoom through the same guarded loader. One previous tile is retained dimmed for continuity. Sparse embedding rows and Micro-Qwen tensors use explicit coordinate controls. The renderer uses WebGL2; deterministic reference math runs in a CPU worker, with WebGPU available for the optional projection comparison.

Source files, benchmark settings, data accounting, architecture references and all tests are included in the project. The README provides run instructions and the same scope distinctions.
