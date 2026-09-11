# LLM anatomy

Four architecture tabs with real miniature-model inference and training: **Qwen3.8**, **OpenAI gpt-oss**, **Meta Llama 3.2**, and **Google Gemma 3**. This is an interactive 3D learning lab. The predictions, activations, expert routes, gradients and updates are computed from each model’s own weights; they are not scripted.

**These are teaching models, not official pretrained checkpoints.** Each miniature ships with weights genuinely trained for 300 steps on the same 30-sentence, 24-word corpus. They demonstrate architecture and learning, not the language quality of the released models. Qwen’s Explore tab additionally provides the original checkpoint’s full metadata and authentic sampled vendor weights. The real Qwen tokenizer remains available as a clearly labelled comparison across tabs.

The added models are materially different:

- **Micro-gpt-oss:** 97,040 parameters, four layers, eight query heads sharing one KV head, alternating local/full attention, learned sinks, and two selected experts out of four. Its router probabilities drive the 3D expert highlights.
- **Micro-Llama:** 35,872 parameters, four dense GQA layers, direct RMSNorm scales, full rotary positions, SwiGLU, and shared embedding/output weights.
- **Micro-Gemma:** 53,888 parameters, five local layers followed by one global layer, Q/K normalization, pre/post branch norms, GELU gates and scaled, shared embeddings.

The three new families each have their own nine-stop tour. Qwen retains its twelve-stop tour. Tabs preserve independent weights and training history for the browser session; a reload restores the shipped starters. Use `?model=gpt-oss`, `?model=llama` or `?model=gemma` to link directly to a family. The persistent **Real computation · teaching weights** button explains the distinction at any time.

Source configurations and exact implementation boundaries are in [family references](docs/families.md) and [micro-family mathematics and benchmarks](docs/micro-families.md). Meta and Google checkpoint configuration downloads were gated; their reference geometry is explicitly derived from pinned official source code/model cards. No Meta, Google or OpenAI vendor weight/tokenizer downloads are represented as present.

**Verified:** 28 automated tests pass, including finite-difference checks for all three new families, deterministic training, learned continuations, and all existing Qwen checks. All new tours completed in the browser. Gemma’s 300-step browser rerun took 2.96 seconds and learned `the` / `mat` / `rug` at 98.5% / 99.1% / 96.3%. These timings are observations on the development Apple M4, not device-independent guarantees.

## Run

Use Node **20.19+** and a desktop browser with WebGL2. Development was tested with Node 24 on macOS.

```sh
cd ~/Documents/ML/llm-anatomy
npm install
npm run dev
```

Open the localhost address printed by Vite. The real metadata, tokenizer, default weight samples and trained micro starter are already bundled; opening the app does **not** download the 27B model.

```sh
npm test                 # Shapes, byte provenance, tokenizer, gradients, determinism and training
npm run test:gradcheck   # Numerical gradient gate alone
npm run build           # TypeScript check and static dist/ build
npm run preview         # Serve the production build
npm run prefetch        # Verify/reuse the real default samples; fetch only missing files
```

Serve `dist/` with an HTTP static server, rather than opening its HTML as a local file. Recreate the Qwen starter with `npx tsx scripts/train-starter.ts`, or all three new-family starters with `npx tsx scripts/train-families.ts`.

## Start exploring

- **Explore:** orbit with drag, pan with right-drag, zoom with the wheel, double-click to focus, and press Escape or Back to move up. Search for `layer 3` or `q_proj`. “Inspect a weight” opens the real first attention layer's query projection. Zoom closer for individual cells and numbers; use head buttons and row/column controls to choose another tile.
- **Inference:** use `the cat sat on`, run the forward pass, and set playback to `0.10×`. Play, pause, step in either direction, scrub, jump to a layer, or detach the camera. The calculation lens shows exact recorded arrays, formulas, a checkable numerical example, DeltaNet memory and causally masked attention. “Next token” appends the chosen word and computes the next pass.
- **Training:** animate one step for the batch, forward operations, loss, backward gradients and actual before/after updates. Reset weights to seed 42, then run **Train 300 fast** to reproduce learning. Batch size, learning rate, loss history and the largest measured updates remain inspectable.
- **Take the tour:** twelve stops connect a single parameter to embeddings, both attention families, feed-forward layers, sampling, gradients and fine-tuning. The glossary explains the vocabulary.

## Qwen reference data and labels

**`[real]`** means downloaded Qwen weight values or token IDs produced by its actual tokenizer. **`[config]`** means dimensions, geometry, parameter counts and storage totals computed from the pinned config, index and all safetensors headers. **`[micro]`** means the separately initialized teaching model's real activations, predictions, gradients and weight updates. Micro-Qwen never substitutes its values for missing Qwen samples.

The pinned checkpoint has **64 language layers**, **1,199 stored tensors**, **27,781,427,952 parameters** and **55,562,855,904 weight bytes** across 18 shards. Vision and multi-token prediction are included in those totals. The configured native context is **262,144 tokens**; the model card's one-million-token extension is not the browser model's capacity.

Real linear weights are stored **[output feature, input feature]**. Layer 3's query projection is `[12288,5120]`: each of 24 heads contains 256 Q rows followed by 256 output-gate rows. The default bundle includes a complete 512-row gated Q head and a complete K head, alongside the requested layer 0/3 tiles and sparse embedding/LM-head rows.

## Actual training, with a reproducible starter

Micro-Qwen has **278,932 parameters**, hidden width 64, eight hybrid layers and a 24-word vocabulary. It preserves the three-DeltaNet/one-attention pattern, six-query/one-KV-head grouped attention, partial rotary positions, gated delta-rule recurrence, normalization, residual connections and gated feed-forward blocks. Its explicit forward and backward implementations use no ML library.

The shipped starter was genuinely trained for 300 steps. From seed 42, batch size 4 and learning rate 0.05, the recorded **Node CPU run took 6.36 seconds on an Apple M4**. The production in-app browser separately reproduced 300 steps after reset in **8.26 seconds**, using the same seed, batch size and learning rate. These are device-specific measurements. Corpus mean loss fell from **3.3403 to 0.5628**, with **77.48%** next-word accuracy. Learned continuations were:

- `the cat sat on` → `the` (**98.45%**).
- `the cat sat on the` → `mat` (**92.68%**).
- `the dog sat on the` → `rug` (**88.51%**).

The browser reset-and-train run reproduced these continuations at 98.4%, 92.7% and 88.5%. These come from model outputs, with no hand-written prediction override. The [benchmark and complete loss history](public/data/micro-benchmark.json) and [starter checkpoint](public/data/micro-checkpoint.json) are included. Finite-difference checks cover every parameter tensor in a tiny configuration; the measured maximum normalized gradient error was approximately **2.30×10⁻⁷**. [Micro-Qwen equations and tests](docs/micro.md) explain the implementation.

## Download budget and reproducibility

The initial bundle used **65,019,823 of 150,000,000 network bytes**. It stores **101 real blocks / 20,278,144 Float32 bytes**, plus the actual 12.8 MB tokenizer and metadata. BF16 source values convert exactly to Float32. Network accounting includes headers, tokenizer and unused columns inside retrieved row bands; cached file size is therefore different from downloaded source size.

The Node prefetch path supplied the bundled defaults. Other tiles use guarded browser HTTP Range requests and IndexedDB caching. Every range must return HTTP 206 with matching offsets and length; a full-shard response is cancelled. Missing values remain visibly unloaded. Browser Range/CORS was verified in the production in-app browser: a previously uncached layer-3 Q-projection tile at output row 512, input column 0 loaded directly from Hugging Face. Its 256 × 256 real values had mean approximately −0.00002399, standard deviation 0.01462, minimum −0.07275 and maximum 0.10400. If browser fetching fails, the app provides a Node fallback command carrying its current `--browser-bytes` usage.

Counters reject malformed state, reserve in-flight bytes and coordinate browser tabs; a process lock protects the Node ledger. Checksums validate cached files. Preserve the accounting files and browser storage; do not interpret their remaining balances as separate 150 MB allowances. Details and fallback commands are in [data provenance](docs/data.md).

## Scope and remaining limits

- The 3D renderer uses **WebGL2**. Model math and the real tokenizer run in workers. An optional **WebGPU calculation-lens check** recomputes selected projections in Float32 and reports their difference from the deterministic CPU reference; it does not accelerate the full training graph. The browser check measured maximum absolute error **5.827 × 10⁻⁷**, with **2 ms** dispatch and readback.
- Live inference/training belongs to Micro-Qwen, with a 64-token context and batches up to 16. Real Qwen inference, fine-tuning, vision computation and multi-token prediction are not executed. Vision/MTP are still present in the real model anatomy.
- Recorded training expands every operation for the **first sentence** in a batch, shows true losses for the other sentences, and retains **aggregate batch gradients and updates for every parameter tensor**. Attention/state previews show **head 0**; recorded operation arrays remain available for numerical inspection. Arithmetic text is rounded; stored reference arrays retain Float64 precision.
- Playback presents operation snapshots. It does not animate every scalar multiply or continuously interpolate every updated cell. At cell-level zoom, panning across a regular real tensor loads the adjacent 256 × 256 tile automatically and retains one previous tile dimmed for orientation. Sparse embedding rows and Micro-Qwen use explicit controls; the view does not retain an unlimited matrix surface.
- Rendering caps visible cell instances at **25,600** and reduces quality after sustained slow frames. In the production in-app browser, the WebGL2 overview, layer and tile views displayed steady **60 FPS**; a view with **640 numerically labelled cells** also displayed 60 FPS, with a transient 55 FPS during camera movement. All twelve guided-tour stops completed. These measurements describe the tested browser/device, not a universal performance guarantee. The [acceptance record](docs/acceptance.md) records the verification evidence.

## Project map

```text
src/data/                 Pinned metadata, Range loader, cache and real tokenizer worker
src/model/micro/          Explicit autograd, hybrid model, corpus and model worker
src/model/gpu.ts          Optional WebGPU projection verification
src/scene/                Three.js anatomy, zoom levels, heatmaps, cells and camera
src/playback/             Recorded-step controller and worker client
src/ui/                   Cards, glossary, tour content and styling
scripts/                  Budgeted real-weight prefetch and micro starter training
public/data/              Config, shard headers, manifests, tokenizer and micro checkpoint
public/weights/           Verified Float32 real-weight samples
tests/                    Shapes, transport, tokenizer, gradients, determinism and training
docs/                     Architecture sources, data provenance, model maths and acceptance
```

Development follows eight phase commits: **1** real data and shapes; **2** Micro-Qwen and gradient gate; **3** model/layer scene; **4** heatmaps/cells; **5** inference/lens; **6** training playback; **7** explanations/tour; **8** performance, verification and documentation. Scene implementation began after the gradient gate passed. The Git history records the completed phase checkpoints.

## Primary sources

The real snapshot is pinned to [`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0): [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json), [weight index](https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json), and the shard headers supplied with this project.

Architecture and recurrent equations follow the [pinned Transformers implementation](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py), the [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464), and the [authors' implementation](https://github.com/NVlabs/GatedDeltaNet). [Architecture reference](docs/architecture-reference.md) names the exact functions and explains head packing, normalization and the different DeltaNet/attention gates.

## Static deployment

Build with `npm run build`. The site needs only static hosting; all miniature model computations run in browser workers. `scripts/deploy-vercel.py` uploads only `dist/`, reads the Vercel token from a local secrets file and never writes it into the site. Its private deployment receipt is excluded from Git. No API key is required by visitors.
