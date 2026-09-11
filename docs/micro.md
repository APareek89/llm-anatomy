# Micro-Qwen: actual calculations in a small teaching model

Micro-Qwen is a separately initialized, separately trained model. It does not use downloaded Qwen weights. Its numbers must always be labelled `[micro]`. It has 278,932 trainable parameters in the default configuration and a 24-word vocabulary (including four special tokens).

The default model has a 64-dimensional residual stream and eight layers. Layers 0, 1, 2, 4, 5 and 6 use Gated DeltaNet. Layers 3 and 7 use gated full attention. Every mixer and every feed-forward block has a pre-normalization and a residual connection. The feed-forward intermediate width is 128, using SiLU(gate) × up, followed by down projection.

Full attention has six query heads, one shared key/value head, and head dimension 16. Its query projection has shape `[192,64]`: each head contributes 16 query coordinates followed by 16 gate coordinates. Query and key vectors use zero-centered RMSNorm, whose learned multiplier is `1 + weight`. The first four coordinates in each head use rotary position embedding. The rotation pairs coordinate 0 with 2, and 1 with 3; this follows the half-split reference convention. Causal scaled dot-product attention uses `softmax(QKᵀ/√16) V`. The attention output is multiplied by a sigmoid gate before the output projection.

DeltaNet uses one query/key head and three value heads, keeping the official 1:3 ratio. Its key and value dimensions are eight. A causal depthwise convolution of kernel size four operates on Q/K/V, followed by SiLU. Queries and keys are L2 normalized. For each value head and token, with a zero initial state, it computes:

```
g = −exp(A_log) × softplus(a + dt_bias)
beta = sigmoid(b)
S_bar = exp(g) × S_previous
prediction = kᵀ × S_bar
delta = beta × (v − prediction)
S_next = S_bar + k ⊗ delta
output = qᵀ × S_next / sqrt(key_dim)
```

The state output is RMS normalized with a conventional learned weight, multiplied by `SiLU(z)`, and projected to the residual width. Unlike the attention gate, this DeltaNet output gate uses SiLU. Full operation order and references are in [architecture-reference.md](./architecture-reference.md).

The source equations are the official Transformers [`torch_recurrent_gated_delta_rule`, `Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, and `Qwen3_5RMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py). This implementation uses the direct recurrent form, not the GPU chunked implementation. The real model's multi-token prediction head and vision encoder are omitted from the micro model. Its tokenizer is word based, not Qwen's tokenizer.

## Explicit forward and backward

`tensor.ts` has a vector-valued reverse-mode tape. Each matrix multiplication, normalization, activation, slice, lookup and residual operation has an explicitly written backward loop. `mixers.ts` has explicitly differentiated causal convolution, rotary embedding, grouped attention and recurrent DeltaNet memory updates. No external machine-learning library is used. Float64 arrays make finite-difference checking and seeded runs stable.

Recording copies exact input and output arrays, shapes, parameter gradients, formulas and arithmetic. The recorder also exposes intermediate causal score matrices, softmax matrices and per-token DeltaNet states. An animated batch records the full forward trace of its first example; the batch card identifies this explicitly. The batch card also records the true probability and −log probability of every correct next word, plus each sentence mean loss and the batch mean. Backward parameter heatmaps and SGD update records use the actual average gradient from every example in the batch. Displayed arithmetic is rounded to seven significant digits; retained arrays contain the complete Float64 values.

The trainer uses cross-entropy at every position in each complete corpus sentence, including BOS input and EOS target. Gradients are averaged across examples. It clips the global gradient norm to one and performs actual SGD:

```
clip_scale = min(1, 1 / gradient_norm)
weight_after = weight_before − learning_rate × clip_scale × gradient
```

The displayed update includes the clipping factor, so the shown arithmetic can be checked against the weight arrays. There is no hidden optimizer, hand-authored prediction distribution, transition table or output override. Only the embedding rows looked up in a batch receive gradients through the embedding operation; the untied output head is a separate parameter matrix.

## Measured training result

On this development Mac, Node 24/tsx ran 300 steps from seed 42 with batch size 4 and learning rate 0.05 in 6.5 seconds, including the initial evaluation. This is a measured CPU reference result, not a guarantee about every browser/device. The deterministic 30-sentence corpus lives in `corpus.ts`.

- Corpus mean loss: 3.34026494 → 0.56279922.
- Next-word accuracy across all corpus positions: 77.48%. Some prefixes have several valid next words, so accuracy cannot reach 100% by memorizing a unique next token.
- `the cat sat on` → `the`, probability 0.98449129.
- `the cat sat on the` → `mat`, probability 0.92678480.
- `the dog sat on the` → `rug`, probability 0.88512755.

Run `npx tsx --test tests/gradcheck.test.ts tests/determinism.test.ts tests/training.test.ts` to verify. The training acceptance test performs all 300 steps and checks the actual learned predictions, then checks checkpoint and RNG continuation. Reproduce the shipped starter weights with `npx tsx scripts/train-starter.ts`; the exact state is written to `public/data/micro-checkpoint.json` and the measured run, hardware/runtime metadata and complete training history are written to `public/data/micro-benchmark.json`. The full tiny architecture checks 250 finite differences across every parameter tensor, including both mixer families, feed-forward blocks, embeddings and head. It also tests nonzero rotary positions, multiple KV groups and recurrent carry across four tokens. The observed maximum normalized whole-model gradient error was `2.30e-7`; the threshold is `3e-5`. The deterministic test compares complete updated parameter arrays after the same seeded batches. Another test verifies every recorded update cell against the actual SGD formula.

## Worker protocol

The browser must use `new Worker(new URL('./model/micro/worker.ts', import.meta.url), {type:'module'})`. All inference, loss, backward, update, evaluation and optional GPU verification run there. Rendering does not execute model calculations.

Requests use `{id,type,...}`. Supported types:

- `init`: optional `config`, `seed`, `checkpoint` or `checkpointUrl`; returns `ready` with `config`, `vocab`, `parameters`, `parameterCount`, `step`, `corpus`, and `backend`.
- `infer`: `prompt`, optional `temperature`, `seed`, and `record`; returns `inference` with `result`. Temperature zero selects argmax. A positive temperature uses a seeded categorical draw.
- `train`: `steps`, `batchSize`, `learningRate`, and `record`; progress every ten steps, final `training` response includes `history` and complete corpus `evaluation`. One animated step records forward, gradients and update. Fast training normally sets `record:false`.
- `stop`: cancels between steps. The worker yields its event loop every two steps.
- `reset`: optional seed; clears training state and reconstructs exact seeded weights.
- `evaluate`: complete corpus loss and accuracy.
- `parameter`: `name`; returns the current parameter values and most recent gradient.
- `export` / `import`: exact checkpoint serialization; includes RNG state and training step.
- `gpu`: `input`, `weight`, `rows`, `inputDim`, `outputDim`, optional `expected`; computes a displayed projection on WebGPU and reports measured discrepancy versus the CPU reference.

Responses with calculations put them under `result`; `ready` and `reset` put metadata at the top level. Every request response echoes its `id`. Failures return `{id,type:'error',error}`.

## Honest limits

The primary execution path is a deterministic CPU worker. WebGPU can independently compute the projection in the calculation lens; it does not accelerate the full autograd training graph. WebGL2 is a rendering fallback, not an implementation of general model training. The context limit is 64 micro tokens and training batches are capped at 16. Detailed recorded traces intentionally consume more memory than fast training, because they retain actual arrays for inspection. Full Qwen inference/training is not provided by this browser model.
