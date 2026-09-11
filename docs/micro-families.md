# Three additional working micro model families

The family tabs run separately initialized and trained models. Their mechanisms follow the released model families; their weights do not come from OpenAI, Meta or Google. All numerical parameters, activations, token IDs, predictions and gradients in these tabs are `[micro]`. Official architectural metadata and source provenance are in `public/data/families.json`.

All three use the same fixed 30-sentence teaching corpus, 24-word vocabulary, explicit Float64 autograd, cross-entropy loss and clipped SGD used by Micro-Qwen. Inference includes the same explicit `<bos>` marker used by training. Each family has its own checkpoint, RNG state and worker session. The existing Qwen model's architecture, initialization, trained weights and predictions are unchanged.

## GPT-OSS miniature

The working model has hidden width 32, four layers, eight query heads sharing one key/value head, head dimension eight, and 97,040 parameters. Even-numbered layers have a four-token sliding window; odd-numbered layers have full causal attention. Each query head has a learned sink logit. The sink is appended after token-score scaling, participates in softmax, and contributes a zero value; real-token probabilities can therefore sum to less than one. Its gradient is computed explicitly.

Each feed-forward block has four learned experts and a router that selects two experts independently for each token. It takes top-k **before** softmax, normalizes only the selected logits, and combines the selected expert outputs. Inactive routing scores have zero gradient within that selection. The selection itself is discrete; the ordinary derivative assumes the selected set stays fixed, and finite-difference tests avoid routing ties.

The expert activation matches the official formula: `g = min(gate, 7)`, `u = clamp(up, −7, 7)`, then `g × sigmoid(1.702g) × (u + 1)`. Query/key/value/output projections, the router and expert projections include learned biases. RMSNorm uses a direct learned scale and epsilon `1e-5`; there is no Q/K normalization or attention output gate. The output head is separate from the embedding matrix.

The mathematical source is the pinned [OpenAI reference implementation, including `sdpa`, `swiglu`, and `MLPBlock`](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py). The tiny reference evaluates all candidate expert outputs, then combines top-k, so it reproduces sparse routing numerics rather than a production sparse-kernel speedup. It uses separate expert projection tensors instead of the packed production layout. Four experts/top-two changes capacity and the selected fraction from the original 32/top-four. The original 8:1 query-to-KV head ratio is retained. MXFP4 quantization, the auxiliary router load-balancing loss and YaRN long-context scaling are omitted. Full ordinary RoPE with base 150,000 is used for the 64-token micro context.

## Llama miniature

The working model has hidden width 32, four layers, four query heads sharing one key/value head, head dimension eight, feed-forward width 64, and 35,872 parameters. Every layer uses full causal attention and an ordinary dense SwiGLU feed-forward block. Both sub-blocks have pre-normalization and a residual connection. RMSNorm uses direct weights with epsilon `1e-5`. The projections have no bias and there is no Q/K normalization or attention output gate.

The input embedding matrix is the **same Tensor object** used by the vocabulary output projection. There is no second output-head parameter or copied weight array. Its gradient accumulates through both lookup and softmax-output paths, so a word absent from the input can still receive an output-head update. This difference is shown in the training explanation and tested.

The source is the pinned [Meta `RMSNorm`, `Attention`, and `FeedForward` reference](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py), with [shared embeddings described in the Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_2/MODEL_CARD.md). The micro uses full ordinary RoPE with base 500,000; it does not implement the checkpoint's long-context frequency rescaling. The 4:1 query-to-KV ratio is retained, while widths, depth and vocabulary are reduced.

## Gemma miniature

The working model has hidden width 32, six layers, four query heads sharing one key/value head, head dimension eight, feed-forward width 64, and 53,888 parameters. Layers 0–4 have a four-token local window, while layer 5 sees the entire causal prefix. This retains the five-local/one-global pattern. Full RoPE uses base 10,000 in local layers and 1,000,000 in global layers.

Input embeddings are multiplied by `sqrt(hidden_width)`. Q and K each receive RMSNorm before rotary embedding. Every RMSNorm is zero-centered: it multiplies by `1 + weight` with epsilon `1e-6`. Each attention and feed-forward branch has both pre-normalization and post-normalization before its residual add:

```
h = x + post_attention_norm(attention(input_norm(x)))
y = h + post_feedforward_norm(FFN(pre_feedforward_norm(h)))
```

The feed-forward gate uses the tanh approximation to GELU, followed by an elementwise product with the up projection and a down projection. All projections are bias free. The output head shares the embedding tensor without applying the embedding-scale factor a second time. As in the Llama micro model, tied embeddings receive gradients from both paths.

The source is the pinned [Google DeepMind model configuration](https://github.com/google-deepmind/gemma/blob/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/gm/nn/_gemma.py), [embedding/FFN/attention modules](https://github.com/google-deepmind/gemma/blob/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/gm/nn/_modules.py), and [Transformers Gemma3 decoder and normalization implementation](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/gemma3/modeling_gemma3.py). The micro attention denominator scales from the real model's `sqrt(256)` to `sqrt(8)` with the smaller head dimension. The model is text-only, like Gemma 3 1B. Its micro window, width, depth and vocabulary are smaller; BF16 rounding of the official embedding scale is not reproduced by the Float64 reference.

## Reproducible results and tests

Run `npx tsx scripts/train-families.ts` to recreate all three exact checkpoints and benchmark files. Each is trained from seed 42 for 300 steps, batch size four, learning rate 0.05. Timings below measure Node CPU training on the development Mac and are not guarantees for every browser/device.

- GPT-OSS: 2.57 seconds; corpus loss 3.316648 → 0.620133; accuracy 76.58%. Learned probabilities for **the / mat / rug** are **98.54% / 94.48% / 92.76%**.
- Llama: 0.96 seconds; corpus loss 6.840212 → 0.735280; accuracy 73.42%. Learned probabilities are **94.37% / 93.37% / 84.78%**.
- Gemma: 1.53 seconds; corpus loss 4.852992 → 0.560866; accuracy 77.48%. Learned probabilities are **98.46% / 99.15% / 96.31%**.

The three probes are `the cat sat on`, `the cat sat on the`, and `the dog sat on the`. All predictions come directly from the trained model softmax; there is no answer table or output override.

`tests/families-gradcheck.test.ts` checks finite differences across every parameter tensor in small complete models, plus isolated sliding attention with sinks, GELU, clipped SwiGLU and top-k routing. Whole-model maximum normalized errors are about `5.7e-10` for GPT, `3.7e-10` for Llama and `4.0e-7` for Gemma. Gemma's repeated post-normalization makes its tiny four-dimensional check sensitive to finite-difference step size, so the common test epsilon is `1e-6` with tolerance `3e-5`.

`tests/families-training.test.ts` performs all 300 steps for each family, checks deterministic seeded batches, verifies the learned the/mat/rug predictions, round-trips checkpoints, validates real routing/sink matrices, checks the five-local/one-global Gemma pattern, checks tied-embedding gradients on an absent input word, and verifies every differentiable recorded input/output gradient has the correct size and finite values.

For complete checks, run `npm test`. The tests also retain the Qwen gradient and 300-step learning regression. Each family checkpoint is stored at `public/data/micro-<family>-checkpoint.json`; the matching benchmark contains complete history, initial/final evaluation and runtime/CPU metadata.
