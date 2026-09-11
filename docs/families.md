# Three additional architecture families

The additional tabs teach **OpenAI GPT-oss-20B**, **Meta Llama 3.2 1B** and **Google Gemma 3 1B**. Their public architecture information is `[config]`; numerical weights, activations, token IDs, gradients and predictions in their teaching simulations are `[micro]`. No checkpoint weights, safetensors headers or production tokenizer were downloaded for these three families. The original Qwen real-weight explorer is unchanged.

## Metadata and provenance

[public/data/families.json](../public/data/families.json) is keyed by `gpt-oss`, `llama` and `gemma`. Every entry supplies `name`, `organization`, `repo`, `revision`, `config`, `features`, `sources`, `limitations` and `license`. Additional `geometry` fields normalize dimensions and layer patterns for the scene. `configStatus`, `configProvenance` and `mechanisms` distinguish direct checkpoint configuration from reference-derived facts. `microImplementation` records the separate working model dimensions and parameter counts.

GPT-oss's public `config.json` was downloaded at revision **6cee5e81ee83917806bbde320786a8fb61efebee**. Its original JSON contents are preserved under `config`; provenance includes the original response's SHA-256 digest and byte length. [Pinned official configuration](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json).

Llama and Gemma configuration downloads returned **HTTP 401** because those checkpoint files are gated. No credentials were sought and no third-party weight mirrors were substituted. Public Hugging Face API revisions are recorded to identify the selected models; **their configuration objects are explicitly labelled `official-reference-derived`**, not downloaded checkpoint files. Llama dimensions come from Meta's pinned SKU list; Gemma dimensions come from Google's pinned `Gemma3_1B` class. Their model cards provide context and shared-embedding details. The exact Llama checkpoint RoPE rescaling values remain unverified and are omitted from the normalized metadata. [Meta SKU source](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py), [Google configuration source](https://github.com/google-deepmind/gemma/blob/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/gm/nn/_gemma.py).

The data file records licenses as **Apache-2.0**, **Llama 3.2 Community License**, and **Gemma Terms of Use**, with official links. These identify the source models' terms; this project does not redistribute their checkpoint weights.

## GPT-oss: choosing expert pathways

The real 20B configuration has 24 decoder layers, hidden width 2,880, 64 query heads, eight KV heads and head dimension 64. Layers alternate a 128-token local attention window and full causal attention, starting with local. Each layer contains 32 feed-forward experts; a token selects four. Embeddings and output weights are untied. The reference uses direct-weight RMSNorm and biases in attention projections, router and expert projections. It has neither Q/K normalization nor an attention output gate. [Official model implementation](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py).

For each token, the router computes expert scores, selects the four highest logits, and applies softmax **over those selected logits**. Their probabilities weight the selected expert outputs. The production sparse design executes selected experts for that token; this micro reference evaluates all candidate experts and combines only the selected outputs. It demonstrates the routing numerics, without reproducing sparse-kernel compute savings. Top-k selection is discrete; the local backward calculation differentiates through the selected paths and router probabilities, not through changes in expert membership.

The expert activation differs from ordinary SwiGLU:

```text
g = min(gate_projection, 7)
u = clamp(up_projection, −7, 7)
activation = g × sigmoid(1.702 × g) × (u + 1)
expert_output = down_projection(activation)
output = sum(selected_expert_probability × expert_output)
```

The gate is upper-clamped only; negative gate values do not receive a −7 clamp. Projection biases are separate from the activation's explicit `+1`. Source functions are OpenAI's `swiglu` and `MLPBlock.forward`.

A learned **attention sink** adds one extra logit per query head after the token scores are scaled and masked. Softmax includes this extra slot, then the value sum uses only token slots. The sink absorbs probability without contributing a value; displayed token probabilities can therefore sum to less than one. The sink logit itself is not multiplied by the attention scaling factor. Source: OpenAI `sdpa`, also [Transformers `GptOssTopKRouter`, `GptOssExperts`, and attention](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/gpt_oss/modeling_gpt_oss.py).

## Llama: dense attention and shared embeddings

The public Meta SKU specifies 16 layers, hidden width 2,048, 32 query heads, eight KV heads and vocabulary 128,256. Head dimension is `2048 / 32 = 64`. Applying Meta's feed-forward rounding formula to its multiplier 1.5 and multiple 256 yields intermediate width 8,192. Every layer uses full causal attention. [Meta SKU](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py).

Each layer follows:

```text
h = x + attention(RMSNorm(x))
y = h + down(SiLU(gate(RMSNorm(h))) × up(RMSNorm(h)))
RMSNorm(x) = x / sqrt(mean(x²) + epsilon) × weight
```

Normalization weights directly scale their inputs and begin at one. Q/K have no additional normalization; attention and feed-forward projections have no biases. Rotary positions cover the full head dimension, with base frequency 500,000 and scaled frequencies in the real model. The micro simulation's ordinary short-context RoPE is an explicit simplification. [Meta `RMSNorm`, `Attention`, `FeedForward`, and `TransformerBlock`](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py).

The 1B text model shares input embeddings and output vocabulary weights. Consequently, its embedding tensor receives both the lookup gradient and the output-projection gradient. A word absent from the input can still receive an update through the output softmax; the original untied Qwen embedding explanation must not be reused here. [Meta Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_2/MODEL_CARD.md).

## Gemma: local/global attention and extra normalization

Google's `Gemma3_1B` source specifies hidden width 1,152, intermediate width `6 × 1152 = 6912`, four query heads, one KV head and head dimension 256. Its 26 layers repeat **five local layers then one global layer**, truncating that pattern at layer 25. The local window is 512 tokens. Local RoPE uses base 10,000; global RoPE uses 1,000,000. This 1B model is text-only, and its documented context is 32K, unlike the larger multimodal Gemma 3 variants. [Google architecture](https://github.com/google-deepmind/gemma/blob/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/gm/nn/_gemma.py), [official model card](https://ai.google.dev/gemma/docs/core/model_card_3).

Gemma normalizes both Q and K before applying RoPE. It also normalizes the input and output of each attention/FFN branch:

```text
h = x + post_attention_norm(attention(input_norm(x)))
y = h + post_ffn_norm(FFN(pre_ffn_norm(h)))
RMSNorm(x) = x / sqrt(mean(x²) + epsilon) × (1 + weight)
FFN(x) = down(GELU_tanh(gate(x)) × up(x))
```

The norm's stored weights start at zero and act through `1 + weight`. The activation uses the tanh approximation to GELU. Attention scales scores by `1 / sqrt(query_pre_attn_scalar)`; the 1B scalar is 256. All projections are bias-free. [Transformers `Gemma3RMSNorm`, `Gemma3Attention`, `Gemma3MLP`, and `Gemma3DecoderLayer`](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/gemma3/modeling_gemma3.py).

The input embedding is multiplied by `sqrt(hidden_size)`. The output head reuses that same embedding matrix without applying the input multiplier a second time. These tied weights receive gradients from both uses. [Google `Embedder.encode`, `Embedder.decode`, and `FeedForward`](https://github.com/google-deepmind/gemma/blob/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/gm/nn/_modules.py).

## Teaching limits

Each simulation uses a small independently initialized model, word vocabulary and local corpus. Its layer count, dimensions, head counts, local window, expert count and routing count are shown as micro dimensions; they are not the pretrained model's capacity. Micro-gpt-oss has 97,040 parameters, four layers, hidden width 32, eight query heads sharing one KV head, and four experts with two selected per token. It preserves the real 8:1 grouped-query ratio while changing the expert selection fraction from 4/32 to 2/4. Micro-Llama has 35,872 parameters and four layers; Micro-Gemma has 53,888 parameters and six layers, so its five-local/one-global cycle is complete. Both use hidden width 32 and four query heads sharing one KV head. All three use head dimension eight and a 24-token teaching vocabulary. All three use inspectable CPU calculations and full-dimension short-context RoPE. GPT-oss YaRN scaling, Llama's exact long-context frequency adjustment, MXFP4 quantization, distributed expert execution and large-scale training objectives are outside this demonstration.

Each family ships with a reproducibly trained 300-step starter. Reset replaces it with actual seed-initialized weights; predictions and gradients then reflect that untrained state until more updates run. [Micro implementation and measured training checks](micro-families.md) records the test cases, learned probabilities, and explicit simplifications.

For a local window of width `w`, the causal mask permits keys `t − w + 1` through `t`, including the current token. Norm conventions, attention sinks, embedding tying and gated-activation equations remain substantive family differences. Applying the same tiny dense Transformer to all tabs and changing only its name would not demonstrate these architectures.
