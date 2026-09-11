# Verified architecture reference

Verified 11 September 2026 against the actual public checkpoint and primary implementation sources. This document explains the structure and equations; it does not claim that loading a weight tile runs the full model.

## Checkpoint and source of truth

The requested `Qwen/Qwen3.8-27B` repository exists. The API, configuration and safetensors index all returned HTTP 200. The inspected checkpoint revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

- [Pinned configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
- [Pinned safetensors index](https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json)
- [Pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)

The index contains 1,199 tensor names assigned to 18 shards. The 18 downloaded shard headers contain exactly the same 1,199 unique names, with no missing, extra or duplicate tensors. Products of their dimensions sum to **27,781,427,952 stored parameters**, including vision and MTP. The index's `metadata.total_size` is 55,562,855,904 bytes, or about 55.56 decimal GB / 51.75 GiB, and matches the summed header byte ranges. The index supplies names and shard assignments, **not shapes**. Each shard's safetensors header supplies shape, dtype and data offsets; shape tests must compare rendered metadata with those headers. Do not infer counts solely from the checkpoint's rounded “27B” name or divide aggregate byte size by two without checking each tensor's dtype.

Configuration-derived text architecture:

- Top-level `model_type` is `qwen3_5`; text `model_type` is `qwen3_5_text`.
- 64 decoder layers, in 16 repetitions of three `linear_attention` layers followed by one `full_attention` layer. Layer indices are zero based: layer 3 is the first full-attention layer.
- Hidden size 5,120; gated feed-forward intermediate size 17,408; vocabulary 248,320. Embedding and LM output weights are untied.
- Full attention has 24 query heads and four key/value heads, each 256 dimensions. Six query heads share each key/value head. RoPE rotates 64 of 256 head dimensions, with base 10,000,000.
- DeltaNet has 16 query/key heads and 48 value heads, each 128 dimensions. Query and key heads repeat three times to match value heads. The depthwise causal convolution has kernel width four.
- The vision encoder has 27 layers and hidden size 1,152. One multi-token-prediction layer is configured and MTP tensor names exist in the index.
- The configured native position limit is 262,144. The model card describes extension to 1,000,000; do not present one million as the default loaded configuration or as a measured browser capability.

## Tensor axes and head bands

PyTorch linear weights are stored `[output feature, input feature]`; the operation is `y = x Wᵀ`. For `W[row, column]`, the column identifies the input feature and the row identifies the output feature. The illustrative index interpretation in the brief reverses these roles and must not be copied.

Verified shard-header examples:

- Layer 3 `self_attn.q_proj.weight`: `[12288, 5120]`.
- Layer 3 `self_attn.k_proj.weight` and `v_proj.weight`: `[1024, 5120]` each.
- Layer 3 `self_attn.o_proj.weight`: `[5120, 6144]`.
- Layer 0 `linear_attn.in_proj_qkv.weight`: `[10240, 5120]`.
- Layer 0 `linear_attn.in_proj_z.weight`: `[6144, 5120]`.
- Layer 0 `linear_attn.in_proj_a.weight` and `in_proj_b.weight`: `[48, 5120]` each.
- Layer 0 `linear_attn.conv1d.weight`: `[10240, 1, 4]`; `norm.weight`: `[128]`; `A_log` and `dt_bias`: `[48]` each.
- Layer 0 `linear_attn.out_proj.weight`: `[5120, 6144]`.
- Layer 0 feed-forward gate/up weights: `[17408, 5120]`; down weight: `[5120, 17408]`.

The full-attention Q projection contains **both query and output-gate parameters**. In `Qwen3_5Attention.forward`, the projection is reshaped to `[..., heads, 2 × head_dim]`, then split along the last dimension. Each real head therefore occupies a contiguous 512-row band: 256 query rows followed by 256 gate rows. Head 0 query rows are `[0, 256)`; its gate rows are `[256, 512)`; head 1 begins at row 512. Showing 24 bands across 12,288 rows is correct only if these query/gate subdivisions are explained. A 256-row tile at the beginning of this matrix contains head 0's query parameters, not a whole gated head. A key head occupies 256 rows.

DeltaNet's combined QKV projection uses three contiguous sections: 2,048 query rows, 2,048 key rows, then 6,144 value rows. It is not split into equal thirds.

Source: [Transformers `Qwen3_5Attention`, `Qwen3_5GatedDeltaNet`, pinned source](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## DeltaNet equations used for a readable recurrent implementation

Use the recurrent formulation in `torch_recurrent_gated_delta_rule`, equivalent to the chunked formulation for the same inputs. For each head and token, state `S` has shape `[key dimension, value dimension]`.

First project normalized hidden states, apply depthwise causal convolution and SiLU to the combined QKV stream, then split Q/K/V. Normalize q and k by their L2 norms, with epsilon. Scale q by `1 / sqrt(key dimension)`. Value heads use the corresponding repeated query/key head.

```
beta = sigmoid(in_proj_b(x))
g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
S_decayed = exp(g) * S_previous
prediction = kᵀ S_decayed
delta = beta * (v - prediction)
S = S_decayed + outer(k, delta)
readout = qᵀ S
output = out_proj(RMSNormGated(readout, in_proj_z(x)))
```

`RMSNormGated(readout, z)` normalizes each value head, multiplies by its learned norm weight, and then multiplies by `SiLU(z)`. The decay occurs **before** the key reads the old memory. The subtraction is the delta correction; replacing it with a plain outer-product accumulation would implement a different model.

The recurrent state changes during inference, but those changes are **not optimizer updates to model weights**. Training gradients backpropagate through the state updates, convolution and all projections.

Sources: [Transformers functions `torch_recurrent_gated_delta_rule`, `l2norm`, `causal_conv1d_fn`; classes `Qwen3_5GatedDeltaNet` and `Qwen3_5RMSNormGated`](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py); [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464); [authors' implementation](https://github.com/NVlabs/GatedDeltaNet).

## Full attention, normalization and feed-forward

Full attention performs the Q/gate projection and K/V projections, Q/K RMSNorm, partial RoPE, grouped-query causal attention, sigmoid gating, then output projection. The gate applies **before** `o_proj`.

```
scores = Q Kᵀ / sqrt(head_dimension) + causal_mask
probabilities = softmax(scores)
mixed = probabilities V
output = o_proj(mixed * sigmoid(gate))
```

For ordinary pre-block, final, Q and K normalization, `Qwen3_5RMSNorm` stores zero-centered learned weights and multiplies normalized inputs by `(1 + weight)`. The Delta output's `Qwen3_5RMSNormGated` instead multiplies directly by its learned weight. These parameter conventions must be preserved when interpreting real norm tensors.

For text-only partial RoPE, `apply_rotary_pos_emb` rotates the first configured fraction of each head and leaves the rest unchanged. `rotate_half` pairs the first half of the rotary subvector with its second half. At a micro head dimension of 16 and fraction 0.25, the rotary subvector is four dimensions: its paired coordinates are 0↔2 and 1↔3.

Feed-forward is `down_proj(SiLU(gate_proj(x)) * up_proj(x))`. The decoder block is `x + mixer(RMSNorm(x))`, followed by a second normalized feed-forward residual addition. A hidden size of 64 with six 16-dimensional query heads is valid: the 96-dimensional attention stream is projected back to 64; there is no requirement that the projection width equal the residual-stream width.

Source: [Transformers `Qwen3_5Attention`, `eager_attention_forward`, `repeat_kv`, `Qwen3_5RMSNorm`, `apply_rotary_pos_emb`, `rotate_half`, `Qwen3_5MLP`, `Qwen3_5DecoderLayer`](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## Resolved swish / sigmoid question

Qwen3.8's configuration includes `output_gate_type: "swish"`. Reading this as the full-attention gate would appear to conflict with Transformers' sigmoid gate. The primary SGLang implementation resolves the scope: `Qwen3_5GatedDeltaNet.__init__` reads `config.output_gate_type` and passes it as the activation for **DeltaNet's RMSNormGated**. The full-attention `self_attention` method separately applies sigmoid. Swish/SiLU for Delta output and sigmoid for full attention are therefore consistent, not an architecture deviation. Transformers v5.8.0 and the inspected current source use those same two activations.

Sources: [SGLang pinned `Qwen3_5GatedDeltaNet` and `self_attention`](https://github.com/sgl-project/sglang/blob/52fecfdf0908dca24f4c6799ff5967125cc4110e/python/sglang/srt/models/qwen3_5.py), [Transformers v5.8.0](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## Real tile transport probe

A request to shard 1 with `Range: bytes=0-7` and `Origin: http://localhost:5173` returned HTTP 206, eight bytes, `Content-Range: bytes 0-7/3966730552`, `Access-Control-Allow-Origin: *`, and exposed response headers. Those eight bytes encoded a 45,392-byte JSON header. A second range request returned exactly those 45,392 header bytes, from which the examples above were read.

This proves the server supports ranged responses and advertises permissive CORS for that probe. A real browser fetch remains a separate integration check. The loader must validate status, byte count and content range before accepting data, and reject any server response that tries to deliver the entire multi-GB shard. Cache keys must include checkpoint revision and tile coordinates; a URL query differentiating ranges can avoid incorrectly reused CDN responses.

For 2-D row-major tensors, a small rectangular tile is generally not contiguous in the shard: fetch one interval per row or a budgeted larger contiguous band and extract columns. Count actual fetched bytes against the download budget, including bytes discarded from a larger band.

## Teaching qualifications

Real tensor values are `[real]`; computations of dimensions, storage and parameter totals are `[config]`; micro model activations, gradients and predictions are `[micro]`. A real weight heatmap does not imply full-model inference. Statistical summaries of tiles describe only their loaded samples.

The statement “an embedding changes only when that word appears in the batch” applies to the lookup gradient in this demonstration when embeddings are untied and plain SGD has no weight decay or momentum. It is not universally true for tied LM-head weights or other optimizers. The backward animation should explain dependency tracing from loss to inputs, rather than suggesting gradient direction is caused by the arbitrary vertical arrangement of the drawing.
