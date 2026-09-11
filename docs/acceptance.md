# Acceptance record

This file records measured results, not intended features. Unchecked items are not verified.

## Numerical foundation

- [ ] Model snapshot pins config, index and all safetensors tensor headers to the same Hugging Face revision.
- [ ] Every index tensor has a validated shape, dtype and byte range.
- [ ] Loaded BF16 values are converted to Float32 without synthesizing data.
- [ ] Data downloads stay within the 150 MB budget.
- [ ] Full Micro-Qwen backward passes finite-difference checks.
- [ ] The same seed produces the same loss curve.
- [ ] Training performance and the cat/dog continuations are measured.

## Browser verification

- [ ] Model view derives layer count and parameter accounting from the snapshot.
- [ ] Layer view exposes exact real tensor names.
- [ ] Layer 3 query projection shows real loaded tiles and correct query/gate head bands.
- [ ] Unloaded tiles remain visibly unloaded and can be fetched.
- [ ] Cell view shows numerical values and correct output-row/input-column semantics.
- [ ] Inference controls play, pause, step forward/back, scrub and change speed.
- [ ] Calculation lens displays recorded numerical operations.
- [ ] DeltaNet state and causally masked attention are visible.
- [ ] Training shows loss, backward gradients and actual before/after updates.
- [ ] Guided tour completes and escape navigation works.
- [ ] Production build loads and worker operations run.
- [ ] Runtime performance is measured; hardware and renderer are recorded.

## Evidence

Pending implementation and measurement.
