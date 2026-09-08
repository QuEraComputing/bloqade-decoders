# Changelog

## 0.7.0-DEV

### Breaking changes

- Standardized decoder construction around `BaseDecoder`, `instantiate`, and
  `_instantiate`.
- Standardized decoder APIs around `decode`, `decode_confidence`, `_decode`, and
  `_decode_confidence` for both single-shot and batched decoding.
- Simplified `TableDecoder.train` so training samples from the provided detector
  error model instead of accepting independent or pre-sampled training data.

### Other changes

- Relaxed the `numpy` floor from `>=2.2.6` to `>=1.26.4` so downstream projects
  pinned to numpy 1.x (e.g. through TensorFlow) can install `bloqade-decoders`.

## Earlier versions

Changelog not kept before this version.
