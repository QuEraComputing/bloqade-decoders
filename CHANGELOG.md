# Changelog

## 0.7.0-DEV

### Breaking changes

- Standardized decoder construction around `BaseDecoder`, `instantiate`, and
  `_instantiate`.
- Standardized decoder APIs around `decode`, `decode_confidence`, `_decode`, and
  `_decode_confidence` for both single-shot and batched decoding.
- Simplified `TableDecoder.train` so training samples from the provided detector
  error model instead of accepting independent or pre-sampled training data.

## Earlier versions

Changelog not kept before this version.
