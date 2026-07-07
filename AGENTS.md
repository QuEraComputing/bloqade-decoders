# Decoder Architecture

This package exposes decoder classes through `bloqade.decoders`. Concrete
decoders live under `src/bloqade/decoders/_decoders/` and inherit from
`BaseDecoder`.

`BaseDecoder` owns the common public API:

- `Decoder(dem, **kwargs)` stores the `stim.DetectorErrorModel`, runs
  `_instantiate(**kwargs)`, then runs `train(**kwargs)`.
- `Decoder.instantiate(dem, **kwargs)` builds an untrained decoder by running
  only `_instantiate(**kwargs)`.
- `decode(...)` and `decode_confidence(...)` handle both single-shot and
  batched detector arrays.
- `_decode(...)` and `_decode_confidence(...)` are the single-shot hooks.
- `num_detectors` and `num_observables` are base properties backed by
  `self.dem`.

Concrete decoders should keep their setup in `_instantiate(...)`, not
`__init__(...)`. Use `train(...)` only when the decoder has a real training
phase. If a decoder provides a batch-optimized implementation, it may override
`decode(...)` while preserving the same public signature.

Prefer `self.dem` over decoder-specific DEM aliases. Only keep a separate DEM
attribute when it has different semantics, such as `GurobiDecoder._flat_dem`,
which stores a flattened copy for iteration.

Current decoder groups:

- `ldpc.py`: wrappers around LDPC package decoders. These build check matrices
  from `self.dem` during `_instantiate(...)`.
- `mwpf.py`: wrapper around MWPF's sinter-compatible decoder.
- `tesseract.py`: wrapper around Tesseract configuration and compiled decoder.
- `mle/decoder.py`: Gurobi MLE decoder. It keeps an optimized batch `decode`.
- `mld/decoder.py`: lookup-table decoder. Its `train(num_shots=...)` samples
  from `self.dem`; `instantiate(...)` is useful for tests or advanced callers
  that want to populate counts manually through `update_det_obs_counts(...)`.

Tests in `test/decoders/test_decoders.py` derive the decoder list from
`BaseDecoder.__subclasses__()`. Add new decoder classes to the package exports
so they are imported before that discovery runs.
