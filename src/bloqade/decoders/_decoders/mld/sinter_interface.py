from __future__ import annotations

import stim
import numpy as np
from sinter import Decoder as _SinterDecoder, CompiledDecoder as _SinterCompiledDecoder

from .decoder import TableDecoder


class _CompiledTableDecoder(_SinterCompiledDecoder):
    """Compiled decoder wrapping TableDecoder for sinter."""

    def __init__(self, decoder: TableDecoder) -> None:
        self._decoder = decoder

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        num_dets = self._decoder.num_detectors
        det_shots = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            bitorder="little",
        )[:, :num_dets].astype(bool)
        obs_predictions = self._decoder.decode(det_shots)
        return np.packbits(
            obs_predictions.astype(np.uint8),
            axis=1,
            bitorder="little",
        )


class SinterTableDecoder(_SinterDecoder):
    """Sinter-compatible adapter for the TableDecoder (MLD).

    Samples from the DEM (not a circuit) to build the lookup table,
    since sinter only provides a DEM in compile_decoder_for_dem. This
    is equivalent to circuit sampling when the DEM faithfully represents
    all error mechanisms.

    Args:
        num_shots: Number of shots to sample for building the table.
    """

    num_shots: int

    def __init__(self, num_shots: int = 2**26) -> None:
        self.num_shots = num_shots

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> _SinterCompiledDecoder:
        sampler = dem.compile_sampler()
        det_data, obs_data, _ = sampler.sample(self.num_shots)
        det_obs_shots = np.concatenate([det_data, obs_data], axis=1)
        decoder = TableDecoder.from_det_obs_shots(dem, det_obs_shots)
        return _CompiledTableDecoder(decoder)
