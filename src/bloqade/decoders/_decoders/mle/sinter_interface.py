from __future__ import annotations

import stim
import numpy as np
from sinter import Decoder as _SinterDecoder, CompiledDecoder as _SinterCompiledDecoder

from .decoder import GurobiDecoder


class _CompiledGurobiDecoder(_SinterCompiledDecoder):
    """Compiled decoder wrapping GurobiDecoder for sinter."""

    def __init__(self, decoder: GurobiDecoder) -> None:
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
        assert isinstance(obs_predictions, np.ndarray)
        return np.packbits(
            obs_predictions.astype(np.uint8),
            axis=1,
            bitorder="little",
        )


class SinterGurobiDecoder(_SinterDecoder):
    """Sinter-compatible adapter for the GurobiDecoder (MLE).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`GurobiDecoder`.
    """

    def __init__(self, **kwargs: object) -> None:
        self._decoder_kwargs = kwargs

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> _SinterCompiledDecoder:
        decoder = GurobiDecoder(dem, **self._decoder_kwargs)
        return _CompiledGurobiDecoder(decoder)
