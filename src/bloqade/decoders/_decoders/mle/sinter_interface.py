from __future__ import annotations

from typing import TYPE_CHECKING

import stim
import numpy as np
from sinter import Decoder as _SinterDecoder, CompiledDecoder as _SinterCompiledDecoder

from .base import BaseMLEDecoder
from .milp_decoder import MILPDecoder
from .gurobi_decoder import GurobiDecoder

if TYPE_CHECKING:
    import pulp


class _CompiledMLEDecoder(_SinterCompiledDecoder):
    """Compiled decoder wrapping an MLE decoder for sinter."""

    def __init__(self, decoder: BaseMLEDecoder) -> None:
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
    """Sinter-compatible adapter for the GurobiDecoder (MLE)."""

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> _SinterCompiledDecoder:
        decoder = GurobiDecoder(dem)
        return _CompiledMLEDecoder(decoder)


class SinterMILPDecoder(_SinterDecoder):
    """Sinter-compatible adapter for the PuLP-based MILPDecoder (MLE).

    Args:
        solver: Name of a PuLP solver (e.g. ``"HiGHS"``, ``"CPLEX_PY"``,
            ``"COPT"``, ``"GUROBI"``) or a ``pulp.LpSolver`` instance.
            Defaults to ``"HiGHS"``.
    """

    def __init__(self, solver: str | pulp.LpSolver = "HiGHS") -> None:
        self._solver = solver

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> _SinterCompiledDecoder:
        decoder = MILPDecoder(dem, solver=self._solver)
        return _CompiledMLEDecoder(decoder)
