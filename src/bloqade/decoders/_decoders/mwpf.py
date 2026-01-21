from typing import Literal

import stim
import numpy as np
import numpy.typing as npt

from .base import BaseDecoder


class MWPFDecoder(BaseDecoder):
    """Hypergraph Minimum Weight Parity Factor decoder wrapper.

    Arguments match mwpf.SinterMWPFDecoder default; defaults are used if not specified.

    Args:
        dem (stim.DetectorErrorModel): The detector error model describing the error structure.
        decoder_type (Literal["SolverSerialJointSingleHair", "SolverSerialUnionFind", "SolverSerialSingleHair"]): Decoder class used to construct the MWPF decoder. All decoder
            types inherit from `SolverSerialPlugins` but provide different plugins for optimizing
            the primal and/or dual solutions. For example, `SolverSerialUnionFind` is the most
            basic solver without any plugin: it only grows clusters until the first valid solution
            appears. More optimized solvers use plugins to further optimize the solution at the
            cost of longer decoding time.  Default is "SolverSerialJointSingleHair".
        cluster_node_limit (int): The maximum number of nodes in a cluster, used to tune
            decoder performance. Default is 50.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        decoder_type: Literal[
            "SolverSerialJointSingleHair",
            "SolverSerialUnionFind",
            "SolverSerialSingleHair",
        ] = "SolverSerialJointSingleHair",
        cluster_node_limit: int = 50,
    ):
        try:
            from mwpf import SinterMWPFDecoder
        except ImportError as e:
            raise ImportError(
                "The mwpf package is required for MWPFDecoder. "
                'You can install it via: pip install "mwpf[stim]"'
            ) from e

        self._dem = dem
        self._sinter_decoder = SinterMWPFDecoder(
            decoder_type=decoder_type,
            cluster_node_limit=cluster_node_limit,
        )
        self._compiled_decoder = self._sinter_decoder.compile_decoder_for_dem(dem=dem)

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        bit_packed = np.packbits(np.array([detector_bits]), axis=-1, bitorder="little")
        prediction = self._compiled_decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=bit_packed
        )
        observables = self._compiled_decoder.predictor.get_observable_bits(prediction)
        return np.array(observables, dtype=bool)
