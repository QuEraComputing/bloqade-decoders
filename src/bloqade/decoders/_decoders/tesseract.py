import sys

import stim
import numpy as np
import numpy.typing as npt

from .base import BaseDecoder


class TesseractDecoder(BaseDecoder):
    """Tesseract decoder wrapper.

    The Tesseract decoder employs A* search to decode the most-likely error
    configuration from a measured syndrome.

    Args:
        dem (stim.DetectorErrorModel): The detector error model that provides
            the logical structure of the quantum error-correcting code,
            including the detectors and relationships between them. This model
            is essential for the decoder to understand the syndrome and
            potential error locations.
        det_beam (int | None): Beam search cutoff. Specifies a threshold for
            the number of "residual detection events" a node can have before
            it is pruned from the search. A lower value makes the search more
            aggressive, potentially sacrificing accuracy for speed. Default is
            no beam cutoff (INF_DET_BEAM).
        beam_climbing (bool): When True, enables a heuristic that
            causes the decoder to try different det_beam values (up to a
            maximum) to find a good decoding path. This can improve the
            decoder's chance of finding the most likely error, even with an
            initial narrow beam search. Default is False.
        no_revisit_dets (bool): When True, activates a heuristic to
            prevent the decoder from revisiting nodes that have the same set
            of leftover detection events as a node it has already visited.
            This can help reduce search redundancy and improve decoding speed.
            Default is False.
        verbose (bool): When True, enables verbose logging for
            debugging and understanding the decoder's internal behavior.
            Default is False.
        pqlimit (int): Limit on the number of nodes in the priority
            queue. This can be used to constrain memory usage. Default is
            sys.maxsize (effectively unbounded).
        det_orders (list[list[int]]): A list of lists of integers,
            where each inner list represents an ordering of the detectors.
            Used for "ensemble reordering," an optimization that tries
            different detector orderings to improve search convergence.
            Default is an empty list (single, fixed ordering).
        det_penalty (float): A cost added for each residual detection
            event. This encourages the decoder to prioritize paths that
            resolve more detection events, steering the search towards more
            complete solutions. Default is 0.0 (no penalty).
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_beam: int | None = None,
        beam_climbing: bool = False,
        no_revisit_dets: bool = False,
        verbose: bool = False,
        pqlimit: int = sys.maxsize,
        det_orders: list[list[int]] = [],
        det_penalty: float = 0.0,
    ):
        try:
            import tesseract_decoder.tesseract as tesseract
        except ImportError as e:
            raise ImportError(
                "The tesseract-decoder package is required for TesseractDecoder. "
                'You can install it via: pip install "tesseract-decoder"'
            ) from e

        if det_beam is None:
            det_beam = tesseract.INF_DET_BEAM

        self._dem = dem
        self._config = tesseract.TesseractConfig(
            dem=dem,
            det_beam=det_beam,
            beam_climbing=beam_climbing,
            no_revisit_dets=no_revisit_dets,
            verbose=verbose,
            pqlimit=pqlimit,
            det_orders=det_orders,
            det_penalty=det_penalty,
        )
        self._decoder = tesseract.TesseractDecoder(self._config)

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits.

        Args:
            detector_bits: 1D numpy array of boolean detector outcomes.

        Returns:
            1D numpy array of boolean observable outcomes.
        """
        return self._decoder.decode(detector_bits)
