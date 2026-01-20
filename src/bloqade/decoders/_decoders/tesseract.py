from typing import Optional

import stim
import numpy as np
import numpy.typing as npt

from .base import BaseDecoder

try:
    import tesseract_decoder.tesseract as tesseract
except ImportError as e:
    raise ImportError(
        "The tesseract-decoder package is required for TesseractDecoder. "
        "Install it with: pip install tesseract-decoder"
    ) from e


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
        beam_climbing (bool | None): When True, enables a heuristic that
            causes the decoder to try different det_beam values (up to a
            maximum) to find a good decoding path. This can improve the
            decoder's chance of finding the most likely error, even with an
            initial narrow beam search. Default is False.
        no_revisit_dets (bool | None): When True, activates a heuristic to
            prevent the decoder from revisiting nodes that have the same set
            of leftover detection events as a node it has already visited.
            This can help reduce search redundancy and improve decoding speed.
            Default is False.
        verbose (bool | None): When True, enables verbose logging for
            debugging and understanding the decoder's internal behavior.
            Default is False.
        pqlimit (int | None): Limit on the number of nodes in the priority
            queue. This can be used to constrain memory usage. Default is
            sys.maxsize (effectively unbounded).
        det_orders (list[list[int]] | None): A list of lists of integers,
            where each inner list represents an ordering of the detectors.
            Used for "ensemble reordering," an optimization that tries
            different detector orderings to improve search convergence.
            Default is an empty list (single, fixed ordering).
        det_penalty (float | None): A cost added for each residual detection
            event. This encourages the decoder to prioritize paths that
            resolve more detection events, steering the search towards more
            complete solutions. Default is 0.0 (no penalty).
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_beam: Optional[int] = None,
        beam_climbing: Optional[bool] = None,
        no_revisit_dets: Optional[bool] = None,
        verbose: Optional[bool] = None,
        pqlimit: Optional[int] = None,
        det_orders: Optional[list[list[int]]] = None,
        det_penalty: Optional[float] = None,
    ):

        self._dem = dem

        # Collect only user-set arguments into a dictionary
        config_kwargs: dict = {"dem": dem}

        if det_beam is not None:
            config_kwargs["det_beam"] = det_beam
        if beam_climbing is not None:
            config_kwargs["beam_climbing"] = beam_climbing
        if no_revisit_dets is not None:
            config_kwargs["no_revisit_dets"] = no_revisit_dets
        if verbose is not None:
            config_kwargs["verbose"] = verbose
        if pqlimit is not None:
            config_kwargs["pqlimit"] = pqlimit
        if det_orders is not None:
            config_kwargs["det_orders"] = det_orders
        if det_penalty is not None:
            config_kwargs["det_penalty"] = det_penalty

        self._config_kwargs = config_kwargs
        self._config = tesseract.TesseractConfig(**config_kwargs)
        self._decoder = tesseract.TesseractDecoder(self._config)

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits.

        Args:
            detector_bits: 1D numpy array of boolean detector outcomes.

        Returns:
            1D numpy array of boolean observable outcomes.
        """
        return self._decoder.decode(detector_bits)
