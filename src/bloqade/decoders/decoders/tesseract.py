import stim
import numpy as np
from typing import Optional

from .base import BaseDecoder

import tesseract_decoder.tesseract as tesseract


class TesseractDecoder(BaseDecoder):
    """Tesseract decoder wrapper.

    Arguments match tesseract_decoder.TesseractConfig; defaults are used if not specified.
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

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        """Decode a single shot of detector bits.

        Args:
            detector_bits: 1D numpy array of boolean detector outcomes.

        Returns:
            1D numpy array of boolean observable outcomes.
        """
        return self._decoder.decode(detector_bits)

