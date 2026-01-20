from abc import ABC, abstractmethod

import stim
import numpy as np
import numpy.typing as npt


class BaseDecoder(ABC):
    def __init__(self, dem: stim.DetectorErrorModel):
        pass

    @abstractmethod
    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits."""
        pass

    def decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a batch or single shot of detector bits.

        This method accepts either an array of booleans (representing a single shot)
        or an array of arrays of booleans (representing a batch of shots).

        """
        if detector_bits.ndim == 1:
            return self._decode(detector_bits)
        else:
            return np.stack([self._decode(row) for row in detector_bits])
