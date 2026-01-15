import stim
import numpy as np
from abc import ABC, abstractmethod


class BaseDecoder(ABC):
    def __init__(self, dem: stim.DetectorErrorModel):
        pass

    @abstractmethod
    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        """Decode a single shot of detector bits."""
        pass

    def decode(self, detector_bits: np.ndarray) -> np.ndarray:
        """Decode a batch or single shot of detector bits."""
        if detector_bits.ndim == 1:
            return self._decode(detector_bits)
        else:
            return np.stack([self._decode(row) for row in detector_bits])
