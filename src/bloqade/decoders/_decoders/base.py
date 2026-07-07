from abc import ABC, abstractmethod
from typing import TypeVar

import stim
import numpy as np
import numpy.typing as npt

DecoderT = TypeVar("DecoderT", bound="BaseDecoder")


class BaseDecoder(ABC):
    def __init__(self, dem: stim.DetectorErrorModel, **kwargs: object) -> None:
        self.dem = dem
        self._instantiate(**kwargs)
        self.train(**kwargs)

    @classmethod
    def instantiate(
        cls: type[DecoderT], dem: stim.DetectorErrorModel, **kwargs: object
    ) -> DecoderT:
        """Configure an untrained decoder from constructor keyword arguments."""
        decoder = cls.__new__(cls)
        decoder.dem = dem
        decoder._instantiate(**kwargs)
        return decoder

    def _instantiate(self, **kwargs: object) -> None:
        """Configure the decoder from constructor keyword arguments."""
        pass

    def train(self, **kwargs: object) -> None:
        """Train the decoder.

        The default is a no-op for decoders that do not need training.
        """
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
        if len(detector_bits) == 0:
            return np.empty((0, self.dem.num_observables), dtype=np.bool_)
        return np.stack([self._decode(row) for row in detector_bits])

    def _decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float]:
        """Decode a single shot of detector bits with a confidence score.

        Confidence is ``0.0`` when a decoder did not converge. This default
        implementation always reports ``1.0`` because it has no convergence
        signal.
        """
        return self._decode(detector_bits), 1.0

    def decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float | npt.NDArray[np.float64]]:
        """Decode a batch or single shot of detector bits with confidence scores.

        Confidence is ``0.0`` when a decoder did not converge. This default
        implementation always reports ``1.0`` because it has no convergence
        signal.
        """
        if detector_bits.ndim == 1:
            return self._decode_confidence(detector_bits)
        if len(detector_bits) == 0:
            return (
                np.empty((0, self.dem.num_observables), dtype=np.bool_),
                np.empty(0, dtype=np.float64),
            )
        corrections, confidences = zip(
            *(self._decode_confidence(row) for row in detector_bits)
        )
        return np.stack(corrections), np.array(confidences, dtype=np.float64)
