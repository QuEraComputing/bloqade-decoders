import stim
import numpy as np
import numpy.typing as npt

from bloqade.decoders import BaseDecoder


class ExampleDecoder(BaseDecoder):
    def _instantiate(self, **kwargs: object) -> None:
        self.flip = bool(kwargs.get("flip", False))

    def train(self, **kwargs: object) -> None:
        self.trained = bool(kwargs.get("trained", True))

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        return np.array([detector_bits[0] ^ self.flip], dtype=np.bool_)


class SometimesUnsureDecoder(ExampleDecoder):
    def _decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float]:
        correction = self._decode(detector_bits)
        confidence = 0.0 if detector_bits[0] else 1.0
        return correction, confidence


def test_constructor_stores_dem_and_runs_hooks():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    decoder = ExampleDecoder(dem, flip=True, trained=True)

    assert decoder.dem is dem
    assert decoder.flip
    assert decoder.trained


def test_instantiate_skips_training():
    decoder = ExampleDecoder.instantiate(stim.DetectorErrorModel("error(0.1) D0 L0"))

    assert decoder.flip is False
    assert not hasattr(decoder, "trained")


def test_decode_confidence_defaults_to_one():
    decoder = ExampleDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"))

    correction, confidence = decoder.decode_confidence(np.array([False]))

    np.testing.assert_array_equal(correction, np.array([False]))
    assert confidence == 1.0


def test_decode_accepts_batches():
    decoder = ExampleDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"), flip=True)
    detector_bits = np.array([[False], [True]])

    corrections = decoder.decode(detector_bits)

    np.testing.assert_array_equal(corrections, np.array([[True], [False]]))


def test_decode_confidence_accepts_batches():
    decoder = SometimesUnsureDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"))
    detector_bits = np.array([[False], [True]])

    corrections, confidences = decoder.decode_confidence(detector_bits)

    np.testing.assert_array_equal(corrections, np.array([[False], [True]]))
    np.testing.assert_array_equal(confidences, np.array([1.0, 0.0]))
