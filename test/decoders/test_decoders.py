import numpy as np
import pytest

from bloqade.decoders import (
    BaseDecoder,
    TableDecoder,
)

from .rep_code_ref import (
    time_error_dem,
    space_error_dem,
    time_error_syndromes,
    space_error_syndromes,
    expected_time_error_decoded_obs,
    expected_space_error_decoded_obs,
)
from .two_logical_ref import (
    two_logical_dem,
    two_logical_syndromes,
    two_logical_expected_decoded_obs,
)


def _all_decoder_subclasses(decoder_cls):
    subclasses = []
    for subclass in decoder_cls.__subclasses__():
        if subclass.__module__.startswith("bloqade.decoders._decoders."):
            subclasses.append(subclass)
        subclasses.extend(_all_decoder_subclasses(subclass))
    return subclasses


DECODERS = sorted(_all_decoder_subclasses(BaseDecoder), key=lambda cls: cls.__name__)
NEEDS_TRAINING = [TableDecoder]

TEST_CASES = [
    (space_error_dem, space_error_syndromes, expected_space_error_decoded_obs),
    (time_error_dem, time_error_syndromes, expected_time_error_decoded_obs),
    (two_logical_dem, two_logical_syndromes, two_logical_expected_decoded_obs),
]


@pytest.mark.parametrize("decoder_cls", DECODERS)
@pytest.mark.parametrize("dem,syndromes,expected", TEST_CASES)
def test_decoder(decoder_cls, dem, syndromes, expected):
    decoder = decoder_cls(dem)
    result = decoder.decode(syndromes)
    np.testing.assert_array_equal(result, expected)


def test_decoders_can_instantiate_without_training():
    for decoder_cls in DECODERS:
        if decoder_cls in NEEDS_TRAINING:
            continue

        decoder = decoder_cls.instantiate(space_error_dem)
        result = decoder.decode(space_error_syndromes)
        np.testing.assert_array_equal(result, expected_space_error_decoded_obs)
