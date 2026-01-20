import numpy as np
import pytest

from bloqade.decoders import (
    MWPFDecoder,
    BpLsdDecoder,
    BpOsdDecoder,
    TesseractDecoder,
    BeliefFindDecoder,
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

DECODERS = [
    TesseractDecoder,
    BeliefFindDecoder,
    BpLsdDecoder,
    BpOsdDecoder,
    MWPFDecoder,
]

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
