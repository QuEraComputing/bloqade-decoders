import numpy as np
import pytest

from bloqade.decoders import (
    MWPFDecoder,
    BpLsdDecoder,
    BpOsdDecoder,
    GurobiDecoder,
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

# TableDecoder is excluded because it needs training data for meaningful
# corrections.
DECODERS = [
    TesseractDecoder,
    BeliefFindDecoder,
    BpLsdDecoder,
    BpOsdDecoder,
    MWPFDecoder,
    GurobiDecoder,
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


def test_mwpf_decoder_can_instantiate_without_training():
    decoder = MWPFDecoder.instantiate(space_error_dem)

    result = decoder.decode(space_error_syndromes)

    np.testing.assert_array_equal(result, expected_space_error_decoded_obs)


def test_tesseract_decoder_can_instantiate_without_training():
    decoder = TesseractDecoder.instantiate(space_error_dem)

    result = decoder.decode(space_error_syndromes)

    np.testing.assert_array_equal(result, expected_space_error_decoded_obs)


def test_belief_find_decoder_can_instantiate_without_training():
    decoder = BeliefFindDecoder.instantiate(space_error_dem)

    result = decoder.decode(space_error_syndromes)

    np.testing.assert_array_equal(result, expected_space_error_decoded_obs)


def test_bp_lsd_decoder_can_instantiate_without_training():
    decoder = BpLsdDecoder.instantiate(space_error_dem)

    result = decoder.decode(space_error_syndromes)

    np.testing.assert_array_equal(result, expected_space_error_decoded_obs)


def test_bp_osd_decoder_can_instantiate_without_training():
    decoder = BpOsdDecoder.instantiate(space_error_dem)

    result = decoder.decode(space_error_syndromes)

    np.testing.assert_array_equal(result, expected_space_error_decoded_obs)
