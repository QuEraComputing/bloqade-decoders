import numpy as np

from bloqade.decoders.decoders import (
    TesseractDecoder,
    BeliefFindDecoder,
    BpLsdDecoder,
    BpOsdDecoder,
)
from .reference import reference_dem, reference_syndromes, decoded_obs


def test_tesseract_decoder():
    decoder = TesseractDecoder(reference_dem)
    result = decoder.decode(reference_syndromes)
    np.testing.assert_array_equal(result, decoded_obs)


def test_belief_find_decoder():
    decoder = BeliefFindDecoder(reference_dem)
    result = decoder.decode(reference_syndromes)
    np.testing.assert_array_equal(result, decoded_obs)


def test_bp_lsd_decoder():
    decoder = BpLsdDecoder(reference_dem)
    result = decoder.decode(reference_syndromes)
    np.testing.assert_array_equal(result, decoded_obs)


def test_bp_osd_decoder():
    decoder = BpOsdDecoder(reference_dem)
    result = decoder.decode(reference_syndromes)
    np.testing.assert_array_equal(result, decoded_obs)
