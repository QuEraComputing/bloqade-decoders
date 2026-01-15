import numpy as np

from bloqade.decoders.decoders import TesseractDecoder
from .reference import reference_dem, reference_syndromes, decoded_obs


def test_tesseract_decoder():
    decoder = TesseractDecoder(reference_dem)
    result = decoder.decode(reference_syndromes)
    np.testing.assert_array_equal(result, decoded_obs)
