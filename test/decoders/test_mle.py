import numpy as np
import stim

from bloqade.decoders import GurobiDecoder
from bloqade.decoders._decoders.base import BaseDecoder


def regular_dem():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D9 D0 L0
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D3 D4
        error(0.1) D4 D5
        error(0.1) D5 D6
        error(0.1) D6 D7
        error(0.1) D7 D8
        error(0.1) D8 D9
        """
    )
    return dem


def regular_samples():
    det_shots = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        bool,
    )
    obs_shots = np.array([[1], [0]], bool)
    return det_shots, obs_shots


def hypergraph_dem():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D9 D0 D1 L0
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D3 D4
        error(0.1) D4 D5
        error(0.1) D5 D6
        error(0.1) D6 D7
        error(0.1) D7 D8
        error(0.1) D8 D9
        """
    )
    return dem


def hyper_samples():
    det_shots = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        bool,
    )
    obs_shots = np.array([[1], [0]], bool)
    return det_shots, obs_shots


def test_is_base_decoder():
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    assert isinstance(decoder, BaseDecoder)


def test_num_detectors():
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    assert decoder.num_detectors == 10


def test_num_observables():
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    assert decoder.num_observables == 1


def test_regular():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = GurobiDecoder(dem)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_hyper():
    dem = hypergraph_dem()
    det_shots, obs_shots = hyper_samples()
    decoder = GurobiDecoder(dem)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_decode_returns_weights():
    dem = regular_dem()
    det_shots, _ = regular_samples()
    decoder = GurobiDecoder(dem)
    obs, weights = decoder.decode(det_shots, return_weights=True)
    assert obs.shape == (2, 1)
    assert weights.shape == (2,)
    assert np.all(weights < 0)


def test_preprocess_populates_fields():
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    assert len(decoder.weights) == 10
    assert len(decoder.detector_vertices) == 10
    assert decoder.max_observable_index == 0
    assert len(decoder.observable_indices) == 1
    assert 0 in decoder.observable_indices[0]


def test_decode_error_shape():
    dem = regular_dem()
    det_shots, _ = regular_samples()
    decoder = GurobiDecoder(dem)
    errors = decoder.decode_error(det_shots)
    assert errors.shape == (2, 10)
    assert errors.dtype == bool


def test_logical_from_error():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = GurobiDecoder(dem)
    errors = decoder.decode_error(det_shots)
    logicals = decoder.logical_from_error(errors)
    assert np.array_equal(logicals, obs_shots)


def test_no_error_syndrome():
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    det_shots = np.zeros((1, 10), dtype=bool)
    result = decoder.decode(det_shots)
    assert np.array_equal(result, np.array([[False]]))


def test_single_shot_decode():
    """Test _decode (single-shot) via the BaseDecoder interface."""
    dem = regular_dem()
    det_shots = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    decoder = GurobiDecoder(dem)
    result = decoder.decode(det_shots)
    assert result.ndim == 1
    assert np.array_equal(result, np.array([True]))


def test_logical_gap():
    dem = regular_dem()
    det_shots, _ = regular_samples()
    decoder = GurobiDecoder(dem)
    decoded_errors = decoder.decode_error(det_shots)
    decoder.weight_from_error(decoded_errors)
    decoded_logicals = decoder.logical_from_error(decoded_errors)

    conditional_logicals = np.logical_not(decoded_logicals)
    conditional_decoder = decoder.generate_conditional_decoder()
    conditional_det_shots = np.concatenate(
        [det_shots, conditional_logicals], axis=1
    )
    flipped_errors = conditional_decoder.decode_error(conditional_det_shots)
    flipped_weights = conditional_decoder.weight_from_error(flipped_errors)
    flipped_logicals = conditional_decoder.logical_from_error(flipped_errors)
    assert np.array_equal(flipped_logicals, conditional_logicals)
    assert np.array_equal(
        flipped_weights, decoder.weight_from_error(flipped_errors)
    )


def test_conditional_decoder_has_extra_detector():
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    conditional_decoder = decoder.generate_conditional_decoder()
    assert (
        len(conditional_decoder.detector_vertices)
        == len(decoder.detector_vertices) + 1
    )
