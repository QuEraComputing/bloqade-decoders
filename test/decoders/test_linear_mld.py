import numpy as np
import pytest
import stim

from bloqade.decoders import LinearTableDecoder

from .rep_code_ref import (
    space_error_dem,
    space_error_syndromes,
    expected_space_error_decoded_obs,
    time_error_dem,
    time_error_syndromes,
    expected_time_error_decoded_obs,
)
from .two_logical_ref import (
    two_logical_dem,
    two_logical_syndromes,
    two_logical_expected_decoded_obs,
)


def _simple_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(
        "error(0.1) D0 L0\n"
        "error(0.1) D1 L0\n"
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_linear_table_no_error_syndrome():
    decoder = LinearTableDecoder(_simple_dem())
    result = decoder.decode(np.array([False, False], dtype=bool))
    np.testing.assert_array_equal(result, np.array([False]))


def test_linear_table_single_detector_correction():
    decoder = LinearTableDecoder(_simple_dem())
    # Each single-detector syndrome should predict L0=1
    result_d0 = decoder.decode(np.array([True, False], dtype=bool))
    result_d1 = decoder.decode(np.array([False, True], dtype=bool))
    np.testing.assert_array_equal(result_d0, np.array([True]))
    np.testing.assert_array_equal(result_d1, np.array([True]))


def test_linear_table_both_detectors_no_correction():
    # Both errors firing simultaneously cancel L0; best correction is 0
    decoder = LinearTableDecoder(_simple_dem())
    result = decoder.decode(np.array([True, True], dtype=bool))
    np.testing.assert_array_equal(result, np.array([False]))


def test_linear_table_batch_decode():
    decoder = LinearTableDecoder(_simple_dem())
    syndromes = np.array(
        [[False, False], [True, False], [False, True], [True, True]], dtype=bool
    )
    result = decoder.decode(syndromes)
    expected = np.array([[False], [True], [True], [False]])
    np.testing.assert_array_equal(result, expected)


def test_linear_table_single_shot_returns_1d():
    decoder = LinearTableDecoder(_simple_dem())
    result = decoder.decode(np.array([True, False], dtype=bool))
    assert result.ndim == 1
    assert result.shape == (1,)


def test_linear_table_batch_returns_2d():
    decoder = LinearTableDecoder(_simple_dem())
    result = decoder.decode(np.array([[True, False]], dtype=bool))
    assert result.ndim == 2
    assert result.shape == (1, 1)


def test_linear_table_empty_batch():
    decoder = LinearTableDecoder(_simple_dem())
    result = decoder.decode(np.empty((0, 2), dtype=bool))
    assert result.shape == (0, 1)


# ---------------------------------------------------------------------------
# Reference DEMs (also covered by test_decoders.py auto-discovery)
# ---------------------------------------------------------------------------


def test_linear_table_space_error_dem():
    decoder = LinearTableDecoder(space_error_dem)
    result = decoder.decode(space_error_syndromes)
    np.testing.assert_array_equal(result, expected_space_error_decoded_obs)


def test_linear_table_time_error_dem():
    decoder = LinearTableDecoder(time_error_dem)
    result = decoder.decode(time_error_syndromes)
    np.testing.assert_array_equal(result, expected_time_error_decoded_obs)


def test_linear_table_two_logical():
    decoder = LinearTableDecoder(two_logical_dem)
    result = decoder.decode(two_logical_syndromes)
    np.testing.assert_array_equal(result, two_logical_expected_decoded_obs)


# ---------------------------------------------------------------------------
# Detector-only errors
# ---------------------------------------------------------------------------


def test_linear_table_detector_only_error_affects_conditional():
    """A detector-only error changes P(L|D) when it shares a detector with
    an observable-affecting error. Ignoring it (as in the original parse_dem
    filter) gives a wrong result in this case."""
    # Error A (p=0.5): flips D0 AND L0
    # Error B (p=0.5): flips D0 only
    # When D0=1: A fired XOR B fired. If just A: L0=1. If just B: L0=0.
    # Both have equal probability 0.25, so P(L0=0|D0=1)=P(L0=1|D0=1)=0.5.
    dem = stim.DetectorErrorModel(
        "error(0.5) D0 L0\n"
        "error(0.5) D0\n"
    )
    decoder = LinearTableDecoder(dem)
    # When D0=0: no correction (neither fired, or both fired cancelling D0)
    result_no_det = decoder.decode(np.array([False], dtype=bool))
    np.testing.assert_array_equal(result_no_det, np.array([False]))

    # Confidence for D0=1 should be 0.5 (genuinely ambiguous)
    _, conf = decoder.decode_confidence(np.array([True], dtype=bool))
    assert conf == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_decode_confidence_values_in_range():
    decoder = LinearTableDecoder(_simple_dem())
    syndromes = np.array(
        [[False, False], [True, False], [False, True], [True, True]], dtype=bool
    )
    _, confidences = decoder.decode_confidence(syndromes)
    assert np.all(confidences >= 0.0)
    assert np.all(confidences <= 1.0)


def test_decode_confidence_single_shot_returns_float():
    decoder = LinearTableDecoder(_simple_dem())
    _, conf = decoder.decode_confidence(np.array([True, False], dtype=bool))
    assert isinstance(conf, float)


def test_decode_confidence_batch_returns_array():
    decoder = LinearTableDecoder(_simple_dem())
    syndromes = np.array([[True, False], [False, True]], dtype=bool)
    _, confs = decoder.decode_confidence(syndromes)
    assert isinstance(confs, np.ndarray)
    assert confs.shape == (2,)


def test_decode_confidence_exact_single_observable():
    """For a DEM where each syndrome maps deterministically to one observable
    pattern, confidence should be 1.0."""
    decoder = LinearTableDecoder(_simple_dem())
    syndromes = np.array(
        [[False, False], [True, False], [False, True], [True, True]], dtype=bool
    )
    corrections, confidences = decoder.decode_confidence(syndromes)
    np.testing.assert_array_almost_equal(confidences, [1.0, 1.0, 1.0, 1.0])


def test_decode_confidence_empty_batch():
    decoder = LinearTableDecoder(_simple_dem())
    corrections, confidences = decoder.decode_confidence(
        np.empty((0, 2), dtype=bool)
    )
    assert corrections.shape == (0, 1)
    assert confidences.shape == (0,)


# ---------------------------------------------------------------------------
# Instantiate (no training needed — identical to full constructor)
# ---------------------------------------------------------------------------


def test_linear_table_instantiate_without_training():
    decoder = LinearTableDecoder.instantiate(_simple_dem())
    result = decoder.decode(np.array([[True, False]], dtype=bool))
    np.testing.assert_array_equal(result, np.array([[True]]))


def test_linear_table_instantiate_same_as_constructor():
    dem = _simple_dem()
    decoder_full = LinearTableDecoder(dem)
    decoder_inst = LinearTableDecoder.instantiate(dem)
    syndromes = np.array(
        [[False, False], [True, False], [False, True], [True, True]], dtype=bool
    )
    np.testing.assert_array_equal(
        decoder_full.decode(syndromes),
        decoder_inst.decode(syndromes),
    )
