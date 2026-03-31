import math

import stim
import numpy as np
import pytest
import sinter

from bloqade.decoders import GurobiDecoder
from bloqade.decoders.sinter_interface import SinterGurobiDecoder

from .conftest import pack_dets, simple_dem, unpack_obs, repetition_circuit


def regular_dem():
    dem = stim.DetectorErrorModel("""
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
        """)
    return dem


def regular_samples():
    det_shots = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        bool,
    )
    obs_shots = np.array([[1], [0]], bool)
    return det_shots, obs_shots


def test_regular():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = GurobiDecoder(dem)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_hyper():
    dem = stim.DetectorErrorModel("""
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
        """)
    det_shots = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        bool,
    )
    obs_shots = np.array([[1], [0]], bool)
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


# --- SinterGurobiDecoder tests ---


def test_sinter_gurobi_is_sinter_decoder():
    decoder = SinterGurobiDecoder()
    assert isinstance(decoder, sinter.Decoder)


def test_sinter_gurobi_compile_returns_compiled_decoder():
    dem = simple_dem()
    decoder = SinterGurobiDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    assert isinstance(compiled, sinter.CompiledDecoder)


def test_sinter_gurobi_decode_shape_and_dtype():
    dem = simple_dem()
    decoder = SinterGurobiDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )

    num_obs_bytes = math.ceil(dem.num_observables / 8)
    assert result.dtype == np.uint8
    assert result.shape == (3, num_obs_bytes)


def test_sinter_gurobi_decode_correctness():
    dem = simple_dem()
    decoder = SinterGurobiDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )
    obs_predictions = unpack_obs(result, dem.num_observables)

    expected = np.array([[True], [True], [False]])
    assert np.array_equal(obs_predictions, expected)


def test_sinter_gurobi_no_error_syndrome():
    dem = simple_dem()
    decoder = SinterGurobiDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.zeros((1, 2), dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )
    obs_predictions = unpack_obs(result, dem.num_observables)

    assert np.array_equal(obs_predictions, np.array([[False]]))


@pytest.mark.slow
def test_sinter_collect_gurobi():
    circuit = repetition_circuit()
    tasks = [
        sinter.Task(
            circuit=circuit,
            decoder="gurobi_mle",
            json_metadata={"d": 3},
        ),
    ]
    stats = sinter.collect(
        num_workers=1,
        tasks=tasks,
        custom_decoders={"gurobi_mle": SinterGurobiDecoder()},
        max_shots=100,
    )
    assert len(stats) == 1
    assert stats[0].shots == 100
    assert stats[0].errors <= stats[0].shots
    assert stats[0].errors < 50


def test_single_shot_decode_with_weights():
    """Single-shot decode with return_weights=True."""
    dem = regular_dem()
    det_shots = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    decoder = GurobiDecoder(dem)
    obs, weights = decoder.decode(det_shots, return_weights=True)
    assert obs.ndim == 1
    assert np.array_equal(obs, np.array([True]))
    assert isinstance(weights, np.ndarray)


def test_separator_targets_rejected():
    """DEM with separator targets raises ValueError."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 ^ D1 L0
        """)
    with pytest.raises(ValueError, match="separator"):
        GurobiDecoder(dem)


def test_multi_observable_dem():
    """DEM with multiple observables."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D1 L1
        """)
    decoder = GurobiDecoder(dem)
    assert decoder.num_observables == 2
    det_shots = np.array([[1, 0], [0, 1]], dtype=bool)
    result = decoder.decode(det_shots)
    assert result.shape == (2, 2)


def test_repeat_block_decode_correctness():
    """DEM with repeat block produces correct decode results after flattening."""
    dem = stim.DetectorErrorModel("""
        repeat 2 {
            error(0.1) D0 D1 L0
        }
        error(0.1) D1 D2
        """)
    decoder = GurobiDecoder(dem)
    assert decoder.num_detectors == 3
    # Syndrome that triggers first error
    det_shots = np.array([[1, 1, 0]], dtype=bool)
    result = decoder.decode(det_shots)
    assert result.shape == (1, 1)
    assert result[0, 0]  # L0 should be flipped


def test_observable_indices_align_with_weights():
    """observable_indices must reference weight-list indices, not raw DEM positions.

    A DEM with detector coordinate annotations has non-error DemInstructions mixed
    in. A naive enumerate()-based counter would increment for those instructions,
    causing observable_indices to reference wrong (out-of-bounds) weight indices.
    """
    dem = stim.DetectorErrorModel("""
        detector(0, 0) D0
        detector(1, 0) D1
        error(0.1) D0 L0
        error(0.1) D0 D1
        """)
    decoder = GurobiDecoder(dem)
    # Verify decoding works correctly (no index errors)
    det_shots = np.array([[1, 0]], dtype=bool)
    result = decoder.decode(det_shots)
    assert result.shape == (1, 1)
    assert result[0, 0]  # L0 should be flipped


def test_prob_zero_error_skipped():
    """error(0.0) is silently dropped and does not affect decoding."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.0) D1 D2 L0
        error(0.05) D0 D2
        """)
    decoder = GurobiDecoder(dem)
    # Syndrome [0, 1, 1] matches D1 D2 which is only the prob=0 error.
    # Since that error is dropped, the solver sees it as noise.
    result = decoder.decode(np.array([[0, 1, 1]], dtype=bool))
    assert result.shape == (1, 1)


def test_model_built_during_init():
    """Model is built eagerly during __init__, not lazily."""
    dem = regular_dem()
    decoder = GurobiDecoder(dem)
    assert decoder._model is not None


def test_model_reused_across_calls():
    """The same model object is reused across multiple decode_error calls."""
    dem = regular_dem()
    det_shots_1, obs_shots_1 = regular_samples()
    decoder = GurobiDecoder(dem)

    # First call
    errors_1 = decoder.decode_error(det_shots_1)
    model_after_first = decoder._model

    # Second call with different syndrome
    det_shots_2 = np.array(
        [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
        dtype=bool,
    )
    errors_2 = decoder.decode_error(det_shots_2)
    model_after_second = decoder._model

    # Same model object reused
    assert model_after_first is model_after_second

    # Both calls produce correct results
    obs_1 = decoder.logical_from_error(errors_1)
    assert np.array_equal(obs_1, obs_shots_1)
    assert errors_2.shape == (1, 10)


def test_close_releases_model():
    """close() releases the model; decoder rebuilds it on next use."""
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = GurobiDecoder(dem)
    assert decoder._model is not None

    decoder.close()
    assert decoder._model is None

    # Decoder still works after close (model is rebuilt)
    result = decoder.decode(det_shots)
    assert np.array_equal(result, obs_shots)
    assert decoder._model is not None


def test_named_params_stored():
    """Named solver params are stored as Gurobi CamelCase keys."""
    dem = regular_dem()
    decoder = GurobiDecoder(dem, time_limit=30.0, threads=2)
    assert decoder._solver_params == {"TimeLimit": 30.0, "Threads": 2}


def test_named_params_applied_to_model():
    """Named solver params are applied to the Gurobi model."""
    dem = regular_dem()
    decoder = GurobiDecoder(dem, time_limit=30.0, threads=2)
    assert decoder._model.getParamInfo("TimeLimit")[2] == 30.0
    assert decoder._model.getParamInfo("Threads")[2] == 2


def test_named_params_none_ignored():
    """None values are not stored in _solver_params."""
    dem = regular_dem()
    decoder = GurobiDecoder(dem, time_limit=None, threads=None)
    assert decoder._solver_params == {}


def test_named_params_decode_correctness():
    """Decoder with named params still produces correct results."""
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = GurobiDecoder(dem, threads=1)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_prob_one_error_pre_applied():
    """error(1.0) always fires and is pre-applied to the syndrome."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(1.0) D1 D2 L0
        error(0.05) D0 D2
        """)
    decoder = GurobiDecoder(dem)
    # The certain error flips D1 and D2, so syndrome [0,1,1] after
    # pre-application becomes [0,0,0] (no errors to solve for).
    result = decoder.decode(np.array([[0, 1, 1]], dtype=bool))
    assert result.shape == (1, 1)
    # The certain error flips L0, so the observable should be True
    assert result[0, 0]
