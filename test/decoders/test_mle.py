import math

import stim
import numpy as np
import pytest
import sinter

from bloqade.decoders import GurobiDecoder, SinterGurobiDecoder
from bloqade.decoders._decoders.base import BaseDecoder

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


def hypergraph_dem():
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
    """Single-shot decode with return_weights=True (covers line 278)."""
    dem = regular_dem()
    det_shots = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    decoder = GurobiDecoder(dem)
    obs, weights = decoder.decode(det_shots, return_weights=True)
    assert obs.ndim == 1
    assert np.array_equal(obs, np.array([True]))
    assert isinstance(weights, np.ndarray)


def test_separator_targets_rejected():
    """DEM with separator targets raises ValueError (covers line 124)."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 ^ D1 L0
        """)
    with pytest.raises(ValueError, match="separator"):
        GurobiDecoder(dem)


def test_multi_observable_dem():
    """DEM with multiple observables covers max_observable_index update (line 82)."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D1 L1
        """)
    decoder = GurobiDecoder(dem)
    assert decoder.max_observable_index == 1
    assert decoder.num_observables == 2
    det_shots = np.array([[1, 0], [0, 1]], dtype=bool)
    result = decoder.decode(det_shots)
    assert result.shape == (2, 2)


def test_repeat_block_dem():
    """DEM with repeat block is flattened and decoded correctly."""
    dem = stim.DetectorErrorModel("""
        repeat 3 {
            error(0.1) D0 L0
        }
        """)
    decoder = GurobiDecoder(dem)
    assert decoder.num_detectors == 1
    assert decoder.num_observables == 1
    # The flattened DEM should have 3 error instructions, producing 3 weights
    assert len(decoder.weights) == 3


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


def test_nested_repeat_block_dem():
    """Nested repeat blocks are fully flattened."""
    dem = stim.DetectorErrorModel("""
        repeat 2 {
            repeat 3 {
                error(0.1) D0 L0
            }
        }
        """)
    decoder = GurobiDecoder(dem)
    assert len(decoder.weights) == 6


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
    assert len(decoder.weights) == 2
    # The first (and only) observable-flipping error is at weight index 0
    assert decoder.observable_indices[0] == [0]
