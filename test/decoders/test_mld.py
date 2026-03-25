import math

import stim
import numpy as np
import pytest
import sinter

from bloqade.decoders import TableDecoder
from bloqade.decoders.sinter_interface import SinterTableDecoder
from bloqade.decoders._decoders.mld.utils import (
    shots_to_counts,
    pack_boolean_array,
    unpack_boolean_array,
    det_obs_shots_to_counts,
)

from .conftest import pack_dets, simple_dem, unpack_obs, repetition_circuit


def test_pack_unpack():
    bm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    packed = pack_boolean_array(bm)
    unpacked = unpack_boolean_array(packed, bm.shape[1])
    assert np.array_equal(bm, unpacked)


def test_pack_boolean_array_values():
    arr = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
    packed = pack_boolean_array(arr)
    assert np.array_equal(packed, np.array([5, 0, 7]))


def test_unpack_boolean_array_values():
    packed = np.array([5, 0, 7])
    unpacked = unpack_boolean_array(packed, 3)
    expected = np.array(
        [[True, False, True], [False, False, False], [True, True, True]]
    )
    assert np.array_equal(unpacked, expected)


def test_pack_unpack_single_bit():
    arr = np.array([[0], [1]], dtype=bool)
    packed = pack_boolean_array(arr)
    unpacked = unpack_boolean_array(packed, 1)
    assert np.array_equal(arr, unpacked)


def test_shots_to_counts():
    shots = np.array([[0, 0], [0, 1], [0, 1], [1, 0]], dtype=bool)
    counts = shots_to_counts(shots)
    assert np.array_equal(counts, np.array([1, 1, 2, 0]))


def test_det_obs_shots_to_counts():
    det_shots = np.array([[0, 0], [1, 0]], dtype=bool)
    obs_shots = np.array([[0], [1]], dtype=bool)
    counts = det_obs_shots_to_counts(det_shots, obs_shots)
    expected = np.zeros(8, dtype=int)
    expected[0] = 1  # (0,0,0) -> 0b000 = 0
    expected[5] = 1  # (1,0,1) -> 0b101 = 5
    assert np.array_equal(counts, expected)


def test_mld_repetition():
    circ = stim.Circuit("""
        R 0 1 2
        X_ERROR(0.1) 0 1 2
        MZZ 0 1
        DETECTOR rec[-1]
        MZZ 1 2
        DETECTOR rec[-1]
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
    """)
    decoder = TableDecoder.from_stim_circuit(circ, 10000)
    assert decoder.num_detectors == 2
    assert decoder.num_observables == 1
    det_shots = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    obs_shots = np.array([[1], [1], [1], [0]])
    correction = decoder.decode(det_shots)
    assert np.array_equal(correction, obs_shots)


def test_decode_obs_det_counts():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0]))
    assert np.array_equal(
        decoder.decode(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
        np.array([[0], [1], [1], [0]]),
    )
    raw_det_obs_counts = np.arange(8)
    decoded_det_obs_counts = decoder.decode_det_obs_counts(raw_det_obs_counts)
    assert np.array_equal(decoded_det_obs_counts, np.array([0, 5, 6, 3, 4, 1, 2, 7]))


def test_no_error_syndrome():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0]))
    result = decoder.decode(np.array([[0, 0]]))
    assert np.array_equal(result, np.array([[0]]))


def test_all_detectors_fired():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0]))
    result = decoder.decode(np.array([[1, 1]]))
    assert result.shape == (1, 1)


def test_single_shot_decode():
    """Test _decode (single-shot) via the BaseDecoder interface."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0]))
    result = decoder.decode(np.array([0, 1], dtype=bool))
    assert result.ndim == 1
    assert np.array_equal(result, np.array([True]))


# --- SinterTableDecoder tests ---


def _repetition_dem():
    return repetition_circuit().detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=True
    )


def test_sinter_table_is_sinter_decoder():
    decoder = SinterTableDecoder()
    assert isinstance(decoder, sinter.Decoder)


def test_sinter_table_compile_returns_compiled_decoder():
    dem = simple_dem()
    decoder = SinterTableDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    assert isinstance(compiled, sinter.CompiledDecoder)


def test_sinter_table_decode_shape_and_dtype():
    dem = _repetition_dem()
    decoder = SinterTableDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )

    num_obs_bytes = math.ceil(dem.num_observables / 8)
    assert result.dtype == np.uint8
    assert result.shape == (3, num_obs_bytes)


def test_sinter_table_decode_correctness():
    dem = _repetition_dem()
    decoder = SinterTableDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )
    obs_predictions = unpack_obs(result, dem.num_observables)

    expected = np.array([[True], [True], [True], [False]])
    assert np.array_equal(obs_predictions, expected)


def test_sinter_table_no_error_syndrome():
    dem = simple_dem()
    decoder = SinterTableDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.zeros((1, 2), dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )
    obs_predictions = unpack_obs(result, dem.num_observables)

    assert np.array_equal(obs_predictions, np.array([[False]]))


@pytest.mark.slow
def test_sinter_collect_table():
    circuit = repetition_circuit()
    tasks = [
        sinter.Task(
            circuit=circuit,
            decoder="table_mld",
            json_metadata={"d": 3},
        ),
    ]
    stats = sinter.collect(
        num_workers=1,
        tasks=tasks,
        custom_decoders={"table_mld": SinterTableDecoder(num_shots=100000)},
        max_shots=100,
    )
    assert len(stats) == 1
    assert stats[0].shots == 100
    assert stats[0].errors <= stats[0].shots
    assert stats[0].errors < 50


def test_update_det_obs_counts_wrong_columns():
    """Wrong column count raises ValueError."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([10, 0, 0, 1]))
    wrong_shots = np.array([[0, 0, 0]], dtype=bool)  # 3 cols, expected 2
    with pytest.raises(ValueError, match="columns"):
        decoder.update_det_obs_counts(wrong_shots)


def test_decode_det_obs_counts_wrong_length():
    """Wrong array length raises ValueError."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([10, 0, 0, 1]))
    wrong_counts = np.array([1, 2, 3])  # length 3, expected 4
    with pytest.raises(ValueError, match="length"):
        decoder.decode_det_obs_counts(wrong_counts)
