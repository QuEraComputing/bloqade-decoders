import math

import numpy as np
import pytest
import sinter
import stim

from bloqade.decoders import SinterGurobiDecoder, SinterTableDecoder


def simple_dem():
    return stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.1) D1 L0
        """
    )


def repetition_circuit():
    return stim.Circuit(
        """
        R 0 1 2
        X_ERROR(0.1) 0 1 2
        MZZ 0 1
        DETECTOR rec[-1]
        MZZ 1 2
        DETECTOR rec[-1]
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
    """
    )


def repetition_dem():
    return repetition_circuit().detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=True
    )


def pack_dets(det_shots: np.ndarray) -> np.ndarray:
    """Pack boolean detector shots into uint8 bit-packed format."""
    return np.packbits(
        det_shots.astype(np.uint8), axis=1, bitorder="little"
    )


def unpack_obs(packed_obs: np.ndarray, num_obs: int) -> np.ndarray:
    """Unpack uint8 bit-packed observable predictions to boolean."""
    unpacked = np.unpackbits(packed_obs, axis=1, bitorder="little")
    return unpacked[:, :num_obs].astype(bool)


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


# --- SinterTableDecoder tests ---


def test_sinter_table_is_sinter_decoder():
    decoder = SinterTableDecoder()
    assert isinstance(decoder, sinter.Decoder)


def test_sinter_table_compile_returns_compiled_decoder():
    dem = simple_dem()
    decoder = SinterTableDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    assert isinstance(compiled, sinter.CompiledDecoder)


def test_sinter_table_decode_shape_and_dtype():
    dem = repetition_dem()
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
    dem = repetition_dem()
    decoder = SinterTableDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array(
        [[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool
    )
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


def test_sinter_table_custom_num_shots():
    decoder = SinterTableDecoder(num_shots=10000)
    assert decoder.num_shots == 10000


def test_sinter_table_default_num_shots():
    decoder = SinterTableDecoder()
    assert decoder.num_shots == 2**26


# --- Integration tests with sinter.collect ---


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
        custom_decoders={
            "table_mld": SinterTableDecoder(num_shots=100000)
        },
        max_shots=100,
    )
    assert len(stats) == 1
    assert stats[0].shots == 100
    assert stats[0].errors <= stats[0].shots
    assert stats[0].errors < 50


@pytest.mark.slow
def test_sinter_collect_both_decoders():
    """Run both decoders on the same circuit via sinter.collect."""
    circuit = repetition_circuit()
    tasks = [
        sinter.Task(
            circuit=circuit,
            decoder="gurobi_mle",
            json_metadata={"decoder": "mle"},
        ),
        sinter.Task(
            circuit=circuit,
            decoder="table_mld",
            json_metadata={"decoder": "mld"},
        ),
    ]
    stats = sinter.collect(
        num_workers=1,
        tasks=tasks,
        custom_decoders={
            "gurobi_mle": SinterGurobiDecoder(),
            "table_mld": SinterTableDecoder(num_shots=100000),
        },
        max_shots=100,
    )
    assert len(stats) == 2
    for s in stats:
        assert s.shots == 100
        assert s.errors < 50
