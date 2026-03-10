import numpy as np
import stim

from bloqade.decoders import TableDecoder
from bloqade.decoders._decoders.base import BaseDecoder
from bloqade.decoders._decoders.mld import (
    det_obs_shots_to_counts,
    pack_boolean_array,
    shots_to_counts,
    unpack_boolean_array,
)


def repetition_stim():
    circ = stim.Circuit(
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
    return circ


def repetition_shots():
    det_shots = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    obs_shots = np.array([[1], [1], [1], [0]])
    return det_shots, obs_shots


def test_is_base_decoder():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([10, 0, 0, 1]))
    assert isinstance(decoder, BaseDecoder)


def test_num_detectors():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    decoder = TableDecoder(
        dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0])
    )
    assert decoder.num_detectors == 2


def test_num_observables():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    decoder = TableDecoder(
        dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0])
    )
    assert decoder.num_observables == 1


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
    decoder = TableDecoder.from_stim_circuit(repetition_stim(), 10000)
    assert decoder.num_detectors == 2
    assert decoder.num_observables == 1
    det_shots, obs_shots = repetition_shots()
    correction = decoder.decode(det_shots)
    assert np.array_equal(correction, obs_shots)


def test_decode_obs_det_counts():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    decoder = TableDecoder(
        dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0])
    )
    assert np.array_equal(
        decoder.decode(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
        np.array([[0], [1], [1], [0]]),
    )
    raw_det_obs_counts = np.arange(8)
    decoded_det_obs_counts = decoder.decode_det_obs_counts(
        raw_det_obs_counts
    )
    assert np.array_equal(
        decoded_det_obs_counts, np.array([0, 5, 6, 3, 4, 1, 2, 7])
    )


def test_from_det_obs_shots():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    det_obs_shots = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    decoder = TableDecoder.from_det_obs_shots(dem, det_obs_shots)
    assert decoder.num_detectors == 2
    assert decoder.num_observables == 1
    assert decoder.det_obs_counts.sum() == 6


def test_update_det_obs_counts_invalidates_cache():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([10, 0, 0, 1]))
    decoder.cache_correction()
    assert decoder.is_cached_correction
    new_shots = np.array([[0, 0], [1, 1]], dtype=bool)
    decoder.update_det_obs_counts(new_shots)
    assert not decoder.is_cached_correction
    assert not decoder.is_cached_df


def test_det_obs_dataframe():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\n")
    decoder = TableDecoder(dem, det_obs_counts=np.array([5, 0, 3, 2]))
    df = decoder.det_obs_dataframe
    assert "det-0" in df.columns
    assert "obs-0" in df.columns
    assert "samples" in df.columns
    assert df["samples"].sum() == 10
    # cached
    assert decoder.is_cached_df
    df2 = decoder.det_obs_dataframe
    assert df is df2


def test_no_error_syndrome():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    decoder = TableDecoder(
        dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0])
    )
    result = decoder.decode(np.array([[0, 0]]))
    assert np.array_equal(result, np.array([[0]]))


def test_all_detectors_fired():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    decoder = TableDecoder(
        dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0])
    )
    result = decoder.decode(np.array([[1, 1]]))
    assert result.shape == (1, 1)


def test_single_shot_decode():
    """Test _decode (single-shot) via the BaseDecoder interface."""
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\nerror(0.1) D1 L0\n"
    )
    decoder = TableDecoder(
        dem, det_obs_counts=np.array([81, 0, 0, 1, 0, 9, 9, 0])
    )
    result = decoder.decode(np.array([0, 1], dtype=bool))
    assert result.ndim == 1
    assert np.array_equal(result, np.array([True]))
