import numpy as np

from bloqade.decoders.bit_packing import (
    shots_to_counts,
    pack_boolean_array,
    packed_bits_to_int,
    unpack_packed_bits,
    unpack_boolean_array,
    det_obs_shots_to_counts,
)


def test_public_bit_packing_helpers_round_trip_little_endian_bits():
    bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

    packed = pack_boolean_array(bits)

    assert packed.tolist() == [0b101, 0b110]
    np.testing.assert_array_equal(unpack_boolean_array(packed, 3), bits.astype(bool))
    assert packed_bits_to_int([1, 0, 1, 1]) == 0b1101
    np.testing.assert_array_equal(
        unpack_packed_bits(0b1101, 4),
        np.array([1, 0, 1, 1], dtype=np.uint8),
    )


def test_public_count_helpers_pack_detector_observable_shots():
    det_shots = np.array([[0, 0], [1, 0]], dtype=bool)
    obs_shots = np.array([[0], [1]], dtype=bool)

    counts = det_obs_shots_to_counts(det_shots, obs_shots)

    expected = np.zeros(8, dtype=np.int64)
    expected[0] = 1
    expected[5] = 1
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(shots_to_counts(det_shots), np.array([1, 1, 0, 0]))
