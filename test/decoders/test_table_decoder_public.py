import stim
import numpy as np

from bloqade.decoders import SparseTableDecoder, TableDecoderWithConfidence


def test_sparse_table_decoder_matches_dense_mld_argmax_semantics():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\nerror(0.5) D1 L0\n")
    shots = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    decoder = SparseTableDecoder.from_det_obs_shots(dem, shots)

    decoded = decoder.decode(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    np.testing.assert_array_equal(decoded, np.array([[0], [1], [1], [0]], dtype=bool))


def test_table_decoder_with_confidence_uses_packed_syndrome_scores():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    decoder = SparseTableDecoder.from_det_obs_shots(
        dem,
        np.array([[0, 0], [1, 1], [1, 1]], dtype=bool),
    )
    wrapped = TableDecoderWithConfidence(
        decoder=decoder,
        syndrome_confidence=np.array([0.25, 0.75], dtype=np.float64),
    )

    correction, confidence = wrapped.decode_with_confidence(np.array([1], dtype=bool))

    np.testing.assert_array_equal(correction, np.array([True]))
    assert confidence == np.float64(0.75)
