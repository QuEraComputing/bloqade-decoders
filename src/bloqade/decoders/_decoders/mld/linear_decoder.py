from typing import Any

import numpy as np
import numpy.typing as npt
import stim

from ..base import BaseDecoder
from .utils import pack_boolean_array, unpack_boolean_array


class LinearTableDecoder(BaseDecoder):
    """Exact maximum likelihood decoder via analytic XOR-permutation table construction.

    Builds the joint (detector, observable) probability table analytically from
    the DEM in O(E · 2^(D+L)) time, where E is the number of error mechanisms,
    D the number of detectors, and L the number of observables.

    Algorithm due to Katsuhiro Endo (AIST): each error mechanism applies a XOR
    permutation to the table ``P[d, l] = P(detector=d AND observable=l)``:

    .. code-block:: none

        P_new[d, l] = (1-p)*P[d, l] + p*P[d ^ det_mask, l ^ obs_mask]

    The lookup table is ``argmax_l P[d, l]`` for each detector pattern d.
    Confidence is the exact Bayesian posterior ``max_l P[d,l] / sum_l P[d,l]``.

    Unlike :class:`TableDecoder`, no sampling is needed — the table is exact.
    Memory usage is O(2^(D+L)); building scales linearly with E.

    Args:
        dem: The detector error model.

    Examples:
        >>> from bloqade.decoders import LinearTableDecoder
        >>> import stim
        >>> dem = stim.DetectorErrorModel(
        ...     '''
        ...     error(0.02) D0 L0
        ...     error(0.1) D1 L0
        ...     '''
        ... )
        >>> decoder = LinearTableDecoder(dem)
    """

    def _instantiate(self, **kwargs: Any) -> None:
        num_D = self.num_detectors
        num_L = self.num_observables

        # P[d, l] = P(detector pattern = d AND observable flip = l)
        P = np.zeros((2**num_D, 2**num_L))
        P[0, 0] = 1.0

        d_range = np.arange(2**num_D)
        l_range = np.arange(2**num_L)

        for instruction in self.dem:
            if instruction.type != "error":
                continue
            targets = instruction.target_groups()[0]
            det_mask = sum(
                2 ** t.val for t in targets if t.is_relative_detector_id()
            )
            obs_mask = sum(
                2 ** t.val for t in targets if t.is_logical_observable_id()
            )
            prob = instruction.args_copy()[0]
            d_perm = d_range ^ det_mask
            l_perm = l_range ^ obs_mask
            P = (1 - prob) * P + prob * P[np.ix_(d_perm, l_perm)]

        self._lookup = np.argmax(P, axis=1)
        row_sums = np.sum(P, axis=1)
        max_vals = np.max(P, axis=1)
        confidence = np.zeros(row_sums.shape, dtype=np.float64)
        # Avoid divide-by-zero for unreachable detector patterns
        np.divide(max_vals, row_sums, out=confidence, where=row_sums > 0)
        self._confidence = confidence

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        packed = int(pack_boolean_array(detector_bits.reshape(1, -1))[0])
        return unpack_boolean_array(
            np.array([self._lookup[packed]]), self.num_observables
        )[0]

    def decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode detector bits (batch-optimized for 2D input).

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            Observable corrections as boolean array.

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import LinearTableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0"
            ... )
            >>> decoder = LinearTableDecoder(dem)
            >>> corrections = decoder.decode(np.array([True, False], dtype=bool))
            >>> corrections
            array([ True])
            >>> corrections_batch = decoder.decode(
            ...     np.array([[True, False], [False, True], [False, False], [True, True]], dtype=bool)
            ... )
            >>> corrections_batch
            array([[ True],
                   [ True],
                   [False],
                   [False]])
        """
        if detector_bits.ndim == 1:
            return self._decode(detector_bits)
        if len(detector_bits) == 0:
            return np.empty((0, self.num_observables), dtype=np.bool_)
        packed = pack_boolean_array(detector_bits)
        return unpack_boolean_array(self._lookup[packed], self.num_observables)

    def _decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float]:
        packed = int(pack_boolean_array(detector_bits.reshape(1, -1))[0])
        correction = unpack_boolean_array(
            np.array([self._lookup[packed]]), self.num_observables
        )[0]
        return correction, float(self._confidence[packed])

    def decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float | npt.NDArray[np.float64]]:
        """Decode detector bits with exact Bayesian MLD confidence scores.

        The confidence for syndrome d is ``max_l P(D=d, L=l) / sum_l P(D=d, L=l)``,
        the exact posterior probability of the most likely logical correction.
        It is always in ``[0.0, 1.0]``. Unreachable syndromes have confidence 0.0.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            A tuple of (observable corrections, confidence score(s)).

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import LinearTableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0"
            ... )
            >>> decoder = LinearTableDecoder(dem)
            >>> corrections, confidence = decoder.decode_confidence(
            ...     np.array([True, False], dtype=bool)
            ... )
        """
        if detector_bits.ndim == 1:
            return self._decode_confidence(detector_bits)
        if len(detector_bits) == 0:
            return (
                np.empty((0, self.num_observables), dtype=np.bool_),
                np.empty(0, dtype=np.float64),
            )
        packed = pack_boolean_array(detector_bits)
        corrections = unpack_boolean_array(self._lookup[packed], self.num_observables)
        return corrections, self._confidence[packed].astype(np.float64)
