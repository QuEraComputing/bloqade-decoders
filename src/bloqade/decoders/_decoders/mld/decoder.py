from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import BaseDecoder
from .utils import shots_to_counts, pack_boolean_array, unpack_boolean_array


class TableDecoder(BaseDecoder):
    """Maximum likelihood decoder from detector-observable lookup table.

    Builds a lookup table mapping detector syndromes to the most likely
    observable correction, using sampled detector-observable data.

    Conventions:
        - Shot convention: each row is [D0, D1, ..., L0, L1, ...].
        - After packing, each row is an integer with bitstring
          representation (..., L1, L0, ..., D1, D0).
        - Counts can be reshaped as (2**L, 2**D) where row index
          is for logical and column for detector.

    Args:
        dem: The detector error model.

    Examples:
        >>> from bloqade.decoders import TableDecoder
        >>> import stim
        >>> dem = stim.DetectorErrorModel(
        ...     '''
        ...     error(0.02) D0 L0
        ...     error(0.1) D1 L0
        ...     '''
        ... )
        >>> mld_decoder_10k = TableDecoder(dem) # Samples from the detector error model with 10,000 shots
        >>> mld_decoder_1mill = TableDecoder(dem, num_shots=1_000_000) # Samples from the detector error model with 1_000_000 shots
    """

    def _instantiate(self, **kwargs: object) -> None:
        self._det_obs_counts = np.zeros(self._det_obs_counts_len(), dtype=int)
        self._df = None
        self._is_cached_df = False
        self._maximum_likelihood_correction: np.ndarray | None = None
        self._is_cached_correction = False

    def _det_obs_counts_len(self) -> int:
        return 2 ** (self.num_detectors + self.num_observables)

    def _set_det_obs_counts(self, det_obs_counts: np.ndarray) -> None:
        expected_len = self._det_obs_counts_len()
        if det_obs_counts.shape != (expected_len,):
            raise ValueError(
                f"det_obs_counts must have shape ({expected_len},) for "
                f"{self.num_detectors} detectors and {self.num_observables} "
                f"observables, got {det_obs_counts.shape}"
            )
        self._det_obs_counts = det_obs_counts.copy()
        self._df = None
        self._is_cached_df = False
        self._maximum_likelihood_correction: np.ndarray | None = None
        self._is_cached_correction = False

    def train(
        self,
        *,
        num_shots: int = 10_000,
        **_kwargs: Any,
    ) -> None:
        """Replace the lookup table with counts sampled from the error model.

        Any detector-observable counts previously collected with
        :meth:`update_det_obs_counts` are discarded.

        Args:
            num_shots: Number of detector-observable samples to collect.

        Example:
            >>> from bloqade.decoders import TableDecoder
            >>> import stim
            >>> dem = stim.DetectorErrorModel(
            ...     '''
            ...     error(0.02) D0 L0
            ...     error(0.1) D1 L0
            ...     '''
            ... )
            >>> mld_decoder_10k = TableDecoder(dem) # Samples from the detector error model with 10,000 shots
            >>> mld_decoder_10k.train(num_shots=1_000_000) # Replaces the table with a new one sampled with 1_000_000 shots
        """
        data_len = self.num_detectors + self.num_observables
        if data_len > 64:
            raise ValueError(
                f"Total data length {data_len} (detectors + observables) "
                "exceeds 64 bits and cannot be packed into int64."
            )

        sampler = self.dem.compile_sampler()
        det_data, obs_data, _ = sampler.sample(num_shots)
        det_obs_shots = np.concatenate([det_data, obs_data], axis=1)
        self._set_det_obs_counts(np.zeros(self._det_obs_counts_len(), dtype=int))
        self.update_det_obs_counts(det_obs_shots)

    @property
    def det_obs_dataframe(self):  # type: ignore[no-untyped-def]
        """Return a tabular view of the sampled detector-observable counts.

        Each row represents a detector-observable bit pattern with a nonzero
        count. Detector bits are stored in columns named ``det-0``, ``det-1``,
        and so on; observable bits are stored in columns named ``obs-0``,
        ``obs-1``, and so on. The ``samples`` column contains the number of
        times that pattern was observed.

        The DataFrame is created lazily and cached until the underlying counts
        are replaced or updated. Patterns with zero samples are omitted, so
        the number of rows is not the total capacity of the lookup table.

        Returns:
            A Polars DataFrame containing the nonzero detector-observable
            counts.

        Raises:
            ImportError: If Polars is not installed.

        Examples:
            >>> import stim
            >>> from bloqade.decoders import TableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0"
            ... )
            >>> decoder = TableDecoder(dem, num_shots=100)
            >>> dataframe = decoder.det_obs_dataframe
            >>> dataframe.columns
            ['det-0', 'det-1', 'obs-0', 'samples']
            >>> dataframe["samples"].sum()
            100
        """
        if not self._is_cached_df:
            try:
                import polars as pl
            except ImportError as e:
                raise ImportError(
                    "The polars package is required for "
                    "det_obs_dataframe. "
                    'Install it via: pip install "polars"'
                ) from e

            det_obs_counts = self._det_obs_counts
            bins_gt_zero = det_obs_counts > 0
            nonzero_bin_ids = np.arange(len(det_obs_counts))[bins_gt_zero]
            keys = [f"det-{i}" for i in range(self.num_detectors)] + [
                f"obs-{i}" for i in range(self.num_observables)
            ]
            cols: dict[str, np.ndarray] = {
                key: (nonzero_bin_ids & (1 << i)).astype(bool)
                for i, key in enumerate(keys)
            }
            cols["samples"] = det_obs_counts[bins_gt_zero]
            df = pl.DataFrame(cols)
            self._df = df
            self._is_cached_df = True
        return self._df

    def update_det_obs_counts(self, det_obs_shots: np.ndarray) -> None:
        """Accumulate detector-observable shots into the lookup table.

        Args:
            det_obs_shots: Boolean array with shape
                ``(num_shots, num_detectors + num_observables)``. Each row
                contains detector bits first, followed by observable bits:
                ``[D0, D1, ..., L0, L1, ...]``.

        Raises:
            ValueError: If the number of columns does not equal the total
                number of detectors and observables.

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import TableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0"
            ... )
            >>> decoder = TableDecoder(dem)
            >>> shots = np.array(
            ...     [
            ...         [False, False, False],
            ...         [True, False, True],
            ...         [True, False, True],
            ...     ],
            ...     dtype=bool,
            ... )
            >>> decoder.update_det_obs_counts(shots)
            >>> decoder.det_obs_dataframe["samples"].sum() # Expect to see 3 more samples added to the table
            10003

        Notes:
            Counts from repeated calls are added to the existing table. Updating
            the counts invalidates cached corrections and the cached
            :attr:`det_obs_dataframe`.
        """
        data_len = self.num_detectors + self.num_observables
        if data_len != det_obs_shots.shape[1]:
            raise ValueError(
                f"Expected {data_len} columns (detectors + observables), "
                f"got {det_obs_shots.shape[1]}"
            )
        step_counts = shots_to_counts(det_obs_shots)
        self._det_obs_counts += step_counts
        self._is_cached_df = False
        self._is_cached_correction = False

    def cache_correction(self) -> None:
        """
        Build the maximum likelihood correction lookup table as well as computes confidence as the empirical conditional fraction
        (max observable count for syndrome / total count for syndrome). Caches the correction lookup table and the confidence scores for later use.
        """
        if not self._is_cached_correction:
            det_obs_counts = self._det_obs_counts
            obs_counts = det_obs_counts.reshape(
                2**self.num_observables, 2**self.num_detectors
            )
            self._maximum_likelihood_correction = np.argmax(obs_counts, axis=0).reshape(
                -1
            )
            max_counts = np.max(obs_counts, axis=0).astype(np.float64, copy=False)
            total_counts = np.sum(obs_counts, axis=0, dtype=np.uint64)
            confidence = np.zeros(total_counts.shape, dtype=np.float64)
            np.divide(
                max_counts,
                total_counts.astype(np.float64, copy=False),
                out=confidence,
                where=total_counts > 0,
            )
            self._correction_confidence = confidence
            self._is_cached_correction = True

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits."""
        self.cache_correction()
        assert self._maximum_likelihood_correction is not None
        packed = pack_boolean_array(detector_bits.reshape(1, -1))
        correction_idx = self._maximum_likelihood_correction[packed[0]]
        return unpack_boolean_array(np.array([correction_idx]), self.num_observables)[0]

    def decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode detector bits (batch-optimized for 2D input). If a syndrome is never seen during training, the correction
        will be all 0's.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            Observable corrections as boolean array.

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import TableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0"
            ... )
            >>> decoder = TableDecoder(dem)
            >>> corrections = decoder.decode(np.array([True, False], dtype=bool))
            >>> corrections
            array([ True])
            >>> corrections_batch = decoder.decode(np.array([[True, False], [False, True], [False, False], [True, True]], dtype=bool))
            >>> corrections_batch
            array([[ True],
                [ True],
                [False],
                [False]])
        """
        if detector_bits.ndim == 1:
            return self._decode(detector_bits)
        self.cache_correction()
        assert self._maximum_likelihood_correction is not None
        packed_det_shots = pack_boolean_array(detector_bits)
        packed_correction = self._maximum_likelihood_correction[packed_det_shots]
        return unpack_boolean_array(packed_correction, self.num_observables)

    def decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float | npt.NDArray[np.float64]]:
        """Decode detector bits with empirical correction confidence.

        For each detector syndrome, the confidence is the sampled count of its
        most likely observable correction divided by the total sampled count
        for that syndrome. It is in ``[0.0, 1.0]`` and is ``0.0`` for an unseen
        syndrome. For an unseen syndrome, the correction will be all 0.

        This empirical fraction is not on the same scale as
        :class:`GurobiDecoder`'s normalized logical-gap confidence, even though
        both are in ``[0.0, 1.0]``. Confidence thresholds are therefore not
        interchangeable between the MLD and MLE decoders without calibration.

        A simple alternative to calibrating the confidences across decoders would be to sort
        the results of various decoders by confidence, and subsequently do thresholding
        based on the accepted fraction of shots instead of by the raw confidence threshold value.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            A tuple where the first element is the observable corrections, and the second element is the confidence score.
            The confidence score is either a float (for 1D inputs) or an array of floats (for 2D inputs).

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import TableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0\\n"
            ...     "error(0.0) D2"
            ... )
            >>> decoder = TableDecoder(dem)
            >>> corrections_confidence = decoder.decode_confidence(np.array([True, False, False], dtype=bool))
            >>> corrections_confidence
            (array([ True]), 1.0)
            >>> corrections_confidence_unseen = decoder.decode_confidence(np.array([True, False, True], dtype=bool))
            >>> corrections_confidence_unseen
            (array([False]), 0.0)
            >>> corrections_batch_confidence = decoder.decode_confidence(np.array([[True, False, False], [False, True, True], [False, False, True], [True, True, False]], dtype=bool))
            >>> corrections_batch_confidence
            (array([[ True],
                    [False],
                    [False],
                    [False]]),
            array([1., 0., 0., 1.]))
        """
        result = self.decode(detector_bits)
        packed = pack_boolean_array(detector_bits.reshape(-1, self.num_detectors))
        confidence = self._correction_confidence[packed]
        if detector_bits.ndim == 1:
            return result, float(confidence[0])
        return result, confidence

    def decode_det_obs_counts(self, raw_det_obs_counts: np.ndarray) -> np.ndarray:
        """Apply the learned corrections to detector-observable counts.

        The input is a flattened histogram over joint detector-observable bit
        patterns. Detector bits form the low-order bits of each packed index,
        followed by the observable bits. Equivalently, the input can be viewed
        as an array of shape ``(2**num_observables, 2**num_detectors)``, with
        observable patterns as rows and detector syndromes as columns.

        For each detector syndrome, this method XORs the learned observable
        correction into the observable label and moves the corresponding count
        to the corrected bin. Count values and detector labels are unchanged.

        Args:
            raw_det_obs_counts: One-dimensional count array with length
                ``2**(num_detectors + num_observables)``.

        Returns:
            A new count array with the same shape and dtype, indexed by the
            corrected observable labels.

        Raises:
            ValueError: If the input length does not match the number of joint
                detector-observable bit patterns.

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import TableDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0"
            ... )
            >>> decoder = TableDecoder(dem)
            >>> raw_counts = np.arange(8)
            >>> decoder.decode_det_obs_counts(raw_counts)
            array([0, 5, 6, 3, 4, 1, 2, 7])
        """
        self.cache_correction()
        assert self._maximum_likelihood_correction is not None
        num_detectors = self.num_detectors
        num_observables = self.num_observables
        expected_len = 1 << (num_detectors + num_observables)
        if expected_len != raw_det_obs_counts.shape[0]:
            raise ValueError(
                f"Expected array of length {expected_len}, "
                f"got {raw_det_obs_counts.shape[0]}"
            )
        appended_correction = self._maximum_likelihood_correction << num_detectors
        labels = np.arange(1 << (num_detectors + num_observables)).reshape(
            1 << num_observables, 1 << num_detectors
        )
        repeated_appended_correction = appended_correction.reshape(1, -1).repeat(
            1 << num_observables, axis=0
        )
        corrected_labels = labels ^ repeated_appended_correction
        decoded_det_obs_counts = raw_det_obs_counts[corrected_labels.reshape(-1)]
        return decoded_det_obs_counts
