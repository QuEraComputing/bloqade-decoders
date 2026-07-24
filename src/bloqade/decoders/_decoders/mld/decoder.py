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
        """Polars DataFrame of nonzero detector-observable counts."""
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
        """Update counts from new detector-observable shots."""
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
        """Build the maximum likelihood correction lookup table."""
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
        """Decode detector bits (batch-optimized for 2D input).

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            Observable corrections as boolean array.
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
        syndrome.

        This empirical fraction is not on the same scale as
        :class:`GurobiDecoder`'s normalized logical-gap confidence, even though
        both are in ``[0.0, 1.0]``. Confidence thresholds are therefore not
        interchangeable between the MLD and MLE decoders without calibration.
        """
        result = self.decode(detector_bits)
        packed = pack_boolean_array(detector_bits.reshape(-1, self.num_detectors))
        confidence = self._correction_confidence[packed]
        if detector_bits.ndim == 1:
            return result, float(confidence[0])
        return result, confidence

    def decode_det_obs_counts(self, raw_det_obs_counts: np.ndarray) -> np.ndarray:
        """Decode raw detector-observable counts.

        Args:
            raw_det_obs_counts: Array of shape ``(2**(D+L),)``.

        Returns:
            Decoded counts array of the same shape.
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
