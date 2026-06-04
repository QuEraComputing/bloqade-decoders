from __future__ import annotations

import logging
from os import PathLike

import stim
import numpy as np
import numpy.typing as npt

from ..base import BaseDecoder
from .utils import shots_to_counts, pack_boolean_array, unpack_boolean_array

logger = logging.getLogger(__name__)

_COUNT_DTYPE = np.uint32
_COUNT_MAX = np.iinfo(_COUNT_DTYPE).max


def _empty_count_table(
    shape: int,
    *,
    memmap_path: str | PathLike[str] | None = None,
) -> npt.NDArray[np.uint32]:
    """Create a zeroed dense count table, optionally backed by memmap."""
    if memmap_path is None:
        return np.zeros(shape, dtype=_COUNT_DTYPE)
    counts = np.memmap(
        memmap_path,
        dtype=_COUNT_DTYPE,
        mode="w+",
        shape=(shape,),
    )
    counts[:] = 0
    counts.flush()
    return counts


def _as_count_table(counts: np.ndarray) -> npt.NDArray[np.uint32]:
    """Coerce a detector-observable count table to the decoder storage dtype."""
    arr = counts if isinstance(counts, np.memmap) else np.asarray(counts)
    if np.any(arr < 0):
        raise ValueError("det_obs_counts cannot contain negative counts.")
    if np.any(arr > _COUNT_MAX):
        raise OverflowError(
            f"det_obs_counts contains a value larger than uint32 max ({_COUNT_MAX})."
        )
    if arr.dtype == _COUNT_DTYPE:
        return arr
    return arr.astype(_COUNT_DTYPE, copy=False)


def _add_grouped_shot_counts(
    count_table: npt.NDArray[np.uint32],
    det_obs_shots: np.ndarray,
) -> None:
    """Add shots by grouping equal packed detector-observable labels."""

    packed = pack_boolean_array(det_obs_shots).astype(np.int64, copy=False)
    labels, label_counts = np.unique(packed, return_counts=True)
    if len(labels) == 0:
        return
    label_indices = labels.astype(np.intp, copy=False)
    if np.any(label_counts > (_COUNT_MAX - count_table[label_indices])):
        raise OverflowError(
            f"TableDecoder count table would exceed uint32 max ({_COUNT_MAX})."
        )
    count_table[label_indices] += label_counts.astype(_COUNT_DTYPE, copy=False)


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
        det_obs_counts: Array of shape ``(2**(D+L),)`` counting
            detector-observable pattern frequencies. This may be an
            ``np.memmap``.
        step_counts_memmap_path: Deprecated compatibility option. Memmap-backed
            decoders now update the persistent table from grouped packed labels
            instead of allocating a temporary dense step-count table.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_obs_counts: np.ndarray,
        *,
        step_counts_memmap_path: str | PathLike[str] | None = None,
    ) -> None:
        super().__init__(dem)
        expected_len = 2 ** (dem.num_detectors + dem.num_observables)
        if det_obs_counts.shape != (expected_len,):
            raise ValueError(
                f"det_obs_counts must have shape ({expected_len},) for "
                f"{dem.num_detectors} detectors and {dem.num_observables} "
                f"observables, got {det_obs_counts.shape}"
            )
        self._dem = dem
        self._det_obs_counts = _as_count_table(det_obs_counts)
        self._step_counts_memmap_path = step_counts_memmap_path
        self._df = None
        self._is_cached_df = False
        self._maximum_likelihood_correction: np.ndarray | None = None
        self._is_cached_correction = False

    @property
    def num_detectors(self) -> int:
        return self._dem.num_detectors

    @property
    def num_observables(self) -> int:
        return self._dem.num_observables

    @classmethod
    def from_stim_circuit(
        cls,
        circuit: stim.Circuit,
        num_shots: int = 10**8,
        seed: int | None = None,
        step_size: int = 65536,
        *,
        memmap_path: str | PathLike[str] | None = None,
        step_counts_memmap_path: str | PathLike[str] | None = None,
    ) -> TableDecoder:
        """Build a TableDecoder by sampling a stim circuit.

        Args:
            circuit: The stim circuit to sample from.
            num_shots: Number of shots to sample.
            seed: Optional random seed.
            step_size: Number of shots per sampling batch.
            memmap_path: Optional path for memmapping the decoder's persistent
                detector-observable count table.
            step_counts_memmap_path: Deprecated compatibility option. Memmap-backed
                decoders use grouped packed-label updates instead of a temporary
                dense step-count table.

        Returns:
            A TableDecoder with counts from the sampled shots.
        """
        try:
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError(
                "The tqdm package is required for "
                "TableDecoder.from_stim_circuit. "
                'Install it via: pip install "tqdm"'
            ) from e

        dem = circuit.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True
        )
        num_observables = dem.num_observables
        num_detectors = dem.num_detectors
        data_len = num_observables + num_detectors
        if data_len > 64:
            raise ValueError(
                f"Total data length {data_len} (detectors + observables) "
                "exceeds 64 bits and cannot be packed into int64."
            )

        sampler = circuit.compile_detector_sampler(seed=seed)
        det_obs_counts = _empty_count_table(2**data_len, memmap_path=memmap_path)

        decoder = cls(
            dem=dem,
            det_obs_counts=det_obs_counts,
            step_counts_memmap_path=step_counts_memmap_path,
        )

        progress_bar_steps = ((num_shots - 1) // step_size) + 1
        total_sampled = 0

        logger.info("Building decoder...")
        for _ in tqdm(range(progress_bar_steps)):
            next_shots = min(step_size, num_shots - total_sampled)
            total_sampled += next_shots
            det_obs_shots = sampler.sample(
                next_shots,
                separate_observables=False,
                append_observables=True,
            )
            if not isinstance(det_obs_shots, np.ndarray):
                raise RuntimeError(
                    "Expected np.ndarray from sampler.sample, "
                    f"got {type(det_obs_shots)}"
                )
            decoder.update_det_obs_counts(det_obs_shots)
        return decoder

    @classmethod
    def from_det_obs_shots(
        cls,
        dem: stim.DetectorErrorModel,
        det_obs_shots: np.ndarray,
        *,
        memmap_path: str | PathLike[str] | None = None,
        step_counts_memmap_path: str | PathLike[str] | None = None,
    ) -> TableDecoder:
        """Build a TableDecoder from pre-sampled detector-observable shots.

        Args:
            dem: The detector error model.
            det_obs_shots: Boolean array of shape (num_shots, D+L).
            memmap_path: Optional path for memmapping the decoder's persistent
                detector-observable count table.
            step_counts_memmap_path: Deprecated compatibility option. Memmap-backed
                decoders use grouped packed-label updates instead of a temporary
                dense step-count table.

        Returns:
            A TableDecoder with counts from the provided shots.
        """
        num_detectors = dem.num_detectors
        num_observables = dem.num_observables
        shape: int = 2 ** (num_detectors + num_observables)
        decoder = cls(
            dem=dem,
            det_obs_counts=_empty_count_table(shape, memmap_path=memmap_path),
            step_counts_memmap_path=step_counts_memmap_path,
        )
        decoder.update_det_obs_counts(det_obs_shots)
        return decoder

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
        if isinstance(self._det_obs_counts, np.memmap):
            _add_grouped_shot_counts(self._det_obs_counts, det_obs_shots)
        else:
            step_counts = shots_to_counts(det_obs_shots)
            remaining = _COUNT_MAX - self._det_obs_counts
            if np.any(step_counts > remaining):
                raise OverflowError(
                    f"TableDecoder count table would exceed uint32 max ({_COUNT_MAX})."
                )
            self._det_obs_counts += step_counts.astype(_COUNT_DTYPE, copy=False)
        if isinstance(self._det_obs_counts, np.memmap):
            self._det_obs_counts.flush()
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
