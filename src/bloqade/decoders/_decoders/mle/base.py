from __future__ import annotations

from abc import abstractmethod
from typing import Any, NamedTuple, cast

import stim
import numpy as np
import numpy.typing as npt
from stim import DemInstruction

from ..base import BaseDecoder


class BaseMLEDecoder(BaseDecoder):
    """Shared machinery for MILP-based most-likely-error decoders.

    Owns the solver-independent parts of the MLE decoder: parsing the
    flattened detector error model into log-odds weights, detector
    hyperedges, and observable indices; pre-applying probability-one
    errors; converting error configurations into logical-observable
    flips; and the logical-gap confidence computation.

    Subclasses provide the solver-specific pieces by implementing
    ``_setup_solver(...)`` (called from ``_instantiate``),
    ``_decode_error(...)``, and ``_solve_single_shot_for_confidence(...)``.

    Does NOT support decomposed error models with separator targets.
    Use ``detector_error_model(decompose_errors=False)`` instead.

    This class is abstract: ``test_decoders.py`` discovers concrete
    decoders via ``__subclasses__()`` and skips abstract ones.
    """

    class _ConfidenceSolveResult(NamedTuple):
        error: np.ndarray
        logical: np.ndarray
        objective: float

    def _setup_solver(self, **kwargs: Any) -> None:
        """Hook for solver-specific setup, called from ``_instantiate``."""

    def _instantiate(self, verbose: bool = False, **kwargs: Any) -> None:
        self._verbose = verbose
        self._setup_solver(verbose=verbose, **kwargs)
        self._flat_dem = self.dem.flattened()
        # Check the flattened DEM: repeat blocks in the original would hide
        # separator targets from the scan.
        self._check_no_separators(self._flat_dem)

        # Single pass over DEM to extract weights, hyperedges, and observables
        weights: list[float] = []
        hyperedge_dets: list[list[int]] = []
        hyperedge_obs: list[list[int]] = []

        # Track errors with probability 1.0 (always fire)
        certain_det_flip = np.zeros(self.num_detectors, dtype=int)
        certain_obs_flip = np.zeros(self.num_observables, dtype=int)

        for instruction in self._flat_dem:  # type: ignore[union-attr]
            if not isinstance(instruction, DemInstruction):
                raise TypeError(
                    "The detector-error model should be already flattened. But still got DemRepeatBlock."
                )
            if instruction.type != "error":
                continue
            probability = instruction.args_copy()[0]
            if probability == 0:
                continue

            det_targets: list[int] = []
            obs_targets: list[int] = []
            for t in instruction.targets_copy():
                target = cast(stim.DemTarget, t)
                if stim.DemTarget.is_relative_detector_id(target):
                    det_targets.append(target.val)
                else:
                    obs_targets.append(target.val)

            if probability == 1:
                # Certain errors always fire: pre-apply their contributions
                for d in det_targets:
                    certain_det_flip[d] ^= 1
                for o in obs_targets:
                    certain_obs_flip[o] ^= 1
            else:
                weights.append(np.log(probability / (1 - probability)))
                hyperedge_dets.append(det_targets)
                hyperedge_obs.append(obs_targets)

        # Invert hyperedge incidence: detector -> error indices touching it
        detector_vertices: list[list[int]] = [[] for _ in range(self.num_detectors)]
        for e_idx, det_targets in enumerate(hyperedge_dets):
            for d in det_targets:
                detector_vertices[d].append(e_idx)

        # Build observable indices (sized from DEM, not max seen index)
        observable_indices: list[list[int]] = [[] for _ in range(self.num_observables)]
        for e_idx, obs_targets in enumerate(hyperedge_obs):
            for obs_val in obs_targets:
                observable_indices[obs_val].append(e_idx)

        self._detector_vertices = detector_vertices
        self._weights = weights
        self._observable_indices = observable_indices
        self._certain_det_flip = certain_det_flip
        self._certain_obs_flip = certain_obs_flip

    def _check_no_separators(self, dem: stim.DetectorErrorModel) -> None:
        """Raise ValueError if the DEM contains separator targets."""
        for instruction in dem:  # type: ignore[union-attr]
            if not isinstance(instruction, DemInstruction):
                continue
            if instruction.type == "error":
                for t in instruction.targets_copy():
                    target = cast(stim.DemTarget, t)
                    if stim.DemTarget.is_separator(target):
                        raise ValueError(
                            f"{type(self).__name__} does not support decomposed "
                            "error models with separator targets. Use "
                            "detector_error_model(decompose_errors=False)"
                            " instead."
                        )

    def weight_from_error(self, error: np.ndarray) -> np.ndarray:
        """Return the log-odds objective value for each error configuration."""
        return np.sum(error * self._weights, axis=1)

    @abstractmethod
    def _decode_error(
        self, det_shots: np.ndarray, confidence: np.ndarray | None = None
    ) -> np.ndarray:
        """Solve the MILP for a batch of detector shots. Solver-specific."""

    def logical_from_error(self, errors: np.ndarray) -> np.ndarray:
        """Convert batched error configurations into logical-observable flips.

        Each row of ``errors`` selects the variable error mechanisms in the
        flattened detector error model. Columns follow the order of error
        instructions with probabilities strictly between zero and one; errors
        with probability one are applied automatically. The logical targets of
        the selected mechanisms are combined modulo two.

        Args:
            errors: Boolean array with shape
                ``(num_shots, num_error_variables)``.

        Returns:
            Boolean array with shape ``(num_shots, num_observables)``.
        """
        num_shots = errors.shape[0]
        observable_indices = self._observable_indices
        # Start from certain error contributions (prob=1.0 errors always fire)
        logicals = np.tile(self._certain_obs_flip, (num_shots, 1)).astype(float)
        for i, error in enumerate(errors):
            for o, observable_index in enumerate(observable_indices):
                if len(observable_index) > 0:
                    logicals[i, o] = (
                        logicals[i, o] + np.sum(error[np.array(observable_index)])
                    ) % 2
        return logicals.astype(bool)

    def _decode_batch(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        confidence = np.ones(len(detector_bits), dtype=np.float64)
        errors = self._decode_error(detector_bits, confidence)
        result = self.logical_from_error(errors)
        result[confidence == 0.0] = False
        return result, confidence

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits."""
        result, _ = self._decode_batch(detector_bits.reshape(1, -1))
        return result[0]

    def decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a batch or single shot of detector bits.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            Observable corrections as boolean array.
        """
        if detector_bits.ndim == 1:
            return self._decode(detector_bits)
        result, _ = self._decode_batch(detector_bits)
        return result

    @abstractmethod
    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> tuple[_ConfidenceSolveResult | None, bool]:
        """Solve one shot, optionally excluding a logical class. Solver-specific."""

    def _decode_with_logical_gap(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = False,
    ) -> tuple[npt.NDArray[np.bool_], np.ndarray]:
        """Decode detector bits and return the logical-gap confidence score."""

        single_shot = detector_bits.ndim == 1
        det_shots = detector_bits.reshape(1, -1) if single_shot else detector_bits

        decoded_obs = np.zeros(
            (det_shots.shape[0], self.num_observables),
            dtype=np.bool_,
        )
        logical_gaps = np.zeros(det_shots.shape[0], dtype=float)

        for shot_idx, detector_shot in enumerate(det_shots.astype(int)):
            best, best_converged = self._solve_single_shot_for_confidence(
                detector_shot,
                verbose=verbose,
            )
            if not best_converged:
                continue
            assert best is not None
            decoded_obs[shot_idx] = best.logical
            second, second_converged = self._solve_single_shot_for_confidence(
                detector_shot,
                verbose=verbose,
                forbidden_logical=best.logical,
            )
            if not second_converged:
                continue
            logical_gaps[shot_idx] = (
                np.inf if second is None else best.objective - second.objective
            )

        if single_shot:
            return decoded_obs[0], logical_gaps
        return decoded_obs, logical_gaps

    def decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float | npt.NDArray[np.float64]]:
        """Decode detector bits and return normalized logical-gap confidence.

        For a detector syndrome, let ``best`` be the most likely error
        configuration and ``alternative`` be the most likely configuration
        with a different logical correction. First compute the logical gap

        ``log(P(best) / P(alternative))``,

        as the difference between their solver objective values. The returned
        confidence is ``tanh(logical_gap / 2)``, a normalized likelihood margin
        in ``[0.0, 1.0]``. It is ``1.0`` when no alternative logical correction
        is feasible and ``0.0`` when the alternatives are equally likely or
        either optimization does not find an optimal solution. If the initial
        solve is not optimal, the default correction is all zeros. If only the
        alternative solve is not optimal, the best correction from the initial
        solve is returned with ``0.0`` confidence.

        This normalized margin is not a calibrated probability and is not on
        the same scale as :class:`TableDecoder`'s empirical confidence.
        Confidence thresholds are therefore not interchangeable between the
        MLE and MLD decoders without calibration.

        A simple alternative to calibrating the confidences across decoders would be to sort
        the results of various decoders by confidence, and subsequently do thresholding
        based on the accepted fraction of shots instead of by the raw confidence threshold value.

        A single detector shot returns one correction and a scalar confidence.
        A batch returns corrections with shape ``(shots, num_observables)``
        and confidence scores with shape ``(shots,)``.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            A tuple where the first element is the observable corrections, and the second element is the confidence score.
            The confidence score is either a float (for 1D inputs) or an array of floats (for 2D inputs).
        """

        single_shot = detector_bits.ndim == 1
        decoded_obs, logical_gaps = self._decode_with_logical_gap(
            detector_bits, verbose=self._verbose
        )

        decoded_obs = decoded_obs.astype(np.bool_)
        logical_gaps = np.asarray(logical_gaps, dtype=np.float64).reshape(-1)
        confidence = np.tanh(np.maximum(logical_gaps, 0.0) / 2.0)

        if single_shot:
            return decoded_obs, float(confidence[0])

        return decoded_obs, confidence
