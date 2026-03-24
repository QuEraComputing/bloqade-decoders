from __future__ import annotations

from typing import Literal, ClassVar, NamedTuple, cast, overload

import stim
import numpy as np
import numpy.typing as npt
from stim import DemInstruction

from .base import BaseDecoder

try:
    from sinter import (
        Decoder as _SinterDecoder,
        CompiledDecoder as _SinterCompiledDecoder,
    )
except ImportError:
    _SinterCompiledDecoder = object  # type: ignore[assignment,misc]
    _SinterDecoder = object  # type: ignore[assignment,misc]


class GurobiDecoder(BaseDecoder):
    """MLE decoder using Gurobi mixed-integer programming solver.

    Finds the most likely error pattern matching an observed syndrome
    by solving a mixed integer program via Gurobi.

    Does NOT support decomposed error models with separator targets.
    Use ``detector_error_model(decompose_errors=False)`` instead.

    Args:
        dem: The detector error model describing the error structure.
    """

    _env: ClassVar[object | None] = None

    @classmethod
    def _get_env(cls) -> object:
        if cls._env is None:
            import gurobipy as gp

            cls._env = gp.Env()
        return cls._env

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        try:
            import gurobipy  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The gurobipy package is required for GurobiDecoder. "
                'You can install it via: pip install "gurobipy"'
            ) from e

        try:
            import scipy.sparse  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The scipy package is required for GurobiDecoder. "
                'You can install it via: pip install "bloqade-decoders[mle]"'
            ) from e

        super().__init__(dem)
        self._dem = dem.flattened()
        self._check_no_separators(dem)

        import scipy.sparse

        # Single pass over DEM to extract weights, hyperedges, and observables
        weights: list[float] = []
        max_observable_index = -1
        hyperedge_dets: list[list[int]] = []
        hyperedge_obs: list[list[int]] = []

        for instruction in self._dem:  # type: ignore[union-attr]
            if not isinstance(instruction, DemInstruction):
                raise Exception(
                    "The detector-error model should be already flattened. But still got DemRepeatBlock."
                )
            if instruction.type == "error" and instruction.args_copy()[0] != 0:
                probability = instruction.args_copy()[0]
                weights.append(np.log(probability / (1 - probability)))

                det_targets: list[int] = []
                obs_targets: list[int] = []
                for t in instruction.targets_copy():
                    target = cast(stim.DemTarget, t)
                    if stim.DemTarget.is_relative_detector_id(target):
                        det_targets.append(target.val)
                    else:
                        obs_targets.append(target.val)
                        if target.val > max_observable_index:
                            max_observable_index = target.val
                hyperedge_dets.append(det_targets)
                hyperedge_obs.append(obs_targets)

        # Build hyperedges matrix and detector vertices
        hyperedges_matrix = scipy.sparse.lil_matrix(
            (len(weights), dem.num_detectors), dtype=bool
        )
        for row_idx, det_targets in enumerate(hyperedge_dets):
            targets_arr = np.asarray(det_targets)
            if len(targets_arr) > 0:
                hyperedges_matrix[row_idx, targets_arr] = 1

        detector_vertices: list[list[int]] = []
        for row in hyperedges_matrix.T:  # type: ignore[union-attr]
            detector_vertices.append(
                [int(x) for x in np.argwhere(row)[:, 1].flatten()]  # type: ignore[arg-type]
            )

        # Build observable indices (sized from DEM, not max seen index)
        observable_indices: list[list[int]] = [[] for _ in range(dem.num_observables)]
        for e_idx, obs_targets in enumerate(hyperedge_obs):
            for obs_val in obs_targets:
                observable_indices[obs_val].append(e_idx)

        self._detector_vertices = detector_vertices
        self._weights = weights
        self._observable_indices = observable_indices
        self._max_observable_index = max_observable_index

    @staticmethod
    def _check_no_separators(dem: stim.DetectorErrorModel) -> None:
        """Raise ValueError if the DEM contains separator targets."""
        for instruction in dem:  # type: ignore[union-attr]
            if not isinstance(instruction, DemInstruction):
                continue
            if instruction.type == "error":
                for t in instruction.targets_copy():
                    target = cast(stim.DemTarget, t)
                    if stim.DemTarget.is_separator(target):
                        raise ValueError(
                            "GurobiDecoder does not support decomposed "
                            "error models with separator targets. Use "
                            "detector_error_model(decompose_errors=False)"
                            " instead."
                        )

    @property
    def num_detectors(self) -> int:
        return self._dem.num_detectors

    @property
    def num_observables(self) -> int:
        return self._dem.num_observables

    @property
    def max_observable_index(self) -> int:
        return self._max_observable_index

    @property
    def detector_vertices(self) -> list[list[int]]:
        return self._detector_vertices

    @property
    def weights(self) -> list[float]:
        return self._weights

    @property
    def observable_indices(self) -> list[list[int]]:
        return self._observable_indices

    def weight_from_error(self, error: np.ndarray) -> np.ndarray:
        return np.sum(error * self._weights, axis=1)

    class _SolveResult(NamedTuple):
        error: np.ndarray
        logical: np.ndarray
        objective: float

    def _solve_single_shot(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> _SolveResult | None:
        import gurobipy as gp
        from gurobipy import GRB

        env = GurobiDecoder._get_env()
        env.setParam("OutputFlag", 1 if verbose else 0)  # type: ignore[union-attr]

        m = gp.Model("mip1", env=env)
        weights = self._weights
        detector_vertices = self._detector_vertices
        observable_indices = self._observable_indices

        error_variables: list[gp.Var] = []
        detector_variables: list[gp.Var] = []
        logical_variables: list[gp.Var] = []
        objective: gp.LinExpr = gp.LinExpr(0)

        for i, weight in enumerate(weights):
            error_variables.append(m.addVar(vtype=GRB.BINARY, name="e" + str(i)))
            objective += weight * error_variables[i]
        m.setObjective(objective, GRB.MAXIMIZE)

        for i, detector_vertex in enumerate(detector_vertices):
            detector_variables.append(
                m.addVar(
                    vtype=GRB.INTEGER,
                    name="h" + str(i),
                    ub=len(detector_vertex),
                    lb=0,
                )
            )
            constraint: gp.LinExpr = gp.LinExpr(0)
            for j in detector_vertex:
                constraint += error_variables[j]
            constraint -= 2 * detector_variables[i]
            m.addConstr(constraint == int(detector_shot[i]), name="c" + str(i))

        for obs_idx, observable_index in enumerate(observable_indices):
            logical_var = m.addVar(vtype=GRB.BINARY, name="l" + str(obs_idx))
            logical_variables.append(logical_var)
            if len(observable_index) == 0:
                m.addConstr(logical_var == 0, name="lfix" + str(obs_idx))
                continue
            slack_var = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=len(observable_index),
                name="u" + str(obs_idx),
            )
            constraint = gp.LinExpr(0)
            for j in observable_index:
                constraint += error_variables[j]
            constraint -= 2 * slack_var
            m.addConstr(constraint == logical_var, name="lpar" + str(obs_idx))

        if forbidden_logical is not None:
            diff_variables: list[gp.Var] = []
            for obs_idx, forbidden_bit in enumerate(forbidden_logical.astype(int)):
                diff_var = m.addVar(vtype=GRB.BINARY, name="d" + str(obs_idx))
                diff_variables.append(diff_var)
                if forbidden_bit:
                    m.addConstr(
                        diff_var + logical_variables[obs_idx] == 1,
                        name="ddiff" + str(obs_idx),
                    )
                else:
                    m.addConstr(
                        diff_var == logical_variables[obs_idx],
                        name="ddiff" + str(obs_idx),
                    )
            m.addConstr(gp.quicksum(diff_variables) >= 1, name="logical_difference")

        m.optimize()
        if m.status == GRB.INFEASIBLE and forbidden_logical is not None:
            m.close()
            return None
        if m.status != GRB.OPTIMAL:
            if verbose:
                print("Did not find optimal solution", m.status)
            m.close()
            raise RuntimeError(
                f"Gurobi did not find an optimal solution. Status: {m.status}"
            )

        error = np.round(
            np.array([var.X for var in error_variables]), decimals=0
        ).astype(bool)
        logical = np.round(
            np.array([var.X for var in logical_variables]), decimals=0
        ).astype(bool)
        objective_value = float(m.ObjVal)
        m.close()
        return self._SolveResult(
            error=error,
            logical=logical,
            objective=objective_value,
        )

    def decode_error(self, det_shots: np.ndarray, verbose: bool = False) -> np.ndarray:
        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)
        for d, detector_shot in enumerate(det_shots.astype(int)):
            result = self._solve_single_shot(detector_shot, verbose=verbose)
            assert result is not None
            errors[d, :] = result.error
        return errors

    def decode_with_logical_gap(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = False,
    ) -> tuple[npt.NDArray[np.bool_], np.ndarray]:
        """Decode detector bits and return the logical-gap confidence score.

        The logical gap is the objective-value difference between the most
        likely error and the best competing error that implies a different
        logical correction. Larger values indicate higher confidence.
        """
        single_shot = detector_bits.ndim == 1
        det_shots = detector_bits.reshape(1, -1) if single_shot else detector_bits

        decoded_obs = np.zeros(
            (det_shots.shape[0], self.num_observables),
            dtype=np.bool_,
        )
        logical_gaps = np.zeros(det_shots.shape[0], dtype=float)

        for shot_idx, detector_shot in enumerate(det_shots.astype(int)):
            best = self._solve_single_shot(detector_shot, verbose=verbose)
            assert best is not None
            second = self._solve_single_shot(
                detector_shot,
                verbose=verbose,
                forbidden_logical=best.logical,
            )
            decoded_obs[shot_idx] = best.logical
            logical_gaps[shot_idx] = (
                np.inf if second is None else best.objective - second.objective
            )

        if single_shot:
            return decoded_obs[0], logical_gaps
        return decoded_obs, logical_gaps

    def logical_from_error(self, errors: np.ndarray) -> np.ndarray:
        num_shots = errors.shape[0]
        observable_indices = self._observable_indices
        logicals = np.zeros((num_shots, self._dem.num_observables))
        for i, error in enumerate(errors):
            for o, observable_index in enumerate(observable_indices):
                if len(observable_index) > 0:
                    logicals[i, o] = (
                        logicals[i, o] + np.sum(error[np.array(observable_index)])
                    ) % 2
        return logicals.astype(bool)

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits."""
        det_2d = detector_bits.reshape(1, -1)
        errors = self.decode_error(det_2d)
        obs = self.logical_from_error(errors)
        return obs[0].astype(np.bool_)

    @overload
    def decode(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = ...,
        return_weights: Literal[False] = ...,
        return_logical_gap: Literal[False] = ...,
    ) -> npt.NDArray[np.bool_]: ...

    @overload
    def decode(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = ...,
        *,
        return_weights: Literal[True],
        return_logical_gap: Literal[False] = ...,
    ) -> tuple[npt.NDArray[np.bool_], np.ndarray]: ...

    @overload
    def decode(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = ...,
        *,
        return_weights: Literal[False] = ...,
        return_logical_gap: Literal[True],
    ) -> tuple[npt.NDArray[np.bool_], np.ndarray]: ...

    @overload
    def decode(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = ...,
        *,
        return_weights: Literal[True],
        return_logical_gap: Literal[True],
    ) -> tuple[npt.NDArray[np.bool_], np.ndarray, np.ndarray]: ...

    def decode(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = False,
        return_weights: bool = False,
        return_logical_gap: bool = False,
    ) -> (
        npt.NDArray[np.bool_]
        | tuple[npt.NDArray[np.bool_], np.ndarray]
        | tuple[npt.NDArray[np.bool_], np.ndarray, np.ndarray]
    ):
        """Decode detector bits, optionally returning weights.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.
            verbose: If True, print Gurobi solver output.
            return_weights: If True, return (observable_corrections, weights).
            return_logical_gap: If True, also return logical-gap confidence.

        Returns:
            Observable corrections, or tuples with weights and/or logical gaps.
        """
        if return_logical_gap:
            decoded_obs, logical_gaps = self.decode_with_logical_gap(
                detector_bits,
                verbose=verbose,
            )
            if return_weights:
                decoded_errors = self.decode_error(
                    (
                        detector_bits.reshape(1, -1)
                        if detector_bits.ndim == 1
                        else detector_bits
                    ),
                    verbose,
                )
                weights = self.weight_from_error(decoded_errors)
                return decoded_obs, weights, logical_gaps
            return decoded_obs, logical_gaps

        if detector_bits.ndim == 1:
            det_2d = detector_bits.reshape(1, -1)
            errors = self.decode_error(det_2d, verbose)
            obs = self.logical_from_error(errors)
            if return_weights:
                return obs[0].astype(np.bool_), self.weight_from_error(errors)
            return obs[0].astype(np.bool_)
        decoded_errors = self.decode_error(detector_bits, verbose)
        decoded_obs = self.logical_from_error(decoded_errors)
        if return_weights:
            return decoded_obs, self.weight_from_error(decoded_errors)
        return decoded_obs


class _CompiledGurobiDecoder(_SinterCompiledDecoder):  # type: ignore[misc]
    """Compiled decoder wrapping GurobiDecoder for sinter."""

    def __init__(self, decoder: GurobiDecoder) -> None:
        self._decoder = decoder

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        num_dets = self._decoder.num_detectors
        det_shots = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            bitorder="little",
        )[:, :num_dets].astype(bool)
        obs_predictions = self._decoder.decode(det_shots)
        assert isinstance(obs_predictions, np.ndarray)
        return np.packbits(
            obs_predictions.astype(np.uint8),
            axis=1,
            bitorder="little",
        )


class SinterGurobiDecoder(_SinterDecoder):  # type: ignore[misc]
    """Sinter-compatible adapter for the GurobiDecoder (MLE)."""

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> _SinterCompiledDecoder:
        decoder = GurobiDecoder(dem)
        return _CompiledGurobiDecoder(decoder)
