from __future__ import annotations

from typing import ClassVar, cast

import stim
import numpy as np
import numpy.typing as npt
from stim import DemInstruction

from ..base import BaseDecoder


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
                'You can install it via: pip install "scipy"'
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

        # Track errors with probability 1.0 (always fire)
        certain_det_flip = np.zeros(dem.num_detectors, dtype=int)
        certain_obs_flip = np.zeros(dem.num_observables, dtype=int)

        for instruction in self._dem:  # type: ignore[union-attr]
            if not isinstance(instruction, DemInstruction):
                raise Exception(
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
                    if target.val > max_observable_index:
                        max_observable_index = target.val

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
        self._certain_det_flip = certain_det_flip
        self._certain_obs_flip = certain_obs_flip

        self._model: object | None = None
        self._error_vars: list | None = None
        self._constraints: list | None = None
        self._build_model()

    def _build_model(self) -> None:
        """Build the Gurobi model once with placeholder RHS=0 constraints."""
        import gurobipy as gp
        from gurobipy import GRB

        if GurobiDecoder._env is None:
            GurobiDecoder._env = gp.Env()
        env = GurobiDecoder._env

        m = gp.Model("mip1", env=env)

        error_vars: list[gp.Var] = []
        objective = gp.LinExpr(0)
        for i, w in enumerate(self._weights):
            v = m.addVar(vtype=GRB.BINARY, name="e" + str(i))
            error_vars.append(v)
            objective += w * v
        m.setObjective(objective, GRB.MAXIMIZE)

        constraints: list[gp.Constr] = []
        for i, dv in enumerate(self._detector_vertices):
            h = m.addVar(vtype=GRB.INTEGER, name="h" + str(i), ub=len(dv), lb=0)
            lhs = gp.LinExpr(0)
            for j in dv:
                lhs += error_vars[j]
            lhs -= 2 * h
            c = m.addConstr(lhs == 0, name="c" + str(i))
            constraints.append(c)

        m.update()

        self._model = m
        self._error_vars = error_vars
        self._constraints = constraints

    def close(self) -> None:
        """Release the cached Gurobi model resources."""
        if getattr(self, "_model", None) is not None:
            self._model.close()  # type: ignore[union-attr]
            self._model = None
            self._error_vars = None
            self._constraints = None

    def __del__(self) -> None:
        self.close()

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

    def weight_from_error(self, error: np.ndarray) -> np.ndarray:
        return np.sum(error * self._weights, axis=1)

    def decode_error(self, det_shots: np.ndarray, verbose: bool = False) -> np.ndarray:
        from gurobipy import GRB

        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)

        if self._model is None:
            self._build_model()

        env = GurobiDecoder._env
        env.setParam("OutputFlag", 1 if verbose else 0)  # type: ignore[union-attr]

        model = self._model
        error_vars = self._error_vars
        constraints = self._constraints

        # Pre-apply certain errors (prob=1.0) to the syndrome
        det_shots = det_shots.astype(int) ^ self._certain_det_flip

        for d, detector_shot in enumerate(det_shots):
            for i, c in enumerate(constraints):  # type: ignore[arg-type]
                c.RHS = float(detector_shot[i])

            model.optimize()  # type: ignore[union-attr]

            if model.status != GRB.OPTIMAL:  # type: ignore[union-attr]
                if verbose:
                    print("Did not find optimal solution", model.status)  # type: ignore[union-attr]
                raise RuntimeError(
                    f"Gurobi did not find an optimal solution. Status: {model.status}"  # type: ignore[union-attr]
                )
            error = np.round(
                np.array([e.X for e in error_vars]), decimals=0  # type: ignore[union-attr]
            ).astype(bool)
            errors[d, :] = error

        return errors

    def logical_from_error(self, errors: np.ndarray) -> np.ndarray:
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

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a single shot of detector bits."""
        det_2d = detector_bits.reshape(1, -1)
        errors = self.decode_error(det_2d)
        obs = self.logical_from_error(errors)
        return obs[0].astype(np.bool_)

    def decode(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = False,
        return_weights: bool = False,
    ) -> npt.NDArray[np.bool_] | tuple[npt.NDArray[np.bool_], np.ndarray]:
        """Decode detector bits, optionally returning weights.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.
            verbose: If True, print Gurobi solver output.
            return_weights: If True, return (observable_corrections, weights).

        Returns:
            Observable corrections, or (corrections, weights) tuple.
        """
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
