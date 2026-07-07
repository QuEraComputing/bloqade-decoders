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
        verbose: If True, print Gurobi solver output.
    """

    _env: ClassVar[object | None] = None

    def _instantiate(self, verbose: bool = False) -> None:
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

        self._verbose = verbose
        self._dem = self.dem.flattened()
        self._check_no_separators(self.dem)

        import scipy.sparse

        # Single pass over DEM to extract weights, hyperedges, and observables
        weights: list[float] = []
        max_observable_index = -1
        hyperedge_dets: list[list[int]] = []
        hyperedge_obs: list[list[int]] = []

        # Track errors with probability 1.0 (always fire)
        certain_det_flip = np.zeros(self.num_detectors, dtype=int)
        certain_obs_flip = np.zeros(self.num_observables, dtype=int)

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
            (len(weights), self.num_detectors), dtype=bool
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
        observable_indices: list[list[int]] = [
            [] for _ in range(self.num_observables)
        ]
        for e_idx, obs_targets in enumerate(hyperedge_obs):
            for obs_val in obs_targets:
                observable_indices[obs_val].append(e_idx)

        self._detector_vertices = detector_vertices
        self._weights = weights
        self._observable_indices = observable_indices
        self._certain_det_flip = certain_det_flip
        self._certain_obs_flip = certain_obs_flip

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

    def weight_from_error(self, error: np.ndarray) -> np.ndarray:
        return np.sum(error * self._weights, axis=1)

    def _decode_error(self, det_shots: np.ndarray) -> np.ndarray:
        import gurobipy as gp
        from gurobipy import GRB

        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)

        if GurobiDecoder._env is None:
            GurobiDecoder._env = gp.Env()
        env = GurobiDecoder._env
        env.setParam("OutputFlag", 1 if self._verbose else 0)  # type: ignore[union-attr]

        weights = self._weights
        detector_vertices = self._detector_vertices
        # Pre-apply certain errors (prob=1.0) to the syndrome
        det_shots = det_shots.astype(int) ^ self._certain_det_flip

        for d, detector_shot in enumerate(det_shots):
            m = gp.Model("mip1", env=env)
            error_variables: list[gp.Var] = []
            detector_variables: list[gp.Var] = []
            objective: gp.LinExpr = gp.LinExpr(0)

            for i, w in enumerate(weights):
                error_variables.append(m.addVar(vtype=GRB.BINARY, name="e" + str(i)))
                objective += w * error_variables[i]
            m.setObjective(objective, GRB.MAXIMIZE)

            for i, dv in enumerate(detector_vertices):
                detector_variables.append(
                    m.addVar(
                        vtype=GRB.INTEGER,
                        name="h" + str(i),
                        ub=len(dv),
                        lb=0,
                    )
                )
                constraint: gp.LinExpr = gp.LinExpr(0)
                for j in dv:
                    constraint += error_variables[j]
                constraint -= 2 * detector_variables[i]
                m.addConstr(constraint == detector_shot[i], name="c" + str(i))

            m.optimize()
            if m.status != GRB.OPTIMAL:
                if self._verbose:
                    print("Did not find optimal solution", m.status)
                m.close()
                raise RuntimeError(
                    f"Gurobi did not find an optimal solution. Status: {m.status}"
                )
            error = np.round(
                np.array([e.X for e in error_variables]), decimals=0
            ).astype(bool)
            errors[d, :] = error
            m.close()
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
        errors = self._decode_error(det_2d)
        obs = self.logical_from_error(errors)
        return obs[0].astype(np.bool_)

    def decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Decode a batch or single shot of detector bits."""
        if detector_bits.ndim == 1:
            return self._decode(detector_bits)
        errors = self._decode_error(detector_bits)
        return self.logical_from_error(errors)

    def decode_confidence(
        self, detector_bits: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.bool_], float | npt.NDArray[np.float64]]:
        """Decode detector bits with the default confidence score."""
        result = self.decode(detector_bits)
        if detector_bits.ndim == 1:
            return result, 1.0
        return result, np.ones(len(detector_bits), dtype=np.float64)
