from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from .base import BaseMLEDecoder

if TYPE_CHECKING:
    from gurobipy import Env as GurobiEnv


class GurobiDecoder(BaseMLEDecoder):
    """MLE decoder using Gurobi mixed-integer programming solver.

    .. deprecated::
        Use :class:`MILPDecoder` instead, which supports multiple solver
        backends (HiGHS, CPLEX, COPT, Gurobi) through PuLP.

    Finds the most likely error pattern matching an observed syndrome
    by solving a mixed integer program via Gurobi.

    Does NOT support decomposed error models with separator targets.
    Use ``detector_error_model(decompose_errors=False)`` instead.

    Args:
        dem: The detector error model describing the error structure.
        verbose: If True, print Gurobi solver output.

    Examples:
        >>> from bloqade.decoders import GurobiDecoder
        >>> import stim
        >>> dem = stim.DetectorErrorModel(
        ...     '''
        ...     error(0.02) D0 L0
        ...     error(0.1) D1 L0
        ...     '''
        ... )
        >>> mle_decoder = GurobiDecoder(dem)
        >>> mle_decoder_verbose = GurobiDecoder(dem, verbose=True)
    """

    _env: ClassVar[GurobiEnv | None] = None

    def _instantiate(self, verbose: bool = False, **kwargs: Any) -> None:
        import warnings

        warnings.warn(
            "GurobiDecoder is deprecated; use MILPDecoder instead, which "
            "supports multiple solver backends (HiGHS, CPLEX, COPT, Gurobi) "
            "through PuLP.",
            DeprecationWarning,
            stacklevel=3,
        )
        super()._instantiate(verbose=verbose, **kwargs)

    def _setup_solver(self, verbose: bool = False, **_kwargs: Any) -> None:
        try:
            import gurobipy  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The gurobipy package is required for GurobiDecoder. "
                'You can install it via: pip install "gurobipy"'
            ) from e

    @classmethod
    def _get_env(cls) -> object:
        import gurobipy as gp

        if cls._env is None:
            cls._env = gp.Env()
        return cls._env

    def _decode_error(
        self, det_shots: np.ndarray, confidence: np.ndarray | None = None
    ) -> np.ndarray:
        import gurobipy as gp
        from gurobipy import GRB

        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)

        if GurobiDecoder._env is None:
            GurobiDecoder._env = gp.Env()
        env = GurobiDecoder._env
        assert env is not None
        env.setParam("OutputFlag", 1 if self._verbose else 0)

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
                if confidence is not None:
                    confidence[d] = 0.0
                continue
            error = np.round(
                np.array([e.X for e in error_variables]), decimals=0
            ).astype(bool)
            errors[d, :] = error
            m.close()
        return errors

    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> tuple[BaseMLEDecoder._ConfidenceSolveResult | None, bool]:
        import gurobipy as gp

        GRB = gp.GRB

        env = cast(Any, self._get_env())
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

        detector_shot = np.asarray(detector_shot, dtype=int) ^ self._certain_det_flip
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
            certain_flip = int(self._certain_obs_flip[obs_idx])
            if len(observable_index) == 0:
                m.addConstr(
                    logical_var == certain_flip,
                    name="lfix" + str(obs_idx),
                )
                continue
            slack_var = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=len(observable_index),
                name="u" + str(obs_idx),
            )
            constraint = gp.LinExpr(certain_flip)
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
        status = m.status
        if status == GRB.INFEASIBLE and forbidden_logical is not None:
            m.close()
            return None, True
        if status != GRB.OPTIMAL:
            if verbose:
                print("Did not find optimal solution", status)
            m.close()
            return None, False

        error = np.round(
            np.array([var.X for var in error_variables]), decimals=0
        ).astype(bool)
        logical = np.round(
            np.array([var.X for var in logical_variables]), decimals=0
        ).astype(bool)
        objective_value = float(m.ObjVal)
        m.close()
        return (
            self._ConfidenceSolveResult(
                error=error,
                logical=logical,
                objective=objective_value,
            ),
            True,
        )
