from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Callable

import numpy as np

from .base import BaseMLEDecoder

if TYPE_CHECKING:
    import pulp


class MILPDecoder(BaseMLEDecoder):
    """MLE decoder using mixed-integer programming via PuLP.

    Finds the most likely error pattern matching an observed syndrome
    by solving a mixed integer program. Supports multiple solver
    backends through PuLP, including HiGHS, CPLEX, COPT, and Gurobi.

    The PuLP model is built once at construction; decoding a shot only
    replaces the right-hand side of the detector constraints before
    re-solving.

    Does NOT support decomposed error models with separator targets.
    Use ``detector_error_model(decompose_errors=False)`` instead.

    Decoding raises ``pulp.PulpSolverError`` when a shot does not solve to
    optimality (timeout, infeasible, numerical trouble, ...). No fallback
    correction is returned in that case; catch the error to retry or skip.

    Args:
        dem: The detector error model describing the error structure.
        solver: Name of a PuLP solver (e.g. ``"HiGHS"``, ``"CPLEX_PY"``,
            ``"COPT"``, ``"GUROBI"``), a ``pulp.LpSolver`` instance, or a
            zero-argument factory returning one. Defaults to ``"HiGHS"``.
            Instances must be deep-copyable: solvers holding native handles
            (such as ``pulp.COPT``, whose coptpy environment cannot be
            pickled) must be passed by name or as a factory instead.
        verbose: If True, print solver output.

    Additional keyword arguments are forwarded to the PuLP solver
    constructor when ``solver`` is a name. PuLP's COPT backend applies
    ``gapRel`` only when truthy, so ``gapRel=0`` is silently ignored; pass
    the native parameters instead, e.g.
    ``MILPDecoder(dem, solver="COPT", RelGap=0.0, AbsGap=0.0)``.

    Examples:
        >>> from bloqade.decoders import MILPDecoder
        >>> import stim
        >>> dem = stim.DetectorErrorModel(
        ...     '''
        ...     error(0.02) D0 L0
        ...     error(0.1) D1 L0
        ...     '''
        ... )
        >>> milp_decoder = MILPDecoder(dem)
        >>> milp_decoder_cbc = MILPDecoder(dem, solver="PULP_CBC_CMD")
    """

    def _setup_solver(
        self,
        solver: str | pulp.LpSolver | Callable[[], pulp.LpSolver] = "HiGHS",
        verbose: bool = False,
        **solver_options: Any,
    ) -> None:
        try:
            import pulp
        except ImportError as e:
            raise ImportError(
                "The pulp package is required for MILPDecoder. "
                'You can install it via: pip install "pulp"'
            ) from e

        self._solver_name: str | None = None
        self._solver_factory: Callable[[], pulp.LpSolver] | None = None

        if isinstance(solver, str):
            # msg and verbose control the same switch; forwarding both to
            # getSolver would be a duplicate keyword argument.
            msg = solver_options.pop("msg", None)
            if msg is not None:
                self._verbose = bool(msg)
            self._solver_name = solver
            self._solver_options = solver_options
            # Validate the name and options eagerly; solving would otherwise
            # fail only on the first decode call.
            self._close_solver(pulp.getSolver(solver, **solver_options))
        elif isinstance(solver, pulp.LpSolver):
            if solver_options:
                raise ValueError(
                    "solver options cannot be forwarded to an LpSolver "
                    "instance; configure the instance itself"
                )
            # Each solve needs a fresh solver (PuLP solver objects cache
            # solve state), provided for instances via deepcopy. Solvers
            # holding native handles (e.g. pulp.COPT's coptpy environment)
            # cannot be deep-copied, so fail fast here with alternatives.
            try:
                copy.deepcopy(solver)
            except Exception as e:
                raise ValueError(
                    f"Solver instance of type {type(solver).__name__} cannot "
                    "be deep-copied (native solver handles are not "
                    'picklable). Pass the solver by name (e.g. solver="COPT") '
                    "or as a zero-argument factory "
                    "(e.g. solver=lambda: pulp.COPT(RelGap=0.0))."
                ) from e
            self._solver_factory = lambda: copy.deepcopy(solver)
        elif callable(solver):
            self._solver_factory = solver
        else:
            raise TypeError(
                "solver must be a PuLP solver name, a pulp.LpSolver "
                "instance, or a zero-argument callable returning a "
                "pulp.LpSolver"
            )

    def _instantiate(self, verbose: bool = False, **kwargs: Any) -> None:
        super()._instantiate(verbose=verbose, **kwargs)
        self._build_model()

    def _build_model(self) -> None:
        """Build the persistent PuLP model (variables and constraints)."""
        import pulp

        self._prob = pulp.LpProblem("mip", pulp.LpMaximize)
        self._error_variables = [
            pulp.LpVariable("e" + str(i), cat=pulp.LpBinary)
            for i in range(len(self._weights))
        ]
        self._prob += pulp.lpSum(
            w * self._error_variables[i] for i, w in enumerate(self._weights)
        )

        self._detector_constraints: list[pulp.LpConstraint] = []
        for i, dv in enumerate(self._detector_vertices):
            detector_variable = pulp.LpVariable(
                "h" + str(i),
                lowBound=0,
                upBound=len(dv),
                cat=pulp.LpInteger,
            )
            self._prob += (
                pulp.lpSum(self._error_variables[j] for j in dv) - 2 * detector_variable
                == 0,
                "c" + str(i),
            )
            self._detector_constraints.append(self._prob.constraints["c" + str(i)])

        self._logical_variables: list[pulp.LpVariable] = []
        for obs_idx, observable_index in enumerate(self._observable_indices):
            logical_var = pulp.LpVariable("l" + str(obs_idx), cat=pulp.LpBinary)
            self._logical_variables.append(logical_var)
            certain_flip = int(self._certain_obs_flip[obs_idx])
            if len(observable_index) == 0:
                self._prob += (
                    logical_var == certain_flip,
                    "lfix" + str(obs_idx),
                )
                continue
            slack_var = pulp.LpVariable(
                "u" + str(obs_idx),
                lowBound=0,
                upBound=len(observable_index),
                cat=pulp.LpInteger,
            )
            self._prob += (
                certain_flip
                + pulp.lpSum(self._error_variables[j] for j in observable_index)
                - 2 * slack_var
                == logical_var,
                "lpar" + str(obs_idx),
            )

        # Persistent diff variables for the forbidden-logical constraints in
        # _solve_single_shot_for_confidence. Creating fresh ones per solve
        # would accumulate unconstrained variables in the persistent model.
        self._diff_variables = [
            pulp.LpVariable("d" + str(obs_idx), cat=pulp.LpBinary)
            for obs_idx in range(self.num_observables)
        ]

    def _set_syndrome(self, detector_shot: np.ndarray) -> None:
        """Replace the detector-constraint RHS with a new shot's syndrome."""
        shot = np.asarray(detector_shot, dtype=int)
        if shot.shape[0] != self.num_detectors:
            raise ValueError(
                f"Expected a syndrome with {self.num_detectors} detector "
                f"bits, got {shot.shape[0]}."
            )
        shot = shot ^ self._certain_det_flip
        # PuLP stores `expr == rhs` as `expr - rhs`, so constant = -rhs.
        for constraint, bit in zip(self._detector_constraints, shot):
            constraint.constant = -int(bit)

    def _fresh_solver(self, verbose: bool | None = None) -> pulp.LpSolver:
        # PuLP solver objects cache solve state (e.g. the GUROBI API
        # backend), and native-API backends like COPT cannot be deep-copied
        # at all, so each solve builds a new solver rather than cloning a
        # template instance.
        import pulp

        show = self._verbose if verbose is None else verbose
        if self._solver_name is not None:
            return pulp.getSolver(self._solver_name, msg=show, **self._solver_options)
        solver = cast(Callable[[], pulp.LpSolver], self._solver_factory)()
        if show and hasattr(solver, "msg"):
            solver.msg = True
        return solver

    @staticmethod
    def _close_solver(solver: pulp.LpSolver) -> None:
        # Native-API backends hold external resources: pulp.COPT keeps a
        # coptpy environment open unless it is closed explicitly.
        env = getattr(solver, "coptenv", None)
        if env is not None:
            env.close()

    def _solve_prob(self, verbose: bool | None = None) -> None:
        solver = self._fresh_solver(verbose)
        try:
            self._prob.solve(solver)
        finally:
            self._close_solver(solver)

    def _decode_error(
        self, det_shots: np.ndarray, confidence: np.ndarray | None = None
    ) -> np.ndarray:
        import pulp

        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)

        for d, detector_shot in enumerate(det_shots):
            self._set_syndrome(detector_shot)
            self._solve_prob()
            status = pulp.LpStatus[self._prob.status]
            if status != "Optimal":
                raise pulp.PulpSolverError(
                    f"MILPDecoder found no optimal solution for shot {d}: "
                    f"solver status is {status!r}. No fallback correction "
                    "is returned; catch this error to retry or skip the shot."
                )
            errors[d, :] = np.round(
                np.array([pulp.value(e) for e in self._error_variables]), decimals=0
            ).astype(bool)
        return errors

    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> tuple[BaseMLEDecoder._ConfidenceSolveResult | None, bool]:
        import pulp

        self._set_syndrome(detector_shot)

        added_constraints: list[str] = []
        if forbidden_logical is not None:
            diff_variables = self._diff_variables
            for obs_idx, forbidden_bit in enumerate(forbidden_logical.astype(int)):
                diff_var = diff_variables[obs_idx]
                name = "ddiff" + str(obs_idx)
                if forbidden_bit:
                    self._prob += (
                        diff_var + self._logical_variables[obs_idx] == 1,
                        name,
                    )
                else:
                    self._prob += (
                        diff_var == self._logical_variables[obs_idx],
                        name,
                    )
                added_constraints.append(name)
            self._prob += (
                pulp.lpSum(diff_variables) >= 1,
                "logical_difference",
            )
            added_constraints.append("logical_difference")

        try:
            self._solve_prob(verbose)
            status = pulp.LpStatus[self._prob.status]
            if status == "Infeasible" and forbidden_logical is not None:
                return None, True
            if status != "Optimal":
                if verbose:
                    print("Did not find optimal solution", status)
                return None, False

            error = np.round(
                np.array([pulp.value(var) for var in self._error_variables]),
                decimals=0,
            ).astype(bool)
            logical = np.round(
                np.array([pulp.value(var) for var in self._logical_variables]),
                decimals=0,
            ).astype(bool)
            objective_value = float(cast(float, pulp.value(self._prob.objective)))
            return (
                self._ConfidenceSolveResult(
                    error=error,
                    logical=logical,
                    objective=objective_value,
                ),
                True,
            )
        finally:
            for name in added_constraints:
                constraint = self._prob.constraints.pop(name, None)
                if constraint is None:
                    continue
                # PuLP's addConstraint also appends to modifiedConstraints;
                # drop that entry too or the temporary constraint is
                # retained for the lifetime of the persistent model.
                # Compare by identity: LpConstraint.__eq__ is expression
                # equality and could match unrelated constraints.
                self._prob.modifiedConstraints = [
                    c for c in self._prob.modifiedConstraints if c is not constraint
                ]
