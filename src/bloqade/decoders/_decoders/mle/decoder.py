from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import stim
import numpy as np
import numpy.typing as npt
from stim import DemInstruction

from ..base import BaseDecoder

if TYPE_CHECKING:
    from gurobipy import Env as GurobiEnv


class GurobiDecoder(BaseDecoder):
    """MLE decoder using Gurobi mixed-integer programming solver.

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

    class _ConfidenceSolveResult(NamedTuple):
        error: np.ndarray
        logical: np.ndarray
        objective: float

    def _instantiate(self, verbose: bool = False, **_kwargs: Any) -> None:
        try:
            import gurobipy  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The gurobipy package is required for GurobiDecoder. "
                'You can install it via: pip install "gurobipy"'
            ) from e

        try:
            import scipy.sparse
        except ImportError as e:
            raise ImportError(
                "The scipy package is required for GurobiDecoder. "
                'You can install it via: pip install "scipy"'
            ) from e

        self._verbose = verbose
        self._flat_dem = self.dem.flattened()
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
                    max_observable_index = max(max_observable_index, target.val)

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
        observable_indices: list[list[int]] = [[] for _ in range(self.num_observables)]
        for e_idx, obs_targets in enumerate(hyperedge_obs):
            for obs_val in obs_targets:
                observable_indices[obs_val].append(e_idx)

        self._detector_vertices = detector_vertices
        self._weights = weights
        self._observable_indices = observable_indices
        self._certain_det_flip = certain_det_flip
        self._certain_obs_flip = certain_obs_flip

    @classmethod
    def _get_env(cls) -> object:
        import gurobipy as gp

        if cls._env is None:
            cls._env = gp.Env()
        return cls._env

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
        """Return the log-odds objective value for each error configuration."""
        return np.sum(error * self._weights, axis=1)

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

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import GurobiDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     '''
            ...     error(0.02) D0 L0
            ...     error(0.1) D1 L0
            ...     '''
            ... )
            >>> decoder = GurobiDecoder(dem)
            >>> errors = np.array(
            ...     [
            ...         [False, False],
            ...         [True, False],
            ...         [False, True],
            ...         [True, True],
            ...     ],
            ...     dtype=bool,
            ... )
            >>> decoder.logical_from_error(errors)
            array([[False],
                   [ True],
                   [ True],
                   [False]])
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
        """
        Decode a batch or single shot of detector bits.

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            Observable corrections as boolean array.

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import GurobiDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     '''
            ...     error(0.02) D0 L0
            ...     error(0.1) D1 L0
            ...     '''
            ... )
            >>> decoder = GurobiDecoder(dem)
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
        result, _ = self._decode_batch(detector_bits)
        return result

    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> tuple[_ConfidenceSolveResult | None, bool]:
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

        as the difference between their Gurobi objective values. The returned
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

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import GurobiDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0\\n"
            ...     "error(0.0) D2"
            ... )
            >>> decoder = GurobiDecoder(dem)
            >>> corrections_confidence = decoder.decode_confidence(np.array([True, False, False], dtype=bool))
            >>> corrections_confidence
            (array([ True]), 1.0)
            >>> corrections_confidence_unseen = decoder.decode_confidence(np.array([True, False, True], dtype=bool)) # Impossible error configuration; initial solve is non-optimal
            >>> corrections_confidence_unseen
            (array([False]), 0.0)
            >>> corrections_batch_confidence = decoder.decode_confidence(np.array([[True, False, False], [False, True, True], [False, False, True], [True, True, False]], dtype=bool))
            >>> # Indices 1 and 2 have impossible error configurations
            >>> corrections_batch_confidence
            (array([[ True],
                    [False],
                    [False],
                    [False]]),
            array([1., 0., 0., 1.]))

        Args:
            detector_bits: 1D (single shot) or 2D (batch) boolean array.

        Returns:
            A tuple where the first element is the observable corrections, and the second element is the confidence score.
            The confidence score is either a float (for 1D inputs) or an array of floats (for 2D inputs).

        Examples:
            >>> import stim
            >>> import numpy as np
            >>> from bloqade.decoders import GurobiDecoder
            >>> dem = stim.DetectorErrorModel(
            ...     "error(0.02) D0 L0\\n"
            ...     "error(0.1) D1 L0\\n"
            ...     "error(0.0) D2"
            ... )
            >>> decoder = GurobiDecoder(dem)
            >>> corrections_confidence = decoder.decode_confidence(np.array([True, False, False], dtype=bool))
            >>> corrections_confidence
            (array([ True]), 1.0)
            >>> corrections_confidence_unseen = decoder.decode_confidence(np.array([True, False, True], dtype=bool)) # Impossible error configuration; initial solve is non-optimal
            >>> corrections_confidence_unseen
            (array([False]), 0.0)
            >>> corrections_batch_confidence = decoder.decode_confidence(np.array([[True, False, False], [False, True, True], [False, False, True], [True, True, False]], dtype=bool))
            >>> # Indices 1 and 2 have impossible error configurations
            >>> corrections_batch_confidence
            (array([[ True],
                    [False],
                    [False],
                    [False]]),
            array([1., 0., 0., 1.]))
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
