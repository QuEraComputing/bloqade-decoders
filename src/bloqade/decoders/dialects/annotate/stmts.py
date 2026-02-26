from kirin import ir, types, lowering
from kirin.decl import info, statement
from kirin.dialects import ilist

from .types import DetectorType, ObservableType, MeasurementResultType
from ._dialect import dialect


@statement(dialect=dialect)
class SetDetector(ir.Statement):
    """Statement for defining a detector from measurement results.

    A detector is defined by a set of measurement results and optional
    coordinates for visualization/debugging purposes.
    """

    traits = frozenset({lowering.FromPythonCall()})

    measurements: ir.SSAValue = info.argument(
        ilist.IListType[MeasurementResultType, types.Any]
    )
    coordinates: ir.SSAValue = info.argument(ilist.IListType[types.Float, types.Any])
    result: ir.ResultValue = info.result(DetectorType)


@statement(dialect=dialect)
class SetObservable(ir.Statement):
    """Statement for defining an observable from measurement results.

    An observable is defined by a set of measurement results. The observable
    index is assigned automatically by the MeasurementIDAnalysis pass.
    """

    traits = frozenset({lowering.FromPythonCall()})

    measurements: ir.SSAValue = info.argument(
        ilist.IListType[MeasurementResultType, types.Any]
    )
    result: ir.ResultValue = info.result(ObservableType)


ReturnT = types.TypeVar("ReturnT")


@statement(dialect=dialect)
class Repeat(ir.Statement):
    """Statement for repeating a zero-argument method a fixed number of times.

    Translates directly to stim's REPEAT block. The method being passed in must take no arguments.
    Returns an IList whose element type matches the method's return type.
    """

    traits = frozenset({lowering.FromPythonCall()})

    num_iterations: ir.SSAValue = info.argument(types.Int)
    method: ir.SSAValue = info.argument(types.MethodType[[], ReturnT])
    result: ir.ResultValue = info.result(ilist.IListType[ReturnT, types.Any])
