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

    An observable is defined by a set of measurement results and an
    index identifying which logical observable this corresponds to.
    """

    traits = frozenset({lowering.FromPythonCall()})

    measurements: ir.SSAValue = info.argument(
        ilist.IListType[MeasurementResultType, types.Any]
    )
    idx: ir.SSAValue = info.argument(types.Int)
    result: ir.ResultValue = info.result(ObservableType)
