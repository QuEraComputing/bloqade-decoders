from kirin import ir, types, lowering
from kirin.decl import info, statement
from kirin.dialects import ilist

from ._dialect import dialect

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
