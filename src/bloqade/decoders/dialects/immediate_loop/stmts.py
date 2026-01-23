from typing import cast

from kirin import ir, types
from kirin.decl import info, statement
from kirin.lowering.python.traits import FromPythonWithSingleItem

from ._dialect import dialect


@statement(dialect=dialect)
class Repeat(ir.Statement):
    """
    Execute a series of statements a fixed number of times. This can be seen as a stricter version of the
    Kirin `scf.For` statement directly mappable to Stim REPEAT semantics.
    """

    name = "repeat"
    traits = frozenset({FromPythonWithSingleItem()})
    count: ir.SSAValue = info.argument(types.Int)
    body: ir.Region = info.region(multi=False)

    def __init__(
        self,
        count: ir.SSAValue,
        body: ir.Region | ir.Block,
    ):
        if body.IS_REGION:
            body_region = cast(ir.Region, body)
            if body_region.blocks:
                body_block = body_region.blocks[0]
            else:
                body_block = None
        else:
            body_block = cast(ir.Block, body)
            body_region = ir.Region(body_block)

        super().__init__(args=(count,), regions=(body_region,), args_slice={"count": 0})
