from kirin import ir
from kirin.prelude import structural_no_opt

from bloqade.decoders.dialects import immediate_loop
from bloqade.decoders.dialects.immediate_loop.stmts import Repeat


@ir.dialect_group(structural_no_opt.add(immediate_loop.dialect))
def imm_loop_kernel(self):
    def run_pass(mt):
        return mt

    return run_pass


@imm_loop_kernel
def use_repeat():
    x = 1
    with immediate_loop.repeat(100):
        x = x + 1
    return x


use_repeat.print()


def test_repeat_lowering():
    stmt = use_repeat.callable_region.blocks[0].stmts.at(-2)
    assert isinstance(stmt, Repeat)
    assert len(stmt.body.blocks) == 1
    assert len(stmt.body.blocks[0].stmts) >= 1
