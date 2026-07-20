from kirin import types, passes
from kirin.prelude import structural_no_opt
from kirin.registry import Registry
from kirin.interp.table import Signature
from kirin.dialects.py.binop import stmts as binop_stmts

from bloqade.decoders.dialects.annotate.types import (
    MeasurementResult,
    MeasurementResultType,
)


def test_measurement_xor_typeinfer_registry():
    registry = Registry(structural_no_opt).interpreter(["typeinfer"])
    signature = Signature(
        binop_stmts.BitXor, (MeasurementResultType, MeasurementResultType)
    )
    assert signature in registry


def test_measurement_xor_typeinfer():
    @structural_no_opt
    def xor_results(a: MeasurementResult, b: MeasurementResult):
        return a ^ b

    passes.TypeInfer(structural_no_opt)(xor_results)

    assert xor_results.return_type.is_structurally_equal(MeasurementResultType)


def test_int_xor_typeinfer_unchanged():
    @structural_no_opt
    def xor_int(a: int, b: int):
        return a ^ b

    passes.TypeInfer(structural_no_opt)(xor_int)

    assert xor_int.return_type.is_subseteq(types.Int)
