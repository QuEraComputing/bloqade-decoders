"""Type inference for BitXor on MeasurementResult."""

from kirin import interp
from kirin.dialects import py
from kirin.dialects.py.binop import dialect as binop_dialect
from kirin.dialects.py.binop.typeinfer import TypeInfer

from bloqade.decoders.dialects.annotate.types import MeasurementResultType


class MeasurementResultBinOpTypeInfer(TypeInfer):
    @interp.impl(py.BitXor, MeasurementResultType, MeasurementResultType)
    def bitxor_measurement(self, interp_, frame, stmt):
        return (MeasurementResultType,)


binop_dialect.interps["typeinfer"] = MeasurementResultBinOpTypeInfer()
