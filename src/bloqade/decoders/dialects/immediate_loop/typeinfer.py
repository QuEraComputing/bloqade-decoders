from kirin import ir, types
from kirin.interp import Frame, MethodTable, impl
from kirin.dialects.ilist import IListType
from kirin.analysis.typeinfer import TypeInference

from .stmts import Repeat
from ._dialect import dialect


@dialect.register(key="typeinfer")
class TypeInfer(MethodTable):
    """Infer result types for immediate-loop statements."""

    @impl(Repeat)
    def repeat(
        self,
        interp: TypeInference,
        frame: Frame[types.TypeAttribute],
        stmt: Repeat,
    ):
        """Propagate a repeated method's return type to the resulting list."""
        method = interp.maybe_const(stmt.method, ir.Method)
        if method is not None:
            if method.inferred:
                return_type = method.return_type
            else:
                _, return_type = interp.call(method.code, interp.method_self(method))
        else:
            method_type = frame.get(stmt.method)
            return_type = (
                method_type.return_type
                if isinstance(method_type, types.FunctionType)
                and method_type.return_type
                else types.Any
            )

        num_iterations = interp.maybe_const(stmt.num_iterations, int)
        dimension = (
            types.Literal(num_iterations) if num_iterations is not None else types.Any
        )

        return (IListType[return_type, dimension],)
