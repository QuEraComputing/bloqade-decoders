from kirin import types, passes
from kirin.prelude import structural_no_opt
from kirin.dialects import ilist

from bloqade.decoders.dialects import immediate_loop

structural_with_immediate_loop = structural_no_opt.union([immediate_loop])


def test_repeat_typeinfer_single_return_type():

    @structural_with_immediate_loop
    def main():
        def body():
            return 42

        return immediate_loop.repeat(num_iterations=3, method=body)

    passes.Fold(structural_with_immediate_loop).fixpoint(main)
    passes.TypeInfer(structural_with_immediate_loop)(main)

    assert main.return_type.is_subseteq(ilist.IListType[types.Int, types.Literal(3)])


def test_repeat_typeinfer_tuple_return_type():

    @structural_with_immediate_loop
    def main():
        def body():
            return (1, 2.0)

        return immediate_loop.repeat(num_iterations=3, method=body)

    passes.Fold(structural_with_immediate_loop).fixpoint(main)
    passes.TypeInfer(structural_with_immediate_loop)(main)

    expected = ilist.IListType[types.Tuple[types.Int, types.Float], types.Literal(3)]
    assert main.return_type.is_subseteq(expected)
