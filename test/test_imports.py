def test_decoder_exports():
    from bloqade import decoders

    assert hasattr(decoders, "BaseDecoder")
    assert hasattr(decoders, "TesseractDecoder")
    assert hasattr(decoders, "BeliefFindDecoder")
    assert hasattr(decoders, "BpLsdDecoder")
    assert hasattr(decoders, "BpOsdDecoder")
    assert hasattr(decoders, "MWPFDecoder")
    assert hasattr(decoders, "dialects")


def test_annotate_exports():
    from bloqade.decoders.dialects import annotate

    # Submodules
    assert hasattr(annotate, "stmts")
    assert hasattr(annotate, "types")

    # Statements (via stmts)
    assert hasattr(annotate.stmts, "SetDetector")
    assert hasattr(annotate.stmts, "SetObservable")

    # Types (via types)
    assert hasattr(annotate.types, "MeasurementResult")
    assert hasattr(annotate.types, "MeasurementResultType")
    assert hasattr(annotate.types, "MeasurementResultValue")
    assert hasattr(annotate.types, "Detector")
    assert hasattr(annotate.types, "DetectorType")
    assert hasattr(annotate.types, "Observable")
    assert hasattr(annotate.types, "ObservableType")

    # Dialect
    assert hasattr(annotate, "dialect")

    # Interface functions
    assert hasattr(annotate, "set_detector")
    assert hasattr(annotate, "set_observable")


def test_measurement_result_value_enum():
    """Test MeasurementResultValue enum values."""
    from bloqade.decoders.dialects.annotate.types import MeasurementResultValue

    assert MeasurementResultValue.Zero == 0
    assert MeasurementResultValue.One == 1
    assert MeasurementResultValue.Lost == 2


def test_interface_functions():
    from bloqade.decoders.dialects import annotate

    assert callable(annotate.set_detector)
    assert callable(annotate.set_observable)
