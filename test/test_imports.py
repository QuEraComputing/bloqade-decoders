"""Test that all expected imports work correctly."""


def test_import_annotate_from_dialects():
    """Test that annotate can be imported from bloqade.decoders.dialects."""
    from bloqade.decoders.dialects import annotate

    assert annotate is not None


def test_annotate_exports():
    """Test that annotate exports all expected symbols."""
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


def test_dialect_name():
    """Test that the dialect has the correct name."""
    from bloqade.decoders.dialects.annotate import dialect

    assert dialect.name == "decoders.annotate"


def test_interface_functions():
    """Test that interface functions are callable."""
    from bloqade.decoders.dialects import annotate

    assert callable(annotate.set_detector)
    assert callable(annotate.set_observable)
