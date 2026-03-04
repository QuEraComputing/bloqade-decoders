from bloqade.decoders.dialects.annotate.types import MeasurementResult, MeasurementResultValue

def test_measurement_result_init():
    result = MeasurementResult(value=MeasurementResultValue.Zero)
    assert result.value == MeasurementResultValue.Zero

    result = MeasurementResult(value=MeasurementResultValue.One)
    assert result.value == MeasurementResultValue.One

    result = MeasurementResult(value=MeasurementResultValue.Lost)
    assert result.value == MeasurementResultValue.Lost