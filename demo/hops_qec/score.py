"""Score full and last-round lookup decoders on external HOPS-QEC trajectories."""

from pathlib import Path

import numpy as np
import stim

from bloqade.decoders import TableDecoder

HERE = Path(__file__).resolve().parent


def last_round(detectors):
    # Reconstruct the last observed checks before removing earlier information.
    second = detectors[:, :2] ^ detectors[:, 2:4]
    return np.column_stack((second, detectors[:, 4:]))


def predict(calibration, calibration_flips, detectors):
    width = calibration.shape[1]
    layout = stim.DetectorErrorModel(f"detector D{width - 1}\nlogical_observable L0")
    decoder = TableDecoder.instantiate(layout)
    decoder.update_det_obs_counts(np.column_stack((calibration, calibration_flips)))
    return decoder.decode(detectors).ravel()


def main():
    model = stim.Circuit.from_file(HERE / "prior.stim").detector_error_model()
    calibration, observables, _ = model.compile_sampler(seed=20261120).sample(100_000)
    # The circuit declares Bell X first and Bell Z second.
    flips = observables[:, 1]
    records = np.genfromtxt(HERE / "held_out.csv", delimiter=",", names=True)
    detectors = np.column_stack([records[f"D{i}"] for i in range(6)]).astype(bool)
    full = predict(calibration, flips, detectors)
    last = predict(last_round(calibration), flips, last_round(detectors))
    # Access terminal Bell outcomes only after both predictions are frozen.
    probability = records["probability_z_flip"]
    for name, prediction in (("full", full), ("last round", last)):
        expected_error = np.where(prediction, 1 - probability, probability)
        sampled_errors = np.count_nonzero(prediction != records["sampled_z_flip"])
        print(f"{name}: {sampled_errors}/{len(records)} sampled errors, "
              f"{expected_error.mean():.6f} expected error per shot")
    print(f"Decoder decisions differed on {np.count_nonzero(full != last)} records.")


if __name__ == "__main__":
    main()
