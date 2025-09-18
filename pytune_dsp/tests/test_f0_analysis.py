import numpy as np
from pytune_dsp.analysis.f0_analysis import stable_f0_detection, calculate_f0_measurements

def test_stable_f0():
    f0s = np.array([440.0, 441.0, 439.5, 440.2, 440.1])
    stable, rate = stable_f0_detection(f0s)
    assert abs(stable - 440) < 1
    assert rate > 0.5

def test_calculate_f0_measurements():
    f0s = np.array([440.0, 441.0, 439.5, 440.2])
    amps = np.ones_like(f0s)
    m = calculate_f0_measurements(f0s, amps, 440.0, 442.0)
    assert m.best_measurement_method in ("stableF0Average", "centroid")