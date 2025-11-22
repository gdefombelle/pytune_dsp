# pytune_dsp/analysis/yin_backend_essentia_frames.py

from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import essentia
import essentia.standard as es

# couper le bruit Essentia
essentia.log.infoActive = False
essentia.log.warningActive = False


def _np_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32, order="C")


def yin_backend_essentia_frames(
    signal: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> Tuple[Optional[float], float]:
    """
    Backend générique "frames" basé STRICTEMENT sur la logique de
    `_yinfft_expected_frames` (pitch_detection_expected.py), mais sans notion
    d'expected_freq : on se contente de travailler dans [fmin, fmax].
    Retourne (f0_median, confidence ∈ [0,1]).
    """
    sr = int(sr)
    y = _np_f32(signal)

    # frameSize adaptatif pour couvrir ~12 périodes de fmin
    periods = 12
    target = int(np.ceil(sr * periods / max(fmin, 1e-3)))
    frame_size = 1 << int(np.ceil(np.log2(max(target, 2048))))
    frame_size = int(min(32768, max(2048, frame_size)))
    hop_size = frame_size // 4

    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    yin = es.PitchYinFFT(
        sampleRate=sr,
        frameSize=frame_size,
        minFrequency=fmin,
        maxFrequency=fmax,
    )

    pitches: List[float] = []
    confs: List[float] = []

    for frame in es.FrameGenerator(y, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        mag = spectrum(window(frame))
        f0, c = yin(mag)
        if f0 > 0:
            pitches.append(float(f0))
            confs.append(float(c))

    if not pitches:
        return None, 0.0

    pitches_arr = np.array(pitches, dtype=np.float64)
    f0_med = float(np.median(pitches_arr))
    spread = float(np.std(pitches_arr) + 1e-9)

    # même formule de confiance que `_yinfft_expected_frames`
    conf_disp = 1.0 / (1.0 + spread / 40.0)
    conf_ess = float(np.mean(confs))
    conf = float(np.clip(0.6 * conf_ess + 0.4 * conf_disp, 0.0, 1.0))

    return f0_med, conf