# pytune_dsp/analysis/pitch_detection_expected.py
from __future__ import annotations

import math
from typing import List

import numpy as np
import essentia.standard as es

from pytune_dsp.types.dataclasses import GuessNoteResult
from pytune_dsp.utils.note_utils import freq_to_midi, freq_to_note

EPS = 1e-12


def _np_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32, order="C")


def _cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(max(a, 1e-12) / max(b, 1e-12))


def _window_to_bounds(expected_freq: float, window_cents: float, sr: int) -> tuple[float, float]:
    """
    Construit [fmin, fmax] autour de expected_freq, borné à ±1 octave max.
    """
    # clamp à ±1 octave
    total = min(abs(window_cents), 600.0)   # 600 cents = 1 octave
    ratio = 2.0 ** (total / 1200.0)

    fmin = expected_freq / ratio
    fmax = expected_freq * ratio

    fmin = max(20.0, fmin)
    fmax = min(fmax, sr * 0.45)

    if fmax <= fmin:
        fmax = min(sr * 0.45, fmin * 1.5)

    return float(fmin), float(fmax)


def _yinfft_expected_frames(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    debug: bool = False,
) -> tuple[float | None, float]:
    """
    PitchYinFFT sur toute la note, limité au voisinage [fmin, fmax].
    Retourne (f0_median, confidence ∈ [0,1]).
    """
    sr = int(sr)
    y = _np_f32(y)

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
        if debug:
            print("[EXP-YIN] no valid frames in band")
        return None, 0.0

    pitches = np.array(pitches, dtype=np.float64)
    f0_med = float(np.median(pitches))
    spread = float(np.std(pitches) + 1e-9)

    # confiance = mélange dispersion + confiance interne Essentia
    conf_disp = 1.0 / (1.0 + spread / 40.0)          # plus la dispersion est faible, plus c ≈ 1
    conf_ess = float(np.mean(confs))
    conf = float(np.clip(0.6 * conf_ess + 0.4 * conf_disp, 0.0, 1.0))

    if debug:
        dc = _cents(f0_med, (fmin * fmax) ** 0.5)
        print(
            f"[EXP-YIN] f0={f0_med:.2f} Hz conf={conf:.2f} "
            f"(spread={spread:.2f} Hz, win≈[{fmin:.1f},{fmax:.1f}] Δ≈{dc:.1f}c)"
        )

    return f0_med, conf


def guess_note_expected_essentia(
    signal: np.ndarray,
    sr: int,
    expected_freq: float,
    window_cents: float = 300.0,
    debug: bool = True,
) -> GuessNoteResult:
    """
    Version Essentia spécialisée "diagnostic" :
    - on connaît expected_freq
    - on fait un PitchYinFFT dans une fenêtre étroite autour de cette fréquence
      (±window_cents, borné à ±1 octave max)
    """
    if expected_freq <= 0:
        # fallback neutre
        return GuessNoteResult(
            midi=None,
            f0=None,
            confidence=0.0,
            method="expected_yinfft",
        )

    fmin, fmax = _window_to_bounds(expected_freq, window_cents, sr)

    if debug:
        print(f"[EXP-YIN] expected={expected_freq:.2f} Hz, window={window_cents}c → "
              f"band=[{fmin:.1f},{fmax:.1f}] Hz")

    f0, conf = _yinfft_expected_frames(signal, sr, fmin, fmax, debug=debug)

    if not f0 or f0 <= 0:
        return GuessNoteResult(
            midi=None,
            f0=None,
            confidence=0.0,
            method="expected_yinfft",
            debug_log=[f"[EXP-YIN] no f0 found in [{fmin:.1f},{fmax:.1f}] Hz"],
            subresults={"yinfft_expected": {"f0": 0.0, "conf": 0.0}},
            envelope_band="expected",
        )

    midi = freq_to_midi(f0)
    note_name = freq_to_note(f0)
    delta_cents = _cents(f0, expected_freq)

    log = [
        f"[EXP-YIN] expected={expected_freq:.2f} Hz ({freq_to_note(expected_freq)})",
        f"[EXP-YIN] band=[{fmin:.1f},{fmax:.1f}] Hz (±{min(abs(window_cents),600)}c, ≤1 octave)",
        f"[EXP-YIN] f0_med={f0:.2f} Hz ({note_name}) conf={conf:.2f} "
        f"Δ={delta_cents:.1f} cents",
    ]

    sub = {
        "yinfft_expected": {
            "f0": float(f0),
            "conf": float(conf),
        }
    }

    return GuessNoteResult(
        midi=midi,
        f0=float(f0),
        confidence=float(conf),
        method="expected_yinfft",
        debug_log=log,
        subresults=sub,
        envelope_band="expected",
    )