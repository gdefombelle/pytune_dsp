"""
f0_analysis.py
==============
Outils pour analyser une note isolée et en extraire une fréquence fondamentale stable.
Inclut version robuste (robust_mode_f0), wrappers rétrocompatibles,
et retour structuré avec SimpleNoteMeasures.
"""

from __future__ import annotations

import numpy as np
from collections import Counter

from pytune_dsp.utils.yin import yin_with_adaptive_window
from pytune_dsp.types.dataclasses import SimpleNoteMeasures


# ============================================================
# Version robuste du "mode" f0
# ============================================================

def robust_mode_f0(
    f0s: np.ndarray,
    amplitudes: np.ndarray | None = None,
    trim_ratio: float = 0.1,
    bin_cents: float = 2.0,
    min_hz: float = 20.0,
    amp_percentile: float | None = 20.0,
) -> tuple[float, float, np.ndarray]:
    """
    Retourne (f0_stable, mode_rate, idx_utilisés).
    - Trim début/fin pour ignorer attaque/queue
    - Gate d'amplitude optionnel
    - Quantification en cents relative à la médiane
    - Mode sur bins, puis moyenne Hz
    """
    f0s = np.asarray(f0s, dtype=float)
    good = np.isfinite(f0s) & (f0s >= min_hz)
    if not np.any(good):
        return 0.0, 0.0, np.array([], dtype=int)

    f0s = f0s[good]
    idx_map = np.flatnonzero(good)

    # Trim attaque/queue
    if 0.0 < trim_ratio < 0.49 and f0s.size >= 10:
        n = f0s.size
        a = int(n * trim_ratio)
        b = n - a
        f0s = f0s[a:b]
        idx_map = idx_map[a:b]

    if f0s.size == 0:
        return 0.0, 0.0, np.array([], dtype=int)

    # Gate amplitude
    if amplitudes is not None and amp_percentile is not None:
        amps = np.asarray(amplitudes, dtype=float)[good]
        if 0.0 < trim_ratio and amps.size >= 10:
            amps = amps[a:b]
        thr = np.nanpercentile(amps, amp_percentile)
        keep = amps >= thr
        if np.any(keep):
            f0s = f0s[keep]
            idx_map = idx_map[keep]

    if f0s.size == 0:
        return 0.0, 0.0, np.array([], dtype=int)

    # Quantification en cents
    f_ref = float(np.median(f0s))
    cents = 1200.0 * np.log2(f0s / f_ref)
    bins = np.round(cents / bin_cents)

    # Mode sur bins
    mode_bin, count = Counter(bins).most_common(1)[0]
    mode_sel = bins == mode_bin
    f0_stable = float(np.mean(f0s[mode_sel]))
    mode_rate = float(count) / float(bins.size)

    return f0_stable, mode_rate, idx_map[mode_sel]


# ============================================================
# Wrappers compatibles (retour tuple)
# ============================================================

def stable_f0_detection(f0s: np.ndarray) -> tuple[float, float]:
    """
    Wrapper rétrocompatible autour de robust_mode_f0.
    Retourne (f0_stable, mode_rate).
    """
    f0, rate, _ = robust_mode_f0(f0s)
    return f0, rate


def stable_f0_detection_for_note(signal: np.ndarray,
                                 sr: int,
                                 fmin: float,
                                 fmax: float,
                                 idx_note: int | None = None,
                                 total_notes: int | None = None) -> tuple[float, float]:
    """
    Extraction YIN + détection robuste du f0 stable.
    """
    f0s = yin_with_adaptive_window(signal, sr, fmin, fmax,
                                   idx_note=idx_note, total_notes=total_notes)
    f0, rate, _ = robust_mode_f0(f0s)
    return f0, rate


# ============================================================
# Version structurée : SimpleNoteMeasures
# ============================================================

def summarize_note(
    f0s: np.ndarray,
    amplitudes: np.ndarray | None,
    ref_hz: float,
    mode_conf_threshold: float = 0.60,
) -> SimpleNoteMeasures:
    """
    Pipeline simple :
      - robust_mode_f0 (f0 stable + mode_rate)
      - centroïde si amplitudes dispo
      - choix du meilleur estimateur
      - déviation vs ref
    """
    stable_f0, mode_rate, _ = robust_mode_f0(f0s, amplitudes)
    if amplitudes is not None and amplitudes.size > 0:
        centroid = float(np.sum(f0s * amplitudes) / np.sum(amplitudes))
    else:
        centroid = stable_f0

    if mode_rate >= mode_conf_threshold and stable_f0 > 0:
        best_method, best_value = "stableF0Average", stable_f0
    else:
        best_method, best_value = "centroid", centroid

    dev_cents = 1200 * np.log2(best_value / ref_hz) if best_value > 0 and ref_hz > 0 else 0.0
    dev_hz = best_value - ref_hz if best_value > 0 and ref_hz > 0 else 0.0

    return SimpleNoteMeasures(
        stable_f0=stable_f0,
        mode_rate=mode_rate,
        centroid=centroid,
        best_method=best_method,
        best_value=best_value,
        dev_cents_vs_ref=dev_cents,
        dev_hz_vs_ref=dev_hz,
    )