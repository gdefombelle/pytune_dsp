# pytune_dsp/tests/test_unison_two_pass.py
"""
Unisson (notes polycordes) — détection 2 f0 avec Essentia
---------------------------------------------------------
Idée : MultiPitchKlapuri (MPK) sur le signal complet + vote multi-frames,
filtre des sous-harmoniques, puis **deuxième passe sur le résidu** après
soustraction sinusoïdale de f0₁. On récupère f0₁ et f0₂ (écart en cents).

Ce fichier fonctionne :
- comme test (pytest),
- ou comme script autonome: `poetry run python pytune_dsp/tests/test_unison_two_pass.py`
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# Essentia (skip propre si non installé)
essentia = pytest.importorskip("essentia", reason="Essentia n'est pas installé")
from essentia import log as eslog
import essentia.standard as es

eslog.infoActive = False  # silence les warnings Essentia


# ---------- utilitaires ----------

def cents(a_hz: float, b_hz: float) -> float:
    """Différence (a vs b) en cents."""
    return 1200.0 * math.log2(max(a_hz, 1e-12) / max(b_hz, 1e-12))

def to_float_list(x) -> list[float]:
    """Aplatis n'importe quelle structure Essentia/NumPy en liste de `float` Python."""
    if np.isscalar(x):
        return [float(x)]
    out: list[float] = []
    stack = [x]
    while stack:
        a = stack.pop()
        try:
            stack.extend(list(a))  # liste/ndarray
        except TypeError:
            out.append(float(a))
    return out

def remove_subharmonics(cands: list[float], cents_tol: float = 30.0) -> list[float]:
    """Supprime f si ~ g/2 (±cents_tol) pour un g présent (filtre sous-harmoniques)."""
    if not cands:
        return []
    cands = sorted(cands)
    keep: list[float] = []
    for f in cands:
        is_sub = any(abs(cents(f * 2.0, g)) <= cents_tol for g in cands if g > f)
        if not is_sub:
            keep.append(f)
    return keep

def cluster_by_cents(cands: list[float], tol: float = 8.0) -> tuple[list[float], list[int]]:
    """
    Clustering 1D simple par proximité en cents (tolérance serrée pour unissons).
    Retourne (centres_médianes, tailles) triés par taille décroissante.
    """
    cands = sorted(cands)
    if not cands:
        return [], []
    clusters: list[list[float]] = []
    for f in cands:
        placed = False
        for cl in clusters:
            if abs(cents(f, float(np.median(cl)))) <= tol:
                cl.append(f); placed = True; break
        if not placed:
            clusters.append([f])
    clusters.sort(key=len, reverse=True)
    centers = [float(np.median(cl)) for cl in clusters]
    sizes   = [len(cl) for cl in clusters]
    return centers, sizes

def proj_remove_sine(x: np.ndarray, sr: int, f0: float) -> np.ndarray:
    """Soustrait la sinusoïde à f0 (projection LS sur sin/cos) pour former un résidu."""
    n = x.size
    t = np.arange(n, dtype=np.float32)
    w = 2.0 * math.pi * f0 / sr
    c = np.cos(w * t, dtype=np.float32)
    s = np.sin(w * t, dtype=np.float32)
    a = (2.0 / n) * float(np.dot(x, c))
    b = (2.0 / n) * float(np.dot(x, s))
    return (x - (a * c + b * s)).astype(np.float32, copy=False)

def mpk_collect_tail(
    y_f32: np.ndarray, sr: int,
    fmin: float, fmax: float,
    frame_size: int = 4096, hop_size: int = 256,
    tail_frames: int = 40
) -> list[float]:
    """
    MPK sur le *signal entier*. On collecte les candidats f0 (Hz) sur
    les `tail_frames` dernières frames non vides.
    """
    y_f32 = np.asarray(y_f32, dtype=np.float32)
    mpk = es.MultiPitchKlapuri(sampleRate=sr, frameSize=frame_size, hopSize=hop_size,
                               minFrequency=fmin, maxFrequency=fmax)
    frames = mpk(y_f32)  # liste de frames -> chaque frame = liste/array de f0
    bag: list[float] = []
    seen = 0
    for fr in reversed(frames):
        vals = to_float_list(fr)
        if vals:
            bag.extend(vals)
            seen += 1
            if seen >= tail_frames:
                break
    return bag


# ---------- détecteur deux passes (stable) ----------

def detect_unison_two_pass(
    y_f32: np.ndarray, sr: int,
    fmin: float = 300.0, fmax: float = 900.0,
    cents_tol: float = 8.0,
    residual_ratio_min: float = 0.08
) -> dict:
    """
    Retourne {f0s, diff_cents, residual_ratio, is_unison_problem}

    Pass 1: MPK -> sac de candidats (fin du signal) -> filtre sous-harm. -> clustering -> f0₁.
    Pass 2: Soustraction sinusoïde(f0₁) -> MPK sur résidu -> même post-traitement -> f0₂.
    Décision: 4–30 cents d'écart et résiduel >= residual_ratio_min.
    """
    rms = lambda x: float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float32)))))

    # --- Pass 1
    bag1 = mpk_collect_tail(y_f32, sr, fmin=fmin, fmax=fmax)
    bag1 = remove_subharmonics(bag1)
    c1, n1 = cluster_by_cents(bag1, tol=cents_tol)
    if not c1:
        return {"f0s": [], "diff_cents": None, "residual_ratio": None, "is_unison_problem": False}
    f0_1 = c1[0]

    # --- Résiduel (pour révéler la 2e composante)
    resid = proj_remove_sine(np.asarray(y_f32, dtype=np.float32), sr, f0_1)
    rr = rms(resid) / max(rms(y_f32), 1e-12)

    # --- Pass 2
    bag2 = mpk_collect_tail(resid, sr, fmin=fmin, fmax=fmax)
    # enlever sous/over-harmoniques proches de f0_1
    bag2 = [f for f in bag2 if abs(cents(f, f0_1)) > (cents_tol + 1.5) and abs(cents(2*f, f0_1)) > 25 and abs(cents(f, 2*f0_1)) > 25]
    bag2 = remove_subharmonics(bag2)
    c2, n2 = cluster_by_cents(bag2, tol=cents_tol)

    # Choix final
    if c2:
        f0_2 = c2[0]
        d = abs(cents(f0_2, f0_1))
        is_prob = (4.0 <= d <= 30.0) and (rr >= residual_ratio_min)
        f0s = sorted([f0_1, f0_2])
        return {"f0s": f0s, "diff_cents": d, "residual_ratio": rr, "is_unison_problem": is_prob}
    else:
        return {"f0s": [float(f0_1)], "diff_cents": None, "residual_ratio": rr, "is_unison_problem": False}


# ---------- tests ----------

def test_two_close_A4_are_found():
    """Deux A4 proches (440 & 443 Hz) -> on doit retrouver ~440 et ~443."""
    sr = 44100
    dur = 2.0
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    y = 0.8 * np.sin(2 * np.pi * 440.0 * t) + 0.7 * np.sin(2 * np.pi * 443.0 * t)  # unisson imparfait

    out = detect_unison_two_pass(y, sr, fmin=300, fmax=900)
    assert len(out["f0s"]) >= 2, f"Deux f0 attendues, obtenu: {out}"
    f1, f2 = out["f0s"][:2]
    assert 430.0 < f1 < 450.0 and 430.0 < f2 < 450.0, f"F0 hors zone A4: {out}"
    assert 8.0 <= out["diff_cents"] <= 20.0, f"écart en cents inattendu: {out}"
    assert out["is_unison_problem"], f"Devrait signaler un unisson imparfait: {out}"

def test_single_A4_is_not_flagged():
    """Une seule composante 440 Hz -> ne doit PAS signaler un problème d'unisson."""
    sr = 44100
    dur = 2.0
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    y = 1.0 * np.sin(2 * np.pi * 440.0 * t)
    out = detect_unison_two_pass(y, sr, fmin=300, fmax=900)
    assert not out["is_unison_problem"], f"Faux positif: {out}"


# ---------- mode script ----------

if __name__ == "__main__":
    sr = 44100
    dur = 2.0
    t = np.arange(int(sr * dur), dtype=np.float32) / sr

    # cas unisson imparfait
    y = 0.8 * np.sin(2 * np.pi * 440.0 * t) + 0.7 * np.sin(2 * np.pi * 443.0 * t)
    res = detect_unison_two_pass(y, sr, fmin=300, fmax=900)
    fmt = lambda v: round(v, 3) if isinstance(v, float) else v
    print("Unison 440/443 ->", {k: (fmt(v) if not isinstance(v, list) else [fmt(x) for x in v]) for k, v in res.items()})

    # cas propre (unisson parfait)
    y2 = 1.0 * np.sin(2 * np.pi * 440.0 * t)
    res2 = detect_unison_two_pass(y2, sr, fmin=300, fmax=900)
    print("Unison parfait 440 ->", {k: (fmt(v) if not isinstance(v, list) else [fmt(x) for x in v]) for k, v in res2.items()})