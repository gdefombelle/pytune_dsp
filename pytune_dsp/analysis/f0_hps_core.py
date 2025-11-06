# pytune_dsp/analysis/f0_hps_core.py
"""
Core HPS (Hypothetical Partial Sequence) pour détection F₀.
Permet d'estimer une fondamentale même absente du spectre.
Compatible avec pitch_detection_hps.py
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# Structures & utilitaires
# =============================================================================

@dataclass
class Peak:
    f_hz: float
    amp: float

@dataclass
class HpsConfig:
    hmax_gen: int = 10        # génération des subharmoniques
    hmax_score: int = 14      # score du peigne
    cents_tol: float = 10.0   # tolérance en cents
    alpha_h: float = 1.5      # pondération 1/h^alpha
    beta_grid: Tuple[float, ...] = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    dedup_cents: float = 5.0  # déduplication des candidats (en cents, espace note)
    use_octave_guard: bool = True

    # --- Nouveaux paramètres ---
    use_note_binning: bool = True     # regrouper des pics proches en "bacs" de note
    note_bin_cents: float = 5.0       # largeur du bac en cents (pour absorber les multiples proches)
    use_pairwise_candidates: bool = True  # générer des candidats F0 à partir de paires de pics (combinatoire contrôlée)
    max_pairs: int = 4000             # limite supérieure de paires évaluées (sécurité perfs)

# =============================================================================
# Conversions
# =============================================================================

def hz_to_midi(f: float) -> float:
    if f <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(f / 440.0)

def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

def cents_ratio(a_hz: float, b_hz: float) -> float:
    return 1200.0 * math.log2(a_hz / b_hz)

# =============================================================================
# Extraction de pics FFT simple
# =============================================================================

def extract_peaks_numpy(mag: np.ndarray, freqs: np.ndarray, k: int = 30, floor_db: float = -40.0) -> List[Peak]:
    eps = 1e-12
    db = 20 * np.log10(mag + eps)
    mask = db >= (db.max() + floor_db)
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return []
    local = []
    for i in range(1, len(idx) - 1):
        j = idx[i]
        if db[j] > db[j - 1] and db[j] > db[j + 1]:
            local.append(j)
    if not local:
        return []
    local = np.array(local)
    vals = db[local]
    topk = np.argsort(vals)[::-1][:k]
    peaks = [Peak(float(freqs[local[i]]), float(mag[local[i]])) for i in topk]
    return peaks

def quantize_peaks_to_note_bins(peaks: List[Peak], bin_cents: float = 5.0) -> List[Peak]:
    """
    Regroupe des pics très proches (en cents) en un seul pic agrégé.
    """
    if not peaks:
        return []
    arr = []
    for p in peaks:
        n = hz_to_midi(p.f_hz)
        arr.append((n, p.f_hz, p.amp))
    arr.sort(key=lambda x: x[0])

    out: List[Peak] = []
    bin_n, bin_fa, bin_a = arr[0][0], arr[0][1] * arr[0][2], arr[0][2]
    for i in range(1, len(arr)):
        n, f, a = arr[i]
        if abs((n - bin_n) * 100.0) <= bin_cents:
            bin_fa += f * a
            bin_a  += a
            bin_n  = (bin_n * (bin_a - a) + n * a) / max(bin_a, 1e-12)
        else:
            f_agg = bin_fa / max(bin_a, 1e-12)
            out.append(Peak(float(f_agg), float(bin_a)))
            bin_n, bin_fa, bin_a = n, f * a, a
    f_agg = bin_fa / max(bin_a, 1e-12)
    out.append(Peak(float(f_agg), float(bin_a)))
    return out

# =============================================================================
# Génération et scoring HPS
# =============================================================================

def build_candidates_note(peaks: List[Peak], cfg: HpsConfig) -> List[float]:
    cands = []
    for p in peaks:
        n = hz_to_midi(p.f_hz)
        for h in range(1, cfg.hmax_gen + 1):
            cands.append(n - 12.0 * math.log2(h))
    cands.sort()
    out = []
    for m in cands:
        if not out or abs((m - out[-1]) * 100.0) > cfg.dedup_cents:
            out.append(m)
    return out

def _dedup_midi_list(m_list: List[float], dedup_cents: float) -> List[float]:
    if not m_list:
        return []
    m_list = sorted(m_list)
    out = [m_list[0]]
    for m in m_list[1:]:
        if abs((m - out[-1]) * 100.0) > dedup_cents:
            out.append(m)
    return out

def build_candidates_note_pairwise(peaks: List[Peak], cfg: HpsConfig) -> List[float]:
    """
    Génère des candidats F0 à partir de paires de pics et d'indices harmoniques plausibles.
    """
    N = len(peaks)
    if N < 2:
        return []
    notes = [hz_to_midi(p.f_hz) for p in peaks]
    amps  = [p.amp for p in peaks]
    pairs_budget = cfg.max_pairs
    cands: List[float] = []
    for i in range(N):
        if pairs_budget <= 0:
            break
        for j in range(i + 1, N):
            if pairs_budget <= 0:
                break
            if amps[i] * amps[j] <= 0:
                continue
            for h1 in range(1, cfg.hmax_gen + 1):
                lh1 = 12.0 * math.log2(h1)
                m_i = notes[i] - lh1
                for h2 in range(1, cfg.hmax_gen + 1):
                    lh2 = 12.0 * math.log2(h2)
                    m_j = notes[j] - lh2
                    if abs((m_i - m_j) * 100.0) <= cfg.dedup_cents:
                        cands.append(0.5 * (m_i + m_j))
            pairs_budget -= 1
    return _dedup_midi_list(cands, cfg.dedup_cents)

def score_one(m_f0: float, peaks: List[Peak], cfg: HpsConfig) -> Tuple[float, float, float]:
    best = (-1e9, m_f0, 0.0)
    f0 = midi_to_hz(m_f0)
    for beta in cfg.beta_grid:
        S = 0.0
        low = 0
        for h in range(1, cfg.hmax_score + 1):
            f_pred = h * f0 * math.sqrt(1.0 + (h * h) * beta)
            best_w = 0.0
            for pk in peaks:
                d = abs(cents_ratio(pk.f_hz, f_pred))
                if d <= cfg.cents_tol:
                    w = (pk.amp / (h ** cfg.alpha_h)) * math.exp(-(d ** 2) / (2 * (cfg.cents_tol / 2) ** 2))
                    if w > best_w:
                        best_w = w
            S += best_w
            if h <= 4 and best_w > 0:
                low += 1
        f1 = f0 * math.sqrt(1.0 + beta)
        has_h1 = any(abs(cents_ratio(pk.f_hz, f1)) <= cfg.cents_tol for pk in peaks)
        if not has_h1:
            S -= 0.3
        if beta != 0.0 and not (1e-6 <= beta <= 3e-3):
            S -= 1.0
        S += 0.2 * low
        if S > best[0]:
            best = (S, m_f0, beta)
    return best

# =============================================================================
# Fonction principale
# =============================================================================

def estimate_f0_hps(peaks: List[Peak],
                    cfg: Optional[HpsConfig] = None,
                    seed_octave: Optional[int] = None) -> Tuple[float, float, dict]:
    cfg = cfg or HpsConfig()
    if not peaks:
        return 0.0, 0.0, {"conf": 0.0, "matched_low": 0}

    if cfg.use_note_binning:
        peaks = quantize_peaks_to_note_bins(peaks, bin_cents=cfg.note_bin_cents)

    if cfg.use_pairwise_candidates:
        cands = build_candidates_note_pairwise(peaks, cfg)
        cands += build_candidates_note(peaks, cfg)
        cands = _dedup_midi_list(cands, cfg.dedup_cents)
    else:
        cands = build_candidates_note(peaks, cfg)

    best = (-1e9, None, 0.0)
    for m in cands:
        if seed_octave is not None:
            if abs((m - 69.0) / 12.0 - seed_octave) > 1.0:
                continue
        S, m0, b0 = score_one(m, peaks, cfg)
        if cfg.use_octave_guard:
            for m_alt in (m0 - 12.0, m0 + 12.0):
                S_alt, m1, b1 = score_one(m_alt, peaks, cfg)
                if S_alt > S:
                    S, m0, b0 = S_alt, m1, b1
        if S > best[0]:
            best = (S, m0, b0)

    S, m_best, beta = best
    f0_hz = midi_to_hz(m_best) if m_best is not None else 0.0
    return f0_hz, beta, {"conf": max(0.0, S)}