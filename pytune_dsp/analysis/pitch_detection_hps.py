# hps_multi.py
# -----------------------------------------------------------------------------
# HPS multi-hypothèses (combinatoire légère) pour notes harmoniques (piano).
# - Détecte plusieurs (f0, B) plausibles à partir d'un spectre de pics.
# - Normalisation PGCD (réduit 2×/3×…), garde séquences harmoniques cohérentes.
# - "tessitura" gère tolérances et kmin (auto par défaut, ou "low"/"mid"/"high").
# - API PyTune-compatible: estimate_f0_hps_multi_wrapper(x, fs, top_k=5)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, math
from math import gcd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.signal import get_window
from bisect import bisect_left
from pytune_dsp.analysis.pitch_detection_hps_helpers import *

# ==== Debug =========================================================
HPS_DEBUG = bool(int(os.getenv("HPS_DEBUG", "0")))
def _dbg(msg: str):
    if HPS_DEBUG:
        print(msg)

# ==== Structures ====================================================
@dataclass
class Peak:
    f: float   # Hz
    a: float   # amplitude lin
    db: float  # dB relatif

@dataclass
class Cluster:
    f_mean: float
    f_med: float
    amp_sum: float
    amp_max: float
    n: int

@dataclass
class HpsCandidate:
    f0: float
    B: float
    score: float
    used_idx: np.ndarray
    assigned_h: np.ndarray
    residuals: np.ndarray
    coverage: float
    conf: float

# ==== Notes & ratios ================================================
def _note_from_freq(freq: float) -> str:
    if freq <= 0:
        return "?"
    midi = 69 + 12 * np.log2(freq / 440.0)
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    i = int(round(midi)) % 12
    o = int((round(midi) // 12) - 1)
    return f"{names[i]}{o}"

def _cents_ratio(r: float) -> float:
    return 1200.0 * np.log2(max(r, 1e-12))

# ==== FFT & peaks ===================================================
def extract_peaks_basic(
    x: np.ndarray, fs: float, n_fft: int = 16384, k: int = 30, floor_db: float = -60.0
) -> List[Peak]:
    y = x.astype(np.float64, copy=False)
    if y.size == 0:
        return []
    y -= y.mean()
    y *= get_window("hann", len(y), fftbins=True)

    spec = np.fft.rfft(y, n_fft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    if freqs.size <= 2:
        return []

    lo, hi = 1, len(freqs) - 1
    mag, freqs = mag[lo:hi], freqs[lo:hi]

    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    mmax = float(mag_db.max())
    mask = mag_db >= (mmax + floor_db)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    # Top-k par amplitude (lin)
    sort_idx = np.argsort(mag[idx])[::-1]
    idx_top = idx[sort_idx][:k]
    peaks = [Peak(float(freqs[i]), float(mag[i]), float(mag_db[i])) for i in idx_top]
    peaks.sort(key=lambda p: p.f)
    return peaks

# ==== Clustering (log-f) ============================================
def cluster_peaks_log(peaks: List[Peak], cents_bin: float = 35.0) -> List[Cluster]:
    if not peaks:
        return []
    F = np.array([p.f for p in peaks], float)
    A = np.array([p.a for p in peaks], float)
    order = np.argsort(F)
    F, A = F[order], A[order]

    clusters: List[Cluster] = []
    cur_idx = [0]
    cur_center = F[0]
    for i in range(1, len(F)):
        if abs(_cents_ratio(F[i] / cur_center)) <= cents_bin:
            cur_idx.append(i)
            cur_center = np.average(F[cur_idx], weights=A[cur_idx])
        else:
            fvals, avals = F[cur_idx], A[cur_idx]
            clusters.append(Cluster(
                float(np.average(fvals, weights=avals)),
                float(np.median(fvals)),
                float(avals.sum()),
                float(avals.max()),
                len(cur_idx),
            ))
            cur_idx = [i]
            cur_center = F[i]
    # flush
    fvals, avals = F[cur_idx], A[cur_idx]
    clusters.append(Cluster(
        float(np.average(fvals, weights=avals)),
        float(np.median(fvals)),
        float(avals.sum()),
        float(avals.max()),
        len(cur_idx),
    ))
    clusters.sort(key=lambda c: c.f_mean)
    return clusters

def _nearest_cluster(clusters: List[Cluster], f_target: float) -> Tuple[int, float]:
    if not clusters:
        return (-1, 1e9)
    Fs = [c.f_mean for c in clusters]
    j = bisect_left(Fs, f_target)
    cand = []
    if j < len(Fs): cand.append(j)
    if j - 1 >= 0:  cand.append(j - 1)
    if not cand:    return (-1, 1e9)
    best = min(cand, key=lambda k: abs(_cents_ratio(f_target / Fs[k])))
    return best, float(_cents_ratio(f_target / Fs[best]))

# ==== Seeds depuis clusters forts (h petits) ========================
def seed_from_clusters(
    clusters: List[Cluster], top_clusters: int = 6, h_small: Tuple[int, int] = (1, 6)
) -> np.ndarray:
    if not clusters:
        return np.array([])
    top = sorted(clusters, key=lambda c: c.amp_sum, reverse=True)[:top_clusters]
    seeds = []
    for c in top:
        for h in range(h_small[0], h_small[1] + 1):
            f0 = c.f_mean / h
            if 15.0 <= f0 <= 4000.0:
                seeds.append(f0)
    if not seeds:
        return np.array([])
    return np.unique(np.round(np.array(seeds, float), 4))

# ==== Modèle inharmonique ==========================================
def _harm_ladder(f0: float, B: float, hmax: int) -> np.ndarray:
    h = np.arange(1, hmax + 1, dtype=float)
    return f0 * h * np.sqrt(1.0 + B * h * h)

# ==== Helpers (PGCD, priors) =======================================
def _gcd_many(vals: np.ndarray) -> int:
    if vals.size == 0:
        return 1
    g = int(abs(vals[0]))
    for v in vals[1:]:
        g = gcd(g, int(abs(v)))
        if g == 1:
            break
    return max(1, g)

def _tessitura_prior(f0: float, tessitura: str) -> float:
    # Gauss mou en log2 autour d'un centre par registre
    if tessitura == "low":
        f_c, w = 55.0, 0.8     # ~A1
    elif tessitura == "high":
        f_c, w = 1760.0, 0.8   # ~A6
    else:
        f_c, w = 440.0, 0.8    # ~A4
    d = abs(math.log2(max(f0, 1e-9) / f_c))
    return math.exp(-(d * d) / (2 * w * w))

# ==== Scoring principal ============================================
def score_candidate_on_clusters(
    clusters: List[Cluster],
    f0: float,
    B: float,
    hmax: int = 16,
    cents_tol: float = 30.0,
    tessitura: str = "mid",
    kmin: int = 3,
) -> HpsCandidate:
    fhar = _harm_ladder(f0, B, hmax)
    used_idx: List[int] = []
    assigned_h: List[int] = []
    residuals: List[float] = []
    amps: List[float] = []
    taken = set()

    for h_i, ft in enumerate(fhar, start=1):
        j, res = _nearest_cluster(clusters, ft)
        if j >= 0 and abs(res) <= cents_tol and j not in taken:
            taken.add(j)
            used_idx.append(j)
            assigned_h.append(h_i)
            residuals.append(res)
            amps.append(clusters[j].amp_sum)

    if not used_idx:
        return HpsCandidate(f0, B, 0.0, np.array([], int), np.array([], int),
                            np.array([], float), 0.0, 0.0)

    A_tot = sum(c.amp_sum for c in clusters) + 1e-12
    coverage = float(np.sum(amps) / A_tot)

    # noyau Cauchy sur résidus (en cents)
    s2 = 25.0 ** 2
    base = np.sum(np.array(amps) / (1.0 + (np.array(residuals) ** 2) / s2))

    H = np.array(assigned_h, int)
    nH = int(np.unique(H).size)
    spanH = (H.max() - H.min()) if H.size else 0
    meanH = float(H.mean()) if H.size else 99.0

    # régularité des hauteurs d'harmoniques (préférence aux suites serrées)
    diffs = np.diff(sorted(np.unique(H)))
    if diffs.size > 0:
        reg_score = 1.0 / (1.0 + np.var(diffs))
        base *= (1.0 + 0.2 * reg_score)

    # pénalités/bonus structurels
    if nH < kmin:
        base *= 0.25
    elif nH == 3:
        base *= 0.85
    base *= (1.0 + 0.08 * max(0, spanH - 1))
    base *= 1.0 / (1.0 + 0.04 * max(0.0, meanH - 2.0))

    # prior tessiture
    base *= _tessitura_prior(f0, tessitura)

    conf = float(min(1.0, 0.55 * coverage + 0.45 * (base / (base + 10.0))))
    return HpsCandidate(
        f0, B, float(base),
        np.array(used_idx, int), np.array(assigned_h, int),
        np.array(residuals, float),
        coverage, conf
    )

# ==== Normalisation PGCD (ramène 2×/4× → 1×) =======================
def _normalize_by_gcd(
    cand: HpsCandidate,
    clusters: List[Cluster],
    hmax: int,
    cents_tol: float,
    tessitura: str
) -> HpsCandidate:
    best = cand
    while True:
        H = best.assigned_h
        g = _gcd_many(H)
        if g <= 1:
            break
        f0_try = best.f0 / g
        alt = score_candidate_on_clusters(
            clusters, f0_try, best.B, hmax=hmax, cents_tol=cents_tol,
            tessitura=tessitura, kmin=3
        )
        if alt.score > best.score * 1.001:
            best = alt
        elif abs(alt.score - best.score) <= 1e-6 and f0_try < best.f0:
            best = alt
            break
        else:
            break
    return best

# ==== Octave guard (si h=2,3 forts sans h=1) ========================
def _octave_guard(cand: HpsCandidate, clusters: List[Cluster], hmax: int,
                  cents_tol: float, tessitura: str, kmin: int) -> HpsCandidate:
    if cand.assigned_h.size == 0:
        return cand
    low_h = cand.assigned_h[cand.assigned_h <= 3]
    ratio_low = float(low_h.size / max(1, cand.assigned_h.size))
    if (1 not in cand.assigned_h) and (ratio_low >= 0.4):
        alt = score_candidate_on_clusters(
            clusters, cand.f0 * 0.5, cand.B,
            hmax=hmax, cents_tol=cents_tol, tessitura=tessitura, kmin=kmin
        )
        if alt.score > cand.score:
            return alt
    return cand

# ==== TESSITURA AUTO =================================================
def _infer_tessitura_from_peaks(peaks: List[Peak]) -> str:
    """
    Heuristique simple:
      - médiane des fréqs de pics forts > ~1200 Hz → "high"
      - < ~120 Hz → "low"
      - sinon → "mid"
    """
    if not peaks:
        return "mid"
    F = np.array([p.f for p in peaks], float)
    A = np.array([p.a for p in peaks], float)
    # pondère par amplitude pour mieux refléter l'énergie dominante
    f_med = float(np.median(np.repeat(F, np.maximum((A / (A.max() + 1e-12) * 10).astype(int), 1))))
    if f_med >= 1200.0:
        return "high"
    if f_med <= 120.0:
        return "low"
    return "mid"

# ==== Main detection ===============================================
def hps_detect_candidates(
    x: np.ndarray,
    fs: float,
    top_k: int = 5,
    cents_bin: float = 35.0,
    cents_tol: float = 30.0,
    hmax: int = 16,
    tessitura: str = "auto"
) -> List[HpsCandidate]:
    peaks = extract_peaks_basic(x, fs, n_fft=16384, k=30, floor_db=-60.0)
    if not peaks:
        return []

    # tessitura auto si demandé
    tz = tessitura
    if tessitura == "auto":
        tz = _infer_tessitura_from_peaks(peaks)

    # paramètres par tessiture
    if tz == "low":
        cents_bin, cents_tol, hmax = 45.0, 38.0, 18
    elif tz == "high":
        cents_bin, cents_tol, hmax = 30.0, 28.0, 14
    else:
        cents_bin, cents_tol, hmax = 35.0, 30.0, 16

    kmin = 2 if tz == "high" else 3  # en aigus, dyade (h=2,4) souvent suffisante

    clusters = cluster_peaks_log(peaks, cents_bin=cents_bin)
    if not clusters:
        return []
   
    # helper -> debug
    print_clusters_as_notes(clusters)
    #    Exemple pour tester A5 ≈ 880–890 Hz (ajuste B si tu veux)
    check_harmonic_consistency(clusters, f0=882.0, B=0.0, hmax=16, cents_tol=30.0)

    seeds = seed_from_clusters(
        clusters,
        top_clusters=6,
        h_small=(1, 4) if tz == "high" else (1, 6)
    )
    if seeds.size == 0:
        return []

    # grille de B (inharmonicité)
    B_grid = np.array([0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])

    # 1) coarse
    coarse: List[HpsCandidate] = []
    for f0 in seeds:
        for B in B_grid:
            c = score_candidate_on_clusters(
                clusters, f0, B, hmax=hmax, cents_tol=cents_tol, tessitura=tz, kmin=kmin
            )
            if c.score > 0:
                coarse.append(c)

    if not coarse:
        return []

    coarse.sort(key=lambda c: (c.score, c.coverage, -c.f0), reverse=True)
    coarse = coarse[:32]  # beam

    # 2) fine: petit balayage f0 ±2 %
    fine: List[HpsCandidate] = []
    for c in coarse:
        f0_grid = c.f0 * (1.0 + np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]))
        best = c
        for f0_try in f0_grid:
            cand = score_candidate_on_clusters(
                clusters, f0_try, c.B, hmax=hmax, cents_tol=cents_tol, tessitura=tz, kmin=kmin
            )
            # tie-break → plus petit f0 si même score
            if (cand.score > best.score) or (abs(cand.score - best.score) <= 1e-6 and cand.f0 < best.f0):
                best = cand
        best = _octave_guard(best, clusters, hmax, cents_tol, tz, kmin)
        best = _normalize_by_gcd(best, clusters, hmax, cents_tol, tz)
        fine.append(best)

    # 3) merge quasi-duplicats
    fine.sort(key=lambda c: (c.score, c.coverage, -c.f0), reverse=True)
    merged: List[HpsCandidate] = []
    for c in fine:
        if not merged:
            merged.append(c)
            continue
        last = merged[-1]
        cents_diff = abs(_cents_ratio(c.f0 / last.f0))
        sameB = abs(c.B - last.B) <= 1e-4
        if cents_diff < 5.0 and sameB:
            if (c.score > last.score) or (abs(c.score - last.score) <= 1e-6 and c.f0 < last.f0):
                merged[-1] = c
        else:
            merged.append(c)

    return merged[:top_k]

# ==== API principale ===============================================
def hps_multi_detect(
    x: np.ndarray, fs: float, top_k: int = 5, tessitura: str = "auto"
) -> Dict[str, Any]:
    # NOTE: tessitura calcule ses tolérances en interne si "auto"
    cands = hps_detect_candidates(x, fs, top_k=top_k, tessitura=tessitura)
    out: List[Dict[str, Any]] = []
    for c in cands:
        out.append(dict(
            f0=float(c.f0),
            B=float(c.B),
            score=float(c.score),
            coverage=float(c.coverage),
            conf=float(c.conf),
            used_idx=c.used_idx.tolist(),
            assigned_h=c.assigned_h.tolist(),
            residuals_cents=c.residuals.tolist(),
            note=_note_from_freq(c.f0),
        ))
    return {"candidates": out}

# ==== Wrapper PyTune-compatible ====================================
def estimate_f0_hps_multi_wrapper(
    x: np.ndarray, fs: float, top_k: int = 5, tessitura: str = "auto"
) -> Dict[str, Any]:
    try:
        return hps_multi_detect(x, fs, top_k=top_k, tessitura=tessitura)
    except Exception as e:
        if HPS_DEBUG:
            print("[HPS multi] error:", e)
        return {"candidates": []}

# ==== Demo rapide ===================================================
if __name__ == "__main__":
    # Démo: A5 (880–890 Hz), fondamentale absente, h=2..6
    fs = 44100
    dur = 1.0
    t = np.arange(int(fs * dur)) / fs
    f0_true = 880.0  # A5

    x = np.zeros_like(t)
    amps = [0.0, 1.0, 0.7, 0.45, 0.3, 0.2]  # h1..h6 (h1=0 → fondamentale masquée)
    for h, a in enumerate(amps, start=1):
        if a > 0:
            x += a * np.sin(2 * np.pi * f0_true * h * t)
    x += 0.01 * np.random.randn(len(t))

    res = hps_multi_detect(x, fs, top_k=5, tessitura="auto")
    print("== Demo ==")
    for i, c in enumerate(res["candidates"]):
        print(f"#{i+1}: f0={c['f0']:.2f} Hz ({c['note']})  B={c['B']:.1e}  "
              f"score={c['score']:.1f}  cov={c['coverage']:.2f}  conf={c['conf']:.2f}  "
              f"h_used={c['assigned_h']}")