# hps_seq.py
# -----------------------------------------------------------------------------
# HPS séquentiel (harmonic-sequence) — rapide & simple
# Objectif : trouver f0 en privilégiant des suites harmoniques plausibles
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy.signal import get_window
from bisect import bisect_left


# ===== Analysis-only spectral tilt (FFT-domain) ==============================
def apply_spectral_tilt(mag: np.ndarray, freqs: np.ndarray,
                        pivot: float = 2000.0, slope_db_per_oct: float = 6.0):
    """Emphase haute fréquence sur |FFT| (gain +slope_db_per_oct par octave > pivot)."""
    if mag.size == 0 or freqs.size == 0:
        return mag
    w = np.maximum(freqs, 1.0)
    ratio = w / float(pivot)
    with np.errstate(divide='ignore', invalid='ignore'):
        tilt_db = slope_db_per_oct * np.log2(ratio)
    tilt_db = np.where(ratio > 1.0, tilt_db, 0.0)
    tilt_db = np.clip(tilt_db, 0.0, 12.0)
    gain = 10.0 ** (tilt_db / 20.0)
    return mag * gain


# ===== Tunables ==============================================================
HPS_PEAKS_K    = int(os.getenv("HPS_PEAKS_K",    "30"))
HPS_NFFT       = int(os.getenv("HPS_NFFT",       "16384"))
HPS_FLOOR_DB   = float(os.getenv("HPS_FLOOR_DB", "-60"))
HPS_HIGH_TOPF  = float(os.getenv("HPS_HIGH_TOPF", "1000"))
HPS_TOL_LOW    = float(os.getenv("HPS_TOL_LOW",  "36"))
HPS_TOL_MID    = float(os.getenv("HPS_TOL_MID",  "32"))
HPS_TOL_HIGH   = float(os.getenv("HPS_TOL_HIGH", "28"))
HPS_HMAX_LOW   = int(os.getenv("HPS_HMAX_LOW",  "18"))
HPS_HMAX_MID   = int(os.getenv("HPS_HMAX_MID",  "16"))
HPS_HMAX_HIGH  = int(os.getenv("HPS_HMAX_HIGH", "14"))
HPS_HSEED_HIGH = int(os.getenv("HPS_HSEED_HIGH","6"))
HPS_DEBUG = bool(int(os.getenv("HPS_DEBUG", "0")))
def _dbg(*a): 
    if HPS_DEBUG: print(*a)


# ===== Structures =============================================================
@dataclass
class Peak:
    f: float
    a: float
    db: float

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
    seq_len: int


# ===== Notes utilitaires ======================================================
_A440 = 440.0
_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def _note_from_freq(freq: float) -> str:
    if freq <= 0: return "?"
    midi = 69 + 12 * np.log2(freq / _A440)
    i = int(round(midi)) % 12
    o = int((round(midi) // 12) - 1)
    return f"{_NAMES[i]}{o}"
def _cents_ratio(r: float) -> float:
    return 1200.0 * np.log2(max(r, 1e-12))


# ===== FFT -> Peaks ===========================================================
def extract_peaks_basic(x: np.ndarray, fs: float,
                        n_fft: int = 16384, k: int = 30,
                        floor_db: float = -60.0) -> List[Peak]:
    """Extraction des pics FFT (avec biais aigus si besoin)."""
    HIGH_HINT       = bool(int(os.getenv("HPS_HIGH_HINT", "1")))
    HIGH_FREQ_GATE  = float(os.getenv("HPS_HIGH_FREQ_GATE", "2000.0"))
    HIGH_FLOOR_DB   = float(os.getenv("HPS_HIGH_FLOOR_DB", "-85.0"))
    HARM_MAX        = int(os.getenv("HPS_HIGH_HARM_MAX", "2"))
    HARM_WIN        = int(os.getenv("HPS_HIGH_HARM_WIN", "3"))

    y = x.astype(np.float64, copy=False)
    if y.size == 0: return []
    y -= y.mean()
    y *= get_window("hann", len(y), fftbins=True)
    spec = np.fft.rfft(y, n_fft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, 1.0/fs)

    # spectral tilt auto si topF > 1kHz
    topF = float(freqs[np.argmax(np.abs(spec))])
    if topF > HPS_HIGH_TOPF:
        mag = apply_spectral_tilt(mag, freqs, pivot=2000.0, slope_db_per_oct=6.0)

    if freqs.size <= 2: return []
    lo, hi = 1, len(freqs)-1
    mag, freqs = mag[lo:hi], freqs[lo:hi]
    mag_db = 20*np.log10(np.maximum(mag, 1e-12))
    if mag_db.size == 0: return []

    jmax = int(np.argmax(mag_db))
    f_top, mmax = float(freqs[jmax]), float(mag_db[jmax])
    idx = np.where(mag_db >= (mmax + floor_db))[0]
    if idx.size == 0: return []
    order = np.argsort(mag[idx])[::-1]
    base_sel = idx[order]

    extras = []
    if HIGH_HINT and f_top >= HIGH_FREQ_GATE:
        nyq = fs * 0.5
        for h in range(2, HARM_MAX+1):
            ft = h * f_top
            if ft >= nyq: break
            j = int(np.argmin(np.abs(freqs - ft)))
            j0, j1 = max(0,j-HARM_WIN), min(len(freqs), j+HARM_WIN+1)
            jloc = j0 + int(np.argmax(mag_db[j0:j1]))
            if mag_db[jloc] >= (mmax + HIGH_FLOOR_DB):
                extras.append(jloc)

    seen, sel_list = set(), []
    for j in extras + base_sel.tolist():
        if len(sel_list) >= k: break
        if j not in seen:
            sel_list.append(j); seen.add(j)
    if len(sel_list) < k:
        for j in np.argsort(mag)[::-1]:
            if len(sel_list) >= k: break
            if j not in seen:
                sel_list.append(j); seen.add(j)
    sel = np.array(sel_list[:k], int)
    peaks = [Peak(float(freqs[i]), float(mag[i]), float(mag_db[i])) for i in sel]
    peaks.sort(key=lambda p: p.f)
    return peaks


# ===== Clustering =============================================================
def cluster_peaks_log(peaks: List[Peak], cents_bin: float = 32.0) -> List[Cluster]:
    if not peaks: return []
    F = np.array([p.f for p in peaks], float)
    A = np.array([p.a for p in peaks], float)
    order = np.argsort(F)
    F, A = F[order], A[order]
    clusters: List[Cluster] = []
    cur_idx = [0]; cur_center = F[0]
    for i in range(1, len(F)):
        if abs(_cents_ratio(F[i]/cur_center)) <= cents_bin:
            cur_idx.append(i)
            cur_center = np.average(F[cur_idx], weights=A[cur_idx])
        else:
            fvals, avals = F[cur_idx], A[cur_idx]
            clusters.append(Cluster(float(np.average(fvals, weights=avals)),
                                    float(np.median(fvals)),
                                    float(avals.sum()), float(avals.max()), len(cur_idx)))
            cur_idx=[i]; cur_center=F[i]
    fvals,avals=F[cur_idx],A[cur_idx]
    clusters.append(Cluster(float(np.average(fvals,weights=avals)),
                            float(np.median(fvals)),float(avals.sum()),
                            float(avals.max()),len(cur_idx)))
    clusters.sort(key=lambda c: c.f_mean)
    return clusters


def _nearest_cluster(clusters: List[Cluster], f_target: float) -> Tuple[int, float]:
    if not clusters: return (-1, 1e9)
    Fs = [c.f_mean for c in clusters]
    j = bisect_left(Fs, f_target)
    cand = []
    if j < len(Fs): cand.append(j)
    if j-1 >= 0: cand.append(j-1)
    best = min(cand, key=lambda k: abs(_cents_ratio(f_target / Fs[k])))
    return best, float(_cents_ratio(f_target / Fs[best]))


# ===== Harmonic model =========================================================
def _fh(f0: float, h: np.ndarray, B: float) -> np.ndarray:
    return f0 * h * np.sqrt(1.0 + B*(h**2))


# ===== Séquence harmonique contiguë ==========================================
def _longest_consecutive(hs: np.ndarray) -> Tuple[int, Tuple[int,int]]:
    if hs.size == 0: return 0,(0,0)
    s = sorted(set(int(h) for h in hs))
    best_len,best_seg=1,(s[0],s[0])
    cur_len,cur_start=1,s[0]
    for i in range(1,len(s)):
        if s[i]==s[i-1]+1: cur_len+=1
        else:
            if cur_len>best_len: best_len,best_seg=cur_len,(cur_start,s[i-1])
            cur_len,cur_start=1,s[i]
    if cur_len>best_len: best_len,best_seg=cur_len,(cur_start,s[-1])
    return best_len,best_seg


# ===== Tessiture heuristique ==================================================
def _resolve_tessitura(tessitura: str, clusters: List[Cluster]) -> Tuple[float, int]:
    if tessitura=="auto":
        topF=max((c.f_mean for c in clusters),default=440.0)
        if topF>=HPS_HIGH_TOPF: tessitura="high"
        elif topF<=180.0: tessitura="low"
        else: tessitura="mid"
    if tessitura=="low": return HPS_TOL_LOW,HPS_HMAX_LOW
    if tessitura=="high": return HPS_TOL_HIGH,HPS_HMAX_HIGH
    return HPS_TOL_MID,HPS_HMAX_MID


# ===== Candidats f0 ===========================================================
def _generate_candidates_from_clusters(clusters: List[Cluster],
                                       top_clusters:int=6,
                                       h_seed_max:int=6)->np.ndarray:
    if not clusters: return np.array([])
    top=sorted(clusters,key=lambda c:c.amp_sum,reverse=True)[:top_clusters]
    seeds=[]
    for c in top:
        for h in range(1,h_seed_max+1):
            f0=c.f_mean/h
            if 15.0<=f0<=4200.0: seeds.append(f0)
    if not seeds: return np.array([])
    return np.unique(np.round(np.array(seeds,float),4))


# ===== Matching & scoring =====================================================
def _assign_and_score(clusters: List[Cluster], f0: float, B: float,
                      cents_tol: float, hmax: int,
                      expected_f0: Optional[float]=None,
                      expected_weight: float=0.15) -> HpsCandidate:
    H=np.arange(1,hmax+1,dtype =float)
    targets=_fh(f0,H,B)
    used_idx,assigned_h,residuals,amps=[],[],[],[]
    taken=set()
    for i,ft in enumerate(targets,start=1):
        j,res=_nearest_cluster(clusters,ft)
        if j>=0 and abs(res)<=cents_tol and j not in taken:
            taken.add(j); used_idx.append(j); assigned_h.append(i)
            residuals.append(res); amps.append(clusters[j].amp_sum)
    if not used_idx:
        return HpsCandidate(f0,B,0.0,np.array([],int),np.array([],int),
                            np.array([],float),0.0,0.0,0)

    A_tot=sum(c.amp_sum for c in clusters)+1e-12
    coverage=float(np.sum(amps)/A_tot)
    s2=25.0**2
    base=float(np.sum(np.array(amps)/(1.0+(np.array(residuals)**2)/s2)))
    H_used=np.array(assigned_h,int)
    seq_len,(h_start,h_end)=_longest_consecutive(H_used)
    meanH=float(H_used.mean()) if H_used.size else 99.0
    has_h1=1 in H_used
    base*=(1.0+0.60*max(0,seq_len-1))
    base*=1.0/(1.0+0.05*max(0.0,meanH-2.0))
    if not has_h1 and h_start>2: base*=0.70

    # --- expected_f0 bonus contextuel ---
    if expected_f0 and expected_weight>0:
        diff_cents=abs(_cents_ratio(f0/expected_f0))
        bonus=math.exp(-0.5*(diff_cents/80.0)**2)
        base*=(1.0+expected_weight*bonus)

    conf=float(min(1.0,0.55*coverage+0.45*(base/(base+10.0))))
    return HpsCandidate(float(f0),float(B),float(base),
                        np.array(used_idx,int),H_used,
                        np.array(residuals,float),
                        coverage,conf,int(seq_len))


# ===== Sélection principale ===================================================
def _select_candidates(clusters: List[Cluster], tessitura: str="auto",
                       top_k:int=5, top_clusters:int=6,
                       h_seed_max:int=6, B_grid:Tuple[float,...]=(0.0,1e-4,3e-4),
                       expected_f0:float|None=None,
                       expected_weight:float=0.15)->List[HpsCandidate]:
    if not clusters: return []
    cents_tol,hmax=_resolve_tessitura(tessitura,clusters)
    seeds=_generate_candidates_from_clusters(clusters,top_clusters, h_seed_max)
    if seeds.size==0: return []
    cands=[]
    for f0 in seeds:
        for B in B_grid:
            c=_assign_and_score(clusters,f0,B,cents_tol,hmax,
                                expected_f0=expected_f0,
                                expected_weight=expected_weight)
            if c.score>0: cands.append(c)
    if not cands: return []
    cands.sort(key=lambda c:(c.seq_len,c.score,c.coverage,(1 in c.assigned_h),-c.f0),reverse=True)
    merged=[]
    for c in cands:
        if not merged: merged.append(c); continue
        last=merged[-1]
        cents_diff=abs(_cents_ratio(c.f0/last.f0))
        same_B=abs(c.B-last.B)<=1e-4
        if cents_diff<5.0 and same_B:
            if (c.seq_len,c.score,c.coverage)>(last.seq_len,last.score,last.coverage):
                merged[-1]=c
        else: merged.append(c)
    return merged[:top_k]


# ===== API publique ===========================================================
def hps_seq_detect(x: np.ndarray, fs: float, top_k:int=5,
                   tessitura:str="auto",
                   expected_f0:float|None=None,
                   expected_weight:float=0.15) -> Dict[str,Any]:
    peaks=extract_peaks_basic(x,fs,n_fft=HPS_NFFT,k=HPS_PEAKS_K,floor_db=HPS_FLOOR_DB)
    if not peaks: return {"candidates":[]}
    clusters=cluster_peaks_log(peaks,cents_bin=32.0)
    if HPS_DEBUG:
        _dbg("\nClusters (Hz / amp_sum / n):")
        for i,c in enumerate(clusters):
            _dbg(f"  {i:2d} | {c.f_mean:9.3f} | {c.amp_sum:9.3f} | {c.n:2d} | {_note_from_freq(c.f_mean)}")
    cands=_select_candidates(clusters,tessitura,top_k,
                             expected_f0=expected_f0,
                             expected_weight=expected_weight)
    out=[]
    for c in cands:
        out.append(dict(f0=c.f0,B=c.B,score=c.score,
                        coverage=c.coverage,conf=c.conf,
                        used_idx=c.used_idx.tolist(),
                        assigned_h=c.assigned_h.tolist(),
                        residuals_cents=c.residuals.tolist(),
                        seq_len=c.seq_len,
                        note=_note_from_freq(c.f0)))
    return {"candidates":out}


# ===== Wrappers PyTune =======================================================
def estimate_f0_hps_multi_wrapper(x: np.ndarray, fs: float,
                                  top_k:int=5, tessitura:str="auto",
                                  expected_f0:float|None=None)->Dict[str,Any]:
    try:
        return hps_seq_detect(x,fs,top_k,tessitura,expected_f0=expected_f0)
    except Exception as e:
        if HPS_DEBUG: _dbg("[HPS-SEQ multi] error:",e)
        return {"candidates":[]}


def estimate_f0_hps_wrapper(x: np.ndarray,
                            fs: float,
                            tessitura: str = "auto",
                            expected_f0: float | None = None,
                            expected_weight: float = 0.15) -> Dict[str, Any]:
    """
    Wrapper 'single' pour compat PyTune (renvoie un seul f0 + qualité).
    Clés: f0, B, quality, deltas (cents), partials_hz (clusters appariés).
    """
    try:
        res = hps_seq_detect(
            x, fs, top_k=1, tessitura=tessitura,
            expected_f0=expected_f0, expected_weight=expected_weight
        )
        if not res["candidates"]:
            return dict(
                f0=0.0, B=0.0, quality=0.0,
                deltas=np.array([]), partials_hz=np.array([])
            )
        c = res["candidates"][0]
        # reconstruit les partiels à partir des h appariés
        H = np.array(c["assigned_h"], int)
        f_partials = c["f0"] * H * np.sqrt(1.0 + c["B"] * (H**2))
        return dict(
            f0=float(c["f0"]),
            B=float(c["B"]),
            quality=float(c["conf"]),
            deltas=np.array(c["residuals_cents"], float),
            partials_hz=f_partials.astype(float)
        )
    except Exception as e:
        if HPS_DEBUG: _dbg("[HPS-SEQ single] error:", e)
        return dict(
            f0=0.0, B=0.0, quality=0.0,
            deltas=np.array([]), partials_hz=np.array([])
        )


# ===== Demo ===================================================================
if __name__ == "__main__":
    # Démo : fondamentale masquée (A5 ~ 880 Hz) + harmoniques 2..6
    fs = 44100
    dur = 1.0
    t = np.arange(int(fs * dur)) / fs
    f0_true = 880.0

    x = np.zeros_like(t)
    for i, a in enumerate([0.0, 0.9, 0.6, 0.45, 0.30, 0.20], start=1):
        # i = 1..6 -> h = i
        if i == 1:   # fondamentale absente
            continue
        x += a * np.sin(2 * np.pi * f0_true * i * t)
    x += 0.01 * np.random.randn(len(t))

    os.environ["HPS_DEBUG"] = "1"
    # Optionnel: test avec un expected_f0 doux
    expected = 880.0

    res = hps_seq_detect(
        x, fs, top_k=5, tessitura="auto",
        expected_f0=expected, expected_weight=0.15
    )
    print("\n== HPS-SEQ Demo ==")
    for i, c in enumerate(res["candidates"]):
        print(
            f"#{i+1}: f0={c['f0']:.2f} Hz ({_note_from_freq(c['f0'])})  "
            f"B={c['B']:.1e}  score={c['score']:.2f}  "
            f"cov={c['coverage']:.2f}  conf={c['conf']:.2f}  "
            f"seq={c['assigned_h']}"
        )