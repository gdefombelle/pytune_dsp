# pytune_dsp/analysis/yin_partitioned.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple
import math
import numpy as np

# ──────────────────────────────────────────────
# DEBUG
# ──────────────────────────────────────────────

DEBUG_YIN_PART = True
EARLY_STOP_DEBUG = True

def log(*args):
    if DEBUG_YIN_PART:
        print("[YIN-PART]", *args)
def elog(*args):
    if EARLY_STOP_DEBUG:
        print("[YIN-PART][EARLY]", *args)

# ──────────────────────────────────────────────
# TYPES & STRUCTURES
# ──────────────────────────────────────────────

PartitionMode = Literal[
    "octaves_8",
    "octaves_4x2",
    "cents_250",   # bandes EXP-YIN centrées, 300c
    "binary_tree",
]

@dataclass
class Band:
    fmin: float
    fmax: float

@dataclass
class BandResult:
    band: Band
    f0: Optional[float]
    score: float

# Backend signature
YinBackend = Callable[
    [np.ndarray, int, float, float],
    Tuple[Optional[float], float],
]

A0_HZ = 27.5
C8_HZ = 4186.01
MIDI_A0 = 21
MIDI_C8 = 108

WINDOW_CENTS_FOR_PARTITION = 300.0   # même fenêtre que EXP-YIN


# ──────────────────────────────────────────────
# OUTILS
# ──────────────────────────────────────────────

def _window_to_bounds(center_freq: float, window_cents: float, sr: int):
    if center_freq <= 0:
        return 20.0, min(20000.0, sr * 0.45)

    total = min(abs(window_cents), 600.0)  # max ±1 octave
    ratio = 2.0 ** (total / 1200.0)

    fmin = max(20.0, center_freq / ratio)
    fmax = min(center_freq * ratio, sr * 0.45)

    if fmax <= fmin:
        fmax = min(sr * 0.45, fmin * 1.5)
    return float(fmin), float(fmax)

def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ──────────────────────────────────────────────
# BANDES
# ──────────────────────────────────────────────

def build_octave_bands_8(fmin=A0_HZ, fmax=C8_HZ):
    bands = []
    cur = fmin
    while cur < fmax and len(bands) < 8:
        nxt = min(cur * 2.0, fmax)
        bands.append(Band(cur, nxt))
        cur = nxt
    return bands

def build_octave_bands_4x2(fmin=A0_HZ, fmax=C8_HZ):
    bands=[]
    cur=fmin
    while cur<fmax and len(bands)<4:
        nxt=min(cur*4.0, fmax)
        bands.append(Band(cur,nxt))
        cur=nxt
    return bands

def build_note_centered_bands_cents(sr: int, window_cents=WINDOW_CENTS_FOR_PARTITION):
    bands=[]
    for midi in range(MIDI_A0, MIDI_C8+1):
        center=_midi_to_freq(midi)
        fmin,fmax=_window_to_bounds(center, window_cents, sr)
        bands.append(Band(fmin,fmax))
    return bands

def build_initial_bands(mode: PartitionMode, sr: int, fmin=A0_HZ, fmax=C8_HZ):
    if mode=="octaves_8":
        return build_octave_bands_8(fmin,fmax)
    if mode=="octaves_4x2":
        return build_octave_bands_4x2(fmin,fmax)
    if mode=="cents_250":
        return build_note_centered_bands_cents(sr)
    if mode=="binary_tree":
        return [Band(fmin,fmax)]
    raise ValueError("Unknown mode "+mode)


# ──────────────────────────────────────────────
# VOTE / BAND PARTITION
# ──────────────────────────────────────────────

def yin_partition_vote(
    signal: np.ndarray,
    sr: int,
    mode: PartitionMode,
    yin_backend: YinBackend,
    focus_window: Optional[Tuple[float,float]]=None,
) -> Tuple[Optional[float], List[BandResult]]:

    bands = build_initial_bands(mode, sr)
    log(f"Mode={mode}, evaluating {len(bands)} bands")

    results: List[BandResult] = []

    # ────────────────────────────────────────────
    # PARAMÈTRES EARLY-STOP
    # ────────────────────────────────────────────
    PEAK_ENTER = 0.85          # score mini pour commencer la surveillance
    DROP_RATIO = 0.40          # chute ≥40% → stop immédiat
    DROP_ABS = 0.15            # ou chute absolue d’au moins -0.15
    PEAK = 0.0                 # meilleur score vu jusque là
    peak_band_idx = -1

    # ────────────────────────────────────────────
    # BOUCLE PRINCIPALE
    # ────────────────────────────────────────────
    for idx, band in enumerate(bands):
        f0, score = yin_backend(signal, sr, band.fmin, band.fmax)
        results.append(BandResult(band,f0,score))
        log(f"  Band [{band.fmin:.1f}-{band.fmax:.1f}] → f0={f0} score={score:.3f}")

        # ——— EARLY-STOP LOGIC ————————————————

        # On ne surveille qu'à partir d'un vrai signal exploitable (>0.85)
        if score >= PEAK_ENTER:
            if score > PEAK:
                PEAK = score
                peak_band_idx = idx
                elog(f"New PEAK {PEAK:.3f} at band {idx}")

        # Dès qu’on a un pic, on surveille la chute
        if PEAK > 0 and score < PEAK:
            rel_drop = (PEAK - score) / PEAK
            abs_drop = PEAK - score

            if (rel_drop >= DROP_RATIO) or (abs_drop >= DROP_ABS):
                elog(
                    f"EARLY-STOP at band {idx}: "
                    f"peak={PEAK:.3f} → now={score:.3f}, "
                    f"rel_drop={rel_drop:.2f}, abs_drop={abs_drop:.2f}"
                )
                break

    # ───────────────────────────────────────────
    # FILTRAGE DES CANDIDATS VALIDES
    # ───────────────────────────────────────────
    candidates = [r for r in results if r.f0 is not None and r.score > 0]
    if not candidates:
        log("No valid band")
        return None, results

    # global best score
    best_score = max(candidates, key=lambda r:r.score)
    smax = best_score.score

    # ───────────────────────────────────────────
    # FILTRE DE COHÉRENCE DE BANDE
    # ───────────────────────────────────────────
    if focus_window is not None:
        fw_min,fw_max=focus_window
        strong=[r for r in candidates if r.score>=0.90]
        if len(strong)>=2:
            in_focus=[r for r in strong if fw_min<=r.f0<=fw_max]
            if in_focus:
                chosen=max(in_focus, key=lambda r:(r.f0,r.score))
                log(f"Winner (band-coherent focus) = f0={chosen.f0:.2f}")
                return chosen.f0, results

    # ───────────────────────────────────────────
    # LOGIQUE HISTORIQUE
    # ───────────────────────────────────────────
    if smax>=0.75:
        log(f"Winner (high confidence) = {best_score.f0:.2f}")
        return best_score.f0, results

    # fallback low-f0
    f_ref=best_score.f0
    filt=[r for r in candidates if r.f0<=2.0*f_ref] or candidates
    best_low=min(filt, key=lambda r:r.f0)
    log(f"Winner (fallback low-f0) = {best_low.f0:.2f}")
    return best_low.f0, results


# ──────────────────────────────────────────────
# BINARY TREE (inchangé)
# ──────────────────────────────────────────────

def split_band(b:Band): mid=math.sqrt(b.fmin*b.fmax); return Band(b.fmin,mid),Band(mid,b.fmax)
def band_width_oct(b:Band): return math.log2(b.fmax/b.fmin)

def yin_partition_binary_tree(
    signal,sr,yin_backend,fmin=A0_HZ,fmax=C8_HZ,target_width_octaves=1.0,max_depth=5
):
    current=Band(fmin,fmax)
    history=[]
    for depth in range(max_depth+1):
        width=band_width_oct(current)
        log(f"[Depth {depth}] band={current} width={width:.2f} oc")
        if depth>0 and width<=target_width_octaves: break
        b1,b2=split_band(current)
        f1,s1=yin_backend(signal,sr,b1.fmin,b1.fmax)
        f2,s2=yin_backend(signal,sr,b2.fmin,b2.fmax)
        r1,r2=BandResult(b1,f1,s1),BandResult(b2,f2,s2)
        history+=[r1,r2]
        valid=[r for r in (r1,r2) if r.f0 and r.score>0]
        if not valid: return None,history
        best=max(valid,key=lambda r:r.score)
        current=best.band
    f0,s=yin_backend(signal,sr,current.fmin,current.fmax)
    history.append(BandResult(current,f0,s))
    log(f"FINAL F0={f0}")
    return f0,history


# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────

def detect_f0_seed_partitioned(
    signal,sr,mode:PartitionMode="octaves_8",yin_backend=None,
    target_width_octaves=1.0,max_depth=5,focus_window=None,
):
    if yin_backend is None:
        raise ValueError("yin_backend must be provided")

    if mode in ("octaves_8","octaves_4x2","cents_250"):
        return yin_partition_vote(signal,sr,mode,yin_backend,focus_window)

    if mode=="binary_tree":
        return yin_partition_binary_tree(
            signal,sr,yin_backend,
            fmin=A0_HZ,fmax=C8_HZ,
            target_width_octaves=target_width_octaves,
            max_depth=max_depth,
        )

    raise ValueError("Unknown mode "+mode)