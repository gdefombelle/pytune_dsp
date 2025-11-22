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


def log(*args):
    if DEBUG_YIN_PART:
        print("[YIN-PART]", *args)


# ──────────────────────────────────────────────
# TYPES & STRUCTURES
# ──────────────────────────────────────────────

PartitionMode = Literal[
    "octaves_8",
    "octaves_4x2",
    "cents_250",   # NOTE: on garde le nom pour compatibilité,
                   # mais la fenêtre est en fait de 300c comme EXP-YIN.
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


# Backend signature : (signal, sr, fmin, fmax) → (f0, score)
YinBackend = Callable[
    [np.ndarray, int, float, float],
    Tuple[Optional[float], float],
]

A0_HZ = 27.5
C8_HZ = 4186.01

MIDI_A0 = 21
MIDI_C8 = 108

# fenêtre utilisée pour les bandes centrées, pour coller à EXP-YIN
WINDOW_CENTS_FOR_PARTITION = 300.0


# ──────────────────────────────────────────────
# OUTILS FRÉQUENCE / FENÊTRE (copié de pitch_detection_expected)
# ──────────────────────────────────────────────

def _window_to_bounds(center_freq: float, window_cents: float, sr: int) -> tuple[float, float]:
    """
    Construit [fmin, fmax] autour de center_freq, borné à ±1 octave max,
    et limité par le Nyquist (≈ 0.45 * sr), comme dans pitch_detection_expected.
    """
    if center_freq <= 0.0:
        # fallback simple (ne devrait pas arriver pour les notes piano)
        return 20.0, min(20000.0, sr * 0.45)

    total = min(abs(window_cents), 600.0)  # 600 cents = 1 octave max
    ratio = 2.0 ** (total / 1200.0)

    fmin = center_freq / ratio
    fmax = center_freq * ratio

    fmin = max(20.0, fmin)
    fmax = min(fmax, sr * 0.45)

    if fmax <= fmin:
        fmax = min(sr * 0.45, fmin * 1.5)

    return float(fmin), float(fmax)


def _midi_to_freq(midi: int) -> float:
    """440 * 2^((midi-69)/12)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ──────────────────────────────────────────────
# BAND GENERATION
# ──────────────────────────────────────────────

def build_octave_bands_8(fmin=A0_HZ, fmax=C8_HZ) -> List[Band]:
    """8 bandes, 1 octave chacune."""
    bands: List[Band] = []
    cur = fmin
    while cur < fmax and len(bands) < 8:
        nxt = min(cur * 2.0, fmax)
        bands.append(Band(cur, nxt))
        cur = nxt
    return bands


def build_octave_bands_4x2(fmin=A0_HZ, fmax=C8_HZ) -> List[Band]:
    """4 bandes, chacune couvrant 2 octaves."""
    bands: List[Band] = []
    cur = fmin
    while cur < fmax and len(bands) < 4:
        nxt = min(cur * 4.0, fmax)
        bands.append(Band(cur, nxt))
        cur = nxt
    return bands


def build_note_centered_bands_cents(
    sr: int,
    window_cents: float = WINDOW_CENTS_FOR_PARTITION,
    midi_min: int = MIDI_A0,
    midi_max: int = MIDI_C8,
) -> List[Band]:
    """
    Bandes centrées sur chaque note du piano (A0..C8),
    avec la même logique de fenêtre que EXP-YIN :
    [fmin, fmax] = _window_to_bounds(center_freq, window_cents, sr).

    → Ex: pour A0 (27.5 Hz, window=300c) :
       band ≈ [23.1, 32.7] Hz, exactement comme dans les logs [EXP-YIN].
    """
    bands: List[Band] = []

    for midi in range(midi_min, midi_max + 1):
        center = _midi_to_freq(midi)
        fmin, fmax = _window_to_bounds(center, window_cents, sr)
        bands.append(Band(fmin, fmax))

    return bands


def build_initial_bands(
    mode: PartitionMode,
    sr: int,
    fmin: float = A0_HZ,
    fmax: float = C8_HZ,
) -> List[Band]:
    """
    Fabrique la liste des bandes initiales en fonction du mode.

    NOTE: on passe `sr` pour pouvoir construire des bandes centrées
    avec la même logique que EXP-YIN (clamp Nyquist, ±1 octave, etc.).
    """
    if mode == "octaves_8":
        return build_octave_bands_8(fmin, fmax)

    if mode == "octaves_4x2":
        return build_octave_bands_4x2(fmin, fmax)

    if mode == "cents_250":
        # Option B : on interprète ce mode comme
        # "bandes serrées centrées sur les notes du piano"
        # avec la même fenêtre que EXP-YIN (300 cents par défaut).
        return build_note_centered_bands_cents(
            sr=sr,
            window_cents=WINDOW_CENTS_FOR_PARTITION,
            midi_min=MIDI_A0,
            midi_max=MIDI_C8,
        )

    if mode == "binary_tree":
        return [Band(fmin, fmax)]

    raise ValueError(f"Unknown mode {mode}")


# ──────────────────────────────────────────────
# 1) STRATÉGIE : VOTE SUR BANDES (MODE OCTAVES / CENTS)
# ──────────────────────────────────────────────

def yin_partition_vote(
    signal: np.ndarray,
    sr: int,
    mode: PartitionMode,
    yin_backend: YinBackend,
) -> Tuple[Optional[float], List[BandResult]]:

    bands = build_initial_bands(mode, sr=sr)
    log(f"Mode={mode}, evaluating {len(bands)} bands")

    results: List[BandResult] = []

    for band in bands:
        f0, score = yin_backend(signal, sr, band.fmin, band.fmax)
        log(f"  Band [{band.fmin:.1f}–{band.fmax:.1f}] → f0={f0} score={score:.3f}")
        results.append(BandResult(band=band, f0=f0, score=score))

    # On garde uniquement les bandes valides
    candidates = [r for r in results if r.f0 is not None and r.score > 0.0]
    if not candidates:
        log("No valid band")
        return None, results

    # A) bande avec meilleur score
    best_score = max(candidates, key=lambda r: r.score)
    smax = best_score.score

    # --- Critère principal : bonne confiance ---
    if smax >= 0.75:
        log(
            "Winner (high confidence) = "
            f"[{best_score.band.fmin:.1f}–{best_score.band.fmax:.1f}] "
            f"f0={best_score.f0:.2f}"
        )
        return best_score.f0, results

    # --- Sinon : stratégie basse fréquence contrôlée (anti-harmonique) ---
    f_ref = best_score.f0
    filt = [r for r in candidates if r.f0 <= 2.0 * f_ref]

    if not filt:
        filt = candidates

    best_low = min(filt, key=lambda r: r.f0)
    log(
        "Winner (fallback low-f0) = "
        f"[{best_low.band.fmin:.1f}–{best_low.band.fmax:.1f}] "
        f"f0={best_low.f0:.2f}"
    )

    return best_low.f0, results


# ──────────────────────────────────────────────
# 2) STRATÉGIE : BINARY TREE (inchangé)
# ──────────────────────────────────────────────

def split_band(band: Band) -> Tuple[Band, Band]:
    mid = math.sqrt(band.fmin * band.fmax)
    return Band(band.fmin, mid), Band(mid, band.fmax)


def band_width_oct(band: Band) -> float:
    return math.log2(band.fmax / band.fmin)


def yin_partition_binary_tree(
    signal: np.ndarray,
    sr: int,
    yin_backend: YinBackend,
    fmin: float = A0_HZ,
    fmax: float = C8_HZ,
    target_width_octaves: float = 1.0,
    max_depth: int = 5,
) -> Tuple[Optional[float], List[BandResult]]:
    current = Band(fmin, fmax)
    history: List[BandResult] = []

    for depth in range(max_depth + 1):
        width = band_width_oct(current)
        log(f"[Depth {depth}] band={current} width={width:.2f} oct")

        if depth > 0 and width <= target_width_octaves:
            break

        b1, b2 = split_band(current)
        f1, s1 = yin_backend(signal, sr, b1.fmin, b1.fmax)
        f2, s2 = yin_backend(signal, sr, b2.fmin, b2.fmax)

        log(f"  Split → B1[{b1.fmin:.1f}-{b1.fmax:.1f}] f0={f1} s={s1:.3f}")
        log(f"           B2[{b2.fmin:.1f}-{b2.fmax:.1f}] f0={f2} s={s2:.3f}")

        r1, r2 = BandResult(b1, f1, s1), BandResult(b2, f2, s2)
        history += [r1, r2]

        valid = [r for r in (r1, r2) if r.f0 is not None and r.score > 0]
        if not valid:
            log("STOP: no valid subband")
            return None, history

        best = max(valid, key=lambda r: r.score)
        current = best.band

    # final eval
    f0, s = yin_backend(signal, sr, current.fmin, current.fmax)
    log(f"FINAL F0 = {f0} score={s:.3f}")
    history.append(BandResult(current, f0, s))
    return f0, history


# ──────────────────────────────────────────────
# 3) ENTRYPOINT
# ──────────────────────────────────────────────

def detect_f0_seed_partitioned(
    signal: np.ndarray,
    sr: int,
    mode: PartitionMode = "octaves_8",
    yin_backend: YinBackend | None = None,
    target_width_octaves: float = 1.0,
    max_depth: int = 5,
) -> Tuple[Optional[float], List[BandResult]]:
    if yin_backend is None:
        raise ValueError("yin_backend must be provided")

    if mode in ("octaves_8", "octaves_4x2", "cents_250"):
        return yin_partition_vote(signal, sr, mode, yin_backend)

    if mode == "binary_tree":
        return yin_partition_binary_tree(
            signal,
            sr,
            yin_backend,
            fmin=A0_HZ,
            fmax=C8_HZ,
            target_width_octaves=target_width_octaves,
            max_depth=max_depth,
        )

    raise ValueError(f"Unknown mode {mode}")