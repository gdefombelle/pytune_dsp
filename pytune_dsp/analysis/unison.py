# pytune_dsp/analysis/unison.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy.signal import get_window, find_peaks

from pytune_dsp.utils.pianos import probability_strings_per_note
from pytune_dsp.utils.note_utils import freq_to_midi

# ---------- Helpers music ----------
A4, MIDI_A4 = 440.0, 69

def hz_to_cents(f: float, f_ref: float) -> float:
    if f <= 0 or f_ref <= 0:
        return 0.0
    return 1200.0 * np.log2(f / f_ref)

def cents_to_hz(delta_cents: float, f_ref: float) -> float:
    return f_ref * (2.0 ** (delta_cents / 1200.0))

# ---------- Dataclasses ----------
@dataclass
class UnisonComponent:
    freq: float
    amp: float
    cents_offset: float

@dataclass
class HarmonicComponents:
    harmonic_index: int
    components: List[UnisonComponent] = field(default_factory=list)

@dataclass
class UnisonAnalysis:
    midi: int
    f0_est: float
    n_strings_prior: Dict[int, float]                 # {1: p, 2: p, 3: p}
    posterior: Dict[int, float]                       # {1: p, 2: p, 3: p}
    detected_n_components: int                        # 1..3 (observé autour de f0)
    components_f0_band: List[UnisonComponent]         # composantes détectées près de f0
    harmonics: List[HarmonicComponents]               # composantes sur H=1..Hmax
    beat_hz_estimate: float                           # estimation battement principal (Hz)
    severity: float                                   # 0..1, 0 parfait, 1 très battant
    recommend_f0_hp: str                              # "none" | "global" | "per-component"
    confidence: float                                 # 0..1 sur la décision

# ---------- Core detection ----------
def _peak_pick_band(
    sig: np.ndarray,
    sr: int,
    f_center: float,
    band_cents: float = 40.0,
    window: str = "hann",
    prominence: float = 0.02,
    pad_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RFFT sur une fenêtre, on récupère les pics dans ±band_cents autour de f_center.
    Retourne (freqs_peaks, amps_peaks).
    """
    n = len(sig)
    if n < 128 or f_center <= 0:
        return np.array([]), np.array([])

    # Fenêtrage + padding
    w = get_window(window, n, fftbins=True)
    y = sig * w
    if pad_factor > 1:
        y = np.pad(y, (0, (pad_factor - 1) * n), mode="constant")

    spec = np.abs(np.fft.rfft(y))
    spec /= (np.max(spec) + 1e-12)
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    # Bande en Hz depuis cents autour de f_center
    f_low = cents_to_hz(-band_cents, f_center)
    f_high = cents_to_hz(+band_cents, f_center)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return np.array([]), np.array([])

    sub_spec = spec[mask]
    sub_freq = freqs[mask]

    if len(sub_spec) < 5:
        return np.array([]), np.array([])

    # Pics avec séparation minimale ~ 6 cents en Hz
    min_hz = max(0.5, f_center * (2**(6/1200) - 1))  # ~6 cents
    idx_peaks, _ = find_peaks(sub_spec, prominence=prominence, distance=max(1, int(min_hz / (sub_freq[1]-sub_freq[0]+1e-12))))

    if idx_peaks.size == 0:
        return np.array([]), np.array([])

    # Tri par amplitude décroissante
    order = np.argsort(sub_spec[idx_peaks])[::-1]
    idx_sorted = idx_peaks[order]
    return sub_freq[idx_sorted], sub_spec[idx_sorted]


def _cluster_components(freqs: np.ndarray, amps: np.ndarray, min_sep_cents: float, f_ref: float, kmax: int = 3) -> List[UnisonComponent]:
    """
    Regroupe des pics très proches en ≤ kmax composantes.
    """
    if len(freqs) == 0:
        return []

    comps: List[UnisonComponent] = []
    for f, a in zip(freqs, amps):
        c = hz_to_cents(f, f_ref)
        placed = False
        for comp in comps:
            if abs(c - comp.cents_offset) < min_sep_cents:
                # merge
                wsum = comp.amp + a
                comp.freq = (comp.freq * comp.amp + f * a) / (wsum + 1e-12)
                comp.amp = wsum
                comp.cents_offset = hz_to_cents(comp.freq, f_ref)
                placed = True
                break
        if not placed:
            comps.append(UnisonComponent(freq=f, amp=a, cents_offset=c))
        if len(comps) >= kmax:
            break

    # tri par amplitude
    comps.sort(key=lambda x: x.amp, reverse=True)
    return comps


def _likelihood_components_across_harmonics(harmonics: List[HarmonicComponents]) -> Dict[int, float]:
    """
    Score de vraisemblance pour k=1,2,3 cordes basé sur la cohérence des offsets en cents sur H=1..Hn.
    Idée: en cents, les offsets de chaque corde sont à peu près constants à travers les harmoniques.
    """
    # Collecte des offsets en cents par H
    offsets_per_H = []
    for H in harmonics:
        offsets_per_H.append([c.cents_offset for c in H.components])

    # Si quasi un seul pic partout -> k=1
    n1 = sum(1 for off in offsets_per_H if len(off) <= 1)
    frac_single = n1 / max(1, len(offsets_per_H))

    # Variance intra-H (si plusieurs pics)
    spreads = []
    for off in offsets_per_H:
        if len(off) > 1:
            spreads.append(max(off) - min(off))  # en cents
    avg_spread = np.median(spreads) if spreads else 0.0

    # Heuristiques simples (bornes empiriques)
    like = {1: 0.0, 2: 0.0, 3: 0.0}
    if frac_single > 0.75 and avg_spread < 6.0:
        like[1] = 1.0
        like[2] = 0.2
        like[3] = 0.1
    else:
        # Si on voit ≥2 pics sur ≥2 harmoniques avec offsets stables
        # approx: compter combien d'harmoniques ont >=2 pics
        multi = sum(1 for off in offsets_per_H if len(off) >= 2)
        if multi >= 2:
            # Deux cordes plus probables
            like[1] = 0.2
            like[2] = 1.0
            like[3] = 0.5 if avg_spread > 12.0 else 0.3
        else:
            # cas mitigé
            like[1] = 0.6
            like[2] = 0.5
            like[3] = 0.2

    # Normaliser
    s = sum(like.values()) + 1e-12
    for k in like:
        like[k] /= s
    return like


def analyze_unison(
    signal: np.ndarray,
    sr: int,
    f0_est: float,
    midi: Optional[int] = None,
    piano_type: str = "upright",
    era: str = "modern",
    Hmax: int = 3,
    f0_band_cents: float = 40.0,
    min_sep_cents: float = 8.0,
    window_sec: float = 0.35,
) -> UnisonAnalysis:
    """
    1) prior sur #cordes via probability_strings_per_note
    2) détection de composantes autour de f0 et des 1..Hmax harmoniques
    3) posterior simple: prior ⊙ likelihood
    4) métriques (beat Hz, sévérité) + reco f0_HP
    """
    if midi is None:
        midi = freq_to_midi(f0_est)

    prior = probability_strings_per_note(midi, piano_type=piano_type, era=era)

    # segment court au début (attaque) pour limiter le flou temporel
    n = len(signal)
    n_win = min(n, max(256, int(window_sec * sr)))
    seg = signal[:n_win].astype(np.float64, copy=False)

    # --- f0 band ---
    f_f0, a_f0 = _peak_pick_band(seg, sr, f_center=f0_est, band_cents=f0_band_cents, pad_factor=2)
    comps_f0 = _cluster_components(f_f0, a_f0, min_sep_cents=min_sep_cents, f_ref=f0_est, kmax=3)

    # --- harmonics 1..Hmax ---
    harmonics: List[HarmonicComponents] = []
    for h in range(1, Hmax + 1):
        fH = f0_est * h
        band = max(30.0, f0_band_cents) if h == 1 else f0_band_cents  # un peu plus large au fondamental
        f_h, a_h = _peak_pick_band(seg, sr, f_center=fH, band_cents=band, pad_factor=2)
        comps_h = _cluster_components(f_h, a_h, min_sep_cents=min_sep_cents, f_ref=fH, kmax=3)
        harmonics.append(HarmonicComponents(harmonic_index=h, components=comps_h))

    # --- likelihood k=1..3 d'après cohérence cents sur harmoniques ---
    like = _likelihood_components_across_harmonics(harmonics)

    # --- posterior = prior * like (passe-bas) ---
    post = {k: prior.get(k, 0.0) * like.get(k, 0.0) for k in (1, 2, 3)}
    s = sum(post.values()) + 1e-12
    post = {k: v / s for k, v in post.items()}

    # --- métriques unisson ---
    detected = max(1, len(comps_f0))
    # battement principal : si ≥2 composantes, diff en Hz des deux plus fortes
    beat_hz = 0.0
    if len(comps_f0) >= 2:
        f1, f2 = comps_f0[0].freq, comps_f0[1].freq
        beat_hz = abs(f2 - f1)

    # sévérité: combinons spread en cents et beat_hz
    spread_cents = 0.0
    if len(comps_f0) >= 2:
        spread_cents = max(c.cents_offset for c in comps_f0) - min(c.cents_offset for c in comps_f0)
    # normalisation soft : 6 cents ~ modéré, 12 cents ~ sévère
    sev = float(np.clip(spread_cents / 12.0 + beat_hz / 3.0, 0.0, 1.0))

    # --- reco f0_HP ---
    # - si posterior[1] dominant et sévérité faible -> raffinement global
    # - si posterior[2/3] significatif et composantes nettes -> per-component
    if post.get(1, 0.0) > 0.7 and sev < 0.35:
        reco = "global"
        conf = min(1.0, post[1] * (1.0 - sev))
    elif (post.get(2, 0.0) > 0.5 or post.get(3, 0.0) > 0.4) and len(comps_f0) >= 2:
        reco = "per-component"
        conf = min(1.0, max(post.get(2, 0.0), post.get(3, 0.0)) * (0.5 + 0.5 * min(1.0, spread_cents / 8.0)))
    else:
        reco = "none"
        conf = 0.5 * (1.0 - sev)

    return UnisonAnalysis(
        midi=int(midi) if midi is not None else -1,
        f0_est=float(f0_est),
        n_strings_prior=prior,
        posterior=post,
        detected_n_components=detected,
        components_f0_band=comps_f0,
        harmonics=harmonics,
        beat_hz_estimate=float(beat_hz),
        severity=float(sev),
        recommend_f0_hp=reco,
        confidence=float(np.clip(conf, 0.0, 1.0)),
    )