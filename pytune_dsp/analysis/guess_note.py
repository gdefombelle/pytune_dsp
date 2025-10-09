# guess_note.py — PyTune DSP v0.6 (band classification + anti-subharmonic fusion)

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import librosa
from scipy.signal import hilbert

from pytune_dsp.types.dataclasses import GuessNoteResult, GuessF0Result

NOTE_A4 = 440.0
MIDI_A4 = 69
A0_FREQ = 27.50
A0_MIDI = 21
LOW_BAND_MAX = 200.0
EPS = 1e-12

# Attack analysis
ATTACK_MS = 80
HF_CUTOFF = 2000.0  # SAR cutoff

def freq_to_midi(f: float) -> Optional[int]:
    if not f or f <= 0:
        return None
    return int(round(MIDI_A4 + 12 * np.log2(f / NOTE_A4)))

def midi_to_freq(m: int) -> float:
    return NOTE_A4 * (2.0 ** ((m - MIDI_A4) / 12.0)) if m else 0.0

def _freq_to_note_name(f: float) -> str:
    if not f or f <= 0:
        return "?"
    m = freq_to_midi(f)
    names = ["C", "C#", "D", "D#", "E", "F", "F#",
             "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"

def _cents_between(f1: float, f2: float) -> float:
    return 1200.0 * np.log2(f1 / f2)

def _hann(N: int) -> np.ndarray:
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / max(N - 1, 1))

def adaptive_fft_size(fmin: float, sr: int, periods: int = 12) -> int:
    target = int(np.ceil(sr * periods / max(fmin, 1e-3)))
    return int(2 ** np.ceil(np.log2(max(target, 2048))))

# ---------- Envelope / band ----------
def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    env = np.abs(hilbert(x))
    env /= np.max(env) + 1e-12
    return env

def classify_band_advanced(x: np.ndarray, sr: int) -> tuple[str, float, float, float, float]:
    """
    Retourne (band, t_attack, scr, zcr, sar)
    - t_attack: temps au pic d'enveloppe (s)
    - scr: spectral centroid / (sr/2)
    - zcr: zero-crossing rate sur les ~80 ms initiaux
    - sar: ratio d'énergie HF (>2 kHz) sur l'attaque (~80 ms)
    """
    # 1) Attack envelope
    env = hilbert_envelope(x)
    idx_peak = int(np.argmax(env))
    t_attack = idx_peak / sr

    # 2) Global centroid
    try:
        centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0].mean()
        scr = float(centroid / (sr / 2))
    except Exception:
        scr = 0.0

    # 3) Attack window
    win_len = int(ATTACK_MS / 1000.0 * sr)
    win = x[: min(len(x), win_len)]
    if np.allclose(win, 0.0):
        return "mid", t_attack, scr, 0.0, 0.0

    # 4) ZCR in attack
    try:
        zcr = float(librosa.feature.zero_crossing_rate(win).mean())
    except Exception:
        zcr = 0.0

    # 5) SAR in attack
    sar = 0.0
    try:
        S = np.abs(librosa.stft(win, n_fft=2048, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr)
        total = float(np.sum(S))
        if total > 0:
            sar = float(np.sum(S[freqs >= HF_CUTOFF]) / total)
    except Exception:
        sar = 0.0

    # 6) Decision (seuils robustes, moins de biais “low”)
    # low si attaque nettement lente OU (son sombre + peu d'HF au départ + zcr faible)
    if (t_attack > 0.12) or ((scr < 0.12) and (sar < 0.12) and (zcr < 0.05)):
        band = "low"
    # high si beaucoup d'HF et structure riche/rapide
    elif (sar > 0.35 and scr > 0.25) or (t_attack < 0.035 and (scr > 0.33 or zcr > 0.08)):
        band = "high"
    else:
        band = "mid"

    return band, t_attack, scr, zcr, sar

# ---------- FFT / HPS ----------
def _rfft_mag(signal: np.ndarray, sr: int, fmin: float = 25.0,
              prefer_low_band: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    x = signal.astype(np.float64)
    w = _hann(len(x))
    xw = x * w
    Nfft = max(adaptive_fft_size(fmin, sr, periods=12), 8192)
    spec = np.abs(np.fft.rfft(xw, n=Nfft))
    freqs = np.fft.rfftfreq(Nfft, 1.0 / sr)
    valid = freqs > 1e-6
    spec *= (1.0 / np.maximum(freqs, 1e-6))
    spec[~valid] = 0.0
    if prefer_low_band:
        hi = int(np.searchsorted(freqs, LOW_BAND_MAX, side="right"))
        spec[hi:] *= 0.2
    return freqs, spec

def guess_f0_fft(signal: np.ndarray, sr: int,
                 n_harmonics: int = 8,
                 fmin: float = 25.0,
                 prefer_low_band: bool = False,
                 debug: bool = True) -> Tuple[Optional[float], float]:
    freqs, spec = _rfft_mag(signal, sr, fmin, prefer_low_band)
    if spec.max() <= 0:
        return None, 0.0
    spec_norm = spec / (spec.max() + EPS)
    best_f0, best_score = None, -1.0
    for m in range(A0_MIDI, 108 + 1):
        f0 = midi_to_freq(m)
        score = 0.0
        for n in range(1, n_harmonics + 1):
            fn = f0 * n
            if fn >= freqs[-1]:
                break
            idx = int(np.argmin(np.abs(freqs - fn)))
            score += spec_norm[idx] / np.sqrt(n)
        if score > best_score:
            best_f0, best_score = f0, score
    if debug and best_f0:
        print(f"[FFT] → {best_f0:.2f} Hz ({_freq_to_note_name(best_f0)}) score={best_score:.3f}")
    return best_f0, float(np.tanh(best_score))

def guess_f0_hps(signal: np.ndarray, sr: int,
                 max_down: int = 5,
                 band: str = "mid",
                 debug: bool = True) -> Tuple[Optional[float], float]:
    freqs, spec = _rfft_mag(signal, sr)
    if spec.max() <= 0:
        return None, 0.0
    S = spec / (spec.max() + EPS) + EPS
    H = np.log(S)

    for d in range(2, max_down + 1):
        Sd = S[::d]
        H[:len(Sd)] += np.log(Sd + EPS)

    # bornes : en mid/high, on ignore les très basses fréquences parasites
    fmin = 40.0 if band != "low" else A0_FREQ
    i_min = int(np.searchsorted(freqs, fmin))
    i_max = int(np.searchsorted(freqs, min(1000.0, freqs[-1])))
    if i_max <= i_min:
        return None, 0.0

    k = i_min + int(np.argmax(H[i_min:i_max]))
    f0 = float(freqs[k])
    conf = float(np.clip((H[k] - np.median(H[i_min:i_max])) /
                         (abs(np.median(H[i_min:i_max])) + 1.0), 0, 1))
    if debug:
        print(f"[HPS] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) conf={conf:.2f}")
    return f0, conf

# ---------- YinFFT ----------
def guess_f0_yinfft(signal: np.ndarray, sr: int,
                    fmin: float = 20.0, fmax: float = 2000.0,
                    debug: bool = True) -> Tuple[Optional[float], float]:
    x = signal.astype(np.float64)
    x -= np.mean(x)
    N = len(x)
    fft_size = int(2 ** np.ceil(np.log2(2 * N)))
    X = np.fft.rfft(x, n=fft_size)
    r = np.fft.irfft(np.abs(X)**2, n=fft_size)
    r = r[:N]

    d = np.zeros(N)
    d[0] = 0.0
    d[1:] = np.cumsum(x[:-1]**2)[-1] + np.cumsum(x[1:]**2)[-1] - 2*r[1:]
    d /= np.max(np.abs(d)) + 1e-12

    cmnd = np.zeros_like(d)
    cmnd[0] = 1
    acc = 0.0
    for tau in range(1, len(d)):
        acc += d[tau]
        cmnd[tau] = d[tau] / (acc / tau)

    tau_min = int(sr / fmax)
    tau_max = int(sr / fmin)
    if tau_max <= tau_min + 1:
        return None, 0.0
    tau = np.argmin(cmnd[tau_min:tau_max]) + tau_min
    f0 = sr / tau if tau > 0 else None
    conf = 1 - cmnd[tau] if f0 else 0.0
    if debug and f0:
        print(f"[YINFFT] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) conf={conf:.2f}")
    return f0, conf

# ---------- Comb (low only) ----------
def comb_search_low(signal: np.ndarray, sr: int,
                    fmin: float = 20.0, fmax: float = 100.0,
                    step_hz: float = 0.1,
                    kmax: int = 6,
                    debug: bool = True) -> Tuple[Optional[float], float]:
    freqs, spec = _rfft_mag(signal, sr, fmin=25.0, prefer_low_band=True)
    if spec.max() <= 0:
        return None, 0.0
    spec_norm = spec / (spec.max() + EPS)
    f0_grid = np.arange(fmin, fmax, step_hz)
    scores = np.zeros_like(f0_grid)
    for i, f0 in enumerate(f0_grid):
        s = 0.0
        for k in range(1, kmax + 1):
            fk = f0 * k
            if fk >= freqs[-1]:
                break
            idx = int(np.argmin(np.abs(freqs - fk)))
            s += spec_norm[idx] / np.sqrt(k)
        scores[i] = s
    imax = int(np.argmax(scores))
    best_f0 = float(f0_grid[imax])
    conf = float(np.tanh(scores[imax]))
    if debug:
        print(f"[COMB] → {best_f0:.2f} Hz ({_freq_to_note_name(best_f0)}) conf={conf:.2f}")
    return best_f0, conf

# ---------- Fusion ----------
def _near_octave(f_lo: float, f_hi: float, cents_tol: float = 30.0) -> bool:
    if f_lo <= 0 or f_hi <= 0:
        return False
    ratio = f_hi / f_lo
    return abs(_cents_between(ratio, 2.0)) < cents_tol

def guess_f0_fusion(signal: np.ndarray, sr: int, debug: bool = True) -> GuessF0Result:
    band, t_attack, scr, zcr, sar = classify_band_advanced(signal, sr)
    if debug:
        print(f"🔍 Envelope: band={band} | t_attack={t_attack*1000:.1f}ms | scr={scr:.3f} | zcr={zcr:.3f} | sar={sar:.3f}")

    max_freq_limit = sr * 0.45
    if band == "low":
        yin_fmin, yin_fmax = (20, min(220, max_freq_limit))   # ↑ 220 Hz pour couvrir A3
    else:
        yin_fmin, yin_fmax = (50, min(3000, max_freq_limit))

    prefer_low = band == "low"

    yin_f0, yin_conf = guess_f0_yinfft(signal, sr, yin_fmin, yin_fmax, debug)
    fft_f0, fft_score = guess_f0_fft(signal, sr, fmin=yin_fmin, prefer_low_band=prefer_low, debug=debug)
    hps_f0, hps_conf = guess_f0_hps(signal, sr, band=band, debug=debug)

    comb_f0, comb_conf = (None, 0.0)
    if band == "low" and fft_f0 and fft_f0 < 140:
        comb_f0, comb_conf = comb_search_low(signal, sr, debug=debug)

    # Candidats initiaux
    cand = [(l, f, c) for (l, f, c) in [
        ("YINFFT", yin_f0, yin_conf),
        ("FFT", fft_f0, float(fft_score)),
        ("HPS", hps_f0, hps_conf),
        ("COMB", comb_f0, comb_conf),
    ] if f and f > 20.0]

    if not cand:
        return GuessF0Result(
            f0=None, confidence=0.0, harmonics=[], matched=[],
            method="none", band=band, components={}, extra={"t_attack": t_attack, "scr": scr, "zcr": zcr, "sar": sar}
        )

    # Anti-subharmonique : si un couple est quasi-octave, favoriser le plus élevé (surtout mid/high)
    # et/ou dévaloriser le plus bas.
    def boost(label: str, base_boost: float) -> float:
        # bonus plafonné
        return min(1.0, base_boost)

    adjusted = []
    for l, f, c in cand:
        adjusted.append([l, f, c])

    for i in range(len(cand)):
        li, fi, ci = cand[i]
        for j in range(i + 1, len(cand)):
            lj, fj, cj = cand[j]
            f_lo, f_hi = (fi, fj) if fi <= fj else (fj, fi)
            if _near_octave(f_lo, f_hi, cents_tol=30.0):
                # Si mid/high : on préfère le plus haut; si low : on ne change rien
                if band != "low":
                    # boost pour le haut, petite pénalité pour le bas
                    if fi > fj:
                        adjusted[i][2] = boost(li, max(ci, cj) + 0.25)
                        adjusted[j][2] = max(0.0, adjusted[j][2] - 0.10)
                    else:
                        adjusted[j][2] = boost(lj, max(ci, cj) + 0.25)
                        adjusted[i][2] = max(0.0, adjusted[i][2] - 0.10)

    # Cas spécifique fréquent : FFT ≈ 2× YIN → renforcer YIN
    if yin_f0 and fft_f0 and _near_octave(yin_f0, fft_f0, cents_tol=30.0):
        for k, (l, f, c) in enumerate(adjusted):
            if l == "YINFFT":
                adjusted[k][2] = boost(l, c + 0.20)

    # Score final
    label, final_f, final_c = max(adjusted, key=lambda t: t[2])

    # Rapprochement si deux candidats proches (< 25 cents)
    for i in range(len(adjusted)):
        for j in range(i + 1, len(adjusted)):
            l1, f1, c1 = adjusted[i]
            l2, f2, c2 = adjusted[j]
            if abs(_cents_between(f1, f2)) < 25:
                f_mean = 0.5 * (f1 + f2)
                c_boost = min(1.0, max(c1, c2) + 0.2)
                if c_boost > final_c:
                    final_f, final_c = f_mean, c_boost
                    label = f"{l1}+{l2}"

    if debug:
        print(f"🎯 Fusion → {final_f:.2f} Hz ({_freq_to_note_name(final_f)}) conf={final_c:.2f} method={label}")

    components = {
        "yinfft": {"f0": yin_f0 or 0.0, "conf": yin_conf},
        "fft": {"f0": fft_f0 or 0.0, "score": float(fft_score)},
        "hps": {"f0": hps_f0 or 0.0, "conf": hps_conf},
        "comb": {"f0": comb_f0 or 0.0, "conf": comb_conf},
    }

    return GuessF0Result(
        f0=final_f, confidence=final_c, harmonics=[], matched=[],
        method=label, band=band, components=components,
        extra={"t_attack": t_attack, "scr": scr, "zcr": zcr, "sar": sar}
    )

# ---------- API ----------
def guess_note(signal: np.ndarray, sr: int, debug: bool = True) -> GuessNoteResult:
    res = guess_f0_fusion(signal, sr, debug=debug)
    if not res.f0:
        return GuessNoteResult(midi=None, f0=None, confidence=0.0, method="none")

    f0_final = float(res.f0)
    conf_final = float(res.confidence)
    fusion_method = res.method
    band = res.band
    extra = getattr(res, "extra", {}) or {}

    yin = res.components.get("yinfft", {})
    fft = res.components.get("fft", {})
    hps = res.components.get("hps", {})
    comb = res.components.get("comb", {})

    yin_f0, yin_conf = yin.get("f0"), yin.get("conf")
    fft_f0, fft_score = fft.get("f0"), fft.get("score")
    hps_f0, hps_conf = hps.get("f0"), hps.get("conf")
    comb_f0, comb_conf = comb.get("f0"), comb.get("conf")

    note_name = _freq_to_note_name(f0_final)
    debug_lines: List[str] = []
    debug_lines.append(f"Envelope band detected: {band}")
    debug_lines.append(
        "Envelope features: "
        f"t_attack={extra.get('t_attack', 0.0)*1000:.1f}ms, "
        f"scr={extra.get('scr', 0.0):.3f}, "
        f"zcr={extra.get('zcr', 0.0):.3f}, "
        f"sar={extra.get('sar', 0.0):.3f}"
    )
    if yin_f0:
        debug_lines.append(f"[YINFFT] → {yin_f0:.2f} Hz ({_freq_to_note_name(yin_f0)}) conf={yin_conf:.2f}")
    if fft_f0:
        debug_lines.append(f"[FFT] → {fft_f0:.2f} Hz ({_freq_to_note_name(fft_f0)}) score={(fft_score or 0):.3f}")
    if hps_f0:
        debug_lines.append(f"[HPS] → {hps_f0:.2f} Hz ({_freq_to_note_name(hps_f0)}) conf={hps_conf:.2f}")
    if comb_f0:
        debug_lines.append(f"[COMB] → {comb_f0:.2f} Hz ({_freq_to_note_name(comb_f0)}) conf={comb_conf:.2f}")
    debug_lines.append(f"🎯 Fusion → {f0_final:.2f} Hz ({note_name}) conf={conf_final:.2f} method={fusion_method}")

    return GuessNoteResult(
        midi=freq_to_midi(f0_final),
        f0=f0_final,
        confidence=conf_final,
        method=fusion_method,
        debug_log=debug_lines,
        subresults=res.components,
        envelope_band=band,
    )