# pytune_dsp/analysis/guess_note_essentia.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Essentia
import essentia
import essentia.standard as es

from pytune_dsp.types.dataclasses import GuessNoteResult, GuessF0Result

NOTE_A4, MIDI_A4 = 440.0, 69
A0_FREQ, A0_MIDI = 27.50, 21
EPS = 1e-12

# Attack / cutoff
ATTACK_MS = 80.0
HF_CUTOFF = 2000.0
NFFT = 2048
HOP = 256
EARLY_MS = 60.0  # fenêtre attaque pour aigus

# --- logging ---
ESS_PREFIX = "[ESS]"
def _ess_log(debug: bool, msg: str):
    if debug:
        print(f"{ESS_PREFIX} {msg}")

# --- utils ---
def freq_to_midi(f: float) -> Optional[int]:
    if not f or f <= 0:
        return None
    return int(round(MIDI_A4 + 12 * np.log2(f / NOTE_A4)))

def _freq_to_note_name(f: float) -> str:
    if not f or f <= 0:
        return "?"
    m = freq_to_midi(f) or 0
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{names[m % 12]}{m // 12 - 1}"

def _cents(a: float, b: float) -> float:
    return 1200.0 * np.log2(max(a, 1e-12) / max(b, 1e-12))

def _np_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32, order="C")

def _slice_early(y: np.ndarray, sr: int, dur_ms: float) -> np.ndarray:
    n = int(sr * (dur_ms / 1000.0))
    return y[: max(1, min(len(y), n))]

# ---------- Band classification ----------
def classify_band_essentia(y: np.ndarray, sr: int) -> Tuple[str, Dict[str, float]]:
    """
    Calibrated version (v2.1) aligned with librosa's classify_band_advanced.
    Corrige les erreurs de bande sur les notes A4–A6 (fausses low).
    """
    y = _np_f32(y)

    # --- Enveloppe / attaque ---
    env = es.Envelope()(y)
    env /= (np.max(env) + EPS)
    idx_peak = int(np.argmax(env))
    t_attack = idx_peak / sr

    # --- Spectral centroid global ---
    w = es.Windowing(type="hann")
    sp = es.Spectrum()
    fc = es.FrameGenerator(y, frameSize=NFFT, hopSize=HOP, startFromZero=True)
    cents = []
    for fr in fc:
        mag = sp(w(fr))
        freqs = np.linspace(0.0, sr * 0.5, len(mag), dtype=np.float32)
        weighted = mag * (freqs ** 0.5)
        denom = float(np.sum(weighted)) + EPS
        num = float(np.sum(freqs * weighted))
        cents.append(num / denom if denom > 0 else 0.0)
    scr = float((np.mean(cents) if cents else 0.0) / (sr * 0.5 + EPS))

    # --- Attack window ---
    win_len = int((ATTACK_MS / 1000.0) * sr)
    win = y[: min(len(y), win_len)]
    if np.allclose(win, 0.0):
        return "mid", {"t_attack": t_attack, "scr": scr, "zcr": 0.0, "sar": 0.0}

    zcr = float(es.ZeroCrossingRate()(win))
    spec = sp(w(win))
    freqs_att = np.linspace(0.0, sr * 0.5, len(spec), dtype=np.float32)
    total = float(np.sum(spec)) + EPS
    sar = float(np.sum(spec[freqs_att >= HF_CUTOFF]) / total)

    # --- Decision logic (revised thresholds) ---
    # 1) Graves
    if (t_attack > 0.12 and scr < 0.15) or (scr < 0.10 and sar < 0.10 and zcr < 0.05):
        band = "low"
    # 2) Aigus
    elif (sar > 0.25 and (scr > 0.18 or zcr > 0.06)) or (t_attack < 0.030 and scr > 0.18):
        band = "high"
    # 3) Par défaut
    else:
        band = "mid"

    # --- Correction dynamique post-pondération ---
    # (évite les A4–A6 classées low à tort)
    if band == "low" and (scr > 0.045 and sar > 0.12):
        band = "mid"
    if band == "mid" and (sar > 0.20 and scr > 0.07 and zcr > 0.04):
        band = "high"

    return band, {"t_attack": t_attack, "scr": scr, "zcr": zcr, "sar": sar}
def _search_bounds(band: str, sr: int) -> Tuple[float, float]:
    max_freq_limit = sr * 0.45
    if band == "low":
        return 20.0, min(220.0, max_freq_limit)
    elif band == "high":
        return 100.0, min(4000.0, max_freq_limit)
    else:
        return 50.0, min(3000.0, max_freq_limit)

# ---------- Helpers spectrum / comb ----------
def _rfft_mag_lowpref(y: np.ndarray, sr: int, nfft: int, prefer_low_band: bool = False):
    w = es.Windowing(type="hann")
    sp = es.Spectrum(size=nfft)
    mag = sp(w(_np_f32(y)))
    freqs = np.linspace(0.0, sr * 0.5, len(mag), dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    mag *= 1.0 / np.maximum(freqs, 1e-6)
    if prefer_low_band:
        hi = int(np.searchsorted(freqs, 200.0, side="right"))
        mag[hi:] *= 0.25
    return freqs, mag

def comb_search_low_ess(y: np.ndarray, sr: int,
                        fmin: float = 20.0, fmax: float = 120.0,
                        step_hz: float = 0.1, kmax: int = 6,
                        prefer_low_band: bool = True, debug: bool = False):
    nfft = max(8192, 1 << int(np.ceil(np.log2(len(y) * 2))))
    freqs, mag = _rfft_mag_lowpref(y, sr, nfft, prefer_low_band=prefer_low_band)
    if mag.max() <= 0:
        return None, 0.0
    mag_norm = mag / (mag.max() + EPS)

    f0_grid = np.arange(fmin, fmax, step_hz, dtype=np.float64)
    scores = np.zeros_like(f0_grid)
    for i, f0 in enumerate(f0_grid):
        s = 0.0
        for k in range(1, kmax + 1):
            fk = f0 * k
            if fk >= freqs[-1]:
                break
            idx = int(np.argmin(np.abs(freqs - fk)))
            s += mag_norm[idx] / np.sqrt(k)
        scores[i] = s

    imax = int(np.argmax(scores))
    best_f0 = float(f0_grid[imax])
    conf = float(np.tanh(scores[imax]))
    if debug:
        _ess_log(True, f"[COMB*] → {best_f0:.2f} Hz ({_freq_to_note_name(best_f0)}) conf={conf:.2f}")
    return best_f0, conf

# ---------- Candidates ----------
def _adaptive_framesize(fmin: float, sr: int, periods: int = 12, min_pow2: int = 4096, max_pow2: int = 32768):
    target = int(np.ceil(sr * periods / max(fmin, 1e-3)))
    nfft = 1 << int(np.ceil(np.log2(max(target, min_pow2))))
    return int(min(max_pow2, max(min_pow2, nfft)))

def cand_yinfft(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool = False):
    sr = int(sr)
    y = _np_f32(y)
    frameSize = _adaptive_framesize(fmin, sr)
    hopSize = 256
    fc = es.FrameGenerator(y, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    w, sp = es.Windowing(type="hann"), es.Spectrum()
    py = es.PitchYinFFT(sampleRate=sr, frameSize=frameSize, minFrequency=fmin, maxFrequency=fmax)

    tail = []
    for fr in fc:
        pitch, _ = py(sp(w(fr)))
        if pitch > 0:
            tail.append(float(pitch))

    if not tail:
        return None, 0.0

    tail = tail[-40:] if len(tail) > 40 else tail
    f0 = float(np.median(tail))
    iqr = float(np.percentile(tail, 75) - np.percentile(tail, 25) + 1e-9)
    conf = float(np.clip(1.0 / (1.0 + iqr / 3.0), 0.0, 1.0))
    _ess_log(debug, f"[YINFFT] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) conf={conf:.2f} (frameSize={frameSize})")
    return f0, conf

def cand_yinfft_early(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool = False):
    y = _np_f32(_slice_early(y, sr, EARLY_MS))
    frameSize = 2048 if fmax >= 1000 else 4096
    hopSize = 128
    fc = es.FrameGenerator(y, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    w, sp = es.Windowing(type="hann"), es.Spectrum()
    py = es.PitchYinFFT(sampleRate=sr, frameSize=frameSize, minFrequency=max(200.0, fmin), maxFrequency=fmax)
    vals = []
    for fr in fc:
        p, _ = py(sp(w(fr)))
        if p > 0:
            vals.append(float(p))
    if not vals:
        return None, 0.0
    f0 = float(np.median(vals))
    spread = float(np.std(vals) + 1e-9)
    conf = float(np.clip(1.0 / (1.0 + spread / 50.0), 0.0, 1.0))
    _ess_log(debug, f"[YINFFT-early] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) conf={conf:.2f}")
    return f0, conf

def _flatten_vecvec(x) -> List[float]:
    out = []
    try:
        for row in x:
            try:
                out.extend([float(v) for v in row])
            except TypeError:
                out.append(float(row))
    except TypeError:
        pass
    return out

def cand_mpk(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool = False):
    y = _np_f32(y)
    frameSize, hopSize = 4096, 256
    mpk = es.MultiPitchKlapuri(sampleRate=sr, frameSize=frameSize, hopSize=hopSize,
                               minFrequency=fmin, maxFrequency=fmax,
                               numberHarmonics=10, binResolution=10)
    frames = mpk(y)
    bag = []
    seen = 0
    for fr in reversed(frames):
        vals = _flatten_vecvec(fr)
        if vals:
            bag.extend([v for v in vals if fmin <= v <= fmax])
            seen += 1
            if seen >= 40:
                break
    if not bag:
        return None, 0.0, []
    bag = sorted(bag)
    clusters = []
    def _is_close(f, c): return abs(_cents(f, c)) <= 8.0
    for f in bag:
        for cl in clusters:
            if _is_close(f, np.median(cl)):
                cl.append(f)
                break
        else:
            clusters.append([f])
    clusters.sort(key=len, reverse=True)
    center = float(np.median(clusters[0]))
    count = len(clusters[0])
    conf = float(np.clip(count / max(len(bag), 1), 0.0, 1.0))
    _ess_log(debug, f"[MPK] → {center:.2f} Hz ({_freq_to_note_name(center)}) conf={conf:.2f}")
    return center, conf, bag

# ---------- Fusion ----------
def _near_octave(f1: float, f2: float, tol_cents: float = 30.0) -> bool:
    if f1 <= 0 or f2 <= 0:
        return False
    lo, hi = (f1, f2) if f1 <= f2 else (f2, f1)
    return abs(_cents(hi, lo * 2.0)) < tol_cents

def guess_f0_essentia(signal: np.ndarray, sr: int, debug: bool = True) -> GuessF0Result:
    sr = int(sr)
    band, feats = classify_band_essentia(signal, sr)
    fmin, fmax = _search_bounds(band, sr)
    _ess_log(debug, "=== Essentia guess_f0 (YINFFT + MPK) ===")
    _ess_log(debug, f"band={band} | t_attack={feats['t_attack']*1000:.1f}ms | scr={feats['scr']:.3f} | zcr={feats['zcr']:.3f} | sar={feats['sar']:.3f}")
    _ess_log(debug, f"search: [{fmin:.1f}, {fmax:.1f}] Hz")

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_yin = ex.submit(cand_yinfft, signal, sr, fmin, fmax, debug)
        fut_mpk = ex.submit(cand_mpk, signal, sr, fmin, fmax, debug)
        fut_comb = ex.submit(comb_search_low_ess, signal, sr, 20.0, 120.0, 0.1, 6, True, debug) if band == "low" else None
        fut_yin_early = ex.submit(cand_yinfft_early, signal, sr, fmin, fmax, debug) if band != "low" else None

        yin_f0, yin_conf = fut_yin.result()
        mpk_f0, mpk_conf, mpk_bag = fut_mpk.result()
        comb_f0, comb_conf = (fut_comb.result() if fut_comb else (None, 0.0))
        yinE_f0, yinE_conf = (fut_yin_early.result() if fut_yin_early else (None, 0.0))

    cands = [
        ("YINFFT", yin_f0, yin_conf),
        ("MPK", mpk_f0, mpk_conf),
        ("COMB*", comb_f0, comb_conf if band == "low" else 0.0),
        ("YINFFT-early", yinE_f0, yinE_conf if band != "low" else 0.0),
    ]
    cands = [(l, f, c) for (l, f, c) in cands if f and f > 0]
    if band == "high":
        cands = [(l, f, c) for (l, f, c) in cands if f >= 150.0]

    if not cands:
        return GuessF0Result(f0=None, confidence=0.0, harmonics=[], matched=[],
                             method="none", band=band, components={}, extra=feats)

    adj = [list(x) for x in cands]
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            fi, ci = adj[i][1], adj[i][2]
            fj, cj = adj[j][1], adj[j][2]
            if _near_octave(fi, fj, tol_cents=25.0):
                low_idx = i if fi < fj else j
                high_idx = j if fi < fj else i
                adj[low_idx][2] = min(1.0, max(ci, cj) + (0.20 if band == "low" else 0.10))
                adj[high_idx][2] = max(0.0, adj[high_idx][2] - 0.05)
            if abs(_cents(fi, fj)) < 25:
                f_mean = 0.5 * (fi + fj)
                c_boost = min(1.0, max(ci, cj) + 0.2)
                adj[i] = [f"{adj[i][0]}+{adj[j][0]}", f_mean, c_boost]

    adj = [(l, f, (c + 0.05) if (band == "high" and "YINFFT-early" in l) else c) for (l, f, c) in adj]
    label, f_final, c_final = max(adj, key=lambda t: t[2])
    _ess_log(debug, f"🎯 Fusion → {f_final:.2f} Hz ({_freq_to_note_name(f_final)}) conf={c_final:.2f} method={label}")

    components = {
        "yinfft": {"f0": yin_f0 or 0.0, "conf": float(yin_conf)},
        "mpk": {"f0": mpk_f0 or 0.0, "conf": float(mpk_conf), "bag_n": len(mpk_bag or [])},
        "comb": {"f0": comb_f0 or 0.0, "conf": float(comb_conf)},
        "yinfft_early": {"f0": yinE_f0 or 0.0, "conf": float(yinE_conf)},
    }

       # ---------- suite et fin de guess_f0_essentia ----------
    # Info mismatch toléré (diagnostic)
    if band == "low" and f_final > 400:
        _ess_log(debug, "[Info] Band mismatch tolerated — high freq found in low band window.")
    if band == "mid" and f_final < 100:
        _ess_log(debug, "[Info] Band mismatch tolerated — low freq found in mid band window.")

    return GuessF0Result(
        f0=float(f_final),
        confidence=float(c_final),
        harmonics=[],
        matched=[],
        method=label,
        band=band,
        components=components,
        extra=feats,
    )

# ---------- API ----------
def guess_note_essentia(signal: np.ndarray, sr: int, debug: bool = True) -> GuessNoteResult:
    """
    Interface publique : retourne un GuessNoteResult à partir d’un signal mono.
    """
    res = guess_f0_essentia(signal, sr, debug=debug)
    if not res.f0:
        return GuessNoteResult(midi=None, f0=None, confidence=0.0, method="none")

    f0_final = float(res.f0)
    conf_final = float(res.confidence)
    note_name = _freq_to_note_name(f0_final)

    feats = res.extra or {}
    comps = res.components or {}
    log = [
        f"Band: {res.band} | "
        f"t_attack={feats.get('t_attack', 0.0)*1000:.1f}ms | "
        f"scr={feats.get('scr', 0.0):.3f} | "
        f"zcr={feats.get('zcr', 0.0):.3f} | "
        f"sar={feats.get('sar', 0.0):.3f}"
    ]

    # logs des sous-composants
    if comps.get("yinfft", {}).get("f0"):
        log.append(f"[YINFFT] → {comps['yinfft']['f0']:.2f} Hz conf={comps['yinfft']['conf']:.2f}")
    if comps.get("yinfft_early", {}).get("f0"):
        log.append(f"[YINFFT-early] → {comps['yinfft_early']['f0']:.2f} Hz conf={comps['yinfft_early']['conf']:.2f}")
    if comps.get("mpk", {}).get("f0"):
        log.append(
            f"[MPK] → {comps['mpk']['f0']:.2f} Hz conf={comps['mpk']['conf']:.2f} "
            f"(bag={comps['mpk']['bag_n']})"
        )
    if comps.get("comb", {}).get("f0"):
        log.append(f"[COMB*] → {comps['comb']['f0']:.2f} Hz conf={comps['comb']['conf']:.2f}")

    log.append(
        f"🎯 Fusion → {f0_final:.2f} Hz ({note_name}) "
        f"conf={conf_final:.2f} method={res.method}"
    )

    return GuessNoteResult(
        midi=freq_to_midi(f0_final),
        f0=f0_final,
        confidence=conf_final,
        method=res.method,
        debug_log=log,
        subresults=comps,     # ✅ pas de asdict() sur dict
        envelope_band=res.band,
    )