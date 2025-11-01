# pytune_dsp/analysis/guess_note_essentia.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Essentia
import essentia
import essentia.standard as es

from pytune_dsp.types.dataclasses import GuessNoteResult, GuessF0Result

# ======================================================
# -------- FFT micro-cache (optionnel et thread-safe) ---
# ======================================================
ENABLE_FFT_CACHE = True
_FFT_CACHE_LOCK = threading.Lock()
# key: (ptr, length, sr, nfft, prefer_low_band) -> (freqs, mag)
_FFT_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

def _key_for_signal(y: np.ndarray) -> tuple[int, int]:
    """Génère une clé stable basée sur le pointeur mémoire + longueur du signal."""
    ptr = int(y.__array_interface__['data'][0])
    return (ptr, y.shape[0])

def _rfft_mag_lowpref_cached(y: np.ndarray, sr: int, nfft: int, prefer_low_band: bool):
    """Version avec cache de _rfft_mag_lowpref."""
    if not ENABLE_FFT_CACHE:
        return _rfft_mag_lowpref(y, sr, nfft, prefer_low_band)

    base = _key_for_signal(y)
    key = (base[0], base[1], int(sr), int(nfft), bool(prefer_low_band))
    with _FFT_CACHE_LOCK:
        hit = _FFT_CACHE.get(key)
        if hit is not None:
            return hit

    freqs, mag = _rfft_mag_lowpref(y, sr, nfft, prefer_low_band)
    with _FFT_CACHE_LOCK:
        if len(_FFT_CACHE) > 24:
            _FFT_CACHE.pop(next(iter(_FFT_CACHE)))
        _FFT_CACHE[key] = (freqs, mag)
    return freqs, mag

# ======================================================

NOTE_A4, MIDI_A4 = 440.0, 69
EPS = 1e-12

# Attack / cutoff
ATTACK_MS = 80.0
HF_CUTOFF = 2000.0
NFFT = 2048
HOP = 256
EARLY_MS = 60.0

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
    y = _np_f32(y)
    env = es.Envelope()(y)
    env /= (np.max(env) + EPS)
    idx_peak = int(np.argmax(env))
    t_attack = idx_peak / sr

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

    win_len = int((ATTACK_MS / 1000.0) * sr)
    win = y[: min(len(y), win_len)]
    if np.allclose(win, 0.0):
        return "mid", {"t_attack": t_attack, "scr": scr, "zcr": 0.0, "sar": 0.0}

    zcr = float(es.ZeroCrossingRate()(win))
    spec = sp(w(win))
    freqs_att = np.linspace(0.0, sr * 0.5, len(spec), dtype=np.float32)
    total = float(np.sum(spec)) + EPS
    sar = float(np.sum(spec[freqs_att >= HF_CUTOFF]) / total)

    if (t_attack > 0.12 and scr < 0.15) or (scr < 0.10 and sar < 0.10 and zcr < 0.05):
        band = "low"
    elif (sar > 0.25 and (scr > 0.18 or zcr > 0.06)) or (t_attack < 0.030 and scr > 0.18):
        band = "high"
    else:
        band = "mid"

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

# ---------- FFT / HPS / COMB (cache-enabled) ----------
def _rfft_mag(y: np.ndarray, sr: int, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    w = es.Windowing(type="hann")
    sp = es.Spectrum(size=nfft)
    mag = sp(w(_np_f32(y)))
    freqs = np.linspace(0.0, sr * 0.5, len(mag), dtype=np.float64)
    return freqs, np.asarray(mag, dtype=np.float64)

def _rfft_mag_lowpref(y: np.ndarray, sr: int, nfft: int, prefer_low_band: bool = False):
    freqs, mag = _rfft_mag(y, sr, nfft)
    mag *= 1.0 / np.maximum(freqs, 1e-6)
    if prefer_low_band:
        hi = int(np.searchsorted(freqs, 200.0, side="right"))
        mag[hi:] *= 0.25
    return freqs, mag

def fft_anchor_fundamental(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool=False):
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(y)))))
    freqs, mag = _rfft_mag_lowpref_cached(y, sr, nfft, prefer_low_band=(fmax<=220))
    lo = int(np.searchsorted(freqs, fmin, side="left"))
    hi = int(np.searchsorted(freqs, min(fmax, 1000.0), side="right"))
    if hi <= lo:
        return None, 0.0
    seg = mag[lo:hi]
    idx = int(np.argmax(seg)) + lo
    f0 = float(freqs[idx])
    peak = float(seg.max())
    base = float(np.median(seg) + 1e-9)
    conf = float(np.clip((peak / base) / 20.0, 0.0, 1.0))
    if debug:
        _ess_log(True, f"[FFT*] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) conf={conf:.2f}")
    return f0, conf

def hps_search(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool=False):
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(y)))))
    freqs, mag = _rfft_mag_lowpref_cached(y, sr, nfft, prefer_low_band=False)
    mag = mag / (mag.max() + EPS)

    hps = mag.copy()
    for r in (2, 3, 4, 5):
        dec = mag[::r]
        hps[:len(dec)] *= dec

    lo = int(np.searchsorted(freqs, fmin, side="left"))
    hi = int(np.searchsorted(freqs, fmax, side="right"))
    if hi <= lo:
        return None, 0.0
    seg = hps[lo:hi]
    if seg.size == 0 or np.all(seg <= 0):
        return None, 0.0
    idx = int(np.argmax(seg)) + lo
    f0 = float(freqs[idx])
    left, right = max(lo, idx - 3), min(hi, idx + 4)
    local = hps[left:right]
    prom = float((hps[idx] - np.median(local)) / (np.max(local) + EPS))
    conf = float(np.clip(0.5 + 0.5 * np.tanh(3.0 * prom), 0.0, 1.0))
    if debug:
        _ess_log(True, f"[HPS] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) conf={conf:.2f}")
    return f0, conf

def comb_search_low_ess(y: np.ndarray, sr: int,
                        fmin: float = 20.0, fmax: float = 120.0,
                        step_hz: float = 0.1, kmax: int = 6,
                        prefer_low_band: bool = True, debug: bool = False):
    nfft = max(8192, 1 << int(np.ceil(np.log2(len(y) * 2))))
    freqs, mag = _rfft_mag_lowpref_cached(y, sr, nfft, prefer_low_band=prefer_low_band)
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

# (le reste du fichier inchangé : cand_yinfft, cand_mpk, guess_f0_essentia, guess_note_essentia, etc.)

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
    _ess_log(debug, "=== Essentia guess_f0 (YINFFT core) ===")
    _ess_log(debug, f"band={band} | t_attack={feats['t_attack']*1000:.1f}ms | scr={feats['scr']:.3f} | zcr={feats['zcr']:.3f} | sar={feats['sar']:.3f}")
    _ess_log(debug, f"search: [{fmin:.1f}, {fmax:.1f}] Hz")

    with ThreadPoolExecutor(max_workers=5) as ex:
        fut_yin  = ex.submit(cand_yinfft,        signal, sr, fmin, fmax, debug)
        fut_fft  = ex.submit(fft_anchor_fundamental, signal, sr, fmin, fmax, debug)
        fut_hps  = ex.submit(hps_search,         signal, sr, max(80.0, fmin), fmax, debug) if band != "low" else None
        fut_comb = ex.submit(comb_search_low_ess, signal, sr, 20.0, 120.0, 0.1, 6, True, debug) if band == "low" else None
        fut_mpk  = ex.submit(cand_mpk,           signal, sr, fmin, fmax, debug) if band != "low" else None
        fut_yine = ex.submit(cand_yinfft_early,  signal, sr, fmin, fmax, debug) if band != "low" else None

        yin_f0,  yin_conf  = fut_yin.result()
        fft_f0,  fft_conf  = fut_fft.result()
        hps_f0,  hps_conf  = (fut_hps.result()  if fut_hps  else (None, 0.0))
        comb_f0, comb_conf = (fut_comb.result() if fut_comb else (None, 0.0))
        mpk_f0,  mpk_conf, mpk_bag = (fut_mpk.result() if fut_mpk else (None, 0.0, []))
        yie_f0,  yie_conf  = (fut_yine.result() if fut_yine else (None, 0.0))

    # --- Post-correction d’octave pour les graves (sécurité Librosa-free) ---
    #
    # Objectif : corriger les rares cas où YINFFT détecte la 2e harmonique (~2× trop haut)
    #             alors que la FFT ou COMB détecte la vraie fondamentale (autour de 50–110 Hz).
    #
    # --- Post-correction d’octave globale (graves & médiums) ---
    #
    # Corrige :
    #  - YINFFT qui saute une octave (2× trop haut sur A0–A2)
    #  - YINFFT qui sous-évalue (½ trop bas sur A4–A5)
    #  - Bandes "low" mal classées quand la fondamentale dépasse 200 Hz
    #
    if yin_f0 and fft_f0:
        ratio = yin_f0 / fft_f0

        # Correction d’octave (double ou demi)
        if band == "low":
            # Cas 1 : YINFFT détecte la 2e harmonique (trop haut)
            if ratio > 1.8 and fft_conf > 0.5:
                _ess_log(debug, f"[Anti-octave LOW] YIN={yin_f0:.2f} / FFT={fft_f0:.2f} → ÷2")
                yin_f0 *= 0.5
            # Cas 2 : YINFFT détecte la moitié (trop bas)
            elif ratio < 0.55 and fft_conf > 0.5:
                _ess_log(debug, f"[Anti-octave LOW] YIN={yin_f0:.2f} / FFT={fft_f0:.2f} → ×2")
                yin_f0 *= 2.0

        # Sécurité : si on est encore très au-dessus du domaine "low", on reclasse en "mid"
        if band == "low" and yin_f0 > 200.0:
            _ess_log(debug, f"[Reclassify] {yin_f0:.2f} Hz déplacé en mid-band")
            band = "mid"

    # Correction de cohérence : en zone mid, YIN ne doit jamais être < 70 Hz ni > 2 kHz
    if band == "mid" and yin_f0:
        if yin_f0 < 70.0:
            _ess_log(debug, f"[Clamp-mid] YINFFT corrigé de {yin_f0:.2f} Hz → 70 Hz")
            yin_f0 = 70.0
        elif yin_f0 > 2000.0:
            _ess_log(debug, f"[Clamp-mid] YINFFT corrigé de {yin_f0:.2f} Hz → 2000 Hz")
            yin_f0 = 2000.0
    # --- Bonus de consensus YIN+FFT (renforce la confiance si proche) ---
    if band != "low" and yin_f0 and fft_f0:
        delta_cents = abs(_cents(yin_f0, fft_f0))
        if delta_cents < 10.0:  # très proche (≈ <0.1 demi-ton)
            yin_conf = min(1.0, yin_conf + 0.1)
            fft_conf = min(1.0, fft_conf + 0.1)
            _ess_log(debug, f"[Consensus] YIN+FFT accordés ({delta_cents:.1f} cents) → +0.1 confiance")
    

    # Candidats valides
    cands: List[Tuple[str, float, float, float]] = []  # (label, f0, conf, weight)
    def add(label, f, c, w): 
        if f and f > 0: cands.append((label, float(f), float(c), float(w)))

    # pondérations par défaut
    add("YINFFT",        yin_f0, yin_conf, 1.00)
    add("FFT*",          fft_f0, fft_conf, 0.70)
    if band != "low":
        add("HPS",       hps_f0, hps_conf, 0.65)
        add("MPK",       mpk_f0, mpk_conf, 0.80)
        add("YINFFT-early", yie_f0, yie_conf, 0.60)
    else:
        add("COMB*",     comb_f0, comb_conf, 0.85)

    # filtre haute bande (sécurité)
    if band == "high":
        cands = [c for c in cands if c[1] >= 150.0]

    if not cands:
        return GuessF0Result(f0=None, confidence=0.0, harmonics=[], matched=[],
                             method="none", band=band, components={}, extra=feats)

    # Anti-octave ciblé en grave : si YIN>120Hz et COMB/HPS ~ moitié → favoriser la basse
    if band == "low" and yin_f0 and yin_f0 > 120.0:
        for (lab, f, c, w) in cands:
            if lab in ("COMB*", "FFT*") and (20.0 <= f <= 120.0) and _near_octave(yin_f0, f*2.0, tol_cents=35.0):
                # boost forte de la basse, baisse la haute
                cands = [ (lab2, f2, (c2+0.35 if lab2==lab else c2-0.10), w2) 
                          for (lab2,f2,c2,w2) in cands ]
                break

    # Fusion : moyenne pondérée par (conf * weight); bonus si quasi-consensus
    best_label, best_f, best_score = None, None, -1.0
    for i in range(len(cands)):
        li, fi, ci, wi = cands[i]
        score_i = ci * wi
        # consensus local
        for j in range(i+1, len(cands)):
            lj, fj, cj, wj = cands[j]
            if abs(_cents(fi, fj)) < 20.0:
                score_i += 0.20 * max(ci*wi, cj*wj)
        if score_i > best_score:
            best_label, best_f, best_score = li, fi, score_i

    # Confiance finale bornée [0,1]
    conf_final = float(np.clip(best_score, 0.0, 1.0))
    _ess_log(debug, f"🎯 Fusion → {best_f:.2f} Hz ({_freq_to_note_name(best_f)}) conf={conf_final:.2f} method={best_label}")

    components = {
        "yinfft": {"f0": yin_f0 or 0.0, "conf": float(yin_conf)},
        "fft": {"f0": fft_f0 or 0.0, "conf": float(fft_conf)},
        "hps": {"f0": hps_f0 or 0.0, "conf": float(hps_conf)},
        "comb": {"f0": comb_f0 or 0.0, "conf": float(comb_conf)},
        "mpk": {"f0": mpk_f0 or 0.0, "conf": float(mpk_conf), "bag_n": len(mpk_bag or [])},
        "yinfft_early": {"f0": yie_f0 or 0.0, "conf": float(yie_conf)},
    }

    # Infos de mismatch
    if band == "low" and best_f > 400:
        _ess_log(debug, "[Info] Band mismatch tolerated — high freq found in low band window.")
    if band == "mid" and best_f < 100:
        _ess_log(debug, "[Info] Band mismatch tolerated — low freq found in mid band window.")

    return GuessF0Result(
        f0=float(best_f),
        confidence=conf_final,
        harmonics=[],
        matched=[],
        method=best_label,
        band=band,
        components=components,
        extra=feats,
    )

# ---------- API ----------
def guess_note_essentia(signal: np.ndarray, sr: int, debug: bool = True) -> GuessNoteResult:
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
    if comps.get("yinfft", {}).get("f0"):
        log.append(f"[YINFFT] → {comps['yinfft']['f0']:.2f} Hz conf={comps['yinfft']['conf']:.2f}")
    if comps.get("yinfft_early", {}).get("f0"):
        log.append(f"[YINFFT-early] → {comps['yinfft_early']['f0']:.2f} Hz conf={comps['yinfft_early']['conf']:.2f}")
    if comps.get("hps", {}).get("f0"):
        log.append(f"[HPS] → {comps['hps']['f0']:.2f} Hz conf={comps['hps']['conf']:.2f}")
    if comps.get("fft", {}).get("f0"):
        log.append(f"[FFT*] → {comps['fft']['f0']:.2f} Hz conf={comps['fft']['conf']:.2f}")
    if comps.get("mpk", {}).get("f0"):
        log.append(f"[MPK] → {comps['mpk']['f0']:.2f} Hz conf={comps['mpk']['conf']:.2f} (bag={comps['mpk']['bag_n']})")
    if comps.get("comb", {}).get("f0"):
        log.append(f"[COMB*] → {comps['comb']['f0']:.2f} Hz conf={comps['comb']['conf']:.2f}")

    log.append(f"🎯 Fusion → {f0_final:.2f} Hz ({note_name}) conf={conf_final:.2f} method={res.method}")

    return GuessNoteResult(
        midi=freq_to_midi(f0_final),
        f0=f0_final,
        confidence=conf_final,
        method=res.method,
        debug_log=log,
        subresults=comps,
        envelope_band=res.band,
    )