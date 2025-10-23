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

# --- console logging helper (Essentia only) ---
ESS_PREFIX = "[ESS]"
def _ess_log(debug: bool, msg: str):
    if debug:
        print(f"{ESS_PREFIX} {msg}")

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

# ---------- Band classification (Essentia features) ----------
def classify_band_essentia(y: np.ndarray, sr: int) -> Tuple[str, Dict[str, float]]:
    """
    Heuristique 'low/mid/high' pour guider les bornes de pitch.
    - t_attack via enveloppe d'amplitude
    - scr: spectral centroid normalisé (0..1)
    - zcr: zero-crossing rate sur ~80ms d'attaque
    - sar: ratio d'énergie > 2 kHz dans l'attaque
    """
    y = _np_f32(y)
    N = len(y)

    # Enveloppe lissée (amplitude follower)
    env = es.Envelope()(y)
    idx_peak = int(np.argmax(env))
    t_attack = idx_peak / sr

    # Fenêtre d'attaque ~80 ms
    ATTACK_MS = 80
    win_len = max(256, int(ATTACK_MS / 1000.0 * sr))
    fr = y[: min(N, win_len)]

    # ZCR (attaque)
    zcr = float(es.ZeroCrossingRate()(fr))

    # Spectral centroid + ratio HF (>2k) sur l'attaque
    w = es.Windowing(type="hann")
    sp = es.Spectrum()
    spec = sp(w(fr))

    scr_hz = float(es.SpectralCentroidTime()(spec))
    scr = scr_hz / (sr * 0.5 + EPS)

    n = len(spec)
    freqs = np.linspace(0, sr * 0.5, n, dtype=np.float32)
    total = float(np.sum(spec)) + EPS
    sar = float(np.sum(spec[freqs >= 2000.0]) / total)

    # Décision
    if (t_attack > 0.12 and scr < 0.15) or (scr < 0.10 and sar < 0.10 and zcr < 0.05):
        band = "low"
    elif (sar > 0.35 and (scr > 0.25 or zcr > 0.08)) or (t_attack < 0.030 and scr > 0.33):
        band = "high"
    else:
        band = "mid"

    return band, {"t_attack": t_attack, "scr": scr, "zcr": zcr, "sar": sar}

def _search_bounds(band: str, sr: int) -> Tuple[float, float]:
    max_freq_limit = sr * 0.45
    if band == "low":
        return 20.0, min(220.0, max_freq_limit)
    elif band == "high":
        return 100.0, min(4000.0, max_freq_limit)
    else:
        return 50.0, min(3000.0, max_freq_limit)

# ---------- Candidates with Essentia ----------
def cand_yinfft(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool = False) -> Tuple[Optional[float], float]:
    """
    PitchYinFFT par frames → médiane sur la queue de note.
    """
    y = _np_f32(y)
    frameSize, hopSize = 4096, 256
    fc = es.FrameGenerator(y, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    w, sp = es.Windowing(type="hann"), es.Spectrum()
    py = es.PitchYinFFT(sampleRate=sr,frameSize=frameSize, minFrequency=fmin, maxFrequency=fmax)

    tail: List[float] = []
    for fr in fc:
        pitch, conf = py(sp(w(fr)))
        if pitch > 0:
            tail.append(float(pitch))

    if not tail:
        return None, 0.0

    tail = tail[-40:] if len(tail) > 40 else tail
    f0 = float(np.median(tail))
    _ess_log(debug, f"[YINFFT] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) (n={len(tail)})")

    # Confiance heuristique : compacité du cluster
    iqr = float(np.percentile(tail, 75) - np.percentile(tail, 25) + 1e-9)
    conf = float(np.clip(1.0 / (1.0 + iqr / 3.0), 0.0, 1.0))
    return f0, conf

def cand_melodia(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool = False) -> Tuple[Optional[float], float]:
    """
    PredominantPitchMelodia (lissage / Viterbi intégré).
    Retourne la médiane des frames voisées (!= 0) sur la queue de note.
    """
    y = _np_f32(y)
    mel = es.PredominantPitchMelodia(
        sampleRate=sr, minFrequency=fmin, maxFrequency=fmax,
        frameSize=2048, hopSize=128,
        voicingTolerance=0.2, timeContinuity=2.0
    )
    pitch, voicing = mel(y)   # vectors
    pitch = np.asarray(pitch, dtype=np.float32)
    voiced = pitch[pitch > 0]

    if voiced.size == 0:
        return None, 0.0

    tail = voiced[-80:] if voiced.size > 80 else voiced
    f0 = float(np.median(tail))
    _ess_log(debug, f"[Melodia] → {f0:.2f} Hz ({_freq_to_note_name(f0)}) (n={tail.size})")

    # Confiance heuristique : ratio de frames voisées + compacité
    voiced_ratio = float(voiced.size / max(pitch.size, 1))
    iqr = float(np.percentile(tail, 75) - np.percentile(tail, 25) + 1e-9)
    conf = float(np.clip(0.5 * voiced_ratio + 0.5 * (1.0 / (1.0 + iqr / 3.0)), 0.0, 1.0))
    return f0, conf

def _flatten_vecvec(x) -> List[float]:
    out: List[float] = []
    try:
        for row in x:
            try:
                out.extend([float(v) for v in row])
            except TypeError:
                out.append(float(row))
    except TypeError:
        pass
    return out

def cand_mpk(y: np.ndarray, sr: int, fmin: float, fmax: float, debug: bool = False) -> Tuple[Optional[float], float, List[float]]:
    """
    MultiPitchKlapuri → collecte des 40 dernières frames non vides,
    clustering (médiane) et sélection du cluster majoritaire.
    """
    y = _np_f32(y)
    frameSize, hopSize = 4096, 256
    mpk = es.MultiPitchKlapuri(
        sampleRate=sr, frameSize=frameSize, hopSize=hopSize,
        minFrequency=fmin, maxFrequency=fmax,
        numberHarmonics=10, binResolution=10
    )
    frames = mpk(y)  # VectorVectorFloat

    bag: List[float] = []
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

    # Clusterise avec une fenêtre ±8 cents autour d'une médiane glissante
    clusters: List[List[float]] = []
    def _is_close(f, center): return abs(_cents(f, center)) <= 8.0

    for f in bag:
        placed = False
        for cl in clusters:
            c = float(np.median(cl))
            if _is_close(f, c):
                cl.append(f)
                placed = True
                break
        if not placed:
            clusters.append([f])

    clusters.sort(key=len, reverse=True)
    center = float(np.median(clusters[0]))
    count = len(clusters[0])
    _ess_log(debug, f"[MPK] → {center:.2f} Hz ({_freq_to_note_name(center)}) (cluster={count}, total={len(bag)})")

    # Confiance heuristique : dominance du cluster majoritaire
    conf = float(np.clip(count / max(len(bag), 1), 0.0, 1.0))
    return center, conf, bag

# ---------- Fusion ----------
def _near_octave(f1: float, f2: float, tol_cents: float = 30.0) -> bool:
    if f1 <= 0 or f2 <= 0:
        return False
    lo, hi = (f1, f2) if f1 <= f2 else (f2, f1)
    return abs(_cents(hi, lo * 2.0)) < tol_cents

def guess_f0_essentia(signal: np.ndarray, sr: int, debug: bool = True) -> GuessF0Result:
    band, feats = classify_band_essentia(signal, sr)
    fmin, fmax = _search_bounds(band, sr)

    _ess_log(debug, "=== Essentia guess_f0 (Melodia + YINFFT + MPK) ===")
    _ess_log(debug, f"band={band} | t_attack={feats['t_attack']*1000:.1f}ms | scr={feats['scr']:.3f} | zcr={feats['zcr']:.3f} | sar={feats['sar']:.3f}")
    _ess_log(debug, f"search: [{fmin:.1f}, {fmax:.1f}] Hz")

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_yin = ex.submit(cand_yinfft, signal, sr, fmin, fmax, debug)
        fut_mel = ex.submit(cand_melodia, signal, sr, fmin, fmax, debug)
        fut_mpk = ex.submit(cand_mpk, signal, sr, fmin, fmax, debug)

        yin_f0, yin_conf = fut_yin.result()
        mel_f0, mel_conf = fut_mel.result()
        mpk_f0, mpk_conf, mpk_bag = fut_mpk.result()

    cands = [
        ("Melodia", mel_f0, mel_conf),
        ("YINFFT",  yin_f0, yin_conf),
        ("MPK",     mpk_f0, mpk_conf),
    ]
    cands = [(l, f, c) for (l, f, c) in cands if f and f > 0]

    if not cands:
        return GuessF0Result(
            f0=None, confidence=0.0, harmonics=[], matched=[],
            method="none", band=band, components={}, extra=feats
        )

    # Anti-octave : boost la plus basse si deux candidats sont à ±1 octave
    adj = [list(x) for x in cands]
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            li, fi, ci = adj[i]
            lj, fj, cj = adj[j]
            if _near_octave(fi, fj, tol_cents=25.0) and min(fi, fj) >= 40:
                if fi < fj:
                    adj[i][2] = min(1.0, max(ci, cj) + 0.20)
                else:
                    adj[j][2] = min(1.0, max(ci, cj) + 0.20)

    # Consensus : si deux méthodes sont à <25 cents → moyenne + boost
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            li, fi, ci = adj[i]
            lj, fj, cj = adj[j]
            if abs(_cents(fi, fj)) < 25:
                f_mean = 0.5 * (fi + fj)
                c_boost = min(1.0, max(ci, cj) + 0.20)
                if ci < cj:
                    adj[i] = [f"{li}+{lj}", f_mean, c_boost]
                else:
                    adj[j] = [f"{lj}+{li}", f_mean, c_boost]

    label, f_final, c_final = max(adj, key=lambda t: t[2])

    _ess_log(
        debug,
        "components: "
        f"Melodia={(mel_f0 and f'{mel_f0:.2f}Hz({mel_conf:.2f})') or 'none'}, "
        f"YINFFT={(yin_f0 and f'{yin_f0:.2f}Hz({yin_conf:.2f})') or 'none'}, "
        f"MPK={(mpk_f0 and f'{mpk_f0:.2f}Hz({mpk_conf:.2f})') or 'none'}"
    )
    _ess_log(debug, f"🎯 Fusion → {float(f_final):.2f} Hz ({_freq_to_note_name(float(f_final))}) "
                    f"conf={float(c_final):.2f} method={label}")

    components = {
        "melodia": {"f0": mel_f0 or 0.0, "conf": float(mel_conf)},
        "yinfft":  {"f0": yin_f0 or 0.0, "conf": float(yin_conf)},
        "mpk":     {"f0": mpk_f0 or 0.0, "conf": float(mpk_conf), "bag_n": len(mpk_bag or [])},
    }

    return GuessF0Result(
        f0=float(f_final), confidence=float(c_final),
        harmonics=[], matched=[], method=label, band=band,
        components=components, extra=feats
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
    log: List[str] = []
    log.append(
        f"Band: {res.band} | "
        f"t_attack={feats.get('t_attack', 0.0)*1000:.1f}ms | "
        f"scr={feats.get('scr', 0.0):.3f} | "
        f"zcr={feats.get('zcr', 0.0):.3f} | "
        f"sar={feats.get('sar', 0.0):.3f}"
    )
    if comps.get("melodia", {}).get("f0"):
        log.append(f"[Melodia] → {comps['melodia']['f0']:.2f} Hz conf={comps['melodia']['conf']:.2f}")
    if comps.get("yinfft", {}).get("f0"):
        log.append(f"[YINFFT]  → {comps['yinfft']['f0']:.2f} Hz conf={comps['yinfft']['conf']:.2f}")
    if comps.get("mpk", {}).get("f0"):
        log.append(f"[MPK]     → {comps['mpk']['f0']:.2f} Hz conf={comps['mpk']['conf']:.2f} (bag={comps['mpk']['bag_n']})")
    log.append(f"🎯 Fusion → {f0_final:.2f} Hz ({note_name}) conf={conf_final:.2f} method={res.method}")

    return GuessNoteResult(
        midi=freq_to_midi(f0_final),
        f0=f0_final,
        confidence=conf_final,
        method=res.method,
        debug_log=log,
        subresults=res.components,
        envelope_band=res.band,
    )