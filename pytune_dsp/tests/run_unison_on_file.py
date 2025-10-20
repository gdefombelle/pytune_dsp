# pytune_dsp/tests/run_unisson_on_file.py
from __future__ import annotations
import os, sys, math
import numpy as np
import essentia, essentia.standard as es
essentia.log.infoActive = False

# --- importe la fonction guess_note où qu'elle se trouve dans le package ---
try:
    from pytune_dsp.analysis.guess_note import guess_note as guess_f0  # v0.7 dans sous-module analysis
except Exception:
    from pytune_dsp.guess_note import guess_note as guess_f0           # fallback si placé à la racine

# ---------- utils ----------
def cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(max(a, 1e-12) / max(b, 1e-12))

def to_float_list(x) -> list[float]:
    if np.isscalar(x): return [float(x)]
    out, stack = [], [x]
    while stack:
        a = stack.pop()
        try:
            stack.extend(list(a))
        except TypeError:
            out.append(float(a))
    return out

def remove_subharmonics(cands: list[float], tol_cents: float = 30.0) -> list[float]:
    cands = sorted(cands); keep = []
    for f in cands:
        if any(abs(cents(f * 2, g)) <= tol_cents for g in cands if g > f):
            continue
        keep.append(f)
    return keep

def cluster_by_cents(cands: list[float], tol: float = 8.0):
    cands = sorted(cands)
    if not cands: return [], []
    clusters = []
    for f in cands:
        placed = False
        for cl in clusters:
            if abs(cents(f, float(np.median(cl)))) <= tol:
                cl.append(f); placed = True; break
        if not placed:
            clusters.append([f])
    clusters.sort(key=len, reverse=True)
    centers = [float(np.median(cl)) for cl in clusters]
    sizes   = [len(cl) for cl in clusters]
    return centers, sizes

def proj_remove_sine(x: np.ndarray, sr: int, f0: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = x.size; t = np.arange(n, dtype=np.float32); w = 2*np.pi*f0/sr
    c = np.cos(w*t, dtype=np.float32); s = np.sin(w*t, dtype=np.float32)
    a = (2.0/n) * float(np.dot(x, c)); b = (2.0/n) * float(np.dot(x, s))
    return (x - (a*c + b*s)).astype(np.float32, copy=False)

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x*x)))

def mono_wav_f32(path: str, sr: int = 44100) -> np.ndarray:
    y = es.MonoLoader(filename=path, sampleRate=sr)()
    return np.asarray(y, dtype=np.float32)

# ---------- MPK helpers ----------
def mpk_collect_tail(y: np.ndarray, sr: int, fmin: float, fmax: float,
                     frameSize: int = 4096, hopSize: int = 256, tail_frames: int = 40) -> list[float]:
    y = np.asarray(y, dtype=np.float32)
    mpk = es.MultiPitchKlapuri(sampleRate=sr, frameSize=frameSize, hopSize=hopSize,
                               minFrequency=fmin, maxFrequency=fmax)
    frames = mpk(y)
    bag: list[float] = []; seen = 0
    for fr in reversed(frames):
        vals = to_float_list(fr)
        if vals:
            bag.extend(vals); seen += 1
            if seen >= tail_frames: break
    return bag

# ---------- multi-f0: jusqu'à 3 cordes ----------
def detect_unison_multi(y: np.ndarray, sr: int, fmin: float, fmax: float,
                        max_f0: int = 3, tail_frames: int = 50,
                        cents_tol: float = 8.0, min_sep: float = 4.0, max_sep: float = 35.0,
                        step_min_energy: float = 0.005):
    y = np.asarray(y, dtype=np.float32)
    res = y.copy(); total_r = rms(y)
    found: list[float] = []; explained: list[float] = []

    for _ in range(max_f0):
        bag = mpk_collect_tail(res if found else y, sr, fmin, fmax, tail_frames=tail_frames)
        # écarte proches/sous-harmoniques des f0 déjà posées
        for fp in found:
            bag = [f for f in bag
                   if abs(cents(f, fp)) > (cents_tol + 1.5)
                   and abs(cents(2*f, fp)) > 25
                   and abs(cents(f, 2*fp)) > 25]
        bag = remove_subharmonics(bag)
        centers, _ = cluster_by_cents(bag, tol=cents_tol)
        if not centers: break

        f0 = centers[0]
        if found:
            sep = min(abs(cents(f0, fp)) for fp in found)
            if sep < min_sep or sep > max_sep:
                break

        before = rms(res)
        res_next = proj_remove_sine(res, sr, f0)
        after = rms(res_next)
        explained_rel = max(0.0, (before - after) / max(total_r, 1e-12))
        if explained_rel < step_min_energy:
            break

        res = res_next
        found.append(float(f0))
        explained.append(explained_rel)

    pairwise = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            pairwise.append(abs(cents(found[j], found[i])))

    is_prob = (len(found) >= 2) and all(min_sep <= d <= max_sep for d in pairwise)

    return {
        "f0s": sorted(found),
        "pairwise_cents": [round(d, 2) for d in sorted(pairwise)],
        "residual_explained": round(sum(explained), 3),
        "is_unison_problem": is_prob,
    }

# ---------- bande depuis guess_note (sans connaître la note) ----------
def band_from_guess(y: np.ndarray, sr: int,
                    floor: float = 50.0, ceil: float = 4200.0) -> tuple[float, float, dict]:
    gr = guess_f0(y, sr, debug=False)   # ← ta fusion multi-méthodes
    m = float(getattr(gr, "f0", 0.0) or 0.0)
    band = getattr(gr, "envelope_band", "mid") or "mid"

    if m <= 0.0 or not np.isfinite(m):
        # fallback large si rien trouvé
        return 100.0, 2000.0, {"source": "fallback", "band": "unknown", "f0": None}

    # fenêtre adaptative selon le band
    if band == "low":
        fmin = max(floor, 0.55*m); fmax = min(ceil, 1.60*m)
    elif band == "high":
        fmin = max(300.0, 0.70*m); fmax = min(ceil, 1.35*m)  # anti-220Hz
    else:  # mid
        fmin = max(floor, 0.60*m)
        if m > 250.0: fmin = max(fmin, 300.0)
        fmax = min(ceil, 1.40*m)

    meta = {"source": "guess_note", "band": band, "f0": m, "confidence": float(getattr(gr, "confidence", 0.0))}
    return float(fmin), float(fmax), meta

# ---------- main ----------
def main():
    sr = 44100
    # fichier: arg1 ou chemins standards
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        for p in ("tests/48.wav", "pytune_dsp/tests/48.wav", "./48.wav"):
            if os.path.exists(p): path = p; break
    if not path or not os.path.exists(path):
        raise FileNotFoundError("WAV introuvable (passe le chemin en argument ou mets tests/48.wav)")

    y = mono_wav_f32(path, sr=sr)

    fmin, fmax, meta = band_from_guess(y, sr)
    print(f"[info] bande (guess_note): band={meta['band']} f0≈{(meta['f0'] or 0):.2f}Hz conf={meta.get('confidence',0):.2f}")
    print(f"[info] fmin={fmin:.1f} Hz  fmax={fmax:.1f} Hz")

    out = detect_unison_multi(y, sr, fmin=fmin, fmax=fmax, max_f0=3)
    print("Résultat unison ->", out)

if __name__ == "__main__":
    main()