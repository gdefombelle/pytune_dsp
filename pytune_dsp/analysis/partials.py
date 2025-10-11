# pytune_dsp/analysis/partials.py — v2 (CZT-free, robust, joint f0+B fit)

from __future__ import annotations
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

# ---------------------------
# Utils
# ---------------------------
def _parabolic_robust(freqs: np.ndarray, mags: np.ndarray, k: int) -> float:
    """
    Parabolic peak interp, mais:
      - on travaille en log-mag (moins biaisé),
      - garde une garde-fou si dégénérée.
    """
    if k <= 0 or k >= len(mags) - 1:
        return float(freqs[k])
    m1, m2, m3 = mags[k - 1], mags[k], mags[k + 1]
    if m2 <= 0 or m1 <= 0 or m3 <= 0:
        return float(freqs[k])
    # parabole sur log-mag
    l1, l2, l3 = np.log(m1), np.log(m2), np.log(m3)
    denom = (l1 - 2.0 * l2 + l3)
    if abs(denom) < 1e-12:
        return float(freqs[k])
    delta = 0.5 * (l1 - l3) / denom
    # clamp si aberrant (>1 bin = suspect)
    delta = float(np.clip(delta, -0.75, 0.75))
    df = freqs[1] - freqs[0]
    return float(freqs[k] + delta * df)


def _local_snr(spec: np.ndarray, k: int, win: int = 6) -> float:
    """
    SNR local très simple autour de k: pic / médiane des voisins.
    """
    N = len(spec)
    i0 = max(0, k - win)
    i1 = min(N, k + win + 1)
    nb = spec[i0:i1]
    if nb.size < 3:
        return 1.0
    noise = np.median(nb)
    noise = max(noise, 1e-12)
    return float(spec[k] / noise)


def _cents(a: float, b: float) -> float:
    return 1200.0 * np.log2(a / b)


# ---------------------------
# Extraction de partiels
# ---------------------------
def compute_partials_fft_peaks(
    signal: np.ndarray,
    sr: int,
    f0_ref: float,
    nb_partials: int = 8,
    n_fft: int = 8192,
    hop_length: int = 512,
    search_width_cents: float = 80.0,
    pad_factor: int = 2,
) -> tuple[list[float], list[tuple[float, float, float, int]], list[float]]:
    """
    Estime les partiels via FFT zero-paddée + recherche locale autour de k*f0_ref
    et interpolation parabolique robuste (log-mag).
    Retourne:
      - harmonics: [k*f0_ref]
      - partials:  [(f_est, amp, snr, k_harm), ...]
      - inharm:    [déviation en cents vs k*f0_ref]
    """
    if f0_ref <= 0.0 or signal.size == 0:
        return [], [], []

    # STFT, spectre global = max temporel (stable sur sons soutenus)
    N = n_fft * pad_factor
    S = np.abs(librosa.stft(signal, n_fft=N, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N)
    spec = np.max(S, axis=1)

    harmonics = [k * f0_ref for k in range(1, nb_partials + 1)]
    partials: list[tuple[float, float, float, int]] = []
    inharm: list[float] = []

    f_max_usable = freqs[-1] * 0.995
    ratio = 2.0 ** (search_width_cents / 1200.0)

    for k_idx, target in enumerate(harmonics, start=1):
        if target >= f_max_usable:
            break

        low = max(freqs[0], target / ratio)
        high = min(f_max_usable, target * ratio)
        if high <= low:
            break

        li = np.searchsorted(freqs, low, side="left")
        hi = np.searchsorted(freqs, high, side="right")
        if hi - li < 3:
            continue

        # stratégie k=1 : on favorise le bin le + proche du target pour éviter un sous-harmonique
        if k_idx == 1:
            k0 = int(np.argmin(np.abs(freqs[li:hi] - target)) + li)
        else:
            k0 = int(np.argmax(spec[li:hi]) + li)

        f_est = _parabolic_robust(freqs, spec, k0)
        # garde-fou: si sorti de fenêtre, reprend k0
        if not (low <= f_est <= high):
            f_est = float(freqs[k0])

        amp = float(spec[k0])
        snr = _local_snr(spec, k0, win=8)
        partials.append((f_est, amp, snr, k_idx))
        inharm.append(_cents(f_est, target) if f_est > 0 else 0.0)

    return harmonics, partials, inharm


# ---------------------------
# Fit conjoint (f0, B)
# ---------------------------
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Médiane pondérée (robuste). Utilisée pour estimer f0(B) de façon stable.
    """
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) / np.sum(w)
    j = np.searchsorted(cw, 0.5)
    return float(v[min(j, len(v) - 1)])


def _f0_given_B(partials_hz: np.ndarray, ks: np.ndarray, weights: np.ndarray, B: float) -> float:
    """
    Pour un B donné, f0 optimal (robuste) via médiane pondérée de:
        f0_k(B) = f_k / ( k * sqrt(1 + B k^2) )
    """
    denom = ks * np.sqrt(1.0 + B * (ks ** 2))
    denom = np.maximum(denom, 1e-12)
    f0_candidates = partials_hz / denom
    return _weighted_median(f0_candidates, weights)


def _robust_loss_cents(err_cents: np.ndarray, delta: float = 6.0) -> np.ndarray:
    """
    Perte Huber en cents (delta ≈ 6 cents par défaut).
    """
    a = np.abs(err_cents)
    quad = 0.5 * (a ** 2)
    lin = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)


def fit_f0_B(
    partials: list[tuple[float, float, float, int]],
    f0_seed: float,
    B_min: float = 0.0,
    B_max: float = 3e-3,
    n_grid: int = 60,
) -> tuple[float, float, float, int]:
    """
    Ajuste (f0, B) par recherche 1D sur B ∈ [B_min, B_max].
    Pour chaque B, f0(B) = médiane pondérée (robuste) puis on minimise
    la somme des pertes Huber en cents.

    Retourne: (f0_best, B_best, rms_cents_best, n_used_partials)
    """
    if not partials or f0_seed <= 0:
        return 0.0, 0.0, 0.0, 0

    f_hz = np.array([p[0] for p in partials], float)
    amps = np.array([p[1] for p in partials], float)
    snrs = np.array([p[2] for p in partials], float)
    ks   = np.array([p[3] for p in partials], int).astype(float)

    # Poids: amplitude * snr / sqrt(k) (limite l'influence des harmoniques élevés)
    w = amps * np.maximum(snrs, 1.0) / np.sqrt(np.maximum(ks, 1.0))
    w = w / (np.max(w) + 1e-12)

    # grille sur B
    Bs = np.linspace(B_min, B_max, n_grid)
    best = (1e9, 0.0, 0.0, 0.0)  # (loss, f0, B, rms_cents)

    for B in Bs:
        f0 = _f0_given_B(f_hz, ks, w, B)
        if f0 <= 0:
            continue
        # erreur en cents vs modèle inharmonique
        pred = ks * f0 * np.sqrt(1.0 + B * (ks ** 2))
        err_c = _cents(f_hz, pred)
        loss = np.sum(_robust_loss_cents(err_c, delta=6.0) * w)
        rms = float(np.sqrt(np.average(err_c ** 2, weights=w)))
        if loss < best[0]:
            best = (loss, f0, B, rms)

    _, f0_best, B_best, rms_cents = best
    used = int(len(partials))
    return float(f0_best), float(B_best), float(rms_cents), used


# ---------------------------
# Filtre utilitaire
# ---------------------------
def bandpass_filter(x: np.ndarray, sr: int, low: float, high: float, order: int = 6) -> np.ndarray:
    nyq = sr * 0.5
    lowc = max(1e-6, low / nyq)
    highc = min(0.99, high / nyq)
    b, a = butter(order, [lowc, highc], btype="band")
    return filtfilt(b, a, x)


# ---------------------------
# API haut-niveau: raffinement à partir du signal
# ---------------------------
def refine_f0_B_from_signal(
    signal: np.ndarray,
    sr: int,
    f0_seed: float,
    nb_partials: int = 6,
    n_fft: int = 8192,
    hop_length: int = 512,
    search_width_cents: float = 80.0,
) -> dict:
    """
    Pipeline complet:
      1) extraction de partiels autour de k*f0_seed,
      2) fit conjoint (f0, B).
    Retourne un dict pratique pour le debug-panel.
    """
    _, parts, _ = compute_partials_fft_peaks(
        signal, sr, f0_seed,
        nb_partials=nb_partials,
        n_fft=n_fft, hop_length=hop_length,
        search_width_cents=search_width_cents, pad_factor=2
    )
    if not parts:
        return {
            "f0_refined": f0_seed,
            "B": 0.0,
            "rms_cents": 0.0,
            "n_partials": 0,
            "partials": []
        }

    f0_refined, B, rms_c, n_used = fit_f0_B(parts, f0_seed)
    return {
        "f0_refined": f0_refined,
        "B": B,
        "rms_cents": rms_c,
        "n_partials": n_used,
        "partials": [{"f": float(f), "amp": float(a), "snr": float(s), "k": int(k)} for (f, a, s, k) in parts],
    }