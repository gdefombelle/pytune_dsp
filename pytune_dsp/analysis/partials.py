# pytune_dsp/analysis/partials.py

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

# --- interpolation quadratique autour d'un pic spectral (3 bins) ---
def _parabolic(freq_bins: np.ndarray, mags: np.ndarray, k: int) -> float:
    """
    Affine la fréquence du pic autour du bin k par fit parabolique (k-1,k,k+1).
    Retourne la fréquence estimée (Hz). Revient à freq_bins[k] si k est en bord
    ou si la parabole est dégénérée.
    """
    if k <= 0 or k >= len(mags) - 1:
        return float(freq_bins[k])
    m1, m2, m3 = mags[k - 1], mags[k], mags[k + 1]
    denom = (m1 - 2.0 * m2 + m3)
    if denom == 0.0:
        return float(freq_bins[k])
    delta = 0.5 * (m1 - m3) / denom  # décalage sous-bin (en bins)
    bin_step = freq_bins[1] - freq_bins[0]
    return float(freq_bins[k] + delta * bin_step)


def compute_partials_fft_peaks(
    signal: np.ndarray,
    sr: int,
    f0_ref: float,
    nb_partials: int = 8,
    n_fft: int = 8192,
    hop_length: int = 512,
    search_width_cents: float = 80.0,
    pad_factor: int = 2,
) -> tuple[list[float], list[tuple[float, float]], list[float]]:
    """
    Estime les partiels via FFT + recherche locale autour de k*f0_ref
    (± search_width_cents) et interpolation quadratique.

    Retourne (harmonics, partials, inharmonicity):
    - harmonics: [k*f0_ref] théoriques
    - partials: [(freq_hz, amplitude), ...]
    - inharmonicity: déviation en cents de chaque partiel vs k*f0_ref
    """
    if f0_ref <= 0.0 or signal.size == 0:
        return [], [], []

    # STFT (fenêtre Hann par défaut)
    S = np.abs(librosa.stft(signal, n_fft=n_fft * pad_factor, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft * pad_factor)

    # agrégation temporelle -> max (spectre global)
    spec = np.max(S, axis=1)

    harmonics = [k * f0_ref for k in range(1, nb_partials + 1)]
    partials: list[tuple[float, float]] = []
    inharm: list[float] = []

    # borne max exploitable (éviter Nyquist)
    f_max_usable = freqs[-1] * 0.995
    ratio = 2.0 ** (search_width_cents / 1200.0)

    for k, target in enumerate(harmonics, start=1):
        if target >= f_max_usable:
            break

        # fenêtre de recherche autour de target
        low = max(freqs[0], target / ratio)
        high = min(f_max_usable, target * ratio)
        if high <= low:
            break

        low_idx = np.searchsorted(freqs, low, side="left")
        high_idx = np.searchsorted(freqs, high, side="right")
        if high_idx - low_idx < 3:
            continue

        sub = spec[low_idx:high_idx]

        # --- logique spéciale pour k=1 ---
        if k == 1:
            # on force à prendre le bin le plus proche de f0_ref
            k0 = np.argmin(np.abs(freqs[low_idx:high_idx] - target)) + low_idx
        else:
            # on prend le maximum local dans la fenêtre
            rel_k = np.argmax(sub)
            k0 = low_idx + rel_k

        f_est = _parabolic(freqs, spec, k0)
        amp = float(spec[k0])

        # clip si hors fenêtre
        if not (low <= f_est <= high):
            f_est = float(freqs[k0])

        partials.append((f_est, amp))
        inharm.append(1200.0 * np.log2(f_est / target) if f_est > 0 else 0.0)

    return harmonics, partials, inharm


def bandpass_filter(x: np.ndarray, sr: int, low: float, high: float, order: int = 6) -> np.ndarray:
    """Filtre passe-bande Butterworth."""
    nyq = sr * 0.5
    lowc = max(1e-6, low / nyq)
    highc = min(0.99, high / nyq)
    b, a = butter(order, [lowc, highc], btype="band")
    return filtfilt(b, a, x)


def estimate_B(f0: float, partials_hz: list[float]) -> float:
    """
    Estime B via r_k^2 - 1 ≈ B k^2 (régression through-origin).
    """
    ks, y = [], []
    for k, fk in enumerate(partials_hz, start=1):
        if fk > 0.0 and f0 > 0.0:
            r = fk / (k * f0)
            ks.append(k)
            y.append(r * r - 1.0)
    if len(ks) < 2:
        return 0.0
    ks = np.asarray(ks, float)
    y = np.asarray(y, float)
    num = np.sum((ks ** 2) * y)
    den = np.sum(ks ** 4)
    return float(num / den) if den > 0.0 else 0.0