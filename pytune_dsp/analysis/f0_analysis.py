"""
f0_analysis.py
==============
Outils pour analyser une note (chunk audio unique) et en extraire une fréquence fondamentale stable.
Inclut analyse mono-canal et multi-canaux.
"""

from collections import Counter
import numpy as np
import librosa
from sklearn.cluster import DBSCAN

from pytune_dsp.types.analysis import NoteMeasurements


# ============================================================
# Helpers fondamentaux
# ============================================================

def stable_f0_detection(f0s: np.ndarray) -> tuple[float, float]:
    """
    Détecte la fréquence fondamentale stable dans une série de f0 estimés.
    
    Returns
    -------
    stable_avg : float
        Moyenne des valeurs autour de la fréquence modale.
    mode_rate : float
        Proportion d’occurrences de la fréquence modale (0..1).
    """
    f0s = np.asarray(f0s, dtype=float)
    f0s = f0s[np.isfinite(f0s)]
    if f0s.size == 0:
        return 0.0, 0.0

    # Arrondi pour limiter le bruit : précision dépend de la fréquence
    round_precision = 0 if f0s[0] >= 750 else (1 if f0s[0] >= 100 else 2)
    f0s_round = np.round(f0s, round_precision)

    mode, count = Counter(f0s_round).most_common(1)[0]
    indices = np.where(f0s_round == mode)[0]

    stable_avg = float(np.mean(f0s[indices]))
    mode_rate = count / f0s_round.size
    return stable_avg, mode_rate


def spectral_centroid(frequencies: np.ndarray, amplitudes: np.ndarray) -> float:
    """
    Calcule le centroïde fréquentiel pondéré par les amplitudes.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if frequencies.size == 0 or amplitudes.size == 0:
        return 0.0
    numer = float(np.sum(frequencies * amplitudes))
    denom = float(np.sum(amplitudes))
    return 0.0 if denom == 0.0 else numer / denom


def amplitude_at_frequencies(signal: np.ndarray, f0s: np.ndarray, sr: int) -> list[float]:
    """
    Estime l'amplitude d'un signal à des fréquences données via STFT.
    """
    D = np.abs(librosa.stft(signal))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])
    amps = []
    for f in f0s:
        if not np.isfinite(f) or f <= 0:
            amps.append(0.0)
            continue
        idx = np.argmin(np.abs(freqs - f))
        amps.append(float(np.max(D[idx, :])))
    return amps


# ============================================================
# Estimation YIN
# ============================================================

def estimate_f0_track(
    signal: np.ndarray,
    sr: int,
    fmin: float = 24.0,
    fmax: float = 4200.0,
    frame_length: int | None = None,
    win_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estime une trajectoire de f0 avec YIN et renvoie (f0s, times).
    """
    if frame_length is None or win_length is None:
        frame_length = 2048
        win_length = 1024

    f0s = librosa.yin(
        signal,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        win_length=win_length,
    )
    times = librosa.times_like(f0s, sr=sr)
    return f0s.astype(float), times


# ============================================================
# Analyse mono-canal
# ============================================================

def calculate_f0_measurements(
    f0s: np.ndarray,
    amplitudes: np.ndarray,
    tempered_freq: float,
    stretched_freq: float,
) -> NoteMeasurements:
    """
    Évalue une note en comparant les mesures de fréquence à la référence
    tempérée et à la référence stretchée.
    """
    m = NoteMeasurements()

    f0s = np.asarray(f0s, dtype=float)
    f0s = f0s[np.isfinite(f0s)]
    amps = np.asarray(amplitudes, dtype=float)

    if f0s.size == 0 or amps.size == 0:
        m.best_measurement = 0.0
        m.best_measurement_method = "none"
        return m

    stable_f0, mode_rate = stable_f0_detection(f0s)
    m.stableF0Average = stable_f0
    m.mode_rate = mode_rate

    centroid = spectral_centroid(f0s, amps)
    m.centroid = centroid

    if mode_rate >= 0.60 and stable_f0 > 0:
        m.best_measurement = stable_f0
        m.best_measurement_method = "stableF0Average"
    else:
        m.best_measurement = centroid
        m.best_measurement_method = "centroid"

    m.target_frequency_tempered = tempered_freq
    m.target_frequency_stretched = stretched_freq

    if stable_f0 > 0 and tempered_freq > 0:
        m.eq_tempered_deviation_cents = 1200 * np.log2(stable_f0 / tempered_freq)
    if stable_f0 > 0 and stretched_freq > 0:
        m.strectched_deviation_cents = 1200 * np.log2(stable_f0 / stretched_freq)

    return m


def analyze_single_channel(
    signal: np.ndarray,
    sr: int,
    tempered_freq: float,
    stretched_freq: float,
    yin_fmin: float = 24.0,
    yin_fmax: float = 4200.0,
    frame_length: int | None = None,
    win_length: int | None = None,
) -> tuple[NoteMeasurements, dict]:
    """
    Analyse un canal unique : YIN -> f0s -> amplitudes -> NoteMeasurements.
    """
    f0s, times = estimate_f0_track(
        signal,
        sr=sr,
        fmin=yin_fmin,
        fmax=yin_fmax,
        frame_length=frame_length,
        win_length=win_length,
    )
    amps = amplitude_at_frequencies(signal, f0s, sr)
    m = calculate_f0_measurements(f0s, np.asarray(amps), tempered_freq, stretched_freq)
    return m, {"f0s": f0s, "times": times, "amplitudes": np.asarray(amps)}


# ============================================================
# Analyse multi-canaux
# ============================================================

def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def combine_measurements_average_hz(
    per_channel_best_hz: list[float],
    tempered_freq: float,
) -> tuple[float, float]:
    """
    Combine plusieurs mesures en moyennant en cents (vs référence tempérée),
    puis re-projette en Hz.
    """
    valid = [f for f in per_channel_best_hz if np.isfinite(f) and f > 0]
    if not valid or tempered_freq <= 0:
        return 0.0, 0.0
    cents = [1200 * np.log2(f / tempered_freq) for f in valid]
    avg_cents = float(np.mean(cents))
    best_hz_avg = float(tempered_freq * (2.0 ** (avg_cents / 1200.0)))
    return best_hz_avg, avg_cents


def analyze_multichannel(
    signals: np.ndarray,
    sr: int,
    tempered_freq: float,
    stretched_freq: float,
    strategy: str = "dominant",  # "dominant" | "average"
    yin_fmin: float = 24.0,
    yin_fmax: float = 4200.0,
    frame_length: int | None = None,
    win_length: int | None = None,
) -> tuple[NoteMeasurements, dict]:
    """
    Analyse multi-canaux : sélectionne le canal dominant ou moyenne en cents.
    """
    sig = np.asarray(signals, dtype=float)

    # Mono → délègue
    if sig.ndim == 1:
        m, extras = analyze_single_channel(
            sig,
            sr,
            tempered_freq,
            stretched_freq,
            yin_fmin=yin_fmin,
            yin_fmax=yin_fmax,
            frame_length=frame_length,
            win_length=win_length,
        )
        return m, {"per_channel": [{"measurements": m, **extras, "rms": _rms(sig)}],
                   "chosen_channel": 0, "strategy": "mono"}

    if sig.ndim != 2:
        raise ValueError("signals must be 1D (mono) or 2D (channels, samples).")

    C = sig.shape[0]
    per_channel = []
    rms_vals = []

    # Analyse canal par canal
    for c in range(C):
        m_c, extras_c = analyze_single_channel(
            sig[c],
            sr,
            tempered_freq,
            stretched_freq,
            yin_fmin=yin_fmin,
            yin_fmax=yin_fmax,
            frame_length=frame_length,
            win_length=win_length,
        )
        r = _rms(sig[c])
        per_channel.append({"measurements": m_c, **extras_c, "rms": r})
        rms_vals.append(r)

    if strategy == "dominant":
        chosen = int(np.argmax(rms_vals)) if rms_vals else 0
        m = per_channel[chosen]["measurements"]
        return m, {"per_channel": per_channel, "chosen_channel": chosen, "strategy": "dominant"}

    elif strategy == "average":
        bests_hz = [pc["measurements"].best_measurement for pc in per_channel]
        best_avg_hz, avg_cents = combine_measurements_average_hz(bests_hz, tempered_freq)
        m = NoteMeasurements()
        m.best_measurement = best_avg_hz
        m.best_measurement_method = "multichannel_avg"
        m.target_frequency_tempered = tempered_freq
        m.target_frequency_stretched = stretched_freq
        if best_avg_hz > 0 and tempered_freq > 0:
            m.eq_tempered_deviation_cents = 1200 * np.log2(best_avg_hz / tempered_freq)
        if best_avg_hz > 0 and stretched_freq > 0:
            m.strectched_deviation_cents = 1200 * np.log2(best_avg_hz / stretched_freq)
        return m, {"per_channel": per_channel, "chosen_channel": None, "strategy": "average", "avg_cents": avg_cents}

    else:
        raise ValueError("strategy must be 'dominant' or 'average'")
    


def detect_multiple_f0s(
    f0s: np.ndarray,
    sr: int,
    eps_cents: float = 10.0,
    min_samples: int = 5,
) -> list[tuple[float, float]]:
    """
    Détecte plusieurs fréquences fondamentales stables dans une série de f0,
    pouvant indiquer plusieurs cordes désaccordées.

    Parameters
    ----------
    f0s : np.ndarray
        Tableau de fréquences estimées (Hz).
    sr : int
        Sample rate (non utilisé directement mais peut servir à pondérer).
    eps_cents : float
        Tolérance de regroupement en cents (ex: 10 cents).
    min_samples : int
        Minimum d’échantillons pour valider un cluster.

    Returns
    -------
    clusters : list of (mean_freq, weight)
        Liste des fréquences moyennes détectées et leur poids relatif (0..1).
    """
    f0s = np.asarray(f0s, dtype=float)
    f0s = f0s[np.isfinite(f0s) & (f0s > 20) & (f0s < 5000)]
    if f0s.size == 0:
        return []

    # Convertir en cents relatifs à la médiane
    ref = np.median(f0s)
    cents = 1200 * np.log2(f0s / ref)
    cents = cents.reshape(-1, 1)

    # Clustering par proximité en cents
    clustering = DBSCAN(eps=eps_cents, min_samples=min_samples).fit(cents)
    labels = clustering.labels_

    clusters = []
    for label in set(labels):
        if label == -1:  # bruit
            continue
        cluster_freqs = f0s[labels == label]
        mean_freq = float(np.mean(cluster_freqs))
        weight = cluster_freqs.size / f0s.size
        clusters.append((mean_freq, weight))

    # Trier par poids décroissant
    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters