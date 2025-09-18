import numpy as np
import librosa
from scipy.signal import find_peaks

def estimate_f0_and_harmonics_fft(
    y: np.ndarray,
    sr: int,
    f0_expected: float,
    n_harmonics: int = 6,
    method: str = "peaks",
    debug: bool = False,
):
    """
    Détecte F0 et ses harmoniques principaux à partir du spectre FFT.

    Parameters
    ----------
    y : np.ndarray
        Signal temporel mono.
    sr : int
        Sample rate.
    f0_expected : float
        Fréquence attendue de la note (Hz).
    n_harmonics : int
        Nombre d’harmoniques à rechercher.
    method : {"peaks", "slopes"}
        Algorithme de détection des maxima.
    debug : bool
        Si True, trace le spectre avec matplotlib.

    Returns
    -------
    f0_detected : float
        Fondamentale estimée (Hz).
    harmonics : list[float]
        Liste des fréquences d’harmoniques détectées.
    """
    # Trim silence
    y, _ = librosa.effects.trim(y)

    # FFT
    spectrum = np.fft.fft(y)
    spectrum = np.abs(spectrum) / np.max(np.abs(spectrum))
    n = len(spectrum)
    freqs = np.linspace(0, sr, n)

    # Fenêtre fréquentielle [f0/2, f0*(n_harmonics+1)]
    mask = (freqs > f0_expected / 2) & (freqs < f0_expected * (n_harmonics + 1))
    freqs = freqs[mask]
    amps = spectrum[mask]

    if method == "peaks":
        idx, _ = find_peaks(amps, height=0.05)  # seuil simple
        peaks = freqs[idx]
    elif method == "slopes":
        grad = np.gradient(amps)
        sign_changes = np.where(np.diff(np.sign(grad)))[0]
        peaks = freqs[sign_changes]
    else:
        raise ValueError("method must be 'peaks' or 'slopes'")

    if len(peaks) == 0:
        return None, []

    f0_detected = float(min(peaks))
    harmonics = [float(p) for p in peaks if p > f0_detected][:n_harmonics]

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, amps)
        plt.scatter(peaks, amps[np.searchsorted(freqs, peaks)], color="red")
        plt.axvline(f0_expected, color="green", linestyle="--", label="f0 expected")
        plt.axvline(f0_detected, color="blue", linestyle="--", label="f0 detected")
        plt.legend()
        plt.show()

    return f0_detected, harmonics