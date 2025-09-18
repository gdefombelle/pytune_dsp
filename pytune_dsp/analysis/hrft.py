import numpy as np
import matplotlib.pyplot as plt

def hrft(signal: np.ndarray, n_fft: int = 4096, window: str = "hann") -> np.ndarray:
    """
    Approximation d'une transformée de Fourier haute résolution (HRFT).
    
    Paramètres
    ----------
    signal : np.ndarray
        Signal temporel (mono).
    n_fft : int
        Taille de la FFT (zero-padding si > len(signal)).
    window : str
        Fenêtre à appliquer ("hann", "blackmanharris", ...).

    Retour
    ------
    spectrum : np.ndarray
        Amplitude spectrale haute résolution.
    """
    # Fenêtrage
    if window:
        win = signal * getattr(np, window)(len(signal))
    else:
        win = signal

    # Zero-padding + FFT
    spectrum = np.fft.fft(win, n=n_fft)
    spectrum = np.abs(spectrum[: n_fft // 2])  # moitié utile (0–Nyquist)
    
    return spectrum


# Exemple d’utilisation
if __name__ == "__main__":
    sr = 1000  # Hz
    t = np.arange(0, 1, 1 / sr)
    signal = np.sin(2 * np.pi * 123.45 * t)  # sinus à 123.45 Hz

    spectrum = hrft(signal, n_fft=16384, window="hanning")

    freqs = np.linspace(0, sr / 2, len(spectrum))
    plt.plot(freqs, spectrum)
    plt.title("HRFT - Résolution accrue")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.show()