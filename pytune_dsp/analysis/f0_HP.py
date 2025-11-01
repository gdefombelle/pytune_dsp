# pytune_dsp/analysis/f0_HP.py
import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


def compute_f0_HP(signal: np.ndarray, sr: int, f0_seed: float, window_factor: int = 4, debug: bool = False):
    """
    Raffine la fréquence fondamentale (f₀) à partir d'une estimation initiale (seed).
    Basé sur FFT à haute résolution et interpolation quadratique.

    Args:
        signal: signal audio mono (np.ndarray)
        sr: sample rate (Hz)
        f0_seed: estimation initiale de la fréquence fondamentale (Hz)
        window_factor: facteur de zero-padding (4 = résolution 4× meilleure)
        debug: affiche un graphique local du raffinement (matplotlib requis)

    Returns:
        f0_refined: fréquence f₀ raffinée (Hz)
        confidence: estimation de confiance [0,1]
    """

    if f0_seed <= 0 or len(signal) < 256:
        return None, 0.0

    # --- Préparation du signal ---
    y = np.asarray(signal, dtype=np.float64)
    y = y - np.mean(y)
    n = len(y)
    win = get_window("hann", n)
    yw = y * win

    # --- FFT haute résolution ---
    n_fft = int(n * window_factor)
    spec = np.abs(rfft(yw, n_fft))
    freqs = rfftfreq(n_fft, 1.0 / sr)

    # --- Fenêtre locale autour de f0_seed ---
    search_bw = f0_seed * 0.15  # ±15 %
    mask = (freqs >= f0_seed - search_bw) & (freqs <= f0_seed + search_bw)
    local_freqs = freqs[mask]
    local_spec = spec[mask]

    if len(local_freqs) < 5:
        return f0_seed, 0.0

    # --- Pic local maximum ---
    idx_max = np.argmax(local_spec)
    if idx_max == 0 or idx_max == len(local_spec) - 1:
        f_peak = local_freqs[idx_max]
    else:
        # Interpolation quadratique locale
        alpha, beta, gamma = local_spec[idx_max - 1], local_spec[idx_max], local_spec[idx_max + 1]
        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-12)
        f_peak = local_freqs[idx_max] + p * (local_freqs[1] - local_freqs[0])

    # --- Confiance = rapport pic / médiane locale ---
    base = np.median(local_spec)
    conf = float(np.clip((local_spec[idx_max] - base) / (local_spec[idx_max] + 1e-12), 0, 1))

    f0_refined = float(f_peak)

    if debug:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 3))
            plt.plot(local_freqs, local_spec, color="gray", label="Local spectrum")
            plt.axvline(f0_seed, color="blue", linestyle="--", label=f"seed {f0_seed:.2f} Hz")
            plt.axvline(f0_refined, color="red", linestyle="-", label=f"refined {f0_refined:.2f} Hz")
            plt.title("f₀ High Precision Refinement")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    print(f"[f0_HP] seed={f0_seed:.2f} → refined={f0_refined:.2f} Hz | Δ={f0_refined - f0_seed:+.3f} Hz | conf={conf:.2f}")
    return f0_refined, conf