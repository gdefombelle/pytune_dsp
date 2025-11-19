# pytune_dsp/analysis/f0_HP_v2.py

import math
import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


def _parabolic_peak_1d(y: np.ndarray) -> float:
    """
    Interpolation quadratique sur 3 points autour du maximum local.

    Args:
        y: tableau 1D de 3 points [y[k-1], y[k], y[k+1]]

    Returns:
        delta: offset (en bins) par rapport au centre (index 1), typiquement [-1, 1]
    """
    if y.shape[0] != 3:
        raise ValueError("Parabolic interpolation needs 3 points.")

    a, b, c = float(y[0]), float(y[1]), float(y[2])
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-18:
        return 0.0

    # Formule classique : delta = 0.5 * (a - c) / (a - 2b + c)
    delta = 0.5 * (a - c) / denom
    # On évite les gros dérapages numériques
    return float(np.clip(delta, -1.0, 1.0))


def compute_f0_HP_v2(
    signal: np.ndarray,
    sr: int,
    f0_seed: float,
    window_factor: int = 4,
    max_shift_cents: float = 25.0,
    neighborhood_bins: int = 8,
    debug: bool = False,
):
    """
    Raffinement haute précision de f₀ à partir d'un seed déjà fiable.

    Idée :
      - FFT avec fenêtre Hann + zero-padding (window_factor)
      - on prend le bin de FFT le plus proche de f0_seed
      - interpolation quadratique sur [k-1, k, k+1] pour affiner la position du pic
      - on calcule la confiance comme contraste pic / médiane locale
      - SI le déplacement dépasse max_shift_cents → on garde f0_seed

    Args:
        signal: signal mono (np.ndarray)
        sr: sample rate (Hz)
        f0_seed: estimation initiale f₀ (Hz), supposée déjà bonne
        window_factor: facteur de zero-padding (4 → résolution x4)
        max_shift_cents: déplacement max autorisé entre seed et f_refined
        neighborhood_bins: nb de bins autour de k0 pour évaluer le bruit de fond
        debug: affiche un plot local si matplotlib dispo

    Returns:
        f0_refined (Hz), confidence [0,1]
    """

    # Sécurité basique
    if f0_seed is None or f0_seed <= 0 or sr <= 0:
        return None, 0.0

    y = np.asarray(signal, dtype=np.float64)
    if y.ndim != 1 or y.size < 256:
        # Pas assez de data pour un raffinement sérieux
        return f0_seed, 0.0

    # Centrage + fenêtre
    y = y - np.mean(y)
    n = y.size
    win = get_window("hann", n, fftbins=True)
    yw = y * win

    # FFT haute résolution
    n_fft = int(2 ** math.ceil(math.log2(n * window_factor)))
    spec = np.abs(rfft(yw, n_fft))
    freqs = rfftfreq(n_fft, 1.0 / sr)

    # Bin le plus proche du seed
    df = sr / n_fft
    k0 = int(round(f0_seed / df))

    # Garde-fous indices
    if k0 <= 1:
        k0 = 1
    if k0 >= spec.size - 2:
        k0 = spec.size - 2

    # Valeurs locales pour l'interpolation quadratique
    local3 = spec[k0 - 1 : k0 + 2]  # [k0-1, k0, k0+1]
    delta_bins = _parabolic_peak_1d(local3)
    k_peak = k0 + delta_bins

    f0_refined = float(k_peak * df)

    # Confiance : contraste pic / médiane locale
    k_min = max(0, k0 - neighborhood_bins)
    k_max = min(spec.size, k0 + neighborhood_bins + 1)
    local_env = spec[k_min:k_max]

    peak_mag = float(local3[1])
    # pour la confiance on prend la médiane de l'environnement
    median_mag = float(np.median(local_env)) if local_env.size > 0 else 0.0

    if median_mag <= 0:
        conf = 0.0
    else:
        ratio = (peak_mag - median_mag) / (peak_mag + 1e-12)
        conf = float(np.clip(ratio, 0.0, 1.0))

    # Vérifier que la dérive reste raisonnable
    if f0_refined <= 0:
        # Invalide → retour au seed
        f0_refined = f0_seed
    else:
        cents_shift = 1200.0 * math.log2(f0_refined / f0_seed)
        if abs(cents_shift) > max_shift_cents:
            # Trop de déplacement → on considère que c'est un faux pic
            # et on revient au seed, en baissant un peu la confiance
            print(
                f"[f0_HP_v2] WARNING: seed={f0_seed:.3f} Hz → candidate={f0_refined:.3f} Hz "
                f"(Δ={cents_shift:+.1f} cents > {max_shift_cents}c) → seed kept."
            )
            f0_refined = f0_seed
            conf = min(conf, 0.5)

    if debug:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 3))
            plt.plot(freqs[k_min:k_max], spec[k_min:k_max], label="Local spectrum")
            plt.axvline(f0_seed, color="blue", linestyle="--", label=f"seed {f0_seed:.2f} Hz")
            plt.axvline(f0_refined, color="red", linestyle="-", label=f"refined {f0_refined:.2f} Hz")
            plt.legend()
            plt.title("f₀ High Precision – v2")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    delta_hz = f0_refined - f0_seed
    print(
        f"[f0_HP_v2] seed={f0_seed:.3f} Hz → refined={f0_refined:.3f} Hz "
        f"| Δ={delta_hz:+.3f} Hz | conf={conf:.2f}"
    )

    return f0_refined, conf