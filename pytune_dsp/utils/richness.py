import numpy as np
from numpy.fft import rfft, rfftfreq


def richness_score(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 1000.0
) -> float:
    """
    Calcule un score de "richesse harmonique" d'un signal audio.

    L'idée est de mesurer la proportion d'énergie spectrale située au-dessus
    d'une fréquence seuil (par défaut 1000 Hz). Cela permet de privilégier
    les signaux qui contiennent plus d'aigus et d'harmoniques, 
    indicateur d'une meilleure qualité de capture pour l'analyse de piano.

    Args:
        signal (np.ndarray): Le signal audio mono (Float32 ou Float64).
        sample_rate (int): Taux d'échantillonnage du signal (en Hz).
        cutoff_hz (float, optional): Seuil en Hz à partir duquel
            on considère les hautes fréquences. Par défaut 1000 Hz.

    Returns:
        float: Score de richesse harmonique, compris entre 0 et 1.
            - 0.0 → aucune énergie au-dessus de cutoff_hz
            - 1.0 → toute l'énergie est au-dessus de cutoff_hz
            - En pratique, un bon micro aura un score plus élevé qu’un micro médiocre.
    """
    if signal is None or len(signal) == 0:
        return 0.0
    
    if signal.ndim > 1:
        # Sécurité : moyenne des canaux si stéréo
        signal = np.mean(signal, axis=0)

    # Transformée de Fourier rapide (partie positive)
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=1.0 / sample_rate)

    # Énergies spectrales
    total_energy = np.sum(spectrum ** 2)
    if total_energy <= 0:
        return 0.0

    high_energy = np.sum(spectrum[freqs > cutoff_hz] ** 2)

    return float(high_energy / total_energy)