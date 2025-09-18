import numpy as np
import librosa
from scipy.signal import wiener
from typing import Tuple, Literal


def select_channel_strategy(
    y: np.ndarray,
    strategy: Literal["mono", "dominant", "parallel"] = "mono"
) -> np.ndarray:
    """
    Gère les signaux multi-canaux avant prétraitement.

    Args:
        y: np.ndarray
            Signal audio brut (1D mono ou 2D multi-canaux).
        strategy: str
            - "mono"     : moyenne des canaux (mix classique).
            - "dominant" : conserve uniquement le canal le plus fort.
            - "parallel" : conserve tous les canaux pour traitement séparé.

    Returns:
        np.ndarray:
            - Si "mono" ou "dominant" → vecteur 1D (mono).
            - Si "parallel" → matrice 2D (nb_canaux x nb_samples).
    """
    if y.ndim == 1:
        return y  # déjà mono

    if strategy == "mono":
        return librosa.to_mono(y)

    elif strategy == "dominant":
        # Calcul de l'énergie RMS par canal
        rms_per_channel = [np.sqrt(np.mean(chan**2)) for chan in y]
        dominant_idx = int(np.argmax(rms_per_channel))
        return y[dominant_idx]

    elif strategy == "parallel":
        # On garde tel quel (chaque canal sera traité indépendamment)
        return y

    else:
        raise ValueError(f"Unknown strategy '{strategy}'")


def trim_audio_signal(y: np.ndarray, top_db: float = 30.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Supprime les silences en début et fin du signal.
    Compatible multi-canaux si strategy="parallel".
    """
    if y.ndim == 1:
        return librosa.effects.trim(y, top_db=top_db)
    else:
        # Cas multi-canaux : on concatène les énergies pour trouver silence commun
        energy = np.mean(y**2, axis=0)
        non_silent = librosa.effects.trim(energy, top_db=top_db)[1]
        return y[:, non_silent[0]:non_silent[1]], non_silent


def shelving_equalization(y: np.ndarray, coef: float = 0.5) -> np.ndarray:
    """Boost basses et aigus (appliqué canal par canal si 2D)."""
    if y.ndim == 1:
        return librosa.effects.preemphasis(y, coef=coef) + librosa.effects.preemphasis(y, coef=-coef)
    else:
        return np.stack([shelving_equalization(chan, coef) for chan in y], axis=0)


def level_to_target(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalise le niveau global vers target dBFS."""
    def normalize_signal(sig):
        rms = np.sqrt(np.mean(sig**2))
        current_db = 20 * np.log10(rms + 1e-9)
        gain = 10 ** ((target_dbfs - current_db) / 20)
        return sig * gain

    if y.ndim == 1:
        return normalize_signal(y)
    else:
        return np.stack([normalize_signal(chan) for chan in y], axis=0)


def denoise_wiener(y: np.ndarray) -> np.ndarray:
    """Réduction de bruit (canal par canal si 2D)."""
    if y.ndim == 1:
        return wiener(y)
    else:
        return np.stack([wiener(chan) for chan in y], axis=0)