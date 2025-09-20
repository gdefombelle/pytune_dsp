import numpy as np
import librosa


def yin_window(expected_freq: float, semitones: float = 0.5) -> tuple[float, float]:
    """
    Donne (fmin, fmax) centrés sur la fréquence attendue, avec une marge en demi-tons.

    Parameters
    ----------
    expected_freq : float
        Fréquence cible (Hz).
    semitones : float, default=0.5
        Marge autour de la fréquence cible (en demi-tons).
        Exemple: 0.5 = ± un demi-ton, 1.0 = ± un ton.

    Returns
    -------
    fmin, fmax : tuple[float, float]
        Intervalle de recherche pour YIN.
    """
    ratio = 2 ** (semitones / 12.0)
    fmin = expected_freq / ratio
    fmax = expected_freq * ratio
    return fmin, fmax


def yin_with_adaptive_window(
    signal: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    idx_note: int | None = None,
    total_notes: int | None = None,
) -> np.ndarray:
    """
    YIN avec choix intelligent de la taille de fenêtre selon la fréquence.

    Parameters
    ----------
    signal : np.ndarray
        Signal audio mono (1D).
    sr : int
        Sample rate.
    fmin : float
        Fréquence minimale attendue (Hz).
    fmax : float
        Fréquence maximale attendue (Hz).
    idx_note : int | None
        Index de la note dans l’échelle (A0=0, A#0=1...).
    total_notes : int | None
        Nombre total de notes, utile pour ajustement heuristique.

    Returns
    -------
    f0s : np.ndarray
        Estimations de fréquence fondamentale par trames.
    """
    # Par défaut : fenêtre basée sur fmin
    frame_length = int((2 * sr) / fmin)
    win_length = frame_length // 2

    # Heuristique : ajustement selon la tessiture
    if idx_note is not None and total_notes is not None:
        if 0 <= idx_note <= 25:  # très graves
            frame_length, win_length = 8192, 4096
        elif 40 < idx_note <= min(50, total_notes):  # graves-médiums
            frame_length, win_length = 4096, 2048
        else:  # médiums-aigus
            frame_length, win_length = 1024, 512

    return librosa.yin(
        signal,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        win_length=win_length,
    )


def yin_track(
    signal: np.ndarray,
    sr: int,
    expected_freq: float,
    semitones: float = 0.5,
    idx_note: int | None = None,
    total_notes: int | None = None,
) -> np.ndarray:
    """
    YIN complet : calcule (fmin, fmax) autour de la fréquence attendue
    puis appelle YIN avec fenêtre adaptative.

    Parameters
    ----------
    signal : np.ndarray
        Signal audio mono.
    sr : int
        Sample rate.
    expected_freq : float
        Fréquence attendue (Hz).
    semitones : float, default=0.5
        Marge autour de la fréquence attendue (en demi-tons).
    idx_note : int | None
        Index de la note dans l’échelle (optionnel).
    total_notes : int | None
        Nombre total de notes (optionnel).

    Returns
    -------
    f0s : np.ndarray
        Estimations de fréquence fondamentale par trames.
    """
    fmin, fmax = yin_window(expected_freq, semitones=semitones)
    return yin_with_adaptive_window(
        signal, sr, fmin, fmax, idx_note=idx_note, total_notes=total_notes
    )