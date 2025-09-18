"""
partials.py
===========
Outils pour analyser les partiels et l’inharmonicité d’une note.
Inclut des méthodes via STFT et un prototype de Yin récursif.
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

from pytune_dsp.types.analysis import NoteDeviationAnalysis
from pytune_dsp.types.analysis import Keyboard  # si tu as remplacé TemperedKeyboard

# -------------------------------
# Partials via STFT
# -------------------------------

def compute_partials_stft(
    nb_partials: int,
    note_analysis: NoteDeviationAnalysis,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[tuple[float, float]]:
    """
    Estime les partiels d’une note par STFT.

    Parameters
    ----------
    nb_partials : int
        Nombre de partiels à calculer.
    note_analysis : NoteDeviationAnalysis
        Analyse contenant le signal et la meilleure F0.
    sr : int
        Sample rate.
    n_fft : int
        Taille de FFT (par défaut 2048).
    hop_length : int
        Décalage entre fenêtres.

    Returns
    -------
    partials : list of (freq, amplitude)
        Liste des fréquences et amplitudes des partiels.
    """
    f0 = note_analysis.measurements.best_measurement
    if f0 <= 0:
        return []

    # STFT
    S = np.abs(librosa.stft(note_analysis.signal, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])

    partials = []
    for i in range(1, nb_partials + 1):
        target = i * f0
        idx = np.argmin(np.abs(freqs - target))
        amp = float(np.max(S[idx, :]))
        partials.append((freqs[idx], amp))

    return partials


# -------------------------------
# Inharmonicité simple
# -------------------------------

def calculate_inharmonicity(note_analysis: NoteDeviationAnalysis) -> list[float]:
    """
    Calcule l’inharmonicité en cents entre les partiels et la F0.

    Parameters
    ----------
    note_analysis : NoteDeviationAnalysis
        Analyse contenant F0 et partiels.

    Returns
    -------
    inharmonicity : list of float
        Déviation en cents de chaque partiel par rapport à la F0.
    """
    f0 = note_analysis.measurements.best_measurement
    if f0 <= 0 or not note_analysis.partials:
        return []

    return [1200 * np.log2(freq / f0) for freq, _ in note_analysis.partials]


# -------------------------------
# Recursive Yin (prototype)
# -------------------------------

def compute_partials_recursive_yin(
    kbd: Keyboard,
    nb_partials: int,
    note_analysis: NoteDeviationAnalysis,
    sr: int,
) -> tuple[list[float], np.ndarray, list[float]]:
    """
    Estimation des partiels via Yin récursif :
    on filtre successivement autour de chaque harmonique
    et on applique Yin dans cette bande.

    Parameters
    ----------
    kbd : Keyboard
        Clavier de référence pour calculer les bornes de recherche.
    nb_partials : int
        Nombre de partiels à analyser.
    note_analysis : NoteDeviationAnalysis
        Signal + F0.
    sr : int
        Sample rate.

    Returns
    -------
    harmonics : list[float]
        Harmoniques idéales (k*f0).
    partials : np.ndarray
        Fréquences estimées des partiels.
    inharmonicity : list[float]
        Déviations en demi-tons (ou cents si adapté).
    """
    f0 = note_analysis.measurements.best_measurement
    if f0 <= 0:
        return [], np.array([]), []

    signal = note_analysis.signal
    partials = np.zeros(nb_partials)
    harmonics = [f0 * i for i in range(1, nb_partials + 1)]

    for i, target in enumerate(harmonics, start=1):
        fmin = kbd.add_semitones_to_note(target, -1)
        fmax = kbd.add_semitones_to_note(target, +1)

        # filtrage passe-bas centré autour de l’harmonique
        filtered_signal = lowpass_filter(signal, sr, cutoff_freq=2 * target)

        Y = librosa.yin(filtered_signal, fmin=fmin, fmax=fmax, sr=sr)
        if len(Y) > 0:
            partials[i - 1] = Y[0]  # TODO: améliorer (moyenne stable etc.)

    inharmonicity = [kbd.distance_semitone(f1, f2) for f1, f2 in zip(partials, harmonics)]
    return harmonics, partials, inharmonicity


# -------------------------------
# Filtrage
# -------------------------------

def lowpass_filter(signal: np.ndarray, sr: int, cutoff_freq: float, order: int = 6) -> np.ndarray:
    """
    Applique un filtre passe-bas Butterworth.

    Parameters
    ----------
    signal : np.ndarray
        Signal temporel.
    sr : int
        Sample rate.
    cutoff_freq : float
        Fréquence de coupure (Hz).
    order : int
        Ordre du filtre.

    Returns
    -------
    filtered : np.ndarray
        Signal filtré.
    """
    b, a = butter(order, cutoff_freq / (sr / 2), btype="low")
    return filtfilt(b, a, signal)