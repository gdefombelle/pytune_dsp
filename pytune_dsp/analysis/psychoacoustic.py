"""
psychoacoustic.py
=================
Modèles psychoacoustiques et d’étirement des octaves (Fletcher-Munson, Helmholtz).
"""

import numpy as np
import librosa
from pytune_dsp.types.enums import StretchingModel


A4_HZ = 440.0
CENTS_PER_OCTAVE = 1200.0


def note_to_midi(note: str) -> int:
    """Convertit une note (ex: 'F4') en numéro MIDI."""
    return librosa.note_to_midi(note)


def fletcher_munson(note: str, cents_offset: float = 0.0) -> float:
    """
    Fréquence d’une note avec le modèle Fletcher-Munson.

    Parameters
    ----------
    note : str
        Nom de la note (ex: "F4").
    cents_offset : float
        Décalage en cents (par défaut 0).

    Returns
    -------
    f : float
        Fréquence corrigée.
    """
    midi_number = note_to_midi(note)
    cents = (midi_number - 69) * 100 + cents_offset
    f = A4_HZ * 2 ** (cents / CENTS_PER_OCTAVE)
    return f * (1 + 0.0007 * f ** 2)


def helmholtz(note: str, cents_offset: float = 0.0) -> float:
    """
    Fréquence d’une note avec le modèle Helmholtz.

    Parameters
    ----------
    note : str
        Nom de la note (ex: "F4").
    cents_offset : float
        Décalage en cents.

    Returns
    -------
    f : float
        Fréquence corrigée.
    """
    midi_number = note_to_midi(note)
    cents = (midi_number - 69) * 100 + cents_offset
    f = A4_HZ * 2 ** (cents / CENTS_PER_OCTAVE)
    return f * (1 + 0.002 * f ** 2)


def apply_stretching(note: str, model: StretchingModel) -> float:
    """
    Applique un modèle de stretching choisi.

    Parameters
    ----------
    note : str
        Nom de la note.
    model : StretchingModel
        Modèle (NO_STRETCHING, FLETCHER_MUNSON, HELMHOLTZ).

    Returns
    -------
    f : float
        Fréquence résultante.
    """
    if model == StretchingModel.NO_STRETCHING:
        return librosa.note_to_hz(note)
    elif model == StretchingModel.FLETCHER_MUNSON:
        return fletcher_munson(note)
    elif model == StretchingModel.HELMHOLTZ:
        return helmholtz(note)
    else:
        raise NotImplementedError(f"{model} not supported yet")