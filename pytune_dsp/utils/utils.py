import numpy as np
from pytune_dsp.core.keyboard import Keyboard


def cents_between(f1: float, f2: float) -> float:
    return 1200 * np.log2(f2 / f1)

def neighbor(freq: float, keyboard: Keyboard):
    """Retourne la note la plus proche dans le clavier."""
    diffs = {note: abs(f - freq) for note, f in keyboard.frequencies.items()}
    best_note = min(diffs, key=diffs.get)
    return best_note, keyboard.frequencies[best_note]

def add_semitones(freq: float, n: int) -> float:
    return freq * (2 ** (n / 12))