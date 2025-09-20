from dataclasses import dataclass
from typing import Dict
import numpy as np
import librosa
import math
from pytune_dsp.types.analysis import Keyboard
from pytune_dsp.types.enums import A4, StretchingModel, TuningMethod





# ---------- Scale generation ----------
def generate_equal_tempered(
    A4_hz: float = A4.STANDARD.value,
    lower_note: str = "A0",
    upper_note: str = "C8",
    stretching: StretchingModel = StretchingModel.NO_STRETCHING,
    start_stretching_at: str = "A4",
    simple_factor: float = 1.0003,  # ex: simple stretching factor
) -> Keyboard:
    """Construit un clavier tempéré (éventuellement stretché)."""
    lower_midi = librosa.note_to_midi(lower_note)
    upper_midi = librosa.note_to_midi(upper_note)

    # A4 en MIDI
    midi_A4 = librosa.note_to_midi("A4")

    freqs = {}
    for midi in range(lower_midi, upper_midi + 1):
        if midi == midi_A4:
            f = A4_hz
        else:
            f = A4_hz * (2 ** ((midi - midi_A4) / 12))

        freqs[librosa.midi_to_note(midi)] = f

    # TODO appliquer stretching en fonction de `stretching`

    # JND (just noticeable difference)
    jnd = {note: 21.4 * (f ** -0.57) for note, f in freqs.items()}

    # nombre de cordes par touche (simplifié)
    strings = {}
    for note in freqs.keys():
        if librosa.note_to_midi(note) < librosa.note_to_midi("D#1"):
            strings[note] = 1
        elif librosa.note_to_midi(note) < librosa.note_to_midi("B1"):
            strings[note] = 2
        else:
            strings[note] = 3

    return Keyboard(
        A4=A4_hz,
        lower_note=lower_note,
        upper_note=upper_note,
        frequencies=freqs,
        jnd=jnd,
        strings_per_note=strings,
    )