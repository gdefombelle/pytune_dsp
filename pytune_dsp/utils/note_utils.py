import numpy as np

# Dictionnaire des noms en notation anglo-saxonne et solfège
NOTE_TO_MIDI = {
    # Anglo-saxon
    "C": 0,
    "C#": 1, "C♯": 1, "Db": 1, "D♭": 1,
    "D": 2,
    "D#": 3, "D♯": 3, "Eb": 3, "E♭": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "F♯": 6, "Gb": 6, "G♭": 6,
    "G": 7,
    "G#": 8, "G♯": 8, "Ab": 8, "A♭": 8,
    "A": 9,
    "A#": 10, "A♯": 10, "Bb": 10, "B♭": 10,
    "B": 11,

    # Solfège latin
    "Do": 0,
    "Do#": 1, "Do♯": 1, "Réb": 1, "Ré♭": 1,
    "Ré": 2,
    "Ré#": 3, "Ré♯": 3, "Mib": 3, "Mi♭": 3,
    "Mi": 4,
    "Fa": 5,
    "Fa#": 6, "Fa♯": 6, "Solb": 6, "Sol♭": 6,
    "Sol": 7,
    "Sol#": 8, "Sol♯": 8, "Lab": 8, "La♭": 8,
    "La": 9,
    "La#": 10, "La♯": 10, "Sib": 10, "Si♭": 10,
    "Si": 11,
}

def note_to_midi(note: str) -> int:
    """
    Convertit une note (ex: 'A4', 'C#3', 'Do♯4', 'Réb5') en numéro MIDI.
    """
    # Séparer nom de note et octave
    name = ''.join([c for c in note if not c.isdigit()])
    octave = int(''.join([c for c in note if c.isdigit()]))

    if name not in NOTE_TO_MIDI:
        raise ValueError(f"Note inconnue: {note}")

    return 12 * (octave + 1) + NOTE_TO_MIDI[name]

def midi_to_freq(midi: int, a4: float = 440.0) -> float:
    """Convertit un numéro MIDI en fréquence (Hz)."""
    return a4 * 2 ** ((midi - 69) / 12)

def note_to_freq(note: str, a4: float = 440.0) -> float:
    """Convertit une note (ex: 'A4', 'Do♯4') en fréquence (Hz)."""
    return midi_to_freq(note_to_midi(note), a4)

def freq_to_midi(freq: float, a4: float = 440.0) -> int:
    """Convertit une fréquence (Hz) en numéro MIDI arrondi."""
    return int(round(69 + 12 * np.log2(freq / a4)))

def freq_to_note(
    freq: float,
    a4: float = 440.0,
    use_solfège: bool = False,
    tol_cents: float = 50.0,
) -> str:
    """
    Convertit une fréquence en nom de note.
    
    - Par défaut en notation anglo-saxonne ('A4')
    - Si use_solfège=True → notation latine ('La4')
    - tol_cents: tolérance en cents pour l’arrondi
    
    Exemple: 445 Hz → 'A4' (écart ~20 cents)
    """
    # Calcul MIDI "flottant"
    midi_exact = 69 + 12 * np.log2(freq / a4)
    midi_rounded = int(round(midi_exact))
    cents_off = 1200 * np.log2(freq / midi_to_freq(midi_rounded, a4))

    # Vérifie si dans la tolérance
    if abs(cents_off) > tol_cents:
        raise ValueError(
            f"Fréquence {freq:.2f} Hz trop éloignée de toute note (écart {cents_off:.1f} cents)"
        )
    """
        freq_to_note(440)             # "A4"
        freq_to_note(440, True)       # "La4"
        freq_to_note(445)             # "A4" (écart ~20 cents)
        freq_to_note(455)             # ValueError (écart ~60 cents > tolérance)
        freq_to_note(277, True)       # "Do♯4"
            """
    octave = midi_rounded // 12 - 1
    pitch_class = midi_rounded % 12

    if use_solfège:
        mapping = {
            0: "Do", 1: "Do♯", 2: "Ré", 3: "Mi♭", 4: "Mi",
            5: "Fa", 6: "Fa♯", 7: "Sol", 8: "Sol♯", 9: "La",
            10: "Si♭", 11: "Si",
        }
    else:
        mapping = {
            0: "C", 1: "C♯", 2: "D", 3: "E♭", 4: "E",
            5: "F", 6: "F♯", 7: "G", 8: "G♯", 9: "A",
            10: "B♭", 11: "B",
        }

    return f"{mapping[pitch_class]}{octave}"

# 🔹 Nouveau : obtenir triplet (midi, nom, freq) directement
def freq_to_midi_note_freq(freq: float, a4: float = 440.0, use_solfège: bool = False):
    midi = freq_to_midi(freq, a4)
    note = freq_to_note(freq, a4, use_solfège=use_solfège)
    return midi, note, midi_to_freq(midi, a4)

def midi_to_note(midi: int, use_solfège: bool = False) -> str:
    """
    Convertit un numéro MIDI en nom de note :
      - 60 → C4 ou Do4
      - 69 → A4 ou La4
      - 56 → G#3 ou Sol♯3

    use_solfège=True : notation latine
    """

    if not (0 <= midi <= 127):
        raise ValueError(f"MIDI invalide: {midi}")

    octave = midi // 12 - 1
    pitch_class = midi % 12

    if use_solfège:
        mapping = {
            0: "Do",  1: "Do♯", 2: "Ré",  3: "Mi♭", 4: "Mi",
            5: "Fa",  6: "Fa♯", 7: "Sol", 8: "Sol♯", 9: "La",
            10: "Si♭", 11: "Si",
        }
    else:
        mapping = {
            0: "C",  1: "C♯", 2: "D",  3: "E♭", 4: "E",
            5: "F",  6: "F♯", 7: "G", 8: "G♯", 9: "A",
            10: "B♭", 11: "B",
        }

    return f"{mapping[pitch_class]}{octave}"