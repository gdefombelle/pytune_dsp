from enum import Enum

class Temperament(str, Enum):
    EQUAL = "Equal Temperament"
    JUST = "Just Intonation"
    PYTHAGOREAN = "Pythagorean"
    MEANTONE = "Meantone (1/4 comma)"
    WELL_WERCKMEISTER = "Well Temperament (Werckmeister)"
    WELL_KIRNBERGER = "Well Temperament (Kirnberger)"
    WELL_VALLOTTI = "Well Temperament (Vallotti)"
    RAILSBACK = "Railsback Adjusted Equal Temperament"

    def description(self) -> str:
        return {
            Temperament.EQUAL: "Modern equal temperament: 12x 100 cents. Octaves pure, others slightly compromised.",
            Temperament.JUST: "Intervals based on pure harmonic ratios (3/2, 5/4). Only good in limited keys.",
            Temperament.PYTHAGOREAN: "Fifths pure, but thirds very wide. Medieval/modal system.",
            Temperament.MEANTONE: "Renaissance system: pure thirds, some unusable 'wolf' intervals.",
            Temperament.WELL_WERCKMEISTER: "Well temperament by Werckmeister: all keys usable, but unequal.",
            Temperament.WELL_KIRNBERGER: "Well temperament by Kirnberger: used by Bach, each key has a color.",
            Temperament.WELL_VALLOTTI: "Well temperament by Vallotti: smoother distribution of errors.",
            Temperament.RAILSBACK: "Equal temperament stretched using piano-specific Railsback curve.",
        }[self]