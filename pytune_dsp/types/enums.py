from enum import Enum

class SampleType(Enum):
    FILE_CHROMATIC_SCALE = 0
    FILE_DIATONIC_SCALE = 1
    FILE_SINGLE_NOTE = 2
    FILE_TWO_NOTES = 3
    TRIAD = 4
    FILE_PARTITION = 5
    FILE_INTERVAL = 6
    NONE = 99

class TuningMethod(Enum):
    EQUAL_TEMPERAMENT = 0     # Tempérament égal pur
    RAILBACK = 1              # aka Reinhard curve (octaves étirées à partir du médium)
    STRETCHED = 2             # Accord basé sur un modèle de stretching
    CUSTOM = 99               # Autre méthode définie par l’utilisateur (si besoin)


class StretchingModel(Enum):
    NO_STRETCHING = 0
    SIMPLE = 1                # Facteur fixe appliqué aux octaves
    FLETCHER_MUNSON = 2       # Modèle psychoacoustique
    HELMHOLTZ = 3             # Modèle basé sur l’acoustique des cordes

class A4(Enum):
    BAROQUE = 415
    VERDI = 432
    VIENNOIS = 435
    STANDARD = 440
    LIGETI = 442
    STOCKHAUSEN = 445
    GLASS = 466
    CUSTOM = 0