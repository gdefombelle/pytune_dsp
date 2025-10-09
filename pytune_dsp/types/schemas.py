# src/models/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Tuple, Dict


class NoteCaptureMeta(BaseModel):
    note_expected: int = Field(..., description="MIDI number of expected note")
    sample_rate: int = Field(..., description="Actual sample rate of the stream (Hz)")
    channels: int = Field(1, description="Number of channels (default mono)")
    dtype: str = Field("float32", description="Data type of audio buffer")
    length: int = Field(..., description="Number of samples in buffer")
    compute_inharm: bool = Field(
        True, description="Whether to compute inharmonicity (False for high notes > C6)"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in (44100, 48000, 88200, 96000):
            raise ValueError(f"Unsupported sample rate: {v}")
        return v


class GuessNoteResult(BaseModel):
    midi: Optional[int] = None
    f0: Optional[float] = None
    confidence: float
    method: str

    # ✅ nouveaux champs
    debug_log: Optional[List[str]] = None     # lignes détaillées
    subresults: Optional[Dict[str, dict]] = None  # yinfft / fft / hps / comb etc.
    envelope_band: Optional[str] = None       # "low", "mid", "high"


class NoteAnalysisResult(BaseModel):
    midi: Optional[int] = None 
    note_name: str
    valid: bool
    f0: Optional[float] = None
    confidence: Optional[float] = None
    deviation_cents: Optional[float] = None   # écart vs expected
    expected_freq: Optional[float] = None     # traçabilité

    harmonics: List[float] = []
    partials: List[float] = []
    inharmonicity: List[float] = []

    spectral_fingerprint: List[float] = []

    harmonic_spectrum_raw: List[Tuple[float, float]] = []
    harmonic_spectrum_norm: List[Tuple[float, float]] = []

    inharmonicity_avg: Optional[float] = None
    B_estimate: Optional[float] = None

    # 🔹 Résultat canonique → choisi comme “meilleure hypothèse”
    guessed_note: Optional["GuessNoteResult"] = None

    # 🔹 Tous les résultats détaillés
    guesses: Dict[str, "GuessNoteResult"] = {}

    response: Optional[dict] = None