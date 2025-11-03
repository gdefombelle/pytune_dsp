# src/models/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Tuple, Dict, Union


# ────────────────────────────────────────────────────────────────────────────
# Capture metadata
# ────────────────────────────────────────────────────────────────────────────
class NoteCaptureMeta(BaseModel):
    note_expected: int = Field(..., description="MIDI number of expected note")
    sample_rate: int = Field(..., description="Actual sample rate (Hz)")
    channels: int = Field(1, description="Number of channels (default mono)")
    dtype: str = Field("float32", description="Audio buffer dtype")
    length: int = Field(..., description="Number of samples in buffer")
    compute_inharm: bool = Field(
        True, description="Whether to compute inharmonicity (False for very high notes > C6)"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in (44100, 48000, 88200, 96000):
            raise ValueError(f"Unsupported sample rate: {v}")
        return v


# ────────────────────────────────────────────────────────────────────────────
# Guess result (librosa / essentia / etc.)
# ────────────────────────────────────────────────────────────────────────────
class GuessNoteResult(BaseModel):
    midi: Optional[int] = None
    f0: Optional[float] = None
    confidence: float
    method: str

    # Optional diagnostics
    debug_log: Optional[List[str]] = Field(default=None)                 # lines of debug
    subresults: Optional[Dict[str, dict]] = Field(default=None)          # yin/fft/hps/...
    envelope_band: Optional[str] = Field(default=None)                   # "low" | "mid" | "high"


# ────────────────────────────────────────────────────────────────────────────
# PFD result (f0 + inharmonicity + diagnostics)
# ────────────────────────────────────────────────────────────────────────────
class PFDResult(BaseModel):
    f0: Optional[float] = None
    B: Optional[float] = None
    quality: Optional[float] = None                         # robust std (MAD-scaled) of Δ_k
    deltas: List[float] = Field(default_factory=list)       # Δ_k (Hz)
    partials_hz: List[float] = Field(default_factory=list)  # theoretical partial centers used


# ────────────────────────────────────────────────────────────────────────────
# Full analysis result for a captured note
# ────────────────────────────────────────────────────────────────────────────
class NoteAnalysisResult(BaseModel):
    midi: Optional[int] = None
    note_name: str
    valid: bool

    # Canonical selection
    f0: Optional[float] = None
    confidence: Optional[float] = None
    deviation_cents: Optional[float] = None          # vs expected
    expected_freq: Optional[float] = None            # traceability

    # Spectral / partials / inharm
    harmonics: List[float] = Field(default_factory=list)
    partials: List[float] = Field(default_factory=list)
    inharmonicity: List[float] = Field(default_factory=list)
    inharmonicity_avg: Optional[float] = None
    B_estimate: Optional[float] = None               # final B (possibly fused with PFD)

    # Fingerprints
    spectral_fingerprint: List[float] = Field(default_factory=list)
    harmonic_spectrum_raw: List[Tuple[float, float]] = Field(default_factory=list)
    harmonic_spectrum_norm: List[Tuple[float, float]] = Field(default_factory=list)

    # PFD (exposed for UI + diagnostics)
    f0_pfd: Optional[float] = None
    B_pfd: Optional[float] = None
    quality_pfd: Optional[float] = None
    pfd_deltas: List[float] = Field(default_factory=list)
    pfd_partials_hz: List[float] = Field(default_factory=list)

    # Canonical chosen guess + all guesses
    guessed_note: Optional["GuessNoteResult"] = None
    guesses: Dict[str, Union["GuessNoteResult", PFDResult, dict]] = Field(default_factory=dict)

    # Timings (ms)
    time_librosa_ms: Optional[float] = None
    time_essentia_ms: Optional[float] = None
    time_pfd_ms: Optional[float] = None
    time_parallel_ms: Optional[float] = None
    time_f0_hp_ms: Optional[float] = None

    # Optional extra data
    response: Optional[dict] = None

    @model_validator(mode="after")
    def _normalize_b_estimate(self) -> "NoteAnalysisResult":
        """
        Si B_estimate est manquant mais un B_pfd plausible existe, l'utiliser.
        'Plausible' : |B| < 1e-2 (borne large pour pianos).
        """
        if (self.B_estimate is None) and (self.B_pfd is not None):
            if abs(self.B_pfd) < 1e-2:
                self.B_estimate = self.B_pfd
        return self