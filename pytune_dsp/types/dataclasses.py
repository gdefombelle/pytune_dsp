from dataclasses import dataclass, field
from pytune_dsp.types.enums import SampleType
import numpy as np
from typing import Dict, List, Optional, Any

@dataclass
class GuessF0Result:
    f0: float | None
    confidence: float
    harmonics: list[float]
    matched: list[tuple[int, float, float]]

    # ✅ nouveaux champs pour la fusion complète
    method: str = "none"                  # ex: "YINFFT+FFT"
    band: str = "unknown"                 # "low" | "mid" | "high"
    components: dict[str, dict] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)  
    # ex:
    # {
    #   "yinfft": {"f0": 27.57, "conf": 0.93},
    #   "fft": {"f0": 27.50, "score": 0.687},
    #   "hps": {"f0": 27.83, "conf": 0.90},
    #   "comb": {"f0": 27.30, "conf": 0.65}
    # }

@dataclass
class GuessNoteResult:
    midi: int | None
    f0: float | None
    confidence: float
    method: str

    # ✅ nouveaux champs pour correspondre au modèle Pydantic
    debug_log: list[str] | None = None
    subresults: dict[str, dict] | None = None
    envelope_band: str | None = None

@dataclass
class NoteAnalysisResult:
    note_name: str
    valid: bool
    f0: float | None
    confidence: float
    deviation_cents: float | None
    expected_freq: float | None
    harmonics: list[float]
    partials: list[float]
    inharmonicity: dict | None
    inharmonicity_avg: float | None
    B_estimate: float | None
    spectral_fingerprint: np.ndarray | None
    harmonic_spectrum_raw: np.ndarray | None
    harmonic_spectrum_norm: np.ndarray | None
    response: dict | None
    guessed_note: GuessNoteResult | None  # <--- nouveau



@dataclass(frozen=True)
class Keyboard:
    A4: float
    lower_note: str
    upper_note: str
    frequencies: Dict[str, float]   # { "A0": 27.5, "A#0": 29.1, ... }
    jnd: Dict[str, float]           # { "A0": 3.5 cents, ... }
    strings_per_note: Dict[str, int]

@dataclass
class PyTuneScanData:
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    sr: int = 0
    notes: List[str] = field(default_factory=list)
    f0s: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_frequency_in_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_frequency_in_cents: np.ndarray = field(default_factory=lambda: np.array([]))
    relative_time_stamps: np.ndarray = field(default_factory=lambda: np.array([]))
    norm_1_amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    max_amplitude: float = 0.0
    duration: float = 0.0
    avg_note_duration: float = 0.0
    filename: str = "PianoScanData.csv"

    def get_note_signal(self, start: int, end: int):
        start = int(self.relative_time_stamps[start] * self.sr)
        end = int(self.relative_time_stamps[end] * self.sr)
        return self.signal[start:end]

@dataclass
class SimpleNoteMeasures:
    stable_f0: float = 0.0
    mode_rate: float = 0.0
    centroid: float = 0.0
    best_method: str = "stableF0Average"
    best_value: float = 0.0
    dev_cents_vs_ref: float = 0.0
    dev_hz_vs_ref: float = 0.0

@dataclass
class NoteMeasurements:
    total_samples_size: int = 0
    used_sample_size: int = 0
    target_frequency_tempered: float = 0.0
    target_frequency_stretched: float = 0.0
    centroid: float = 0.0
    stableF0Average: float = 0.0
    mode_rate: float = 0.0
    best_measurement_method: Optional[str] = None
    best_measurement: float = 0.0
    # Déviations par rapport au tempéré / stretché
    eq_tempered_deviation_cents: float = 0.0
    stretched_deviation_cents: float = 0.0
    eq_tempered_deviation_hz: float = 0.0
    stretched_deviation_hz: float = 0.0
    perceptible_jnd_tempered: bool = False
    perceptible_jnd_stretched: bool = False


@dataclass
class NoteDeviationAnalysis:
    key: str = ""
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    idx_scan_data: int = -1
    scanned: bool = False
    nb_f0s: int = 0
    tempered_expected_frequency: float = 0.0
    stretched_expected_frequency: float = 0.0
    measurements: Optional[NoteMeasurements] = None
    nb_of_strings: int = 1


@dataclass
class AnalysisResult:
    notes_deviations: List[NoteDeviationAnalysis] = field(default_factory=list)
    sample_type: SampleType = SampleType.NONE
    sample_duration: float = 0.0
    notes_per_second: float = 0.0
    spectral_deviation_tempered_hz: float = 0.0
    spectral_deviation_stretched_hz: float = 0.0
    spectral_deviation_tempered_cents: float = 0.0
    spectral_deviation_stretched_cents: float = 0.0
    std_eq_tempered_deviations_cents: float = 0.0
    std_eq_tempered_deviations_hz: float = 0.0
    std_stretched_deviations_cents: float = 0.0
    std_stretched_deviations_hz: float = 0.0
    missing_notes: List[str] = field(default_factory=list)
    score: int = 0

    @property
    def A4_measurements(self) -> Optional[NoteMeasurements]:
        return next((n.measurements for n in self.notes_deviations if n.key == "A4"), None)