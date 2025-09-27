from __future__ import annotations
from pytune_dsp.types.dataclasses import *
import numpy as np
from scipy.signal import find_peaks


# ---------- utilitaires internes ----------

NOTE_A4 = 440.0
MIDI_A4 = 69

def freq_to_midi(f: float) -> int:
    return int(round(MIDI_A4 + 12 * np.log2(f / NOTE_A4)))


def _fft_peaks(signal: np.ndarray,
               sr: int,
               pad_factor: int = 4,
               peak_ratio: float = 0.03,
               min_distance_bins: int = 3):
    """FFT + détection de pics."""
    N = max(2048, int(len(signal) * pad_factor))
    spec = np.abs(np.fft.rfft(signal, n=N))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    thr = float(spec.max()) * peak_ratio if spec.size and spec.max() > 0 else 0.0
    peaks, _ = find_peaks(spec, height=thr, distance=min_distance_bins)
    return freqs, spec, peaks


# ---------- 1) Guess simple par cohérence harmonique (FFT) ----------

def guess_f0_fft(signal,
                 sr,
                 expected_freq: float | None = None,
                 n_harmonics: int = 8,
                 tol_cents: float = 15.0) -> tuple[float | None, float]:
    """Retourne (f0, confidence)."""
    freqs, mag, pk = _fft_peaks(signal, sr)
    if pk.size == 0:
        return None, 0.0

    peak_freqs = freqs[pk]
    peak_amps = mag[pk]

    # Candidats = sous-harmoniques des pics
    candidates: list[tuple[float, float]] = []
    for f, a in zip(peak_freqs, peak_amps):
        for d in range(1, 9):
            f0 = f / d
            if 20.0 < f0 < sr / 2.0:
                candidates.append((f0, a / d))
    if not candidates:
        return None, 0.0

    def score_f0(f0: float) -> tuple[float, int]:
        score, hits = 0.0, 0
        for n in range(1, n_harmonics + 1):
            target = f0 * n
            if target >= sr / 2:
                break
            diffs = 1200.0 * np.log2(peak_freqs / target)
            i = int(np.argmin(np.abs(diffs)))
            if abs(diffs[i]) <= tol_cents:
                score += (peak_amps[i] / max(1.0, np.sqrt(n)))
                hits += 1
        return score, hits

    scored = [(f0, *score_f0(f0)) for f0, _ in candidates]
    if not scored:
        return None, 0.0

    # tri par (#hits, score)
    scored.sort(key=lambda t: (t[2], t[1]), reverse=True)
    best_f0, best_score, best_hits = scored[0]
    conf = min(1.0, best_hits / float(n_harmonics))
    return float(best_f0), float(conf)


# ---------- 2) Guess "pattern" robuste ----------

def guess_f0_pattern(signal,
                     sr,
                     n_harmonics: int = 10,
                     tol_cents: float = 15.0) -> GuessF0Result:
    freqs, mag, pk = _fft_peaks(signal, sr, pad_factor=8, peak_ratio=0.02, min_distance_bins=4)
    if pk.size == 0:
        return GuessF0Result(None, 0.0, [], [])

    peak_freqs = freqs[pk]
    peak_amps = mag[pk]

    # Banque de candidats par votes de sous-harmoniques
    votes: dict[float, float] = {}
    for f, a in zip(peak_freqs, peak_amps):
        for d in range(1, 13):
            f0 = f / d
            if 20.0 < f0 < sr / 2.0:
                votes[f0] = votes.get(f0, 0.0) + a / d

    if not votes:
        return GuessF0Result(None, 0.0, [], [])

    candidates = sorted(votes.items(), key=lambda t: t[1], reverse=True)
    best_f0 = candidates[0][0]

    matched: list[tuple[int, float, float]] = []
    hits = 0
    for n in range(1, n_harmonics + 1):
        target = best_f0 * n
        if target >= sr / 2:
            break
        diffs = 1200.0 * np.log2(peak_freqs / target)
        i = int(np.argmin(np.abs(diffs)))
        err = diffs[i]
        if abs(err) <= tol_cents:
            matched.append((n, peak_freqs[i], err))
            hits += 1

    conf = hits / float(n_harmonics) if n_harmonics > 0 else 0.0
    return GuessF0Result(float(best_f0), float(conf), list(peak_freqs), matched)


# ---------- 3) Fusion (quand on a un hint MIDI) ----------

def guess_f0_fusion(signal,
                    sr,
                    midi_hint: int | None = None) -> GuessF0Result:
    """Combine FFT et Pattern pour une estimation robuste."""
    f0_fft, conf_fft = guess_f0_fft(signal, sr)
    pat_res = guess_f0_pattern(signal, sr)

    f0_pat, conf_pat = pat_res.f0, pat_res.confidence

    f0_final, conf_final = None, 0.0

    if midi_hint is not None:
        # Graves : FFT prioritaire
        if midi_hint < 45:
            f0_final, conf_final = f0_fft, conf_fft
        # Aigus : Pattern prioritaire
        elif midi_hint > 75:
            f0_final, conf_final = f0_pat, conf_pat
        # Medium : fusion
        else:
            if f0_fft and f0_pat:
                diff = 1200 * np.log2(f0_fft / f0_pat)
                if abs(diff) < 50:
                    f0_final = (f0_fft * conf_fft + f0_pat * conf_pat) / (conf_fft + conf_pat + 1e-6)
                    conf_final = max(conf_fft, conf_pat)
                else:
                    if conf_fft >= conf_pat:
                        f0_final, conf_final = f0_fft, conf_fft
                    else:
                        f0_final, conf_final = f0_pat, conf_pat
            else:
                f0_final, conf_final = (f0_fft, conf_fft) if f0_fft else (f0_pat, conf_pat)
    else:
        # Pas d'indice MIDI → garder celui avec meilleure confiance
        if conf_fft >= conf_pat:
            f0_final, conf_final = f0_fft, conf_fft
        else:
            f0_final, conf_final = f0_pat, conf_pat

    return GuessF0Result(f0_final, conf_final,
                         pat_res.harmonics if pat_res.harmonics else [],
                         pat_res.matched if pat_res.matched else [])


# ---------- 4) API haut-niveau : guess_note ----------

def guess_note(signal, sr) -> GuessNoteResult:
    """
    Détermine la note jouée à partir du signal brut.
    - Utilise Pattern en priorité
    - FFT en secours
    """
    pat = guess_f0_pattern(signal, sr)
    if pat.f0 and pat.confidence >= 0.2:  # seuil minimal
        return GuessNoteResult(
            midi=freq_to_midi(pat.f0),
            f0=pat.f0,
            confidence=pat.confidence,
            method="pattern"
        )

    f0_fft, conf_fft = guess_f0_fft(signal, sr)
    if f0_fft:
        return GuessNoteResult(
            midi=freq_to_midi(f0_fft),
            f0=f0_fft,
            confidence=conf_fft,
            method="fft"
        )

    return GuessNoteResult(None, None, 0.0, "none")