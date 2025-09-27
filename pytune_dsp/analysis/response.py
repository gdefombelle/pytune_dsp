# pytune_dsp/analysis/response.py
import numpy as np
from scipy.signal import hilbert

def compute_response(signal: np.ndarray, sr: int, f0: float | None = None) -> dict:
    """
    Calcule une réponse temporelle et spectrale d'une note tenue.
    
    Args:
        signal: signal audio mono
        sr: sample rate
        f0: optionnel, fondamentale estimée (pour extraire des métriques centrées sur f0)

    Returns:
        dict avec:
            - envelope: courbe d’amplitude normalisée
            - decay_time: temps de décroissance (90% -> 10%)
            - sustain_level: niveau moyen plateau après attaque
            - spectrum: spectre moyen normalisé
            - partials (si f0 fourni): amplitudes relatives des harmoniques
    """
    # --- Étape 1 : amplitude (Hilbert transform → enveloppe)
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope /= np.max(envelope) if np.max(envelope) > 0 else 1.0

    # --- Étape 2 : temps de décroissance (90% → 10%)
    above_90 = np.argmax(envelope >= 0.9)
    below_10 = np.argmax(envelope[::-1] <= 0.1)
    decay_time = (len(envelope) - below_10 - above_90) / sr if below_10 > 0 else None

    # --- Étape 3 : niveau moyen sustain (zone centrale)
    sustain_slice = envelope[len(envelope)//3 : 2*len(envelope)//3]
    sustain_level = float(np.mean(sustain_slice)) if sustain_slice.size else 0.0

    # --- Étape 4 : spectre moyen
    N = 8192
    spec = np.abs(np.fft.rfft(signal, n=N))
    spec /= np.max(spec) if np.max(spec) > 0 else 1.0
    freqs = np.fft.rfftfreq(N, 1/sr)

    # --- Étape 5 : partiels si f0 fourni
    partials = []
    if f0:
        for n in range(1, 10):
            target = f0 * n
            if target >= sr/2:
                break
            idx = np.argmin(np.abs(freqs - target))
            partials.append((n, freqs[idx], spec[idx]))

    return {
        "envelope": envelope.tolist(),   # série temporelle normalisée
        "decay_time": decay_time,
        "sustain_level": sustain_level,
        "spectrum": spec.tolist(),
        "freqs": freqs.tolist(),
        "partials": partials,
    }