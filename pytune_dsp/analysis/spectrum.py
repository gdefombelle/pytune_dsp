import numpy as np

def harmonic_spectrum_fft(
    signal: np.ndarray,
    sr: int,
    f0: float,
    nb_harmonics: int = 8,
    search_cents: float = 40.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Retourne deux empreintes harmoniques :
    - brut : [(freq, amplitude), ...]
    - normalisé (0–1) : [(freq, amplitude_norm), ...]

    Amplitudes normalisées par rapport au max global.
    """
    if f0 <= 0 or signal.size == 0:
        return [], []

    # FFT brute
    S = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)

    ratio = 2 ** (search_cents / 1200.0)
    raw = []

    for k in range(1, nb_harmonics + 1):
        target = k * f0
        low, high = target / ratio, target * ratio
        mask = (freqs >= low) & (freqs <= high)

        if not np.any(mask):
            raw.append((target, 0.0))
            continue

        idx = np.argmax(S[mask])
        freqs_masked = freqs[mask]
        S_masked = S[mask]

        raw.append((float(freqs_masked[idx]), float(S_masked[idx])))

    # Normalisation sur le max global
    max_amp = max(a for _, a in raw) if raw else 1.0
    norm = [(f, a / max_amp if max_amp > 0 else 0.0) for f, a in raw]

    return raw, norm