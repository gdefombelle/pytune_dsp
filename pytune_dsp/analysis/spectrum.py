import numpy as np
import librosa

def harmonic_spectrum(
    signal: np.ndarray,
    sr: int,
    f0: float,
    nb_harmonics: int = 8,
    n_fft: int = 16384,
    search_width_cents: float = 60.0,
) -> list[tuple[float, float]]:
    """
    Retourne [(freq_est, amplitude)] pour f0 et ses harmoniques.
    Même si certains partiels sont très faibles, on garde l’ordre.
    """
    if f0 <= 0 or signal.size == 0:
        return []

    # FFT magnitude moyenne
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=n_fft//4))
    spec = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    ratio = 2 ** (search_width_cents / 1200.0)
    results = []

    for k in range(1, nb_harmonics+1):
        target = k * f0
        low, high = target/ratio, target*ratio
        idx_low = np.searchsorted(freqs, low)
        idx_high = np.searchsorted(freqs, high)

        if idx_high <= idx_low:
            results.append((target, 0.0))
            continue

        sub = spec[idx_low:idx_high]
        rel_idx = int(np.argmax(sub))
        k0 = idx_low + rel_idx

        # interpolation quadratique
        if 0 < k0 < len(spec)-1:
            m1, m2, m3 = spec[k0-1], spec[k0], spec[k0+1]
            denom = (m1 - 2*m2 + m3)
            if denom != 0:
                delta = 0.5 * (m1 - m3) / denom
                freq_est = freqs[k0] + delta * (freqs[1] - freqs[0])
            else:
                freq_est = freqs[k0]
        else:
            freq_est = freqs[k0]

        amp = float(spec[k0])
        results.append((freq_est, amp))

    return results

def harmonic_spectrum_fft(signal: np.ndarray, sr: int, f0: float, nb_harmonics: int = 8, search_cents: float = 40):
    S = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/sr)

    ratio = 2 ** (search_cents / 1200.0)
    out = []

    for k in range(1, nb_harmonics + 1):
        target = k * f0
        low, high = target / ratio, target * ratio
        mask = (freqs >= low) & (freqs <= high)

        if not np.any(mask):
            out.append((target, 0.0))
            continue

        idx = np.argmax(S[mask])
        freqs_masked = freqs[mask]
        S_masked = S[mask]

        out.append((freqs_masked[idx], float(S_masked[idx])))

    return out