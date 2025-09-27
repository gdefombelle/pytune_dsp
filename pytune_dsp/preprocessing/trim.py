import numpy as np

def trim_signal(signal: np.ndarray, sr: int, 
                threshold_db: float = -25.0,
                min_duration: float = 0.3) -> np.ndarray:
    """
    Nettoie le signal audio d'une note de piano :
    - Coupe le bruit d'attaque mécanique
    - Coupe la fin bruitée (release)
    - Applique un fade-in/out
    """
    # --- Normalisation ---
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    # --- Enveloppe RMS ---
    frame = int(0.01 * sr)  # 10 ms
    hop = frame // 2
    rms = np.array([
        np.sqrt(np.mean(signal[i:i+frame]**2))
        for i in range(0, len(signal)-frame, hop)
    ])
    rms_db = 20 * np.log10(rms / np.max(rms) + 1e-9)

    # --- Détection début ---
    start_idx = next(
        (i for i, val in enumerate(rms_db) if val > threshold_db), 
        0
    ) * hop

    # --- Détection fin ---
    end_idx = len(signal)
    for i in range(len(rms_db)-1, -1, -1):
        if rms_db[i] > threshold_db:
            end_idx = i * hop + frame
            break

    # --- Découpage ---
    trimmed = signal[start_idx:end_idx]

    # --- Durée minimale (sécurité) ---
    if len(trimmed) < int(min_duration * sr):
        return signal  # on garde tout si trop court

    # --- Fade-in/out ---
    fade = int(0.02 * sr)  # 20 ms
    if len(trimmed) > 2 * fade:
        win = np.ones(len(trimmed))
        win[:fade] = np.linspace(0, 1, fade)
        win[-fade:] = np.linspace(1, 0, fade)
        trimmed = trimmed * win

    return trimmed