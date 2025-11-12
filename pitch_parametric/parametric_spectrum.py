import numpy as np

def inharmonic_map(fk: np.ndarray, f0: float, beta: float) -> np.ndarray:
    """Remappe les fréquences fk selon la loi d’inharmonicité."""
    h2 = (fk / f0) ** 2
    return fk / np.sqrt(1 + beta * (h2 - 1))

def beta_estimate(f0: float) -> float:
    """Renvoie un coefficient β typique pour un piano selon la note."""
    # Valeurs approchées (Fletcher & Rossing)
    if f0 < 100:
        return 3e-4
    elif f0 < 500:
        return 2e-4
    elif f0 < 2000:
        return 1e-4
    return 5e-5