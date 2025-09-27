# pytune_dsp/analysis/inharmonicity.py

from __future__ import annotations
import numpy as np
from typing import List, Optional


def compute_inharmonicity_avg(inharmonicity: List[float]) -> Optional[float]:
    """
    Calcule la moyenne simple des écarts en cents pour une note.
    ❌ Pas de pondération par amplitude (trop dépendant de la prise de son).
    
    Args:
        inharmonicity: liste des écarts en cents pour chaque partiel mesuré.

    Returns:
        Moyenne (float) ou None si liste vide.
    """
    if not inharmonicity:
        return None
    return float(np.mean(inharmonicity))


def estimate_B(f0: float, partials_hz: List[float], k_start: int = 2) -> Optional[float]:
    """
    Estime le facteur d'inharmonicité B pour une corde de piano.

    Modèle théorique :
        f_k ≈ k * f0 * sqrt(1 + B * k^2)

    On linéarise :
        (f_k / (k * f0))^2 - 1 ≈ B * k^2

    Puis on ajuste B par régression linéaire sur les partiels mesurés.

    Args:
        f0: fréquence fondamentale mesurée (Hz).
        partials_hz: fréquences des partiels mesurés (Hz).
        k_start: index du premier partiel utilisé (par défaut 2, car k=1 est trop bruité).

    Returns:
        B (float) ou None si estimation impossible.
    """
    if not partials_hz or f0 <= 0:
        return None

    x_vals = []
    y_vals = []

    for k, f_k in enumerate(partials_hz, start=1):
        if k < k_start:  # on saute k=1
            continue
        ratio = f_k / (k * f0)
        y = ratio**2 - 1
        x = k**2
        x_vals.append(x)
        y_vals.append(y)

    if len(x_vals) < 2:
        return None

    # Régression linéaire (moindres carrés)
    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)
    B, _, _, _ = np.linalg.lstsq(x_vals[:, None], y_vals, rcond=None)

    return float(B[0])