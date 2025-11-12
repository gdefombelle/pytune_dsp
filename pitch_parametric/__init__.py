"""
PyTune DSP — Pitch Parametric Estimator
---------------------------------------

Ce sous-module implémente une version inspirée du modèle paramétrique
de Badeau–Emiya–David (ICASSP 2007), combinant :
- modélisation de l’inharmonicité (β(f₀)),
- analyse spectrale paramétrique (gaussienne pondérée),
- fonction temporelle reconstruite (R_inh),
- fusion R×U pour l’estimation de f₀.

Structure :
    inharmonicity.py     → loi β(f₀) + remappage fréquentiel
    parametric_spectrum.py → construction S(f)
    pitch_estimation.py  → fonctions R_inh, U_inh et fusion
    esprit_analysis.py   → (optionnel) extraction des pôles haute résolution
    test_pipeline.py     → démonstration synthétique
"""

from .inharmonicity import inharmonic_map, beta_estimate
from .parametric_spectrum import parametric_spectrum
from .pitch_estimation import R_inh, U_inh, estimate_f0

__all__ = [
    "inharmonic_map",
    "beta_estimate",
    "parametric_spectrum",
    "R_inh",
    "U_inh",
    "estimate_f0",
]