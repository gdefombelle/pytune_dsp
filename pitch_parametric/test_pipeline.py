import numpy as np
from pitch_parametric.pitch_estimation import estimate_f0
from pitch_parametric.inharmonicity import beta_estimate

# Exemple synthétique : A3 = 220 Hz
f0_true = 220
fks = np.array([f0_true * n for n in range(1, 6)])
Eks = np.exp(-0.5 * np.arange(5))  # décroissance d’énergie
pks = Eks.copy()

f_candidates = np.geomspace(50, 1000, 2000)
beta = beta_estimate(f0_true)
sigma = f_candidates[0] / 4

f0_est = estimate_f0(f_candidates, fks, Eks, pks, beta, sigma)
print(f"True f0 = {f0_true:.2f} Hz, estimated = {f0_est:.2f} Hz")