import numpy as np
from .inharmonicity import inharmonic_map

def R_inh(f, fks, pks, beta, f0):
    mapped = inharmonic_map(fks, f0, beta)
    tau = 1 / f
    return np.sum(pks * np.cos(2 * np.pi * mapped * tau))

def U_inh(f, fks, Eks, beta, sigma):
    mapped = inharmonic_map(fks, f, beta)
    gauss = np.exp(-0.5 * ((f - mapped)/sigma)**2)
    return np.sum(np.sqrt(Eks) * gauss)

def estimate_f0(f_candidates, fks, Eks, pks, beta, sigma):
    R_vals = np.array([R_inh(f, fks, pks, beta, f) for f in f_candidates])
    U_vals = np.array([U_inh(f, fks, Eks, beta, sigma) for f in f_candidates])
    combined = R_vals * U_vals
    return f_candidates[np.argmax(combined)]