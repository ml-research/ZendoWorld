import numpy as np
from numpy.random import random as _np_random

def systematic_resample(weights, N, rng=None):
    """Systematic resampling for particle filters: N evenly spaced positions with one shared random offset."""
    rand_val = rng.random() if rng is not None else _np_random()
    positions = (rand_val + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def feedback_generator(structures, labels):
    txt = ""
    for idx, (structure, label) in enumerate(zip(structures, labels)):
        flip_label = 'yes' if label == 'no' else 'no'
        txt += f"{idx + 1}. {structure.to_text()}"
        txt += f"Correct output: {label}\n"
        txt += f"Rule's output: {flip_label}\n\n"
    return txt


def find_c(weights, N):
    sorted_weights = np.sort(weights)
    B_val = 0.0
    A_val = len(weights)
    for i in range(len(sorted_weights)):
        chi = sorted_weights[i]
        A_val -= 1
        B_val += chi
        if B_val / chi + A_val - N <= 1e-12:
            return (N - A_val) / B_val
    return N


# Source: https://github.com/probcomp/hfppl/blob/main/hfppl/inference/smc_steer.py
def resample_optimal(weights, N, rng=None):
    c = find_c(weights, N)
    deterministic = np.where(c * weights >= 1)[0]
    stochastic = np.where(c * weights < 1)[0]
    n_stochastic = len(stochastic)
    n_resample = N - len(deterministic)
    if n_resample == 0:
        return deterministic, np.array([], dtype=int), c
    K = np.sum(weights[stochastic]) / (n_resample)
    u = rng.uniform(0, K) if rng is not None else np.random.uniform(0, K)
    i = 0
    stoch_resampled = np.array([], dtype=int)
    while i < n_stochastic:
        u = u - weights[stochastic[i]]
        if u <= 0:
            stoch_resampled = np.append(stoch_resampled, stochastic[i])
            u = u + K
            i = i + 1
        else:
            i += 1

    resampled = np.concatenate((deterministic, stoch_resampled))
    return resampled