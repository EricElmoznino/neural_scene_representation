import numpy as np
from behavioural.plotting import *


def flatten_matrix(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('The input must be a square matrix')
    n = mat.shape[0]
    flat_vector = np.zeros((n * (n - 1)) // 2)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            flat_vector[count] = mat[i, j]
            count += 1
    return flat_vector


def rdm_corr(a, b, flatten=True):
    # Pearson correlation between two RDMs (a and b)
    if flatten:
        a = flatten_matrix(a)
        b = flatten_matrix(b)
    corr = np.corrcoef(a, b)[0, 1]
    return corr


if __name__ == '__main__':
    gqn_rdm = np.load('behavioural/results/gqn_rdm.npy')
    vae_rdm = np.load('behavioural/results/vae_rdm.npy')
    human_rdm = np.load('behavioural/results/human_rdm.npy')

    gqn_human_corr = rdm_corr(gqn_rdm, human_rdm)
    vae_human_corr = rdm_corr(vae_rdm, human_rdm)
    gqn_vae_corr = rdm_corr(gqn_rdm, vae_rdm)

    plot_bar({'GQN - Human': gqn_human_corr, 'VAE - Human': vae_human_corr, 'GQN - VAE': gqn_vae_corr})
