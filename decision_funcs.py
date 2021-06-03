import numpy as np
import scipy.stats

val_data = np.loadtxt('./val_data.txt')
clusters = np.loadtxt('./clusters_ss.txt')

mu = np.loadtxt('./em_mu.txt')
phi = np.loadtxt('./em_phi.txt')
sigma = np.zeros((3, 7, 7))
for i in range(3):
    si = np.loadtxt('./em_sigma{}.txt'.format(i))
    sigma[i] = si

def k_means_semi_sup(xi):
    norms = np.array([np.linalg.norm(xi - cl) for cl in clusters])
    min_idx = np.argmin(norms)
    return min_idx

def em_predict(xi):
    pdfs = np.array([scipy.stats.multivariate_normal.pdf(xi, mean=i, cov=j) * k for i, j, k in zip(mu, sigma, phi)]).T
    w = pdfs / np.sum(pdfs)
    return np.argmin(w)
