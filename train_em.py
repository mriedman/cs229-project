from training_observer import TrainingObserver
import numpy as np
import os
from copy import deepcopy
import scipy.stats

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 3           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    it = 0
    ll = prev_ll = None
    ct = 0
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # E-step
        pdfs = np.array([scipy.stats.multivariate_normal.pdf(x, mean=i, cov=j) * k for i, j, k in zip(mu, sigma, phi)]).T
        w = pdfs / np.sum(pdfs, axis=1).reshape((-1,1))

        # M-step
        phi = np.sum(w, axis=0) / x.shape[0]
        mu = np.array([np.sum(w[:, j:j+1] * x, axis=0) for j in range(w.shape[1])])
        mu /= np.sum(w, axis=0).reshape((-1, 1))
        sigma = [(w[:, j:j+1] * (x - mu[j])).T @ (x - mu[j]) for j in range(w.shape[1])]
        sigma /= np.sum(w, axis=0).reshape((K, 1, 1))

        # ll
        prev_ll = ll

        print([np.linalg.det(j) for j in sigma])
        if any(np.linalg.det(j) < 1e-28 for j in sigma):
            break

        pdfs = np.array([scipy.stats.multivariate_normal.pdf(x, mean=i, cov=j) * k for i, j, k in zip(mu, sigma, phi)]).T
        ll = np.sum(np.log(np.sum(pdfs, axis=1))).squeeze()
        ct += 1
        print('Iteration', ct, 'likelihood:', ll)

    return w

def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    alpha = 100  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    it = 0
    ll = prev_ll = None

    # Created w array for x_tilde which is 1 if zsi=j and 0 otherwise to allow for parallel code structue
    w_sup = np.zeros((x_tilde.shape[0], phi.shape[0]))
    for i, j in enumerate(z_tilde):
        w_sup[i, int(j)] = 1
    ct = 0

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        unsup_n = x.shape[0]
        sup_n = x_tilde.shape[0]

        # E-step
        # Same as in the earlier part
        pdfs = np.array([scipy.stats.multivariate_normal.pdf(x, mean=i, cov=j) * k for i, j, k in zip(mu, sigma, phi)]).T
        w = pdfs / np.sum(pdfs, axis=1).reshape((-1, 1))

        # M-step
        phi = (np.sum(w, axis=0) + alpha * np.sum(w_sup, axis=0)) / (unsup_n + alpha * sup_n)
        mu = np.array([np.sum(w[:, j:j + 1] * x, axis=0) for j in range(w.shape[1])])
        mu += alpha * np.array([np.sum(w_sup[:, j:j + 1] * x_tilde, axis=0) for j in range(w.shape[1])])
        mu /= (np.sum(w, axis=0).reshape((-1, 1)) + alpha * np.sum(w_sup, axis=0).reshape((-1, 1)))
        sigma = [
            (w[:, j:j + 1] * (x - mu[j])).T @ (x - mu[j]) +
            alpha * (w_sup[:, j:j + 1] * (x_tilde - mu[j])).T @ (x_tilde - mu[j])
            for j in range(w.shape[1])]
        sigma /= (np.sum(w, axis=0).reshape((K, 1, 1)) + alpha * np.sum(w_sup, axis=0).reshape((K, 1, 1)))

        # ll
        prev_ll = ll

        pdfs = np.array(
            [scipy.stats.multivariate_normal.pdf(x, mean=i, cov=j) * k
             for i, j, k in zip(mu, sigma, phi)]
        ).T
        pdfs_sup = np.array(
            [scipy.stats.multivariate_normal.pdf(xj, mean=mu[int(j)], cov=sigma[int(j)]) for j, xj in zip(z_tilde, x_tilde)]
        ).T
        ll = np.sum(np.log(np.sum(pdfs, axis=1))).squeeze()
        sup_ll = np.sum(np.log(pdfs_sup))
        ll += alpha * sup_ll
        ct += 1
        print('Iteration', ct, 'likelihood:', ll)

        # *** END CODE HERE ***

    return w

def prep_em(x):
    a = 200
    x_shuff_labels = deepcopy(x)
    # np.random.shuffle(x_shuff_labels)
    x_tilde = x_shuff_labels[-a:, :-1]  # Labeled examples
    z_tilde = x_shuff_labels[-a:, -1]  # Corresponding labels
    x_shuff = x_shuff_labels[:, :-1]
    x = x_shuff[:-a, :]

    mu = np.zeros((K, x.shape[1]))
    sigma = np.zeros((K, x.shape[1], x.shape[1]))
    n = x.shape[0]
    for j in range(K):
        xj = x_shuff[(n * j) // K:(n * (j + 1)) // K, :]
        mu[j] = np.mean(xj, axis=0)
        sigma[j] = np.cov(xj.T)

    phi = np.ones((K,)) / K
    n = x.shape[0]
    w = np.ones((n, K)) / K

    # w = run_em(x, w, phi, mu, sigma)
    w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)

    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    # np.savetxt('./ests.txt', np.concatenate([z_pred.reshape((-1,1)), x_shuff_labels[a:, -1].reshape((-1,1))], axis=1))
    '''hits = sum(1 for i in range(len(ests)) if ests[i] == x_shuff_labels[i, -1])
    avg_ll = sum(np.log(1e-6 + w[i, int(x_shuff_labels[i, -1])]) for i in range(len(ests)))

    print(hits / len(ests))
    print(avg_ll / len(ests))'''

    np.savetxt('./em_mu.txt', mu)
    np.savetxt('./em_phi.txt', phi)
    for i in range(sigma.shape[0]):
        np.savetxt('./em_sigma{}.txt'.format(i), sigma[i])

    return z_pred


train_data = np.loadtxt('./train_data.txt')
prep_em(train_data)



