import numpy as np
from scipy.special import digamma, polygamma

def initialize_lda(documents, K, V):
    """
    Initialize the LDA model parameters and variables.
    """
    D = len(documents)  # Number of documents
    n_dk = np.zeros((D, K))
    n_kv = np.zeros((K, V))
    n_k = np.zeros(K)
    z_dn = [[np.random.choice(K) for _ in doc] for doc in documents]

    # Initialize counts
    for d, doc in enumerate(documents):
        for n, word_id in enumerate(doc):
            topic = z_dn[d][n]
            n_dk[d, topic] += 1
            n_kv[topic, word_id] += 1
            n_k[topic] += 1

    return n_dk, n_kv, n_k, z_dn

def compute_expectations(alpha, beta, n_dk, n_kv, n_k, D, V, K):
    """
    Compute the expectations using Gaussian approximations.
    """
    E_log_theta_dk = digamma(n_dk + alpha) - digamma(np.sum(n_dk, axis=1)[:, np.newaxis] + K*alpha)
    E_log_phi_kv = digamma(n_kv + beta) - digamma(n_k + V*beta)

    # Gaussian approximations for the variances
    var_theta_dk = polygamma(1, n_dk + alpha)
    var_phi_kv = polygamma(1, n_kv + beta)

    return E_log_theta_dk, E_log_phi_kv, var_theta_dk, var_phi_kv

def cvb_update(documents, z_dn, n_dk, n_kv, n_k, alpha, beta, V, K, E_log_theta_dk, E_log_phi_kv):
    """
    Update the CVB assignments for each word in each document.
    """
    D = len(documents)

    for d in range(D):
        doc = documents[d]
        for n, word_id in enumerate(doc):
            old_topic = z_dn[d][n]

            # Remove current word's influence
            n_dk[d, old_topic] -= 1
            n_kv[old_topic, word_id] -= 1
            n_k[old_topic] -= 1

            # Update expectations
            E_log_theta_dk[d], E_log_phi_kv[:, word_id], _, _ = compute_expectations(alpha, beta, n_dk, n_kv, n_k, D, V, K)

            # Compute the topic assignment probabilities
            p_z_dn = np.exp(E_log_theta_dk[d] + E_log_phi_kv[:, word_id])
            p_z_dn /= np.sum(p_z_dn)

            # Sample a new topic based on the computed probabilities
            new_topic = np.random.choice(K, p=p_z_dn)

            # Update counts with the new topic assignment
            n_dk[d, new_topic] += 1
            n_kv[new_topic, word_id] += 1
            n_k[new_topic] += 1

            # Update the topic assignment
            z_dn[d][n] = new_topic

    return z_dn, n_dk, n_kv, n_k

if __name__ == "__main__":
    # Example usage
    documents = [['word1', 'word2', 'word3'], ['word2', 'word3', 'word4'], ['word3', 'word4', 'word1']]
    K = 3  # Number of topics    V = ...  # Vocabulary size
    V = 4
    alpha, beta = 0.1, 0.1

    n_dk, n_kv, n_k, z_dn = initialize_lda(documents, K, V)
    E_log_theta_dk, E_log_phi_kv, var_theta_dk, var_phi_kv = compute_expectations(alpha, beta, n_dk, n_kv, n_k, len(documents), V, K)

    for iteration in range(100):  # Number of iterations
        z_dn, n_dk, n_kv, n_k = cvb_update(documents, z_dn, n_dk, n_kv, n_k, alpha, beta, V, K, E_log_theta_dk, E_log_phi_kv)
        # Optionally: Check for convergence
