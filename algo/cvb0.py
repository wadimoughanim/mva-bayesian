#### ACTUAL
import numpy as np
from collections import defaultdict
from scipy.special import digamma

def initialize_lda(documents, K):
    word_to_id = defaultdict(lambda: len(word_to_id))
    id_to_word = {}
    doc_word_ids = []

    # Convert words in documents to unique IDs
    for doc in documents:
        doc_ids = [word_to_id[word] for word in doc]
        doc_word_ids.append(doc_ids)

    # Invert word_to_id to get id_to_word mapping
    id_to_word = {id_: word for word, id_ in word_to_id.items()}

    V = len(word_to_id)  # Vocabulary size
    n_d_k = np.zeros((len(documents), K))
    n_k_t = np.zeros((K, V))
    n_k = np.zeros(K)
    z_d_i = [[np.random.randint(K) for _ in doc] for doc in documents]  # Fixed

    for d, doc_ids in enumerate(doc_word_ids):
        for i, word_id in enumerate(doc_ids):

            topic = z_d_i[d][i]  # Corrected indexing
            n_d_k[d, topic] += 1
            n_k_t[topic, word_id] += 1
            n_k[topic] += 1

    return n_d_k, n_k_t, n_k, z_d_i, word_to_id, id_to_word, V, doc_word_ids




def compute_expectation_terms(gamma_ijk, alpha, beta, n_d_k, n_k_t, n_k, W):
    # (equation 16)
    E_gamma_ijk = np.sum(gamma_ijk, axis=0)
    Var_gamma_ijk = np.sum(gamma_ijk * (1 - gamma_ijk), axis=0)

    # (equation 17)
    E_log_alpha_n_j_dot_k = digamma(alpha + E_gamma_ijk) - digamma(alpha * K + np.sum(n_d_k))
    E_log_beta_n_dot_k_x_ij = digamma(beta + n_k_t) - digamma(beta * W + n_k)
    E_log_W_beta_n_dot_k_dot = digamma(W * beta + n_k) - digamma(W * beta * K + np.sum(n_k))
    
    taylor_approx_n_j_dot_k = E_log_alpha_n_j_dot_k - Var_gamma_ijk / (2 * (alpha + E_gamma_ijk)**2)
    taylor_approx_n_dot_k_x_ij = E_log_beta_n_dot_k_x_ij - n_k_t * (1 - n_k_t / n_k) / (2 * (beta + n_k_t)**2)
    taylor_approx_n_dot_k_dot = E_log_W_beta_n_dot_k_dot - n_k * (1 - n_k / np.sum(n_k)) / (2 * (W * beta + n_k)**2)
    
    return taylor_approx_n_j_dot_k, taylor_approx_n_dot_k_x_ij, taylor_approx_n_dot_k_dot

def cvb0_exact_update(doc_word_ids, n_d_k, n_k_t, n_k, alpha, beta, V, K, z_d_i):
    # Initialize gamma_ijk
    gamma_ijk = np.full((len(doc_word_ids), V, K), 1.0 / K)

    for d, doc_ids in enumerate(doc_word_ids):
        for i, word_id in enumerate(doc_ids):
            old_topic = z_d_i[d][i]  # Ensure this is an integer. It should be, based on your initialization.
            n_d_k[d, old_topic] -= 1
            n_k_t[old_topic, word_id] -= 1
            n_k[old_topic] -= 1

            # Compute the expectation terms
            E_log_alpha_n_j_dot_k, E_log_beta_n_dot_k_x_ij, E_log_W_beta_n_dot_k_dot = compute_expectation_terms(
                gamma_ijk[d, :, :], alpha, beta, n_d_k[d, :], n_k_t[:, word_id], n_k, W=V
            )

            # Update gamma_ijk using the computed expectations (equation 18)
            for k in range(K):
                gamma_ijk[d, word_id, k] = np.exp(
                    E_log_alpha_n_j_dot_k[k] +
                    E_log_beta_n_dot_k_x_ij[k] -
                    E_log_W_beta_n_dot_k_dot[k]
                )
            
            # Normalize gamma_ijk
            gamma_ijk[d, word_id, :] /= np.sum(gamma_ijk[d, word_id, :])
        
            # Sample a new topic for the word
            new_topic = np.random.choice(K, p=gamma_ijk[d, word_id, :])
            z_d_i[d][i] = new_topic

            # Update the counts with the new topic assignment
            n_d_k[d, new_topic] += 1
            n_k_t[new_topic, word_id] += 1
            n_k[new_topic] += 1
            
    return gamma_ijk, z_d_i

if __name__ == '__main__':
    documents = [['word1', 'word2', 'word3'], ['word2', 'word3', 'word4'], ['word3', 'word4', 'word1']]
    K = 3  # Number of topics
    n_d_k, n_k_t, n_k, z_d_i, word_to_id, id_to_word, V , doc_word_ids= initialize_lda(documents, K)

    alpha, beta = 0.1, 0.1
    max_iters = 100
    for _ in range(max_iters):
        gamma_ijk, z_d_i = cvb0_exact_update(doc_word_ids, n_d_k, n_k_t, n_k, alpha, beta, V, K, z_d_i)  # Updated to include z_d_i

    print('Document-topic counts:', n_d_k)
    print('Topic-term counts:', n_k_t)
    print('Topic counts:', n_k)
    print('Topic assignments for words in documents:', z_d_i)
    print('Word to ID mapping:', word_to_id)
    print('ID to word mapping:', id_to_word)
    print('Vocabulary size:', V)
    print('Done')
