import numpy as np
from init import initialize_lda

###MOCK
def cvb0_update_mock(doc_word_ids, n_d_k, n_k_t, n_k, z_d_i, alpha, beta, V, K):
    for d, doc in enumerate(doc_word_ids):
        for i, word_id in enumerate(doc):
            old_topic = z_d_i[d][i]
            n_d_k[d, old_topic] -= 1
            n_k_t[old_topic, word_id] -= 1
            n_k[old_topic] -= 1
            
            topic_probs = np.zeros(K)
            for k in range(K):
                topic_probs[k] = (n_k_t[k, word_id] + beta) / (n_k[k] + V * beta) * (n_d_k[d, k] + alpha)
            topic_probs /= np.sum(topic_probs)  # Normalize
            
            new_topic = np.random.choice(np.arange(K), p=topic_probs)
            z_d_i[d][i] = new_topic
            
            n_d_k[d, new_topic] += 1
            n_k_t[new_topic, word_id] += 1
            n_k[new_topic] += 1
    return z_d_i


def run_cvb0(documents, K, alpha, beta, max_iters=100):
    doc_word_ids, n_d_k, n_k_t, n_k, word_to_id, id_to_word, V, z_d_i = initialize_lda(documents, K)
    for _ in range(max_iters):
        z_d_i = cvb0_update_mock(doc_word_ids, n_d_k, n_k_t, n_k, z_d_i, alpha, beta, V, K)
    
    return doc_word_ids, n_d_k, n_k_t, n_k, word_to_id, id_to_word, V, z_d_i

if __name__ == '__main__':
    from data.mock import documents
    K =3 
    _, n_d_k, n_k_t, _, _, _, _, _ = run_cvb0(documents, K, alpha=0.1, beta=0.1)
    
    #
    print('Done')

#### ACTUAL
    
import numpy as np
from scipy.special import digamma, gammaln

def compute_expectation_terms(gamma_ijk, alpha, beta, n_j_dot_k, n_dot_k_x_ij, n_dot_k_dot, W):
    """
    Compute expectation terms using Gaussian approximation and second-order Taylor expansion.
    """
    # equation 16)
    E_gamma_ijk = np.sum(gamma_ijk, axis=0)
    Var_gamma_ijk = np.sum(gamma_ijk * (1 - gamma_ijk), axis=0) 
    
    #(equation 17)
    E_log_alpha_n_j_dot_k = np.log(alpha + E_gamma_ijk) - Var_gamma_ijk / (2 * (alpha + E_gamma_ijk)**2)
    E_log_beta_n_dot_k_x_ij = np.log(beta + n_dot_k_x_ij) - n_dot_k_x_ij * (1 - n_dot_k_x_ij) / (2 * (beta + n_dot_k_x_ij)**2)
    E_log_W_beta_n_dot_k_dot = np.log(W * beta + n_dot_k_dot) - n_dot_k_dot * (1 - n_dot_k_dot) / (2 * (W * beta + n_dot_k_dot)**2)
    
    return E_log_alpha_n_j_dot_k, E_log_beta_n_dot_k_x_ij, E_log_W_beta_n_dot_k_dot

def cvb0_exact_update(doc_word_ids, n_d_k, n_k_t, n_k, alpha, beta, V, K, z_d_i):
    """
    Perform the exact CVB0 update using Gaussian approximations for efficiency.
    """
    gamma_ijk = np.full_like(n_d_k, 1.0 / K)
    
    for d, doc in enumerate(doc_word_ids):
        for i, word_id in enumerate(doc):
            old_topic = z_d_i[d][i]
            n_d_k[d, old_topic] -= 1
            n_k_t[old_topic, word_id] -= 1
            n_k[old_topic] -= 1
            
            E_log_alpha_n_j_dot_k, E_log_beta_n_dot_k_x_ij, E_log_W_beta_n_dot_k_dot = compute_expectation_terms(
                gamma_ijk[d], alpha, beta, n_d_k[d], n_k_t[:, word_id], n_k, V
            )
            
            # Compute the new topic probabilities using equation (18)
            gamma_ijk[d] = np.exp(E_log_alpha_n_j_dot_k + E_log_beta_n_dot_k_x_ij - E_log_W_beta_n_dot_k_dot)
            gamma_ijk[d] /= np.sum(gamma_ijk[d])  #Normalize the probabilities
            new_topic = np.random.choice(K, p=gamma_ijk[d])
            z_d_i[d][i] = new_topic
            
            # Update the counts using the new topic assignment
            n_d_k[d, new_topic] += 1
            n_k_t[new_topic, word_id] += 1
            n_k[new_topic] += 1
            
    return gamma_ijk, z_d_i

if __name__ == '__main__':
    from data.mock import documents
    K =3 
    _, n_d_k, n_k_t, _, _, _, V, z_d_i = initialize_lda(documents, K)
    gamma_ijk, z_d_i = cvb0_exact_update(documents, n_d_k, n_k_t, _, alpha=0.1, beta=0.1, V=V, K=K, z_d_i=z_d_i)
    print('Done')
