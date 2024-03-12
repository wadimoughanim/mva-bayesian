import numpy as np
from init import initialize_lda

def cvb0_update(doc_word_ids, n_d_k, n_k_t, n_k, z_d_i, alpha, beta, V, K):
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
        z_d_i = cvb0_update(doc_word_ids, n_d_k, n_k_t, n_k, z_d_i, alpha, beta, V, K)
    
    return doc_word_ids, n_d_k, n_k_t, n_k, word_to_id, id_to_word, V, z_d_i

if __name__ == '__main__':
    from data.mock import documents
    K =3 
    _, n_d_k, n_k_t, _, _, _, _, _ = run_cvb0(documents, K, alpha=0.1, beta=0.1)
    
    #
    print("Document-topic counts shape:", n_d_k.shape)
    print("Topic-term counts shape:", n_k_t.shape)