### CVB
from algo.cvb0 import cvb0_update, run_cvb0
from init import initialize_lda
import matplotlib.pyplot as plt
from data.mock import documents

def plot_metrics(n_d_k, n_k_t):
    plt.figure(figsize=(10, 4))
    plt.bar(range(n_d_k.shape[1]), n_d_k[0], tick_label=range(n_d_k.shape[1]))
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.title('Topic Distribution for Document 0')
    plt.show()
    plt.figure(figsize=(10, 4))
    plt.bar(range(n_k_t.shape[1]), n_k_t[0], tick_label=range(n_k_t.shape[1]))
    plt.xlabel('word')
    plt.ylabel('count')
    plt.title('word Distrib for Topic 0')
    plt.show()

if __name__ == '__main__':
    K=3
    alpha, beta = 0.1, 0.1
    max_iters=100
    doc_word_ids, n_d_k, n_k_t, n_k, word_to_id, id_to_word, V, z_d_i = run_cvb0(documents, K, alpha, beta, max_iters)
    plot_metrics(n_d_k, n_k_t)
