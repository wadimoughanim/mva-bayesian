import numpy as np

def initialize_lda(documents, K):
    """
    Initialize the LDA model with random topic assignments and compute necessary counts.
    
    Parameters:
    - documents: A list of documents, each document is a list of preprocessed words.
    - K: The number of topics.
    
    Returns:
    - doc_word_ids: Encoded documents with word IDs.
    - n_d_k: Document-topic counts.
    - n_k_t: Topic-term counts.
    - n_k: Total count of words assigned to each topic.
    - word_to_id: A dictionary mapping words to unique IDs.
    - id_to_word: A dictionary mapping unique IDs to words.
    - V: The size of the vocabulary.
    - z_d_i: The initial random topic assignment for each word in each document.
    """
    word_to_id = {}
    id_to_word = {}
    current_id = 0

    doc_word_ids = []
    for doc in documents:
        doc_ids = []
        for word in doc:
            if word not in word_to_id:
                word_to_id[word] = current_id
                id_to_word[current_id] = word
                current_id += 1
            doc_ids.append(word_to_id[word])
        doc_word_ids.append(doc_ids)

    V = len(word_to_id)  # Vocabulary size

    # Randomly assign topics to each word in each document
    z_d_i = [[np.random.randint(K) for _ in doc] for doc in doc_word_ids]

    # Initialize count matrices and topic totals
    n_d_k = np.zeros((len(documents), K))  #topic count
    n_k_t = np.zeros((K, V))  # term counts
    n_k = np.zeros(K)  # words assigned to each topic

    # Count matrices b: randam
    for d, doc in enumerate(doc_word_ids):
        for i, word_id in enumerate(doc):
            topic = z_d_i[d][i]
            n_d_k[d, topic] += 1
            n_k_t[topic, word_id] += 1
            n_k[topic] += 1

    return doc_word_ids, n_d_k, n_k_t, n_k, word_to_id, id_to_word, V, z_d_i

if __name__ == '__main__':
    # mock test -> unit test si on est chaud
    from data.mock import documents
    K = 3
    # Correct the order of unpacked values to match the function's return
    doc_word_ids, n_d_k, n_k_t, n_k, word_to_id, id_to_word, V, z_d_i = initialize_lda(documents, K)
    print(doc_word_ids)
    print(n_d_k)
    print(n_k_t)
    print(n_k)
    print(word_to_id)
    print(id_to_word)
    print(V)
    print(K)
    print(n_d_k.shape)
    print(n_k_t.shape)