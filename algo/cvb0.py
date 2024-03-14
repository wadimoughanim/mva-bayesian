import numpy as np
from scipy.special import digamma, polygamma

class CollapsedVB:
    def __init__(self, documents, K, alpha=0.1, beta=0.1):
        self.documents = documents
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.V = self._build_vocabulary()
        self.D = len(documents)
        
        # Initialize counts and φ
        self.n_dk, self.n_kv, self.n_k = self._initialize_counts()
        self.phi_dnv = self._initialize_phi()

    def _build_vocabulary(self):
        vocabulary = set(word for doc in self.documents for word in doc)
        self.word_to_id = {word: id_ for id_, word in enumerate(vocabulary)}
        return len(vocabulary)

    def _initialize_counts(self):
        n_dk = np.zeros((self.D, self.K))
        n_kv = np.zeros((self.K, self.V))
        n_k = np.zeros(self.K)
        
        # Random initialization for counts
        for d, doc in enumerate(self.documents):
            for word in doc:
                word_id = self.word_to_id[word]
                topic = np.random.randint(0, self.K)
                n_dk[d, topic] += 1
                n_kv[topic, word_id] += 1
                n_k[topic] += 1
        return n_dk, n_kv, n_k

    def _initialize_phi(self):
        # Initialize φ with uniform probabilities
        phi_dnv = {}
        for d, doc in enumerate(self.documents):
            phi_dnv[d] = np.full((len(doc), self.K), 1.0 / self.K)
        return phi_dnv

    def _update_phi(self):
        for d, doc in enumerate(self.documents):
            for n, word in enumerate(doc):
                word_id = self.word_to_id[word]
                gamma_dn = np.zeros(self.K)
                for k in range(self.K):
                    # The sums here are not directly from the paper but are necessary to compute the means
                    # for each document-topic (n_dk), topic-word (n_kv), and topic (n_k) after excluding
                    # the current assignment of word n to topic k (phi_dnv)
                    n_dk_sum = np.sum(self.n_dk[d, :]) - self.phi_dnv[d][n, k]
                    n_kv_sum = np.sum(self.n_kv[k, :]) - self.phi_dnv[d][n, k]
                    n_k_sum = self.n_k[k] - self.phi_dnv[d][n, k]

                    # The means are computed as per equation (16) from the paper
                    # Mean of the Bernoulli variables for document-topic (n_dk),
                    # topic-word (n_kv), and topic (n_k)
                    E_n_dk = self.n_dk[d, k] - self.phi_dnv[d][n, k]
                    E_n_kv = self.n_kv[k, word_id] - self.phi_dnv[d][n, k]
                    E_n_k = self.n_k[k] - self.phi_dnv[d][n, k]

                    # Variance of the Bernoulli variables as per equation (16) from the paper
                    Var_n_dk = E_n_dk * (1 - E_n_dk / n_dk_sum)
                    Var_n_kv = E_n_kv * (1 - E_n_kv / n_kv_sum)
                    Var_n_k = E_n_k * (1 - E_n_k / n_k_sum)

                    # Compute the expectation of the logarithm of a sum, using a Taylor approximation
                    # as per equation (17) from the paper.
                    # This uses the digamma function to approximate the expectation
                    log_term_n_dk = digamma(self.alpha + E_n_dk) - digamma(self.alpha * self.K + n_dk_sum)
                    log_term_n_kv = digamma(self.beta + E_n_kv) - digamma(self.beta * self.V + n_kv_sum)
                    log_term_n_k = digamma(self.beta * self.V + E_n_k) - digamma(self.beta * self.V * self.K + n_k_sum)

                    # Correction terms for the variance in the fields as per the second term in equation (17) from the paper
                    correction_n_dk = -Var_n_dk / (2 * (self.alpha + E_n_dk)**2)
                    correction_n_kv = -Var_n_kv / (2 * (self.beta + E_n_kv)**2)
                    correction_n_k = -Var_n_k / (2 * (self.beta * self.V + E_n_k)**2)

                    # Updating gamma for document n and topic k as per equation (15) from the paper,
                    # which includes the correction factors for the variance.
                    gamma_dn[k] = np.exp(log_term_n_dk + correction_n_dk + log_term_n_kv + correction_n_kv + log_term_n_k + correction_n_k)

                # Normalize γ to ensure it sums to 1 across all topics for a given word in a document
                gamma_dn /= np.sum(gamma_dn)
                self.phi_dnv[d][n, :] = gamma_dn


    def run(self, iterations=100):
        for it in range(iterations):
            self._update_phi()
            if it % 10 == 0:
                print(f"Iteration {it}")

    def get_topic_distributions(self):
        theta_dk = self.n_dk + self.alpha
        theta_dk /= np.sum(theta_dk, axis=1, keepdims=True)
        return theta_dk

    def get_word_distributions(self):
        phi_kv = self.n_kv + self.beta
        phi_kv /= np.sum(phi_kv, axis=1, keepdims=True)
        return phi_kv

    def calculate_log_likelihood(self):
        """
        Calculate the approximate log-likelihood of the entire corpus.
        """
        log_likelihood = 0.0
        # Loop over all documents and words to calculate the likelihood
        for d, doc in enumerate(self.documents):
            for n, word in enumerate(doc):
                word_id = self.word_to_id[word]
                theta_d = self.n_dk[d, :] + self.alpha
                theta_d /= np.sum(theta_d)
                phi_k = self.n_kv[:, word_id] + self.beta
                phi_k /= np.sum(phi_k)
                log_likelihood += np.log(np.dot(theta_d, phi_k))
        return log_likelihood

    def calculate_perplexity(self):
        """
        Calculate the perplexity of the model on the data.
        """
        corpus_log_likelihood = self.calculate_log_likelihood()
        N = sum(len(doc) for doc in self.documents)  # Total number of words in the corpus
        perplexity = np.exp(-corpus_log_likelihood / N)
        return perplexity


if __name__ == "__main__":
    documents = [
        ["apple", "orange", "banana", "apple"],
        ["orange", "banana", "apple", "fruit"],
        ["berry", "fruit", "apple", "banana"],
        ["apple", "berry", "orange", "fruit"]
    ]
    K = 3
    cvb = CollapsedVB(documents, K)
    cvb.run(100)
    
    theta_dk = cvb.get_topic_distributions()
    phi_kv = cvb.get_word_distributions()

    print("Topic distributions per document:", theta_dk)
    print("Word distributions per topic:", phi_kv)
