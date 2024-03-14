import numpy as np
from scipy.special import digamma, polygamma

class CollapsedVB:
    def __init__(self, documents, K, alpha=0.01, beta=0.01):
        self.documents = documents
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.V = self._build_vocabulary()
        self.D = len(documents)
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
        
        for d, doc in enumerate(self.documents):
            for word in doc:
                word_id = self.word_to_id[word]
                topic = np.random.randint(0, self.K)
                n_dk[d, topic] += 1
                n_kv[topic, word_id] += 1
                n_k[topic] += 1
        return n_dk, n_kv, n_k

    def _initialize_phi(self):
        phi_dnv = {}
        for d, doc in enumerate(self.documents):
            phi_dnv[d] = np.full((len(doc), self.K), 1.0 / self.K)
        return phi_dnv

    def _update_phi(self):
        epsilon = 1e-10 # Aavoid division by zero

        for d, doc in enumerate(self.documents):
            for n, word in enumerate(doc):
                word_id = self.word_to_id[word]
                gamma_dn = np.zeros(self.K)

                for k in range(self.K):
                    # Excluding the current word from the counts
                    phi_dnv_excluded = self.phi_dnv[d][n, k]

                    # Compute expected values with smoothing
                    E_n_dk = self.n_dk[d, k] - phi_dnv_excluded + epsilon 
                    E_n_kv = self.n_kv[k, word_id] - phi_dnv_excluded + epsilon
                    E_n_k = self.n_k[k] - phi_dnv_excluded + epsilon
                    #Compute variances with smoothing
                    Var_n_dk = E_n_dk * (1 - E_n_dk / (self.n_dk[d, :].sum() - phi_dnv_excluded + epsilon))
                    Var_n_kv = E_n_kv * (1 - E_n_kv / (self.n_kv[k, :].sum() - phi_dnv_excluded + epsilon))
                    Var_n_k = E_n_k * (1 - E_n_k / (self.n_k.sum() - phi_dnv_excluded + epsilon))

                    # Use the Gaussian approximation for the digamma function around the expected values
                    # This replaces the correction_n_* terms from your current code
                    approx_digamma_n_dk = digamma(self.alpha + E_n_dk)
                    approx_digamma_n_kv = digamma(self.beta + E_n_kv)
                    approx_digamma_n_k = digamma(self.beta * self.V + E_n_k)
                    
                    # Adjust gamma_dn with Gaussian approximations
                    gamma_dn[k] = np.exp(
                        approx_digamma_n_dk - digamma(self.alpha * self.K + self.n_dk[d, :].sum() + epsilon)
                        + approx_digamma_n_kv - digamma(self.beta * self.V + self.n_kv[k, :].sum() + epsilon)
                        + approx_digamma_n_k - digamma(self.beta * self.V * self.K + self.n_k.sum() + epsilon)
                        # Incorporate the variance approximations here
                        - Var_n_dk / (2 * (self.alpha + E_n_dk)**2)
                        - Var_n_kv / (2 * (self.beta + E_n_kv)**2)
                        - Var_n_k / (2 * (self.beta * self.V + E_n_k)**2)
                    )

                    


                # Normalizitaion
                if np.sum(gamma_dn) == 0:
                    print(f"Sum of gamma is zero for document {d}, word {n}")
                    gamma_dn += epsilon
                gamma_dn /= np.sum(gamma_dn)
                self.phi_dnv[d][n, :] = gamma_dn 

                # Debug output
                if np.any(np.abs(self.phi_dnv[d][n, :] - gamma_dn) > 1e-3):
                    print(f"Significant change in phi for Document {d}, Word {n}: {gamma_dn}")


    # def run(self, iterations=100):
    #     for it in range(iterations):
    #         self._update_phi()
    #         if it % 10 == 0:
    #             print(f"Iteration {it}")

    def get_topic_distributions(self):
        theta_dk = self.n_dk + self.alpha
        theta_dk /= np.sum(theta_dk, axis=1, keepdims=True)
        return theta_dk

    def get_word_distributions(self):
        phi_kv = self.n_kv + self.beta
        phi_kv /= np.sum(phi_kv, axis=1, keepdims=True)
        return phi_kv

    def run(self, iterations=100):
        # Initializing lists to store statistics at each iteration
        self.log_likelihoods = []
        self.perplexities = []
        self.word_ELBO = []  # Track ELBO for the first word in the first document

        for it in range(iterations):
            self._update_phi()
            print(self.phi_dnv)
            word_ELBO = self.calculate_word_ELBO(0, 0)
            self.word_ELBO.append(word_ELBO)
            log_likelihood = self.calculate_log_likelihood()
            self.log_likelihoods.append(log_likelihood)
            perplexity = self.calculate_perplexity(log_likelihood)
            self.perplexities.append(perplexity)

            if it % 10 == 0:
                print(f"Iteration {it}: Log Likelihood: {log_likelihood}, Perplexity: {perplexity}, Word ELBO: {word_ELBO}")

    def calculate_word_ELBO(self, d, n):
        word_id = self.word_to_id[self.documents[d][n]]
        term1 = digamma(self.n_kv[:, word_id] + self.beta) - digamma(self.n_k + self.beta * self.V)
        term2 = digamma(self.n_dk[d, :] + self.alpha) - digamma(self.n_dk[d, :].sum() + self.alpha * self.K)
        word_ELBO = np.dot(self.phi_dnv[d][n, :], term1 + term2)
        return word_ELBO
    
    def calculate_log_likelihood(self):
        log_likelihood = 0.0
        for d, doc in enumerate(self.documents):
            for n, word in enumerate(doc):
                word_id = self.word_to_id[word]
                # Adding self.beta to avoid division by zero
                phi_kv = (self.n_kv[:, word_id] + self.beta) / (self.n_k + self.beta * self.V)
                # Ensure no zero values in the dot product argument
                phi_kv = np.clip(phi_kv, 1e-10, 1.0)  # Prevent phi_kv from having 0 values
                if np.isnan(np.log(np.dot(self.phi_dnv[d][n, :], phi_kv))):
                    pass
                else:
                    log_likelihood += np.log(np.dot(self.phi_dnv[d][n, :], phi_kv))
        return log_likelihood


    def calculate_perplexity(self, log_likelihood):
        N = sum(len(doc) for doc in self.documents)  #Total number of words in the corpus
        #Check if log_likelihood is not NaN
        if not np.isnan(log_likelihood) and N > 0:
            perplexity = np.exp(-log_likelihood / N)
        else:
            perplexity = float('inf')  #Assign infinity if the log likelihood is NaN
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


    import matplotlib.pyplot as plt
    plt.plot(cvb.log_likelihoods, label='Log Likelihood')
    plt.plot(cvb.perplexities, label='Perplexity')
    plt.xlabel('Iterations')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))

    iterations = list(range(1, 101))  # 100 iterations
    cvb_metrics = cvb.log_likelihoods 
    plt.plot(iterations, cvb_metrics, label='CVB Log Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for d in range(4):
        ax[d // 2, d % 2].bar(range(K), theta_dk[d])
        ax[d // 2, d % 2].set_title(f"Document {d}")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for k in range(3):
        ax[k].bar(range(cvb.V), phi_kv[k])
        ax[k].set_title(f"Topic {k}")

    plt.show()

    plt.figure()
    plt.plot(range(1, 101), cvb.word_ELBO, label='Per-word ELBO')
    plt.xlabel('Iterations')
    plt.ylabel('Per-word ELBO')
    plt.title('Per-word Variational Bounds over Iterations')
    plt.legend()
    plt.show()