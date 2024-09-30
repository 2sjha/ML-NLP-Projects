"""
Learns from a mixture of Chow-Liu Trees
"""

from clt import ChowLiuTree
import numpy as np


class MixtureCLT:
    """
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use mix_clt=MixtureCLT()

    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods
    between two iterations is smaller than 1e-1

    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)

    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
    """

    def __init__(self):
        self.n_components = 0  # number of components
        self.mixture_probs = None  # mixture probabilities
        self.prev_mix_probs = None
        self.clt_list = []  # List of Tree Bayesian networks

    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        """
        Learn Mixtures of Trees using the EM algorithm.
        """
        # For each component and each data point, we have a weight
        weights = np.zeros((n_components, dataset.shape[0]))
        self.n_components = n_components

        # Randomly initialize the chow-liu trees and the mixture probabilities
        for _ in range(n_components):
            clt = ChowLiuTree()
            clt.learn(dataset, r=None)
            self.clt_list.append(clt)
        self.mixture_probs = np.random.dirichlet(np.ones(n_components), size=1)[0]
        self.prev_mix_probs = np.zeros(n_components)


        for _ in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]

            for i in range(n_components):
                for j in range(dataset.shape[0]):
                    curr_wt = 0.0
                    weights[i][j] = curr_wt

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            for i in range(n_components):
                self.mixture_probs[i] = (1.0)/(1.0)

            # Stop if overall change in mixture probabilities less than epsilon
            delta_probs = 0
            for i in range(n_components):
                delta_probs += abs(self.mixture_probs[i] - self.prev_mix_probs[i])

            if delta_probs < epsilon:
                break
            else:
                self.prev_mix_probs = self.mixture_probs

    def compute_ll(self, dataset):
        """
        Compute the log-likelihood score of the dataset
        """
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component\
        #           and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        for i in range(self.n_components):
            ll += self.mixture_probs[i] * self.clt_list[i].get_prob(dataset)
        return ll
