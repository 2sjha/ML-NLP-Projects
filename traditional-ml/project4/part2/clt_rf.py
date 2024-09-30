"""
Chow-Liu Tree Random Forest class
"""

from clt import ChowLiuTree
import numpy as np
from scipy.stats import gmean, hmean


class CLTRandomForest:
    """
    Implements Random Forest with Chow-Liu Trees
    """

    def __init__(self, num_trees, r):
        self.num_trees = num_trees
        self.r = r
        self.random_forest = []

    def learn(self, dataset):
        """
        Creates num_trees bootstrap samples from the dataset
        and CLTree instances. ith CLTree learns on the ith bootstrap sample
        """
        for _ in range(self.num_trees):
            bootstrap_sample = dataset[
                np.random.choice(dataset.shape[0], size=dataset.shape[0], replace=True)
            ]
            clt = ChowLiuTree()
            clt.learn(bootstrap_sample, self.r)
            self.random_forest.append(clt)

    def compute_ll(self, dataset):
        """
        Computes aggregate log-likelihood score by choosing the best value
        from the AM, GM and HM of the individual LL scores
        """
        log_likelihoods = []
        for clt in self.random_forest:
            log_likelihoods.append(clt.compute_ll(dataset) / dataset.shape[0])

        # Default method where p_i = 1/k i.e arithmetic mean of all the individual LL scores
        am = np.mean(log_likelihoods)
        # Taking Geometric mean and Harmonic mean of absolute values of all the individual LL scores
        gm = -1 * gmean(np.abs(log_likelihoods))
        hm = -1 * hmean(np.abs(log_likelihoods))

        return max(am, gm, hm)
