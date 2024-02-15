"""
Creates Logistic Regression model from the Bernoulli, Bag-of-words representations of the datasets
"""

import numpy as np
from scipy.special import expit
from representations import Representations


class LogisticRegression:
    """
    Contains data from Logistic Regression model
    """

    dtst_names = ["enron1", "enron2", "enron4"]

    def __init__(self, train_representations: Representations):
        self.weights = {"enron1": [], "enron2": [], "enron4": []}
        self.penalty_lambdas = {"enron1": 0.0, "enron2": 0.0, "enron4": 0.0}

        self.train_on_bernoulli(train_representations.bernoulli)
        self.test_on_bernoulli(train_representations.bernoulli)

        self.train_on_bow(train_representations.bag_of_words)
        self.test_on_bernoulli(train_representations.bag_of_words)

    def prob_y_equals_1_given_x_arr(self, dtst, x_arr) -> float:
        """
        Calculates probability(Y|x) given x sample using the sigmoid function
        """
        weighted_sum = 0.0
        weights_len = len(x_arr)
        for i in range(0, weights_len):
            weighted_sum += self.weights[dtst][i] * x_arr[i]

        exp_weighted_sum = expit(weighted_sum)
        return exp_weighted_sum / (1 + exp_weighted_sum)

    def gradient_ascent(
        self, dtst, train_rep, wt_idx, dataset_start, dataset_end, penalty_lambda
    ):
        """
        Performs Gradient Ascent for ith weight parameter
        """
        # Choose appropriate learning rate, and Put a hard limit on no. of iterations
        learning_rate = 0.1
        max_iterations = 500
        dataset_len = len(train_rep[dtst])

        for _i in range(0, max_iterations):
            prediction_sum = 0.0
            for j in range(
                np.floor(dataset_start * dataset_len).astype(int),
                np.floor(dataset_end * dataset_len).astype(int),
            ):
                prediction_sum += train_rep[dtst][j][wt_idx] * (
                    train_rep[dtst][j][-1]
                    - self.prob_y_equals_1_given_x_arr(dtst, train_rep[dtst][j][0:-1])
                )

            self.weights[dtst][wt_idx] += learning_rate * (
                prediction_sum
                - (
                    penalty_lambda  # L2 regularization
                    * self.weights[dtst][wt_idx]
                    * self.weights[dtst][wt_idx]
                )
            )
            print("weight " + str(wt_idx) +" updated to " + str(self.weights[dtst][wt_idx]))

    def train_on_bernoulli(self, bernoulli_train_rep):
        """Learns weights from the berboulli representation"""

        for dtst in self.dtst_names:
            # No bias weight
            # Init all weights with 1.0
            weights_len = len(bernoulli_train_rep[dtst][0]) - 1
            self.weights[dtst] = np.ones(weights_len)

            for i in range(0, weights_len):
                # Learn weights from 70% of the training data
                # Learn wih no penalty, lambda = 0
                self.gradient_ascent(dtst, bernoulli_train_rep, i, 0, 0.7, 0)

        # # Learn Lambda from the remaining training data
        # for dtst in self.dtst_names:
        #     for i in range(0, weights_len):
        #         # Learn for penalty lambda from remaining 30% of the training data

        #         penalty_lambda = 1.0
        #         self.gradient_ascent(
        #             dtst, i, bernoulli_train_rep, 0.7, 1.0, penalty_lambda
        #         )
        #         # update penalty_lambda

        # # Use this learned lambda to relearn weights from the full training set
        # for dtst in self.dtst_names:
        #     for i in range(0, weights_len):
        #         self.gradient_ascent(
        #             dtst, i, bernoulli_train_rep, 0, 1.0, self.penalty_lambdas[dtst]
        #         )

    def test_on_bernoulli(self, bernoulli_train_rep):
        """Tests learned weights from bernoulli representation against test dataset"""

    def train_on_bow(self, bow_train_rep):
        """learns weights from the bag-of-words representation"""

    def test_on_bow(self, bow_train_rep):
        """Tests learned weights from the bag-of-words representation against the test dataset"""
