"""
This is the main module to run all the models
"""
from representations import Representations
from discrete_naive_bayes import DiscreteNaiveBayes
from multinomial_naive_bayes import MultinomialNaiveBayes
from sgd_classifier import MySGDClassifier
# from logistic_regression import LogisticRegression



class AllModels:
    """
    Contains data from all the models required
    """

    def __init__(self):
        self.train_representations = Representations("train")

        self.train_representations.create_bernoulli()
        self.discrete_naive_bayes = DiscreteNaiveBayes(self.train_representations)

        self.train_representations.create_bag_of_words()
        self.multinomial_naive_bayes = MultinomialNaiveBayes(self.train_representations)

        # self.logistic_regression = LogisticRegression(self.train_representations)

        self.sgd_classifier = MySGDClassifier(self.train_representations)


if __name__:
    all_models = AllModels()
