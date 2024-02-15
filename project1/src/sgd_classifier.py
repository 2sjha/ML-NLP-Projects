"""
Creates SGD Classifier model from the Bernoulli, Bag-of-words representations of the datasets
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from representations import Representations
from datasets import Datasets
from accuracy import print_accuracy_data


class MySGDClassifier:
    """
    Contains data from SGD Classifier model
    """

    dtst_names = ["enron1", "enron2", "enron4"]

    def __init__(self, train_representations: Representations):
        self.clf = {
            "enron1": "",
            "enron2": "",
            "enron4": "",
        }

        self.test_datasets = Datasets("test")

        self.train_on_bernoulli(train_representations.bernoulli)
        self.test_on_bernoulli(train_representations)

        self.train_on_bow(train_representations.bag_of_words)
        self.test_on_bow(train_representations)

    def train_on_bernoulli(self, bernoulli_train_rep):
        """Trains from the berboulli representation"""
        for dtst in self.dtst_names:
            self.clf[dtst] = make_pipeline(
                StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)
            )
            features = []
            for data in bernoulli_train_rep[dtst]:
                features.append(data[0:-1])
            classification = []
            for data in bernoulli_train_rep[dtst]:
                classification.append(data[-1])

            self.clf[dtst].fit(features, classification)

    def test_on_bernoulli(self, train_rep):
        """Tests learned parameters from bernoulli representation against test dataset"""

        overall_true_positive = 0
        overall_true_negative = 0
        overall_false_positive = 0
        overall_false_negative = 0

        for dtst in self.dtst_names:
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            train_vocab = train_rep.datasets.vocabulary[dtst]
            vocab_len = len(train_vocab)

            for _ham_file, ham_email_words in self.test_datasets.ham_dataset[
                dtst
            ].items():
                ham_email_words = set(ham_email_words)

                sample_features = np.zeros(vocab_len)

                for word in train_vocab:
                    if word in ham_email_words:
                        sample_features[train_vocab[word]] = 1

                prediction = self.clf[dtst].predict([sample_features])

                if prediction == 1:
                    true_positive += 1
                    overall_true_positive += 1
                else:
                    false_negative += 1
                    overall_false_negative += 1

            for _spam_file, spam_email_words in self.test_datasets.spam_dataset[
                dtst
            ].items():
                spam_email_words = set(spam_email_words)

                sample_features = np.zeros(vocab_len)

                for word in train_vocab:
                    if word in spam_email_words:
                        sample_features[train_vocab[word]] = 1

                prediction = self.clf[dtst].predict([sample_features])

                if prediction == 0:
                    true_negative += 1
                    overall_true_negative += 1
                else:
                    false_positive += 1
                    overall_false_positive += 1

            print(
                "SGD Classifier stats for Bernoulli representation of "
                + dtst
                + " dataset:"
            )
            print_accuracy_data(
                true_positive, false_negative, true_negative, false_positive
            )

        # Print Overall Statistics
        print("Overall SGD Classifier stats for Bernoulli representation:")
        print_accuracy_data(
            overall_true_positive,
            overall_false_negative,
            overall_true_negative,
            overall_false_positive,
        )

    def train_on_bow(self, bow_train_rep):
        """Trains from the bag-of-words representation"""

        for dtst in self.dtst_names:
            clf = make_pipeline(
                StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)
            )
            features = []
            for data in bow_train_rep[dtst]:
                features.append(data[0:-1])
            classification = []
            for data in bow_train_rep[dtst]:
                classification.append(data[-1])

            clf.fit(features, classification)

    def test_on_bow(self, train_rep):
        """Tests learned parameters from the bag-of-words representation against the test dataset"""

        overall_true_positive = 0
        overall_true_negative = 0
        overall_false_positive = 0
        overall_false_negative = 0

        for dtst in self.dtst_names:
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            train_vocab = train_rep.datasets.vocabulary[dtst]
            vocab_len = len(train_vocab)

            for _ham_file, ham_email_words in self.test_datasets.ham_dataset[
                dtst
            ].items():
                ham_email_words = set(ham_email_words)

                sample_features = np.zeros(vocab_len)

                for word in train_vocab:
                    if word in ham_email_words:
                        sample_features[train_vocab[word]] = 1

                prediction = self.clf[dtst].predict([sample_features])

                if prediction == 1:
                    true_positive += 1
                    overall_true_positive += 1
                else:
                    false_negative += 1
                    overall_false_negative += 1

            for _spam_file, spam_email_words in self.test_datasets.spam_dataset[
                dtst
            ].items():
                spam_email_words = set(spam_email_words)

                sample_features = np.zeros(vocab_len)

                for word in train_vocab:
                    if word in spam_email_words:
                        sample_features[train_vocab[word]] = 1

                prediction = self.clf[dtst].predict([sample_features])

                if prediction == 0:
                    true_negative += 1
                    overall_true_negative += 1
                else:
                    false_positive += 1
                    overall_false_positive += 1

            print(
                "SGD Classifier stats for Bag-of-Words representation of "
                + dtst
                + " dataset:"
            )
            print_accuracy_data(
                true_positive, false_negative, true_negative, false_positive
            )

        # Print Overall Statistics
        print("Overall SGD Classifier stats for Bag-of-Words representation:")
        print_accuracy_data(
            overall_true_positive,
            overall_false_negative,
            overall_true_negative,
            overall_false_positive,
        )
