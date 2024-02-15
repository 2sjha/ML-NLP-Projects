"""
Creates the Bernoulli and Bag-of-Words representations of the datasets
"""
import numpy as np
from datasets import Datasets


class Representations:
    """
    Contains the Bernoulli and Bag-of-Words representation of the datasets
    Final class = 1 for HAM
    Final class = 0 for SPAM
    """

    def __init__(self, test_train: str):
        self.datasets = Datasets(test_train)
        self.dtst_names = ["enron1", "enron2", "enron4"]
        self.bernoulli = {"enron1": [], "enron2": [], "enron4": []}
        self.bag_of_words = {"enron1": [], "enron2": [], "enron4": []}

    def create_bernoulli(self):
        """
        Creates a bernoulli matrix of features(col) X samples(row).
        Each row in the matrix contains a boolean vector of the words present in the sample
        """
        for dtst_name, dt_st in self.datasets.ham_dataset.items():
            # Iterate accross ham samples from enron1, enron2 & enron4
            for ham_email_words in dt_st.values():
                # For each sample, create a vocab_len size vector with 0s
                bernoulli_sample = np.zeros(self.datasets.vocab_len[dtst_name])
                # For each word in the email, set 1 for that word in the vector
                for word in ham_email_words:
                    idx = self.datasets.vocabulary[dtst_name][word]
                    bernoulli_sample[idx] = 1

                # Append HAM = 1 as the final class variable
                bernoulli_sample = np.append(bernoulli_sample, np.array([1]))
                self.bernoulli[dtst_name].append(bernoulli_sample)

        for dtst_name, dt_st in self.datasets.spam_dataset.items():
            # Iterate accross spam samples from enron1, enron2 & enron4
            for spam_email_words in dt_st.values():
                # For each sample, create a vocab_len size vector with 0s
                bernoulli_sample = np.zeros(self.datasets.vocab_len[dtst_name])
                # For each word in the email, set 1 for that word in the vector
                for word in spam_email_words:
                    idx = self.datasets.vocabulary[dtst_name][word]
                    bernoulli_sample[idx] = 1

                # Append SPAM = 0 as the final class variable
                bernoulli_sample = np.append(bernoulli_sample, np.array([0]))
                self.bernoulli[dtst_name].append(bernoulli_sample)

        # Export the bernoulli data into CSV files
        # for dtst_name in self.dtst_names:
        #     np.savetxt(
        #         "./../bernoulli-" + dtst_name + ".csv",
        #         self.bernoulli[dtst_name],
        #         delimiter=",",
        #         fmt="%d",
        #     )

    def create_bag_of_words(self):
        """
        Creates a bag-of-words matrix of features(col) X samples(row)
        Each row in the matrix contains a vector of count of words present in the sample
        """
        for dtst_name, dt_st in self.datasets.ham_dataset.items():
            # Iterate accross ham samples from enron1, enron2 & enron4
            for ham_email_words in dt_st.values():
                # For each sample, create a vocab_len size vector with 0s
                bow_sample = np.zeros(self.datasets.vocab_len[dtst_name])
                # For each word in the email, set 1 for that word in the vector
                for word in ham_email_words:
                    idx = self.datasets.vocabulary[dtst_name][word]
                    bow_sample[idx] += 1

                # Append HAM = 1 as the final class variable
                bow_sample = np.append(bow_sample, np.array([1]))
                self.bag_of_words[dtst_name].append(bow_sample)

        for dtst_name, dt_st in self.datasets.spam_dataset.items():
            # Iterate accross spam samples from enron1, enron2 & enron4
            for spam_email_words in dt_st.values():
                # For each sample, create a vocab_len size vector with 0s
                bow_sample = np.zeros(self.datasets.vocab_len[dtst_name])
                # For each word in the email, set 1 for that word in the vector
                for word in spam_email_words:
                    idx = self.datasets.vocabulary[dtst_name][word]
                    bow_sample[idx] += 1

                # Append SPAM = 0 as the final class variable
                bow_sample = np.append(bow_sample, np.array([0]))
                self.bag_of_words[dtst_name].append(bow_sample)

        # Export the Bag-of-Words data into CSV files
        # for dtst_name in self.dtst_names:
        #     np.savetxt(
        #         "./../bag_of_words-" + dtst_name + ".csv",
        #         self.bag_of_words[dtst_name],
        #         delimiter=",",
        #         fmt="%d",
        #     )
