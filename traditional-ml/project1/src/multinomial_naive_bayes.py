"""
Creates Multinomial Naive Bayes model from the Bag-of-words representation of the datasets
"""
from math import log
from representations import Representations
from datasets import Datasets
from accuracy import print_accuracy_data


class MultinomialNaiveBayes:
    """
    Contains data from Multinomial Naive Bayes model
    """

    dtst_names = ["enron1", "enron2", "enron4"]

    def __init__(self, train_representations: Representations):
        # Index 0 = SPAM & 1 = HAM
        # Stores parameters in log-space
        self.class_parameters = {
            "enron1": [0.0, 0.0],
            "enron2": [0.0, 0.0],
            "enron4": [0.0, 0.0],
        }

        # Index 0 : Xj, Y = 0 (SPAM)
        # Index 1 : Xj, Y = 1 (HAM)
        # Stores parameters in log-space
        self.conditional_parameters = {
            "enron1": [[], []],
            "enron2": [[], []],
            "enron4": [[], []],
        }
        # Separate dataset for testing data
        self.test_datasets = ""
        self.train_representations = train_representations

        self.train(self.train_representations.bag_of_words)
        self.test()

    def train(self, train_bow_representation: dict):
        """
        Learns required parameters from bernoulli data representation
        """

        for dtst_name in self.dtst_names:
            dtst_rep = train_bow_representation[dtst_name]
            total_dtst_size = len(dtst_rep)
            # last element is the class variable
            vocab_len = len(dtst_rep[0]) - 1

            # Calculating class parameter
            count_ham = 0
            count_spam = 0
            for i in range(0, total_dtst_size):
                if dtst_rep[i][-1] == 0:
                    count_spam += 1
                else:
                    count_ham += 1
            self.class_parameters[dtst_name][0] = log(count_spam) - log(total_dtst_size)
            self.class_parameters[dtst_name][1] = log(count_ham) - log(total_dtst_size)

            # Calculating feature parameters
            total_word_count_ham = 0
            total_word_count_spam = 0
            for row in range(0, total_dtst_size):
                if dtst_rep[row][-1] == 0:
                    for col in range(0, vocab_len):
                        total_word_count_spam += dtst_rep[row][col]
                else:
                    for col in range(0, vocab_len):
                        total_word_count_ham += dtst_rep[row][col]

            for col in range(0, vocab_len):
                word_count_ham = 0
                word_count_spam = 0

                for row in range(0, total_dtst_size):
                    if dtst_rep[row][-1] == 0:
                        word_count_spam += dtst_rep[row][col]
                    else:
                        word_count_ham += dtst_rep[row][col]

                self.conditional_parameters[dtst_name][0].append(
                    log(word_count_spam + 1) - log(vocab_len + total_word_count_spam)
                )

                self.conditional_parameters[dtst_name][1].append(
                    log(word_count_ham + 1) - log(vocab_len + total_word_count_ham)
                )

    def test(self):
        """
        Validates learned parameters from bernoulli data representation against the test data
        """

        self.test_datasets = Datasets("test")
        overall_true_positive = 0
        overall_true_negative = 0
        overall_false_positive = 0
        overall_false_negative = 0

        # Test against Ham dataset
        for dtst_name, dt_st in self.test_datasets.ham_dataset.items():
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            train_vocab = self.train_representations.datasets.vocabulary[dtst_name]

            # Iterate accross ham samples from enron1, enron2 & enron4
            for ham_email_words in dt_st.values():
                # Create bag of words for test document
                ham_email_bag_of_words = {}
                for word in set(ham_email_words):
                    ham_email_bag_of_words[word] = 0

                for word in ham_email_words:
                    ham_email_bag_of_words[word] += 1

                # Calculate estimate for SPAM
                predict_spam = 0.0
                # Add class parameter of SPAM
                predict_spam += self.class_parameters[dtst_name][0]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in ham_email_bag_of_words:
                        # Add conditional parameter of word present, class = SPAM
                        predict_spam += (
                            ham_email_bag_of_words[word]
                            * self.conditional_parameters[dtst_name][0][word_idx]
                        )

                # Calculate estimate for HAM
                predict_ham = 0.0
                # Add class parameter of HAM
                predict_ham += self.class_parameters[dtst_name][1]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in ham_email_bag_of_words:
                        # Add conditional parameter of word present, class = HAM
                        predict_ham += (
                            ham_email_bag_of_words[word]
                            * self.conditional_parameters[dtst_name][1][word_idx]
                        )

                # Print predicted vs Actual = HAM
                if predict_ham > predict_spam:
                    # print("Correct prediction, Actual = HAM, Predicted = HAM")
                    # Correctly predicted Positive instances. Positive = HAM, Negative = SPAM
                    true_positive += 1
                    overall_true_positive += 1
                else:
                    # print("Incorrect prediction, Actual = HAM, Predicted = SPAM")
                    # Incorrectly predicted Negative instances. Positive = HAM, Negative = SPAM
                    false_negative += 1
                    overall_false_negative += 1

        # Test against Ham dataset
        for dtst_name, dt_st in self.test_datasets.spam_dataset.items():
            train_vocab = self.train_representations.datasets.vocabulary[dtst_name]

            # Iterate accross ham samples from enron1, enron2 & enron4
            for spam_email_words in dt_st.values():
                # Create bag of words for test document
                spam_email_bag_of_words = {}
                for word in set(spam_email_words):
                    spam_email_bag_of_words[word] = 0
                for word in spam_email_words:
                    spam_email_bag_of_words[word] += 1

                # Calculate estimate for SPAM
                predict_spam = 0.0
                # Add class parameter of SPAM
                predict_spam += self.class_parameters[dtst_name][0]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in spam_email_bag_of_words:
                        # Add conditional parameter of word present, class = SPAM
                        predict_spam += (
                            spam_email_bag_of_words[word]
                            * self.conditional_parameters[dtst_name][0][word_idx]
                        )

                # Calculate estimate for HAM
                predict_ham = 0.0
                # Add class parameter of HAM
                predict_ham += self.class_parameters[dtst_name][1]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in spam_email_bag_of_words:
                        # Add conditional parameter of word present, class = HAM
                        predict_ham += (
                            spam_email_bag_of_words[word]
                            * self.conditional_parameters[dtst_name][1][word_idx]
                        )

                # Print predicted vs Actual = HAM
                if predict_spam > predict_ham:
                    # print("Correct prediction, Actual = SPAM, Predicted = SPAM")
                    # Correctly predicted Negative instances. Positive = HAM, Negative = SPAM
                    true_negative += 1
                    overall_true_negative += 1
                else:
                    # print("Incorrect prediction, Actual = SPAM, Predicted = HAM")
                    # Incorrectly predicted Positive instances. Positive = HAM, Negative = SPAM
                    false_positive += 1
                    overall_false_positive += 1

            print("Multinomial Naive-Bayes stats for " + dtst_name + " dataset:")
            print_accuracy_data(
                true_positive, false_negative, true_negative, false_positive
            )

        # Print Overall Statistics
        print("Overall Multinomial Naive-Bayes stats:")
        print_accuracy_data(
            overall_true_positive,
            overall_false_negative,
            overall_true_negative,
            overall_false_positive,
        )
