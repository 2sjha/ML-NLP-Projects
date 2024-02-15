"""
Creates Discrete Naive Bayes model from the Bernoulli representation of the datasets
"""
from math import log
from representations import Representations
from datasets import Datasets
from accuracy import print_accuracy_data


class DiscreteNaiveBayes:
    """
    Contains data from Discrete Naive Bayes model
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

        # Index 0 : Xj = 0 (word not present), Y = 0 (SPAM)
        # Index 1 : Xj = 0 (word not present), Y = 1 (HAM)
        # Index 2 : Xj = 1 (word present), Y = 0 (SPAM)
        # Index 3 : Xj = 1 (word present), Y = 1 (HAM)
        # Stores parameters in log-space
        self.conditional_parameters = {
            "enron1": [[], [], [], []],
            "enron2": [[], [], [], []],
            "enron4": [[], [], [], []],
        }
        # Separate dataset for testing data
        self.test_datasets = ""
        self.train_representations = train_representations

        self.train(self.train_representations.bernoulli)
        self.test(self.train_representations.bernoulli)

    def train(self, train_bernoulli_representation: dict):
        """
        Learns required parameters from bernoulli data representation
        """

        for dtst_name in self.dtst_names:
            dtst_rep = train_bernoulli_representation[dtst_name]
            count_ham = 0
            count_spam = 0
            total_dtst_size = len(dtst_rep)
            # last element is the class variable
            vocab_len = len(dtst_rep[0]) - 1

            # Calculating class parameter
            for i in range(0, total_dtst_size):
                if dtst_rep[i][-1] == 0:
                    count_spam += 1
                else:
                    count_ham += 1
            self.class_parameters[dtst_name][0] = log(count_spam) - log(total_dtst_size)
            self.class_parameters[dtst_name][1] = log(count_ham) - log(total_dtst_size)

            # Calculating feature parameters
            for col in range(0, vocab_len):
                count_not_present_spam = 0
                count_not_present_ham = 0
                count_present_spam = 0
                count_present_ham = 0

                for row in range(0, total_dtst_size):
                    if dtst_rep[row][col] == 0 and dtst_rep[row][-1] == 0:
                        count_not_present_spam += 1
                    elif dtst_rep[row][col] == 0 and dtst_rep[row][-1] == 1:
                        count_not_present_ham += 1
                    elif dtst_rep[row][col] == 1 and dtst_rep[row][-1] == 0:
                        count_present_spam += 1
                    else:
                        count_present_ham += 1

                if count_not_present_spam == 0:
                    self.conditional_parameters[dtst_name][0].append(-1000)
                else:
                    self.conditional_parameters[dtst_name][0].append(
                        log(count_not_present_spam) - log(count_spam)
                    )
                if count_not_present_ham == 0:
                    self.conditional_parameters[dtst_name][1].append(-1000)
                else:
                    self.conditional_parameters[dtst_name][1].append(
                        log(count_not_present_ham) - log(count_ham)
                    )
                if count_present_spam == 0:
                    self.conditional_parameters[dtst_name][2].append(-1000)
                else:
                    self.conditional_parameters[dtst_name][2].append(
                        log(count_present_spam) - log(count_spam)
                    )
                if count_present_ham == 0:
                    self.conditional_parameters[dtst_name][3].append(-1000)
                else:
                    self.conditional_parameters[dtst_name][3].append(
                        log(count_present_ham) - log(count_ham)
                    )

    def test(self, train_bernoulli_representation: dict):
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

            train_dataset = train_bernoulli_representation[dtst_name]
            train_dataset_sz = len(train_dataset)
            train_vocab = self.train_representations.datasets.vocabulary[dtst_name]

            # Iterate accross ham samples from enron1, enron2 & enron4
            for ham_email_words in dt_st.values():
                ham_email_words = set(ham_email_words)
                unseen_words = [
                    word for word in ham_email_words if word not in train_vocab.keys()
                ]

                # Assume we've seen each of these words once in the training dataset
                # 1-laplace-smoothing for each of these unseen words
                if len(unseen_words) > 0:
                    laplace_smoothing = log(len(unseen_words)) - log(train_dataset_sz)
                else:
                    laplace_smoothing = 0

                # Calculate estimate for SPAM
                predict_spam = 0.0
                predict_spam += laplace_smoothing
                # Add class parameter of SPAM
                predict_spam += self.class_parameters[dtst_name][0]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in ham_email_words:
                        # Add conditional parameter of word present, class = SPAM
                        predict_spam += self.conditional_parameters[dtst_name][2][
                            word_idx
                        ]
                    else:
                        # Add conditional parameter of word not present, class = SPAM
                        predict_spam += self.conditional_parameters[dtst_name][0][
                            word_idx
                        ]

                # Calculate estimate for HAM
                predict_ham = 0.0
                predict_ham += laplace_smoothing
                # Add class parameter of HAM
                predict_ham += self.class_parameters[dtst_name][1]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in ham_email_words:
                        # Add conditional parameter of word present, class = HAM
                        predict_ham += self.conditional_parameters[dtst_name][3][
                            word_idx
                        ]
                    else:
                        # Add conditional parameter of word not present, class = HAM
                        predict_ham += self.conditional_parameters[dtst_name][1][
                            word_idx
                        ]

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

        # Test against Spam Dataset
        for dtst_name, dt_st in self.test_datasets.spam_dataset.items():
            train_dataset = train_bernoulli_representation[dtst_name]
            train_dataset_sz = len(train_dataset)
            train_vocab = self.train_representations.datasets.vocabulary[dtst_name]

            # Iterate accross spam samples from enron1, enron2 & enron4
            for spam_email_words in dt_st.values():
                spam_email_words = set(spam_email_words)
                unseen_words = [
                    word for word in spam_email_words if word not in train_vocab.keys()
                ]

                # Assume we've seen each of these words once in the training dataset
                # 1-laplace-smoothing for each of these unseen words
                if len(unseen_words) > 0:
                    laplace_smoothing = log(len(unseen_words)) - log(train_dataset_sz)
                else:
                    laplace_smoothing = 0

                # Calculate estimate for SPAM
                predict_spam = 0.0
                predict_spam += laplace_smoothing
                # Add class parameter of SPAM
                predict_spam += self.class_parameters[dtst_name][0]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in spam_email_words:
                        # Add conditional parameter of word present, class = SPAM
                        predict_spam += self.conditional_parameters[dtst_name][2][
                            word_idx
                        ]
                    else:
                        # Add conditional parameter of word not present, class = SPAM
                        predict_spam += self.conditional_parameters[dtst_name][0][
                            word_idx
                        ]

                # Calculate estimate for HAM
                predict_ham = 0.0
                predict_ham += laplace_smoothing
                # Add class parameter of HAM
                predict_ham += self.class_parameters[dtst_name][1]

                for word in train_vocab:
                    word_idx = train_vocab[word]
                    if word in spam_email_words:
                        # Add conditional parameter of word present, class = HAM
                        predict_ham += self.conditional_parameters[dtst_name][3][
                            word_idx
                        ]
                    else:
                        # Add conditional parameter of word not present, class = HAM
                        predict_ham += self.conditional_parameters[dtst_name][1][
                            word_idx
                        ]

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

            print("Discrete Naive-Bayes stats for " + dtst_name + " dataset:")
            print_accuracy_data(
                true_positive, false_negative, true_negative, false_positive
            )

        # Print Overall Statistics
        print("Overall Discrete Naive-Bayes stats:")
        print_accuracy_data(
            overall_true_positive,
            overall_false_negative,
            overall_true_negative,
            overall_false_positive,
        )
