"""
Uses scikit-learn Decision Tree Classifier
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class DecisionTreeClsfr:
    """
    Uses scikit-learn Decision Tree Classifier to run experiments
    """

    def __init__(self, dataset, compute_f1=True):
        self.classifier = ""

        self.train(dataset)
        self.tune_parameters(dataset, compute_f1)
        self.train_after_tuning(dataset)
        self.test(dataset, compute_f1)

    def train(self, dataset):
        """
        Trains the classifier on training data
        """
        x_train = dataset["train"][:, :-1]
        y_train = dataset["train"][:, -1]

        self.classifier = DecisionTreeClassifier(random_state=0)
        self.classifier.fit(x_train, y_train)

    def tune_parameters(self, dataset, compute_f1=True):
        """
        Tries multiple parameters and chooses the best set of parameters for the classifier
        """
        classifiers = [
            # change criterion
            DecisionTreeClassifier(
                random_state=0, criterion="log_loss", splitter="best"
            ),
            DecisionTreeClassifier(
                random_state=0, criterion="entropy", splitter="best"
            ),
            # change splitter
            DecisionTreeClassifier(
                random_state=0, criterion="log_loss", splitter="random"
            ),
            DecisionTreeClassifier(
                random_state=0, criterion="entropy", splitter="random"
            ),
            # change min_samples_split & min_samples_leaf
            DecisionTreeClassifier(
                random_state=0,
                criterion="log_loss",
                splitter="best",
                min_samples_split=5,
                min_samples_leaf=2,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="entropy",
                splitter="best",
                min_samples_split=10,
                min_samples_leaf=5,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="log_loss",
                splitter="random",
                min_samples_split=5,
                min_samples_leaf=2,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="entropy",
                splitter="random",
                min_samples_split=10,
                min_samples_leaf=5,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="log_loss",
                splitter="best",
                min_samples_split=5,
                max_features="sqrt",
                min_samples_leaf=2,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="entropy",
                splitter="best",
                min_samples_split=10,
                max_features="log2",
                min_samples_leaf=5,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="log_loss",
                splitter="random",
                min_samples_split=5,
                max_features="sqrt",
                min_samples_leaf=2,
            ),
            DecisionTreeClassifier(
                random_state=0,
                criterion="entropy",
                splitter="random",
                min_samples_split=10,
                max_features="log2",
                min_samples_leaf=3,
            ),
        ]

        x_train = dataset["train"][:, :-1]
        y_train = dataset["train"][:, -1]

        x_valid = dataset["valid"][:, :-1]
        y_valid = dataset["valid"][:, -1]

        # Predict using the default classifier
        y_pred = self.classifier.predict(x_valid)
        best_accuracy = accuracy_score(y_valid, y_pred)
        print("Tuning default accuracy: " + str(best_accuracy))

        if compute_f1:
            best_f1 = f1_score(y_valid, y_pred)
            print("Tuning default f1 score: " + str(best_f1))

        for clsf in classifiers:
            clsf.fit(x_train, y_train)
            y_pred = clsf.predict(x_valid)

            accuracy = accuracy_score(y_valid, y_pred)

            if compute_f1:
                f1_scr = f1_score(y_valid, y_pred)

            # If this classifier config is better, then choose it
            if accuracy > best_accuracy:
                if compute_f1:
                    if f1_scr > best_f1:
                        best_accuracy = accuracy
                        best_f1 = f1_scr
                        self.classifier = clsf
                else:
                    best_accuracy = accuracy
                    self.classifier = clsf

        print("Tuning best accuracy: " + str(best_accuracy))
        if compute_f1:
            print("Tuning best F1 score: " + str(best_f1))

        params = self.classifier.get_params()
        print("\nBest Decision Tree Classifier Parameters:")
        for key, value in params.items():
            print(f"{key}: {value}")

    def train_after_tuning(self, dataset):
        """
        Merge Train and Validation data and Train the classifier
        """
        x_train = dataset["train"][:, :-1]
        y_train = dataset["train"][:, -1]
        x_valid = dataset["valid"][:, :-1]
        y_valid = dataset["valid"][:, -1]

        x_train = np.concatenate((x_train, x_valid), axis=0)
        y_train = np.append(y_train, y_valid)
        self.classifier.fit(x_train, y_train)

    def test(self, dataset, compute_f1=True):
        """
        Report Accuracy & F1 score on Test Data
        """
        x_test = dataset["test"][:, :-1]
        y_test = dataset["test"][:, -1]

        y_pred = self.classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("\nTest accuracy: " + str(accuracy))

        if compute_f1:
            f1_scr = f1_score(y_test, y_pred)
            print("Test F1 score: " + str(f1_scr))
