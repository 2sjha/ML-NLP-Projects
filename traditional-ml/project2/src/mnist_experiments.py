"""
Fetches MNIST dataset and runs all models on it
"""
import sys
import numpy as np
from sklearn.datasets import fetch_openml
from decision_tree_classifier import DecisionTreeClsfr
from bagging_classifier import BaggingClsfr
from random_forest_classifier import RandomForestClsfr
from gradient_boosting_classifier import GradientBoostingClsfr


class MNISTExperiments:
    """
    Fetches MNIST dataset and runs all models on it
    """

    def __init__(self, models="all"):
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        x = x / 255

        x_train, x_valid, x_test = x[:40000], x[40000:60000], x[60000:]
        y_train, y_valid, y_test = y[:40000], y[40000:60000], y[60000:]

        dataset = {
            "train": np.column_stack((x_train, y_train)),
            "valid": np.column_stack((x_valid, y_valid)),
            "test": np.column_stack((x_test, y_test)),
        }

        if models == "all" or "dtree" in models:
            print("Running Decision Tree Classifier on MNIST Dataset")
            print("*************************************************\n")
            self.decision_tree_classifier = DecisionTreeClsfr(dataset, compute_f1=False)
            print("---------------------------------------------")

        if models == "all" or "bagging" in models:
            print("Running Bagging Classifier on MNIST Dataset")
            print("*************************************************\n")
            self.bagging_classifier = BaggingClsfr(dataset, compute_f1=False)
            print("---------------------------------------------")

        if models == "all" or "randomforest" in models:
            print("Running Random Forest Classifier on MNIST Dataset")
            print("*************************************************\n")
            self.random_forest_classifier = RandomForestClsfr(dataset, compute_f1=False)
            print("---------------------------------------------")

        if models == "all" or "gradientboost" in models:
            print("Running Gradient Boost Classifier on MNIST Dataset")
            print("*************************************************\n")
            self.gradient_boosting_classifier = GradientBoostingClsfr(
                dataset, compute_f1=False, multi_class=True
            )
            print("---------------------------------------------")


if __name__:
    # Run all models passed as arguments
    if len(sys.argv) >= 2:
        MODELS = " ".join(sys.argv[1:])
        all_models = MNISTExperiments(MODELS)
    else:
        print(
            'Argument ("all" or "dtree" or "bagging" or "randomforest" or "gradientboost") needed.',
            file=sys.stderr,
        )
