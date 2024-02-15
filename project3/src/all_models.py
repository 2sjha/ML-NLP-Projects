"""
Instantiates and Runs all models
"""

import zipfile
import sys
from os.path import exists
from collections import defaultdict
import numpy as np
from sklearn.datasets import fetch_openml
from collab_filtering import CollabFilter
from svm_classifier import SVMClassifier
from knn_classifier import KNNClassifier


class AllModels:
    """
    Instantiates and Runs all models
    """

    def __init__(self, models="all"):
        self.netflix_dataset = {
            "movies": {},
            "ratings": {
                "train": defaultdict(dict),
                "test": defaultdict(dict),
            },
        }
        dataset_path = "./../netflix/"
        self.read_netflix_dataset(dataset_path)

        if models == "all" or "collab" in models:
            print("Running Collaborative Filtering on Netflix Dataset")
            print("*************************************************\n")
            CollabFilter(self.netflix_dataset)
            print("*************************************************\n")

        mnist_dataset = defaultdict()
        if models == "all" or "svm" in models or "knn" in models:
            # Fetch MNIST dataset for SVM and KNN
            x, y = fetch_openml("mnist_784", version=1, return_X_y=True)
            x = x / 255

            x_train, x_test = x[:60000], x[60000:]
            y_train, y_test = y[:60000], y[60000:]

            mnist_dataset["train"] = np.column_stack((x_train, y_train))
            mnist_dataset["test"] = np.column_stack((x_test, y_test))

        if models == "all" or "svm" in models:
            print("Running SVM Experiments on MNIST Dataset")
            print("*************************************************\n")
            SVMClassifier(mnist_dataset)
            print("*************************************************\n")

        if models == "all" or "knn" in models:
            print("Running K Nearest Neighbors Experiments on MNIST Dataset")
            print("*************************************************\n")
            KNNClassifier(mnist_dataset)
            print("*************************************************\n")

    def read_netflix_dataset(self, dataset_path):
        """
        Extracts and Reads txt files from the Netflix dataset
        """
        if not exists(dataset_path):
            with zipfile.ZipFile("./../netflix.zip", "r") as zip_ref:
                zip_ref.extractall(dataset_path)

        movies_dataset = dataset_path + "movie_titles.txt"
        train_ratings = dataset_path + "TrainingRatings.txt"
        test_ratings = dataset_path + "TestingRatings.txt"

        with open(movies_dataset, "r", encoding="latin-1") as mv:
            lines = mv.readlines()
            for line in lines:
                mv_data = line.split(",")
                mv_id = mv_data[0]
                mv_year = mv_data[1]
                mv_name = mv_data[2].strip()
                self.netflix_dataset["movies"][mv_id] = {
                    "year": mv_year,
                    "name": mv_name,
                }

        with open(train_ratings, "r", encoding="utf-8") as trn:
            lines = trn.readlines()
            for line in lines:
                trn_data = line.split(",")
                mv_id = trn_data[0]
                user_id = trn_data[1]
                user_mv_rating = float(trn_data[2].strip())
                self.netflix_dataset["ratings"]["train"][user_id][
                    mv_id
                ] = user_mv_rating

        with open(test_ratings, "r", encoding="utf-8") as tst:
            lines = tst.readlines()
            for line in lines:
                tst_data = line.split(",")
                mv_id = tst_data[0]
                user_id = tst_data[1]
                user_mv_rating = float(tst_data[2].strip())
                self.netflix_dataset["ratings"]["test"][user_id][mv_id] = user_mv_rating


if __name__:
    # Run all models passed as arguments
    if len(sys.argv) >= 2:
        MODELS = " ".join(sys.argv[1:])
        all_models = AllModels(MODELS)
    else:
        print(
            'Argument ("all" or "collab" or "knn" or "svm") needed.',
            file=sys.stderr,
        )
