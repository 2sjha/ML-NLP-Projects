"""
Instantiates and Runs all models
"""

import zipfile
import sys
from os.path import exists, join
from os import listdir
from collections import defaultdict
import numpy as np
from decision_tree_classifier import DecisionTreeClsfr
from bagging_classifier import BaggingClsfr
from random_forest_classifier import RandomForestClsfr
from gradient_boosting_classifier import GradientBoostingClsfr


class AllModels:
    """
    Instantiates and Runs all models
    """

    def __init__(self, models="all"):
        self.dataset_path = "./../all_data/"
        self.dataset = defaultdict(lambda: defaultdict(dict))
        self.read_datasets()

        if models == "all" or "dtree" in models:
            print("Running Decision Tree Classifier")
            print("*************************************************\n")
            self.decision_tree_classifiers = defaultdict(
                lambda: defaultdict(lambda: DecisionTreeClsfr)
            )

            for n_clauses in self.dataset:
                for n_examples in self.dataset[n_clauses]:
                    print(n_clauses + "->" + n_examples)
                    print("---------------------------------------------")
                    self.decision_tree_classifiers[n_clauses][
                        n_examples
                    ] = DecisionTreeClsfr(self.dataset[n_clauses][n_examples])
                    print("---------------------------------------------\n")

        if models == "all" or "bagging" in models:
            print("Running Bagging Classifier")
            print("*************************************************\n")
            self.bagging_classifiers = defaultdict(
                lambda: defaultdict(lambda: BaggingClsfr)
            )

            for n_clauses in self.dataset:
                for n_examples in self.dataset[n_clauses]:
                    print(n_clauses + "->" + n_examples)
                    print("---------------------------------------------")
                    self.bagging_classifiers[n_clauses][n_examples] = BaggingClsfr(
                        self.dataset[n_clauses][n_examples]
                    )
                    print("---------------------------------------------\n")

        if models == "all" or "randomforest" in models:
            print("Running Random Forest Classifier")
            print("*************************************************\n")
            self.random_forest_classifiers = defaultdict(
                lambda: defaultdict(lambda: RandomForestClsfr)
            )

            for n_clauses in self.dataset:
                for n_examples in self.dataset[n_clauses]:
                    print(n_clauses + "->" + n_examples)
                    print("---------------------------------------------")
                    print(n_clauses + "->" + n_examples)
                    self.random_forest_classifiers[n_clauses][
                        n_examples
                    ] = RandomForestClsfr(self.dataset[n_clauses][n_examples])
                    print("---------------------------------------------\n")

        if models == "all" or "gradientboost" in models:
            print("Running Gradient Boost Classifier")
            print("*************************************************\n")
            self.gradient_boosting_classifiers = defaultdict(
                lambda: defaultdict(lambda: GradientBoostingClsfr)
            )

            for n_clauses in self.dataset:
                for n_examples in self.dataset[n_clauses]:
                    print(n_clauses + "->" + n_examples)
                    print("---------------------------------------------")
                    self.gradient_boosting_classifiers[n_clauses][
                        n_examples
                    ] = GradientBoostingClsfr(self.dataset[n_clauses][n_examples])
                    print("---------------------------------------------\n")

    def read_datasets(self):
        """
        Extracts and Reads csv files from the dataset
        """
        if not exists(self.dataset_path):
            with zipfile.ZipFile("./../project2_data.zip", "r") as zip_ref:
                zip_ref.extractall("./..")

        dataset_csv_files = listdir(self.dataset_path)
        for csv_file in dataset_csv_files:
            name_parts = csv_file.split("_")
            d_type = name_parts[0]
            d_clauses = name_parts[1]
            d_examples = name_parts[2].split(".")[0]
            csv_file_path = join(self.dataset_path, csv_file)
            self.dataset[d_clauses][d_examples][d_type] = np.genfromtxt(
                csv_file_path, delimiter=","
            )


if __name__:
    # Run all models passed as arguments
    if len(sys.argv) >= 2:
        MODELS = " ".join(sys.argv[1:])
        all_models = AllModels(MODELS)
    else:
        print(
            'Argument ("all" or "dtree" or "bagging" or "randomforest" or "gradientboost") needed.',
            file=sys.stderr,
        )
