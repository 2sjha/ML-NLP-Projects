"""
Driver code for Subpart 3
"""

import os
from mixture_clt import MixtureCLT
from util import Util


def subpart_2():
    """
    Iterates over all datasets and learns the training dataset
    with a Mixture of Chow-Liu Trees
    Uses the validation dataset multiple times to to choose
    the best value for hyper parameter k 
    """
    k = [2, 5, 10, 20]

    datasets_dir = os.fsencode("./dataset")
    for file in os.listdir(datasets_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".ts.data"):
            train_data_name = os.path.join("./dataset", filename)

            print("Learning " + train_data_name + " dataset using Mixture CLT\n")
            train_data = Util.load_dataset(train_data_name)
            best_ll = -10000
            best_k = 0
            best_mclt = ""
            for k_ in k:
                mclt = MixtureCLT()
                mclt.learn(train_data, n_components=k_)

                validation_data_name = train_data_name[:-8] + ".valid.data"
                print(
                    "Computing log-likelihood for Validation dataset "
                    + validation_data_name
                    + " with Mixture CLT k="
                    + str(k_)
                )
                validation_data = Util.load_dataset(validation_data_name)
                log_likelihood = mclt.compute_ll(validation_data)
                print(str(log_likelihood) + "\n")

                if log_likelihood > best_ll:
                    best_ll = log_likelihood
                    best_k = k_
                    best_mclt = mclt
            print(
                "Best log-likelihood across Validation datasets is "
                + str(best_ll)
                + "\n"
            )

            test_data_name = train_data_name[:-8] + ".test.data"
            print(
                "Computing log-likelihood for Test dataset "
                + test_data_name
                + " with Mixture CLT k="
                + str(best_k)
            )
            test_data = Util.load_dataset(test_data_name)
            test_log_likelihood = best_mclt.compute_ll(test_data)
            print(test_log_likelihood)
            print("---------------------------------------------------\n")


if __name__:
    subpart_2()
