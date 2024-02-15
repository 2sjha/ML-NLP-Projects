"""
Driver code for Subpart 3
"""

import os
from clt_rf import CLTRandomForest
from util import Util


def subpart_3():
    """
    Iterates over all datasets and learns the training dataset
    with a Random Forest of Chow-Liu Trees
    Uses the validation dataset multiple times to to choose
    the best value for hyper parameters num_trees(k) and r
    """
    k = [20, 50, 100]
    r = [0.05, 0.1, 0.2, 0.3]

    datasets_dir = os.fsencode("./dataset")
    for file in os.listdir(datasets_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".ts.data"):
            train_data_name = os.path.join("./dataset", filename)

            print("Learning " + train_data_name + " dataset using CLT Random Forest\n")
            train_data = Util.load_dataset(train_data_name)
            best_ll = -10000
            best_k = 0
            best_r = 0
            best_cltrf = ""
            for k_ in k:
                for r_ in r:
                    cltrf = CLTRandomForest(num_trees=k_, r=r_)
                    cltrf.learn(train_data)

                    validation_data_name = train_data_name[:-8] + ".valid.data"
                    print(
                        "Computing log-likelihood for Validation dataset "
                        + validation_data_name
                        + " with CLT Random Forest k="
                        + str(k_)
                        + ", r="
                        + str(r_)
                    )
                    validation_data = Util.load_dataset(validation_data_name)
                    log_likelihood = cltrf.compute_ll(validation_data)
                    print(str(log_likelihood) + "\n")

                    if log_likelihood > best_ll:
                        best_ll = log_likelihood
                        best_k = k_
                        best_r = r_
                        best_cltrf = cltrf
            print(
                "Best log-likelihood across Validation datasets is "
                + str(best_ll)
                + "\n"
            )

            test_data_name = train_data_name[:-8] + ".test.data"
            print(
                "Computing log-likelihood for Test dataset "
                + test_data_name
                + " with Random Forest k="
                + str(best_k)
                + ", r="
                + str(best_r)
            )
            test_data = Util.load_dataset(test_data_name)
            test_log_likelihood = best_cltrf.compute_ll(test_data)
            print(test_log_likelihood)
            print("---------------------------------------------------\n")


if __name__:
    subpart_3()
