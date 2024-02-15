"""
Driver code for Subpart 1
"""

import os
from clt import ChowLiuTree
from util import Util


def subpart_1():
    """
    Iterates over all the datasets and learns them using the ChowLiuTree algorithm,
    Finally reports test-set log likelihood for each dataset
    """

    datasets_dir = os.fsencode("./dataset")
    for file in os.listdir(datasets_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".ts.data"):
            train_data_name = os.path.join("./dataset", filename)

            print("Learning " + train_data_name + " dataset.")
            train_data = Util.load_dataset(train_data_name)
            clt = ChowLiuTree()
            clt.learn(train_data, r=None)

            test_data_name = train_data_name[:-8] + ".test.data"
            print("Computing log-likelihood for " + test_data_name + " dataset.")
            test_data = Util.load_dataset(test_data_name)
            log_likelihood = clt.compute_ll(test_data) / test_data.shape[0]
            print(test_data.shape[0])
            print(str(log_likelihood) + "\n")


if __name__:
    subpart_1()
