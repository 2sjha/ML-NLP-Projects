# Project 3

## Description

This project implements the **Collaborative Filtering algorithm** on the [Netflix movie ratings dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and also performs hyper-parameter tuning for **K-Nearest Neighbors** model and the **SVM model** on the [MNIST dataset](https://www.openml.org/d/554).

## Source code

The source code for this project is available as a [Jupyer Notebook](./Project3.ipynb) and Python scripts in the [src directory](./src). To execute the scripts for the project, I can use the [run.sh](./run.sh) or [run.bat](./run.bat) files. I added a run.bat file to test on Windows because using a .sh file through WSL leads to worse performance than running natively on Windows. You might need to set up a virtual env and install dependencies to run the python scripts. Both run scripts have commented lines to run in a virtualenv and install dependencies.

### [Collaborative Filtering](./src/collab_filtering.py)

This Python script implements the Collaborative Filtering algorithm in this [paper](https://dl.acm.org/doi/10.5555/2074094.2074100). Since there are a huge number of user pairs for which weights need to be calculated, this can easily lead to memory issues, so I calculate the weights of other users for a particular user and store them on disk. I then load this stored weights dict at prediction time to calculate the rating for any movie. I also report the mean absolute error and root mean square error of the predicted ratings.

### [K-Nearest Neighbors Experiments on MNIST Dataset](./src/knn_classifier.py)

This Python script has a list of [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) objects from scikit-learn with different parameter settings so that I can measure the impact of each parameter on accuracy/error rate.

### [SVM Experiments on MNIST Dataset](./src/svm_classifier.py)

This Python script has a list of [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) objects from scikit-learn with different parameter settings so that I can measure the impact of each parameter on accuracy/error rate.

### [Driver Script](./src/all_models.py)

This Python script extracts the dataset zip file, reads the Netflix dataset and sets up the MNIST dataset described in the project PDF. Then it instantiates objects of each of the above classes with the necessary dataset so that they can do their jobs.

### Report

- [Report](./Report.odt)