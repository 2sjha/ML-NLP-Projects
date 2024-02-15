# Project 1

## Description

This project is based on a subset of Enron emails dataset available [here](https://www.cs.cmu.edu/~./enron/). This is an implementation of Naive bayes and Logistic Regression for text classification.

Firstly, I created a Bag of words model and a Bernoulli model from the vocabulary in the emails. Then I implemented the **Multinomial Naive Bayes algorithm**, **Discrete Naive Bayes algorithm** with Laplace smoothing, **MCAP Logistic Regression algorithm** with L2 regularization to classify emails as **SPAM** or **HAM**. I also used the **SGDClassifier** from scikit-learn to do the same. To reach the best model I also performed hyper-paramter tuning to select the best performing parameters.


## Setup

- Maybe Setup Virtualenv `python -m venv env`
- Install all dependencies `pip install -r requirements.txt`
- Install NLTK Data `python -m nltk.downloader popular`

## Execution

- If virtual env set up, then `source ./env/bin/activate`
- Run `./run.sh`

## Report

- [Report](./Report.pdf)
