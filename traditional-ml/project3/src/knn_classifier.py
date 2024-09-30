"""
Uses scikit-learn K nearest neighbors Classifier
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNClassifier:
    """
    Uses scikit-learn K nearest neighbors Classifier to run experiments
    """

    def __init__(self, dataset):
        # Ran experiments and found that Accuracy does NOT change
        # if algorithm or leaf_size is altered, thus choosing algorithm = auto (default)
        self.classifiers = [
            # Changing n_neighbors
            KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=7, weights="uniform", p=2, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=9, weights="uniform", p=2, n_jobs=-1),
            # Changing weights, now closer neighbors have larger weights
            KNeighborsClassifier(n_neighbors=3, weights="distance", p=2, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=5, weights="distance", p=2, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=7, weights="distance", p=2, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=9, weights="distance", p=2, n_jobs=-1),
            # Changing l_p for minkowski distance
            KNeighborsClassifier(n_neighbors=3, weights="distance", p=3, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=5, weights="distance", p=3, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=5, weights="distance", p=4, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=7, weights="distance", p=4, n_jobs=-1),
            KNeighborsClassifier(n_neighbors=9, weights="distance", p=4, n_jobs=-1),
        ]

        self.classifier = ""
        for classifier in self.classifiers:
            self.classifier = classifier
            self.train(dataset)
            self.test(dataset)

    def train(self, dataset):
        """
        Trains the classifier
        """
        x_train = dataset["train"][:, :-1]
        y_train = dataset["train"][:, -1]

        self.classifier.fit(x_train, y_train)

    def test(self, dataset):
        """
        Report Error Metrics on Test Data
        """
        x_test = dataset["test"][:, :-1]
        y_test = dataset["test"][:, -1]

        y_pred = self.classifier.predict(x_test)

        params = self.classifier.get_params()
        print("K-Nearest Neighbors Parameters:")
        for key, value in params.items():
            print(f"{key}: {value}")

        accuracy = accuracy_score(y_test, y_pred)
        print("\nAccuracy: " + str(accuracy))
        error_rate = round(1.0 - accuracy, 4)
        print("Error rate: " + str(error_rate))
        print("-------------------------------------------------\n")
