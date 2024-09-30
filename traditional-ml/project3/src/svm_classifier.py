"""
Uses scikit-learn SVM Classifier
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMClassifier:
    """
    Uses scikit-learnSVM Classifier to run experiments
    """

    def __init__(self, dataset):
        self.classifiers = [
            # Changing Penalty Regularization
            SVC(C=1.0, kernel="rbf", gamma="scale", random_state=1),
            SVC(C=2.0, kernel="rbf", gamma="scale", random_state=1),
            SVC(C=0.5, kernel="rbf", gamma="scale", random_state=1),
            # Changing Kernel
            SVC(C=1.0, kernel="sigmoid", gamma="scale", random_state=1),
            SVC(C=2.0, kernel="sigmoid", gamma="scale", random_state=1),
            SVC(C=1.0, kernel="linear", gamma="scale", random_state=1),
            SVC(C=2.0, kernel="linear", gamma="scale", random_state=1),
            # Using Poly Kernel with different degree values
            SVC(C=1.0, kernel="poly", degree=3, gamma="scale", random_state=1),
            SVC(C=1.0, kernel="poly", degree=4, gamma="scale", random_state=1),
            SVC(C=1.0, kernel="poly", degree=5, gamma="scale", random_state=1),
            # Changing Gamma
            SVC(C=1.0, kernel="rbf", gamma="auto", random_state=1),
            SVC(C=1.0, kernel="linear", gamma="auto", random_state=1),
            SVC(C=1.0, kernel="poly", gamma="auto", random_state=1),
            SVC(C=1.0, kernel="sigmoid", gamma="auto", random_state=1),
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
        print("SVC Parameters:")
        for key, value in params.items():
            print(f"{key}: {value}")

        accuracy = accuracy_score(y_test, y_pred)
        print("\nAccuracy: " + str(accuracy))
        error_rate = round(1.0 - accuracy, 4)
        print("Error rate: " + str(error_rate))
        print("-------------------------------------------------\n")
