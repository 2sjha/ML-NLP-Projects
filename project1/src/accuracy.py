def print_accuracy_data(true_positive, false_negative, true_negative, false_positive):
    """
    Calcualtes & prints Accuracy, Precision, Recall & F1 Score
    """
    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )
    print("Accuracy = " + str(accuracy))

    precision = (true_positive) / (true_positive + false_positive)
    print("Precision = " + str(precision))

    recall = (true_positive) / (true_positive + false_negative)
    print("Recall = " + str(recall))

    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("F1 score = " + str(f1_score))
    print("\n")
