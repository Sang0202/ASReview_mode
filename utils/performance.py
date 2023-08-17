from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def evaluate_performance(actual_label, predicted_label):
    print('count_value_predict:',predicted_label.value_counts())
    conf_matrix = confusion_matrix(actual_label, predicted_label)
    print("Confusion Matrix: ")
    print(conf_matrix)

    precision_per_class = precision_score(actual_label, predicted_label, average=None)
    accuracy = accuracy_score(actual_label, predicted_label)
    recall_per_class = recall_score(actual_label, predicted_label, average=None)

    print("Precision per class:")
    print(precision_per_class)

    print("Accuracy:")
    print(accuracy)

    print("Recall per class:")
    print(recall_per_class)

    return conf_matrix, accuracy, precision_per_class, recall_per_class