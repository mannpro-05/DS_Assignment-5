from scipy.stats import norm
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import weak_learner_module

sequence_map = {
    1: 'sepal-length-label',
    2: "sepal-width-label",
    3: "petal-length-label",
    4: "petal-width-label"
}

'''
@:param testing_dataset : PandasDataframe containing all the testing values for both the classes.
@:param statistical_measurement : Dictionary containing the feature wise sigma and mu values.
@:Processing : It labels the testing dataset based on the normal distribution calculated for both the classes.
@:return testing_dataset: PandasDataFrame: testing dataset with predicted column values.
'''


def labeling_dataset(testing_dataset, statistical_measurement):
    features = ["sepal-length", "sepal-width",
                "petal-length", "petal-width"]
    for feature in features:
        predicted_labels = []
        flower_data = testing_dataset[feature]
        for flower in flower_data:
            p_0 = norm.pdf((flower - statistical_measurement[feature]["µ0"]) / statistical_measurement[feature]["σ0"])
            p_1 = norm.pdf((flower - statistical_measurement[feature]["µ1"]) / statistical_measurement[feature]["σ1"])
            if p_0 >= p_1:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)
        testing_dataset[feature + "-label"] = predicted_labels

    return testing_dataset


'''
@:param testing_dataset : PandasDataframe containing all the testing values for both the classes.
@:param pattern_list : List: containing list of pattern for the ensemble.
@:Processing : It calculates the confusion matrix for all the pattern(hyperparameter) by using the ensemble method of predicting.
@:return none
'''


def calculate_confusion_matrix(testing_dataset, pattern_list):
    pattern_prediction = {}
    column_name = []
    for pattern in pattern_list:
        predicted_label = []
        for col in range(len(pattern)):
            column_name.append(sequence_map[pattern[col]])
            pattern[col] = str(pattern[col])

        data = np.array(testing_dataset[column_name])
        for value in data:
            predicted_label.append(int(np.where(np.count_nonzero(value == 1) >= 2, 1, 0)))
        matrix = confusion_matrix(testing_dataset["label"], predicted_label)
        matrix = matrix.flatten()
        accuracy = (matrix[3] + matrix[0]) / matrix.sum()
        pattern_prediction[" ".join(pattern)] = {
            "TP": matrix[3],
            "TN": matrix[0],
            "FP": matrix[1],
            "FN": matrix[2],
            "Accuracy": round(accuracy * 100, 3)
        }
    pattern_dataframe = pd.DataFrame(pattern_prediction).transpose()
    print(pattern_dataframe)


data, versicolor_training_dataset, virginica_training_dataset = weak_learner_module.get_training_datasets()
statistical_measurement = weak_learner_module.calculate_statistical_measurement(data, versicolor_training_dataset,
                                                                                virginica_training_dataset)

versicolor_testing_dataset, virginica_testing_dataset = weak_learner_module.get_testing_datasets()

testing_dataset = versicolor_testing_dataset.append(virginica_testing_dataset, statistical_measurement)
modified_testing_dataset = labeling_dataset(testing_dataset, statistical_measurement)
pattern_list = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
print("Confusion matrix for the density learner by ensemble method.")
calculate_confusion_matrix(modified_testing_dataset, pattern_list)
