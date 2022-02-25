import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import norm

'''
@:param None
@:Processing : Generates the Training dataset for the versicolor and the virginica class.
@:return data: PandasDataframe containing the labels and all the classes
@:return versicolor_training_dataset: PandasDataframe containing the labels and training data of the versicolor class.
@:return versicolor_testing_dataset: PandasDataframe containing the labels and Testing data of the versicolor class.
@:return virginica_training_dataset: PandasDataframe containing the labels and training data of the virginica class.
@:return virginica_testing_dataset: PandasDataframe containing the labels and Testing data of the virginica class.
'''


def get_training_and_testing_datasets():
    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url, names=[
        "sepal-length", "sepal-width",
        "petal-length", "petal-width",
        "Class"])

    data = data[data["Class"] != "Iris-setosa"]
    flower_name = data["Class"]

    flower_name = np.where(flower_name == "Iris-versicolor", 0, 1)

    data["label"] = flower_name
    versicolor_dataset = data[data["label"] == 0]
    virginica_dataset = data[data["label"] == 1]
    versicolor_training_dataset, versicolor_testing_dataset, virginica_training_dataset, virginica_testing_dataset = train_test_split(
        versicolor_dataset,
        virginica_dataset,
        random_state=21,
        test_size=0.5)

    return data, versicolor_training_dataset, versicolor_testing_dataset, virginica_training_dataset, virginica_testing_dataset


'''
@:param data : PandasDataframe containing all the values for both the classes.
@:param versicolor_training_dataset : PandasDataframe containing the training values for the versicolor classes.
@:param virginica_training_dataset : PandasDataframe containing the training values for the virginica class.
@:Processing : Calculates the sigma and mu for both the classes separately and also calculates it for 
both of them together.  
@:return statistical_measurement: Dictionary with all the calculated sigma and mu.
'''


def calculate_statistical_measurement(data, versicolor_training_dataset, virginica_training_dataset):
    statistical_measurement = {}

    for column in data.columns:
        if column != "Class" and column != "label":
            statistical_measurement[column] = {
                "µ0": np.mean(versicolor_training_dataset[column]),
                "σ0": np.std(versicolor_training_dataset[column]),
                "µ1": np.mean(virginica_training_dataset[column]),
                "σ1": np.std(virginica_training_dataset[column]),
                "µall": np.mean(data[column]),
                "σall": np.std(data[column])
            }
    return statistical_measurement


'''
@:param versicolor_testing_dataset : PandasDataframe containing the testing values for the versicolor classes.
@:param virginica_testing_dataset : PandasDataframe containing the testing values for the virginica class.
@:Processing : Predicts/labeling the flower based on the hardcoded values determined by looking at the histograms. 
After labeling the flowers the confusion matrix is generated which is mapped to a Dataframe which displays the 
matrix data in readable form for all the classifiers. 
@:return final_testing_dataset: PandasDataFrame: Clubbed testing dataset for both the classes.
@:return final_cm_dataframe: PandasDataFrame: Containing the confusion matrix data in readable form.
'''


def calculate_confusion_matrix_weak_learner(versicolor_testing_dataset, virginica_testing_dataset):
    versicolor_testing_dataset["sepal-length-label"] = np.where(versicolor_testing_dataset["sepal-length"] < 6.3, 0, 1)
    virginica_testing_dataset["sepal-length-label"] = np.where(virginica_testing_dataset["sepal-length"] < 6.3, 0, 1)

    versicolor_testing_dataset["sepal-width-label"] = np.where(versicolor_testing_dataset["sepal-width"] < 2.8, 0, 1)
    virginica_testing_dataset["sepal-width-label"] = np.where(virginica_testing_dataset["sepal-width"] < 2.8, 0, 1)

    versicolor_testing_dataset["petal-length-label"] = np.where(versicolor_testing_dataset["petal-length"] < 4.9, 0, 1)
    virginica_testing_dataset["petal-length-label"] = np.where(virginica_testing_dataset["petal-length"] < 4.9, 0, 1)

    versicolor_testing_dataset["petal-width-label"] = np.where(versicolor_testing_dataset["petal-width"] < 1.7, 0, 1)
    virginica_testing_dataset["petal-width-label"] = np.where(virginica_testing_dataset["petal-width"] < 1.7, 0, 1)

    final_testing_dataset = versicolor_testing_dataset.append(virginica_testing_dataset)
    final_testing_dataset.to_csv('final_testing_dataset.csv', index=False)

    sepal_length_cm = confusion_matrix(final_testing_dataset["label"],
                                       final_testing_dataset["sepal-length-label"])

    sepal_width_cm = confusion_matrix(final_testing_dataset["label"],
                                      final_testing_dataset["sepal-width-label"])

    petal_length_cm = confusion_matrix(final_testing_dataset["label"],
                                       final_testing_dataset["petal-length-label"])

    petal_width_cm = confusion_matrix(final_testing_dataset["label"],
                                      final_testing_dataset["petal-width-label"])

    cm_list = {
        "sepal_length": sepal_length_cm,
        "sepal_width": sepal_width_cm,
        "petal_length": petal_length_cm,
        "petal_width": petal_width_cm
    }

    final_cm_table = {}

    for key, cm in cm_list.items():
        cm = cm.flatten()
        accuracy = (cm[3] + cm[0]) / cm.sum()
        final_cm_table[key] = {
            "TP": cm[3],
            "TN": cm[0],
            "FP": cm[1],
            "FN": cm[2],
            "Accuracy": round(accuracy * 100, 3)
        }
    final_cm_dataframe = pd.DataFrame(final_cm_table).transpose()

    return final_testing_dataset, final_cm_dataframe


'''
@:param data : PandasDataframe containing all the values for both the class.
@:Processing : Calculates the correlation matrix for both the classes separately.
@:return correlation_matrix_versicolor: PandasDataFrame: correlation matrix for the versicolor class.
@:return correlation_matrix_virginica: PandasDataFrame: correlation matrix for the virginica class.
'''


def get_correlation_matrix(data):
    correlation_matrix_versicolor = data[data["Class"] == "Iris-versicolor"].iloc[:, :-1].corr()
    correlation_matrix_virginica = data[data["Class"] == "Iris-virginica"].iloc[:, :-1].corr()
    return correlation_matrix_versicolor, correlation_matrix_virginica


'''
@:param versicolor_testing_dataset : PandasDataframe containing the testing values for the versicolor classes.
@:param virginica_testing_dataset : PandasDataframe containing the testing values for the virginica class.
@:Processing : Plots histograms for all the classifiers with both the classes.   
@:return none
'''


def plot_features(versicolor_training_dataset, virginica_training_dataset):
    features = ["sepal-length", "sepal-width",
                "petal-length", "petal-width"]

    for feature in features:
        plt.hist(versicolor_training_dataset[feature], label="versicolor", alpha=0.8, bins=10)
        plt.hist(virginica_training_dataset[feature], label="virginica", alpha=0.8, bins=10)
        plt.title(feature)
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.legend()
        plt.show()


'''
@:param final_testing_dataset : PandasDataframe containing all the testing values for both the classes.
@:param pattern_list : List: containing list of pattern for the ensemble.
@:Processing : It calculates the confusion matrix for all the pattern(hyperparameter) by using the ensemble method of predicting.
@:return none
'''


def calculate_confusion_matrix_pattern(final_testing_dataset, pattern_list):
    sequence_map = {
        1: 'sepal-length-label',
        2: "sepal-width-label",
        3: "petal-length-label",
        4: "petal-width-label"
    }
    pattern_prediction = {}
    column_name = []
    for pattern in pattern_list:
        predicted_label = []
        for col in range(len(pattern)):

            column_name.append(sequence_map[pattern[col]])
            pattern[col] = str(pattern[col])

        data = np.array(final_testing_dataset[column_name])
        for value in data:
            predicted_label.append(int(np.where(np.count_nonzero(value == 1) >= 2, 1, 0)))
        matrix = confusion_matrix(final_testing_dataset["label"], predicted_label)
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
@:Processing : It calculates the confusion matrix for the predicted labels in the testing dataset and calculate its accuracy.
@:return none
'''


def calculate_confusion_matrix_density(testing_dataset):
    density_cm_values = {}
    predicted_labels_list = ["sepal-length-label", "sepal-width-label", "petal-length-label", "petal-width-label"]
    for predicted_label in predicted_labels_list:
        cm = confusion_matrix(testing_dataset['label'], testing_dataset[predicted_label])
        cm = cm.flatten()
        accuracy = (cm[3] + cm[0]) / cm.sum()
        density_cm_values[predicted_label] = {
            "TP": cm[3],
            "TN": cm[0],
            "FP": cm[1],
            "FN": cm[2],
            "Accuracy": round(accuracy * 100, 3)
        }
    density_weak_classifier = pd.DataFrame(density_cm_values).transpose()
    print(density_weak_classifier)