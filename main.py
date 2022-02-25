import weak_learner_module
import pandas as pd

"""<-------------------- Weak Learner code ----------------------------->"""

data, versicolor_training_dataset, versicolor_testing_dataset, virginica_training_dataset, virginica_testing_dataset = weak_learner_module.get_training_and_testing_datasets()
statistical_measurement = weak_learner_module.calculate_statistical_measurement(data, versicolor_training_dataset,
                                                                                virginica_training_dataset)
print("""<---------------------------- Weak Learner code --------------------------------->""")
print()
print("Statistical measurements for the weak learner.")
weak_learner_stats = round(pd.DataFrame(statistical_measurement).transpose(), 2)
weak_learner_stats.to_csv("Weak Learner Stats")
print(weak_learner_stats)
print()
correlation_matrix_versicolor, correlation_matrix_virginica = weak_learner_module.get_correlation_matrix(data)
print("Correlation Matrix for versicolor:")
weak_learner_versicolor_corr = pd.DataFrame(round(correlation_matrix_versicolor, 2))
print(weak_learner_versicolor_corr)
print()
print("Correlation Matrix for virginica:")
weak_learner_virginica_corr = pd.DataFrame(round(correlation_matrix_virginica, 2))
print(weak_learner_virginica_corr)
print()
weak_learner_module.plot_features(versicolor_training_dataset, virginica_training_dataset)
final_testing_dataset, final_cm_dataframe = weak_learner_module.calculate_confusion_matrix_weak_learner(
    versicolor_testing_dataset, virginica_testing_dataset)

print("Confusion matrix for the weak learner.")
print(final_cm_dataframe)

"""<--------------------------- Weak Learner ensemble ----------------------------->"""
print()
print("""<---------------------------- Weak Learner Ensemble Code --------------------------------->""")
print()
pattern_list = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
print("Confusion matrix for the weak learner by ensemble method.")
weak_learner_module.calculate_confusion_matrix_pattern(final_testing_dataset, pattern_list)

"""<--------------------------- Density Weak Learner ----------------------------->"""
print()
print("""<---------------------------- Density Weak Learner --------------------------------->""")
print()
print("Statistical measurements for the density weak learner.")
print(round(pd.DataFrame(statistical_measurement).transpose(), 2))
print()

testing_dataset = versicolor_testing_dataset.append(virginica_testing_dataset)

modified_testing_dataset = weak_learner_module.labeling_dataset(testing_dataset, statistical_measurement)
print(modified_testing_dataset.columns)
print("Confusion matrix for the density weak learner.")
weak_learner_module.calculate_confusion_matrix_density(modified_testing_dataset)

"""<--------------------------- Density ensemble ----------------------------->"""

print()
print("""<---------------------------- Density Ensemble Code --------------------------------->""")
print()
print("Confusion matrix for the density learner by ensemble method.")
pattern_list = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
weak_learner_module.calculate_confusion_matrix_pattern(modified_testing_dataset, pattern_list)
