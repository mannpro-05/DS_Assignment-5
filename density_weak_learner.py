import weak_learner_module

import pandas as pd
from sklearn.metrics import confusion_matrix









data, versicolor_training_dataset, virginica_training_dataset = weak_learner_module.get_training_datasets()
statistical_measurement = weak_learner_module.calculate_statistical_measurement(data, versicolor_training_dataset,
                                                                                virginica_training_dataset)

