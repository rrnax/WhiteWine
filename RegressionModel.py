from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
# import  matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV

class RegressionModel:
    def __init__(self, X_train, X_test, y_train, y_test, random):
        # parametrs = {"max_depth": range(7, 11), "min_samples_split": [30, 60, 90, 120, 150], "min_samples_leaf": [2, 3, 4], "max_leaf_nodes" : [500, 1000, 1500]}
        # self.model = GridSearchCV(DecisionTreeRegressor(random_state=random), parameters, n_jobs=-1)
        self.variables = X_train.columns
        self.real_test = y_test
        self.model = DecisionTreeRegressor(random_state=random,
                                           max_depth=7,
                                           max_leaf_nodes=500,
                                           min_samples_leaf=2,
                                           min_samples_split=150)
        self.model.fit(X_train, y_train)
        self.train_prediction = self.model.predict(X_train)
        self.test_prediction = self.model.predict(X_test)

    def get_model_statistics(self):
        print("Accuracy: ", round(accuracy_score(self.real_test, np.round(self.test_prediction).astype(int)), 3))
        print("Deviation accuracy (+/-)1: ", round(self.adjusted_accurancy(self.real_test, np.round(self.test_prediction).astype(int)), 3))
        print("MAE: ", round(mean_absolute_error(self.real_test, self.test_prediction), 3))

    def adjusted_accurancy(self, true_set, predict_set):
        class_amount = len(true_set.unique())
        matrix = confusion_matrix(true_set, predict_set)
        fp_deviation_sum = sum([matrix[i, j] for i in range(class_amount)
                                for j in range(class_amount) if abs(i - j) == 1])
        return (sum([matrix[i, i] for i in range(class_amount)])
                + fp_deviation_sum) / len(true_set)

    def get_variables_importance(self):
        importance = pd.Series(self.model.feature_importances_, index=self.variables)
        importance.sort_values(inplace=True)
        print("Importance of variables in classification model:")
        print(importance)
        # plt.figure(figsize=(10,6))
        # importance.plot(kind='barh')
        # plt.xlabel("Importance")
        # plt.ylabel("Variables")
        # plt.show()