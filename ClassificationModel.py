import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
#import  matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV

def get_tree_size(estimator):
    print("Amount of nodes: ", estimator.tree_.node_count)
    print("Amount of leaces: ", estimator.get_n_leaves())
    print("Level: ", estimator.get_depth())


class ClassificationModel:
    def __init__(self, X_train, X_test, y_train, y_test, random):
        self.variables = X_train.columns
        self.target_variable_train = pd.Categorical(y_train)
        self.target_variable_test = pd.Categorical(y_test)
        # parametrs = {"max_depth": range(7, 11), "min_samples_split": [30, 60, 90, 120, 150], "min_samples_leaf": [2, 3, 4], "max_leaf_nodes" : [500, 1000, 1500]}
        # self.model = GridSearchCV(ExtraTreesClassifier(random_state=random), parameters, n_jobs=-1)
        self.model = ExtraTreesClassifier(random_state=random,
                                          n_jobs=-1,
                                          max_depth=9,
                                          max_leaf_nodes=500,
                                          min_samples_leaf=2,
                                          min_samples_split=30)
        self.model.fit(X_train, self.target_variable_train)
        self.train_prediction = self.model.predict(X_train)
        self.test_prediction = self.model.predict(X_test)

    def adjusted_accurancy(self, true_set, predict_set):
        class_amount = len(true_set.unique())
        matrix = confusion_matrix(true_set, predict_set)
        fp_deviation_sum = sum([matrix[i, j] for i in range(class_amount)
                                for j in range(class_amount) if abs(i - j) == 1])
        return (sum([matrix[i, i] for i in range(class_amount)])
                + fp_deviation_sum) / len(true_set)

    def get_model_statistics(self):
        print("Accuracy: ", round(accuracy_score(self.target_variable_test, self.test_prediction), 3))
        print("Deviation accuracy (+/-)1: ", round(self.adjusted_accurancy(self.target_variable_test, self.test_prediction), 3))
        print("MAE: ", round(mean_absolute_error(self.target_variable_test, self.test_prediction), 3))

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