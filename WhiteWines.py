import pandas as pd
from sklearn.model_selection import train_test_split

# Data Exploration
white_wines = pd.read_csv("white_wine.csv", sep=";", decimal=",", skipinitialspace=True, header=0)
white_wines.columns = ["Fixed acidity",
                       "Volatile acidity",
                       "Citric acid",
                       "Residual sugar",
                       "Chlorides",
                       "Free sulfur dioxide",
                       "Total sulfur dioxide",
                       "Density",
                       "pH",
                       "Sulphates",
                       "Alcohol",
                       "Quality"]
white_wines.drop_duplicates(inplace=True)
white_wines.drop("Citric acid", axis=1, inplace=True)
white_wines.drop("Free sulfur dioxide", axis=1, inplace=True)

# Split data for study and test sets
seed_value = 308200
X = white_wines.drop("Quality", axis=1)
y = white_wines["Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed_value)
