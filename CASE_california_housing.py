from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets as datasets
import sklearn.metrics as metric
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.mvodel_selection as modsel
import numpy as np

data_pkg = datasets.fetch_california_housing()
dir(data_pkg)

housing_features = pd.DataFrame(data_pkg.data)
housing_features.columns = data_pkg.feature_names

housing_target = pd.DataFrame(data_pkg.target)
housing_target.columns = ['avgHouseval']

#Data Summary
housing_features.head(n=10)
housing_target.head(n=10)

feature_data = housing_features.values
target_data = housing_target.values[:,0]

#EDA
plt.hist(housing_features)

#Data Split
X_train, X_test, y_train, y_test = modsel.train_test_split(
     feature_data, target_data, test_size=0.2)

gbc = GradientBoostingClassifier(n_estimators=200,max_depth=3)
gbc.fit(X,y)
gbc_pred = gbc.predict(X)
gbc_class_prob = gbc.predict_proba(X)

#Tuning of Hyperparameters vs grid search
param_grid = {  'learning_rate':[0.1,0.05,0.02,0.01],
                'max_depth':[4,6],
                'min_samples_leaf':[3,5,9,17],
                'max_features': [1.0,0.3,0.1]}

gs_cv = GridSearchCV(gbc,param_grid).fit(X,y)
gs_cv.best_params_
gbc.feature_importances_
