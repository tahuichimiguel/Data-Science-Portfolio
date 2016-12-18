from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets as datasets

import sklearn.metrics as metric
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as modsel
import numpy as np
import timeit

#Deliverables
##Plots
# Histograms: HouseAge, Latitude, Longitude, MedInc {DONE}
# Geospatial: Scatterplot of data on map of California 
# Error Convergence: test & train error vs n_estimators
# Variable Importance Barplot for GBRT
# Partial Dependence Plots for GBRT
# Partial Dependence Chloropleth Plots for GBRT
##Tables
# Comparison of Mean, RidgeReg, SVR, RF, & GBRT
#      Train time, Test time
#      Mean Absolute Error(MAE)

data_pkg = datasets.fetch_california_housing()
housing_features = pd.DataFrame(data_pkg.data)
housing_features.columns = data_pkg.feature_names

housing_target = pd.DataFrame(data_pkg.target)
housing_target.columns = ['avgHouseval']

#Data Summary
#   - Data Types
#   - Data shape
#   - Head of both dataframes

type(housing_features)
type(housing_target)

housing_features.head(n=10)
housing_target.head(n=10)

feature_data = housing_features.values
target_data = housing_target.values[:,0]

#EDA
plt.subplot(221)
plt.hist(housing_features['HouseAge'],bins=25)
plt.grid(which='major')
plt.title('HouseAge')

plt.subplot(222)
plt.hist(housing_features['Latitude'],bins=25)
plt.grid(which='major')
plt.title('Latitude')

plt.subplot(223)
plt.hist(housing_features['Longitude'],bins=30)
plt.grid(which='major')
plt.title('Longitude')

plt.subplot(224)
plt.hist(housing_features['MedInc'],bins=30)
plt.grid(which='major')
plt.title('MedInc')

plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=1.0)

#Set Random Seed
seed = 123

#Data Split
X_train, X_test, y_train, y_test = modsel.train_test_split(feature_data, 
                                                           target_data, 
                                                           test_size=0.2,
                                                           random_state=seed)

#MLAs
ridge=linear_model.Ridge(alpha = 0.5)
svr = svm.SVR(kernel='rbf')
forest = RandomForestRegressor(n_estimators=3000, max_features='sqrt' )
gbr = GradientBoostingRegressor(n_estimators=3000, max_depth=4,
                                random_state=seed)


##Training
ridge_time = timeit.default_timer()
ridge.fit(X_train,y_train)
ridge_time= timeit.default_timer()-ridge_time

svr_time = timeit.default_timer()
svr.fit(X_train,y_train)
svr_time= timeit.default_timer()-svr_time

forest_time = timeit.default_timer()
forest.fit(X_train,y_train)
forest_time= timeit.default_timer()-forest_time

gbr_time = timeit.default_timer()
gbr.fit(X_train,y_train)
gbr_time= timeit.default_timer()-gbr_time


##Testing
ridge_testtime = timeit.default_timer()
ridge_pred_test = ridge.predict(X_test)
ridge_testtime= timeit.default_timer()-ridge_testtime

svr_testtime = timeit.default_timer()
svr_pred_test = svr.predict(X_test)
svr_testtime= timeit.default_timer()-svr_testtime

forest_testtime = timeit.default_timer()
forest_pred_test = forest.predict(X_test)
forest_testtime= timeit.default_timer()-forest_testtime

gbr_testtime = timeit.default_timer()
gbr_pred_test = gbr.predict(X_test)
gbr_testtime= timeit.default_timer()-gbr_testtime

##Calculate MAE
ridgeMAE = metric.mean_absolute_error(ridge_pred_test,y_test)
svrMAE = metric.mean_absolute_error(svr_pred_test,y_test)
forestMAE = metric.mean_absolute_error(forest_pred_test,y_test)
gbrMAE = metric.mean_absolute_error(gbr_pred_test,y_test)


compare_results = pd.DataFrame({'MLA':['Ridge','SVR','RF','GBRT'],
                                'Train Time[s]':[ridge_time,svr_time,forest_time,
                                                                     gbr_time],
                                'Test Time[s]':[ridge_testtime,svr_testtime,
                                            forest_testtime,gbr_testtime],
                                'MAE':[ridgeMAE,svrMAE, forestMAE,gbrMAE]})

compare_results = compare_results[['MLA','Train Time[s]','Test Time[s]','MAE']]
print(compare_results)

#Train/Test Error vs n_estimators

error = np.empty(len(gbr.estimators_))
for i,pred in enumerate(gbr.staged_predict(X_test)):
    error[i] = gbr.loss_(pred,y_test)

plt.plot(np.arange(3000)+1,error,label = 'Test')
plt.plot(np.arange(3000)+1,gbr.train_score_,label = 'Train')
plt.legend(loc='upper right', shadow=True)
plt.show()
    
    
    
    
###############################
    
#Tuning of Hyperparameters vs grid search
param_grid = {  'learning_rate':[0.1,0.05,0.02,0.01],
                'max_depth':[4,6],
                'min_samples_leaf':[3,5,9,17],
                'max_features': [1.0,0.3,0.1]}

gs_cv = GridSearchCV(gbr,param_grid).fit(X,y)
gs_cv.best_params_
gbc.feature_importances_
