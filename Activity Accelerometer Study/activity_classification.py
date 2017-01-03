import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/Mikey/Documents/ML-Case-Studies/Activity Accelerometer Study'
os.chdir(path)

#Load Data
df_train = pd.read_csv('pml-training.csv')
df_test = pd.read_csv('pml-testing.csv')


#Data Structure & Summary

##Testing
test_descrip = df_test.describe()
test_nulcol = pd.isnull(df_test).any()
print(df_test.shape)

for i in range(0,len(test_descrip.columns),8):
    print('\n')
    print('Predictors ' + str(i) + ' to ' + str(i+8)+ '\n')
    print(test_descrip.iloc[:,i:(i+8)])

##Training
descrip = df_train.describe()
nulcol = pd.isnull(df_train).any()
print(df_train.shape)

for i in range(0,len(descrip.columns),8):
    print('\n')
    print('Predictors ' + str(i) + ' to ' + str(i+8)+ '\n')
    print(descrip.iloc[:,i:(i+8)])
    
print(df_train.classe.unique())
outcome_freq=df_train['classe'].value_counts()
    
###Note: Only 52 Of the 160 documented predictors are not NaN
good_fields = ['roll_belt','pitch_belt','yaw_belt','total_accel_belt',
                 'gyros_belt_x','gyros_belt_y','gyros_belt_z',
                 'accel_belt_x','accel_belt_y','accel_belt_z',
                 'magnet_belt_x','magnet_belt_y','magnet_belt_z',
                 
                 'roll_arm','pitch_arm','yaw_arm',  'total_accel_arm', 
                 'gyros_arm_x','gyros_arm_y','gyros_arm_z',
                 'accel_arm_x','accel_arm_y','accel_arm_z',
                 'magnet_arm_x','magnet_arm_y','magnet_arm_z',
                 
                 'roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell',
                 'gyros_dumbbell_x' ,'gyros_dumbbell_y','gyros_dumbbell_z', 
                 'accel_dumbbell_x' ,'accel_dumbbell_y' ,'accel_dumbbell_z' ,
                 'magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z',
                 
                 'roll_forearm','pitch_forearm' ,'yaw_forearm','total_accel_forearm',
                 'gyros_forearm_x'  ,'gyros_forearm_y', 'gyros_forearm_z',  
                 'accel_forearm_x'  ,'accel_forearm_y'  ,'accel_forearm_z',
                 'magnet_forearm_x' ,'magnet_forearm_y' ,'magnet_forearm_z',
                 
                 'classe'
                 ]

clean_train = df_train[good_fields]
print(clean_train.columns)

belt_names =['roll_belt','pitch_belt','yaw_belt','total_accel_belt',
                 'gyros_belt_x','gyros_belt_y','gyros_belt_z',
                 'accel_belt_x','accel_belt_y','accel_belt_z',
                 'magnet_belt_x','magnet_belt_y','magnet_belt_z'
                 ]
            
arm_names = [ 'roll_arm','pitch_arm','yaw_arm',  'total_accel_arm', 
                 'gyros_arm_x','gyros_arm_y','gyros_arm_z',
                 'accel_arm_x','accel_arm_y','accel_arm_z',
                 'magnet_arm_x','magnet_arm_y','magnet_arm_z'
                 ]
                 
dumbell_names =  ['roll_dumbbell','pitch_dumbbell','yaw_dumbbell','total_accel_dumbbell',
                 'gyros_dumbbell_x' ,'gyros_dumbbell_y','gyros_dumbbell_z', 
                 'accel_dumbbell_x' ,'accel_dumbbell_y' ,'accel_dumbbell_z' ,
                 'magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z'
                 ]
        
forearm_names = ['roll_forearm','pitch_forearm' ,'yaw_forearm','total_accel_forearm',
                 'gyros_forearm_x'  ,'gyros_forearm_y', 'gyros_forearm_z',  
                 'accel_forearm_x'  ,'accel_forearm_y'  ,'accel_forearm_z',
                 'magnet_forearm_x' ,'magnet_forearm_y' ,'magnet_forearm_z'
                 ]
                 
activity = ['class']


#Exploratory Data Analysis
sns.set()

##Data Subsetting
df_belt= clean_train[belt_names]
df_arm= clean_train[arm_names]
df_dumbell= clean_train[dumbell_names]
df_forearm= clean_train[forearm_names]

##Correlation Plots 
plt.figure()
sns.pairplot(df_belt)
_ = pd.tools.plotting.scatter_matrix(df_belt,figsize=(10, 10),alpha =0.2)

plt.figure()
_ = pd.tools.plotting.scatter_matrix(df_arm,figsize=(10, 10),alpha =0.2)

plt.figure()
_ = pd.tools.plotting.scatter_matrix(df_dumbell,figsize=(10, 10),alpha =0.2)

plt.figure()
_ = pd.tools.plotting.scatter_matrix(df_forearm,figsize=(10, 10),alpha =0.2)


plt.figure()
_ = plt.bar([0,1,2,3,4],[outcome_freq[0],outcome_freq[1],
                     outcome_freq[3],outcome_freq[4],outcome_freq[2]],
                     tick_label=['A','B','C','D','E'],align='center')
plt.title('Proportions of Activity Classifications',fontsize=15)



















