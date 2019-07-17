#Vaibhav - vaibhav7.vc@gmail.com
#Geeta - geetzzz95@gmail.com
#Abhishek - abhisheksingh7004@gmail.com

# after analysing the data
# the final program in which we have predicted that whether the list of users fill faollowback or not using randomforestclassifier

import csv
import numpy as np
import pandas as pd

from tkinter import filedialog as fd
import webbrowser
from sklearn.model_selection import train_test_split
# open file dialog box to get the csv file for which we want to predict followback
# input file contains the details of users for which you want to predict whether they will followback or not
name= fd.askopenfilename(title="select file",filetypes= (("csv files","*.csv"),)) 
predfile = name
# reading the csv file named predfile
data1=pd.read_csv(predfile)
# reading the csv file for training the model
data=pd.read_csv('indata.csv')
data2=data.copy()

data2['Private'] = data2['Private'].map({True: 1,False: 0})
data1['Private'] = data1['Private'].map({True: 1,False: 0})

data2['fol']=data2['Followers']/(data2['Followers']-data2['Follows'])
data1['fol']=data1['Followers']/(data1['Followers']-data1['Follows'])

data2['folfol']=data2['Follows']/data2['Followers']
data1['folfol']=data1['Follows']/data1['Followers']

# standardization of data1
mean=np.mean(data1['fol'])
std=np.std(data1['fol'])
mean=float(mean)
std=float(std)
data1['fol']=(data1['fol']-mean)/std
mean=np.mean(data1['folfol'])
std=np.std(data1['folfol'])
mean=float(mean)
std=float(std)
data1['folfol']=(data1['folfol']-mean)/std

# standardization of data2
mean=np.mean(data2['fol'])
std=np.std(data2['fol'])
mean=float(mean)
std=float(std)
data2['fol']=(data2['fol']-mean)/std
mean=np.mean(data2['folfol'])
std=np.std(data2['folfol'])
mean=float(mean)
std=float(std)
data2['folfol']=(data2['folfol']-mean)/std

# splitting the data
train,test=train_test_split(data2,test_size=0.2)

train = train[['Username','Followers','Follows','Posts','Private','ExternalURL','mutuals','fol','folfol','followback']]
data1 = data1[['Username','Followers','Follows','Posts','Private','ExternalURL','mutuals','fol','folfol','followback']]

from sklearn.ensemble import RandomForestClassifier

important_test = data1[['fol','Follows','folfol','mutuals']]
important_train = train[['fol','Follows','folfol','mutuals']]
new_clf = RandomForestClassifier(n_estimators=500,random_state=5,max_depth=2)
new_clf.fit(important_train, train.iloc[:,-1])
y_pred = new_clf.predict(important_test)
# taking the predicting result as int in output
output = y_pred.astype(int)
# taking the dataframe
df_output = pd.DataFrame()
aux = pd.read_csv(predfile)
# creating the csv file for the output
df_output['Username'] = aux['Username']
df_output['followback'] = np.vectorize(lambda s: '1' if s == 1 else '0')(output)
df_output[['Username','followback']].to_csv('output.csv', index=False)

# to open the output csv file
webbrowser.open('output.csv')


