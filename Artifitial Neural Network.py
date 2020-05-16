#Artifitial Neural Network

#Part 1-Data Preprocessing
#import the libraries
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset
dataset=pd.read_csv("Churn_Modelling.csv")
x= dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:, 1]=labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2=LabelEncoder()
x[:, 2]=labelencoder_x_2.fit_transform(x[:, 2])
OneHotEncoder=OneHotEncoder(categorical_features=[1])
x=OneHotEncoder.fit_transform(x).toarray()
x=x[:,1:]

#splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Part-2-Make the ANN
#Importing the Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier=Sequential()

#Adding the imput layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation ='relu',input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation ='relu'))

#Adding output layer
classifier.add(Dense(output_dim=1, init='uniform', activation ='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics= ['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(x_train,y_train,batch_size=10, nb_epoch=100)

#Part-3-Making the prediction and evaluating the model
#predicting the test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
