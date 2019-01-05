# -*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.model_selection import StratifiedKFold

import numpy as np

seed = 5
np.random.seed(seed)

(feature_train, target_train), (feature_test,target_test) = mnist.load_data() #We dont need test data in cross validation

train_predictors = feature_train.reshape(feature_train.shape[0],28,28,1)

train_predictors = train_predictors.astype('float32')

train_predictors /= 255

classes = np_utils.to_categorical(target_train,10)

kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)

results = []

a = np.zeros(5)
b = np.zeros(shape=(classes.shape[0],1))

for train_index,test_index in kfold.split(train_predictors,np.zeros(shape=(classes.shape[0],1))):
    
       #print('Train index: ', train_index, 'Test index: ', test_index) 
       classifier = Sequential()

       classifier.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
       classifier.add(MaxPooling2D(pool_size=(2,2)))
       classifier.add(Flatten()) 
        
       classifier.add(Dense(units = 128,activation='relu'))
        
       classifier.add(Dense(units = 10,activation='softmax')) 
       classifier.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
       
       classifier.fit(train_predictors[train_index],classes[train_index],batch_size=128,epochs=5)
       
       precision = classifier.evaluate(train_predictors[test_index],classes[test_index])
       
       results.append(precision[1])