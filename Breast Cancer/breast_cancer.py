# -*- coding: utf-8 -*-

import matplotlib.pyplot as plot
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from scipy import misc

from tensorflow import set_random_seed

from numba import cuda


from translate import translate,eval_result
from test_io import allocate_result

import pandas as pd
import numpy as np



set_random_seed(5)

dataset = pd.read_csv('data/test/dataset.csv')


img_ids_train = dataset.head(1800)['file'].values
target_train = dataset.head(1800)['severity'].values

img_ids_test = dataset.tail(281)['file'].values
target_test = dataset.tail(281)['severity'].values



feature_train = []
feature_test = []

target_train = translate(target_train)
target_test = translate(target_test)

#Image loading
for index,img_id in np.ndenumerate(img_ids_train):
     
    
    
    img = misc.imread(img_id)

    if img.shape[0] != 460:
        target_train = np.delete(target_train,index)
        img_ids_train = np.delete(img_ids_train,index)
    else:
        feature_train.append(np.array([img]))  
        
    

    
for index,img_id in np.ndenumerate(img_ids_test):
    
    img = misc.imread(img_id)
    
    if img.shape[0] != 460:
        target_test = np.delete(target_test,index)
        img_ids_test = np.delete(img_ids_test,index)
    else:
        feature_test.append(np.array([img]))   


#feature_train = np.array(feature_train)
#feature_test = np.array(feature_test)

feature_train = np.concatenate(feature_train,axis=0)
feature_test = np.concatenate(feature_test,axis=0)


train_predictors = feature_train.reshape(feature_train.shape[0],460,700,3)
test_predictors = feature_test.reshape(feature_test.shape[0],460,700,3)

train_predictors = train_predictors.astype('float32')
test_predictors = test_predictors.astype('float32')

train_predictors /= 255
test_predictors /= 255

train_classes = np_utils.to_categorical(target_train,2)
test_classes = np_utils.to_categorical(target_test,2)

classifier = Sequential()


classifier.add(Conv2D(32,(8,8),input_shape=(460,700,3),activation='relu'))
classifier.add(BatchNormalization())


classifier.add(MaxPooling2D(pool_size=(4,4)))

classifier.add(Conv2D(32,(8,8),activation='relu'))
classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(4,4)))

#Flattening
classifier.add(Flatten())

#Dense NN
classifier.add(Dense(units = 128,activation='relu'))
 
classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 128,activation='relu'))
 
classifier.add(Dense(units = 128,activation='relu'))
 

classifier.add(Dense(units = 2,activation='softmax')) 
classifier.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


classifier.fit(train_predictors,train_classes,batch_size=20,epochs=14,validation_data=(test_predictors,test_classes))


classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")

#results = []

#index = 0

#for predictor in test_predictors:
    
#
#    
#    result_class = classifier.predict(np.array([predictor,]),verbose=True)
#    
#    result = eval_result(result_class)
#    
#    allocate_result(img_ids_test[index],result)
#    
#    results.append(result_class)
#    
#    index = index + 1
#
cuda.close()