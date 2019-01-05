# -*- coding: utf-8 -*-

import matplotlib.pyplot as plot
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

#Pre processing
(feature_train, target_train), (feature_test,target_test) = mnist.load_data()

train_predictors = feature_train.reshape(feature_train.shape[0],28,28,1)
test_predictors = feature_test.reshape(feature_test.shape[0],28,28,1)

train_predictors = train_predictors.astype('float32')
test_predictors = test_predictors.astype('float32')

train_predictors /= 255
test_predictors /= 255

train_classes = np_utils.to_categorical(target_train,10)
test_classes = np_utils.to_categorical(target_test,10)

classifier = Sequential()


#Convolution 1
classifier.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
classifier.add(BatchNormalization())

#Pooling 1
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Convolution 2
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(BatchNormalization())

#Pooling 2
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening 
classifier.add(Flatten()) #Just 1 flattening at the end

#Dense NN
classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dropout(0.2)) #This layer will make some input values 0 to help avoiding overfitting (20% of them is this case)
classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 10,activation='softmax')) 
classifier.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

classifier.fit(train_predictors,train_classes,batch_size=128,epochs=5,validation_data=(test_predictors,test_classes))

result = classifier.evaluate(test_predictors,test_classes)