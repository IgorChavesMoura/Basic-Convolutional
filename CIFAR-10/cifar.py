# -*- coding: utf-8 -*-

import matplotlib.pyplot as plot
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization



(feature_train, target_train), (feature_test, target_test) = cifar10.load_data()




feature_train = feature_train.astype('float32')
feature_test = feature_test.astype('float32')

feature_train /= 255
feature_test /= 255

#plot.imshow(feature_train[1])

train_classes = np_utils.to_categorical(target_train,10)
test_classes = np_utils.to_categorical(target_test,10)

classifier = Sequential()

#Convolution 1
classifier.add(Conv2D(64,(3,3),input_shape=(32,32,3),activation='relu'))
classifier.add(BatchNormalization())

#Pooling 1
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Convolution 2
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(BatchNormalization())

#Pooling 2
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Flattening
classifier.add(Flatten())

#Dense NN
classifier.add(Dense(units = 96,activation='relu'))
classifier.add(Dropout(0.3)) 
classifier.add(Dense(units = 96,activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 96,activation='relu'))
classifier.add(Dropout(0.3))

classifier.add(Dense(units = 10,activation='softmax')) 
classifier.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

classifier.fit(feature_train,train_classes,batch_size=96,epochs=5,validation_data=(feature_test,test_classes))