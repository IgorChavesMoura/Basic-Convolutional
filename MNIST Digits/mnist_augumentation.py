# -*- coding: utf-8 -*-


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

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



classifier.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten()) 


classifier.add(Dense(units = 128,activation='relu'))


classifier.add(Dense(units = 10,activation='softmax')) 
classifier.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


train_generator = ImageDataGenerator(rotation_range=7,horizontal_flip=True,shear_range=0.2,height_shift_range=0.07,zoom_range=0.2)

test_generator = ImageDataGenerator()

train_database = train_generator.flow(train_predictors,train_classes, batch_size=128)

test_database = test_generator.flow(test_predictors,test_classes, batch_size=128)

classifier.fit_generator(train_database,steps_per_epoch = 60000/128,epochs=5,validation_data=test_database,validation_steps=10000/128, use_multiprocessing=True,workers=3)