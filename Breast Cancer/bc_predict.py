# -*- coding: utf-8 -*-
import matplotlib.pyplot as plot
from keras.datasets import cifar10
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from scipy import misc

from tensorflow import set_random_seed

from numba import cuda


from translate import translate,eval_result
from test_io import allocate_result,copy_replace

import pandas as pd
import numpy as np

json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("classifier.h5")
print("Loaded model from disk")

dataset = pd.read_csv('data/test/dataset.csv')

img_ids_test = dataset.tail(281)['file'].values
target_test = dataset.tail(281)['severity'].values

feature_test = []

target_test = translate(target_test)

for index,img_id in np.ndenumerate(img_ids_test):
    
    img = misc.imread(img_id)
    
    if img.shape[0] != 460:
        target_test = np.delete(target_test,index)
        img_ids_test = np.delete(img_ids_test,index)
    else:
        feature_test.append(np.array([img]))   


copy_replace(img_ids_test)

feature_test = np.concatenate(feature_test,axis=0)


test_predictors = feature_test.reshape(feature_test.shape[0],460,700,3)

test_predictors = test_predictors.astype('float32')

test_predictors /= 255

test_classes = np_utils.to_categorical(target_test,2)


results = []

index = 0

for predictor in test_predictors:
    
   
    result_class = classifier.predict(np.array([predictor,]),verbose=True)
   
    result = eval_result(result_class)
   
    allocate_result(img_ids_test[index],result)
    
    results.append(result_class)
    
    index = index + 1