# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


import os
from glob import glob
results1 = [y for x in os.walk('data/test/benign') for y in glob(os.path.join(x[0], '*.png'))]

data = []

for result in results1:
    
    data.append([result,'benign'])
    

results2 = [y for x in os.walk('data/test/malignant') for y in glob(os.path.join(x[0], '*.png'))]

for result in results2:
    
    data.append([result,'malignant'])
    
data = np.array(data)

totalRows = data.shape[0]

benignRows = len(results1)
malignantRows = len(results2)

benignPerc = benignRows/totalRows
malignantPerc = malignantRows/totalRows

np.random.shuffle(data)

columns = ['file','severity']

df = pd.DataFrame(data,columns=columns)

df.to_csv('data/test/dataset.csv',index=False)