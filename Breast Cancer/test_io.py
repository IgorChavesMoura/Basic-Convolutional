# -*- coding: utf-8 -*-

from shutil import copyfile,copy2

import os




def allocate_result(filepath,result):
    
    filename = filepath.split('/')
    
    filename = filename[len(filename) - 1]
    
    if not os.path.exists('data/test/result'):
        os.makedirs('data/test/result')
    
    if result == 0:
        
        copyfile(filepath,'data/test/result/benign/' + filename)
        
    elif result == 1:
        
        copyfile(filepath,'data/test/result/malignant/' + filename)
        
def copy_replace(filepaths):
    
    if not os.path.exists('data/samples'):
        os.makedirs('data/samples')
    
    for index,path in enumerate(filepaths):
        
        filename = path.split('/')
    
        filename = filename[len(filename) - 1]
        
        copyfile(path,'data/samples/' + filename)
        
        filepaths[index] = 'data/samples/' + filename