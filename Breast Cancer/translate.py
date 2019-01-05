# -*- coding: utf-8 -*-

import numpy as np

#translate_map = { 
    
#    'CALC':0,
#    'CIRC':1,
#    'SPIC':2,
#    'MISC':3,
#    'ARCH':4,
#    'ASYM':5,
#    'NORM':6         
    
#}

#translate_map_inverse = { 
#    
#    0:'CALC',
#    1:'CIRC',
#    2:'SPIC',
#    3:'MISC',
#    4:'ARCH',
#    5:'ASYM',
#    6:'NORM'         
    
#}


translate_map = { 
    
    'benign':0,
    'malignant':1,
         
    
}
def translate(target):
        
    return translate_map[target]

translate = np.vectorize(translate)

def eval_result(r_class):
    
    if r_class[0][0] > r_class[0][1]:
        
        return 0
    
    else:
        
        return 1