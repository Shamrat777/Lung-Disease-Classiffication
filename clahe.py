# -*- coding: utf-8 -*-
"""
@author: Shamrat
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

#input = r''
i = 0
#start = time.time()
for img in glob.glob(input + '/*.jpeg'):
    #choose the directory file of the images
    image = cv2.imread(r''%i,0)
    #Creating CLAHE 
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10,10))
    #Apply CLAHE to the original image
    image_clahe = clahe.apply(image)
    #cv2.imwrite(r''%i,image_clahe)
    i += 1