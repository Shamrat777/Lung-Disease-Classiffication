# -*- coding: utf-8 -*-
"""

@author: Shamrat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('')

df

from scipy.stats import wilcoxon
models = ['AlexNet','GoogLeNet','InceptionV3','ResNet50','MobileNetV2','EfficientNetB7','DenseNet121','VGG16','LungNet22'] 
models_pair = [(a, b) for idx, a in enumerate(models) for b in models[idx + 1:]]
models_pair


wilcoxons = {
    'AlexNet': ['-'],
    'GoogLeNet': ['-', '-'],
    'InceptionV3': ['-', '-', '-'],
    'ResNet50': ['-', '-', '-', '-'],
    'MobileNetV2': ['-', '-', '-', '-', '-'],
    'EfficientNetB7': ['-', '-', '-', '-', '-', '-'],
    'DenseNet121': ['-', '-', '-', '-', '-', '-', '-'],
    'VGG16': ['-', '-', '-', '-', '-', '-', '-','-'],
    'LungNet22': ['-', '-', '-', '-', '-', '-', '-','-','-'],
}
for models in models_pair:
  f = str(models[0])
  s = str(models[1])
  w, p = wilcoxon(df[f], df[s], correction=True)
  wilcoxons[f].append(p)
  print('Wilcoxon value of {} and {}: {}'.format(f, s, p))
  
  wilcoxons
  pdf = pd.DataFrame(wilcoxons)
  pdf
  pdf.to_csv('Wilcoxon_ModelComparison.csv', index=False)
  
from google.colab import files
files.download('Wilcoxon_ModelComparison.csv') 
pdf
