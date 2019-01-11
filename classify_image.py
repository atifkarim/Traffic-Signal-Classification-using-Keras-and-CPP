#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:50:57 2019

@author: atif
"""

#############################
##Importing Library##########
#############################

import numpy as np
from skimage import io, color, exposure, transform
from skimage.color import rgb2gray
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split  #it came from update scikit learn. https://stackoverflow.com/questions/40704484/importerror-no-module-named-model-selection
import os
import glob
import h5py

from matplotlib import pyplot as plt
#%matplotlib inline

NUM_CLASSES = 9 #Used class for the training
IMG_SIZE = 48 #required size. This size has also maintained during training. User defined value