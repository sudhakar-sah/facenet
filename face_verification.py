from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model 
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_transform
from keras.engine.topology import Layer
from keras import backend as K

# K.set_image_data_format('channels_first')

import cv2
import os
import numpy as np 
from numpy import getfromtxt 
import pandas as pd 
import tensorflow as tensorflow
from fr_utils import *
from inception_blocks_v2 import * 

%matplotlib inline 
%load_ext autoreload 
%autoreload 2

np_setprintoptions(threshold=np.nan)

