import random
import pandas as pd
import time 
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
plt.style.use('ggplot')

