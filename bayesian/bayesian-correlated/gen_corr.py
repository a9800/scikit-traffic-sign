# producing correlation matrix of pixels

import pandas
import matplotlib.pyplot as plt
import numpy as np
import glob

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# df_pixels = pandas.read_csv("../data/x_train_gr_smpl.csv")

# Reduced dataset for faster prototyping
df_pixels = pandas.read_csv("../../data/x_train_gr_smpl.csv")
y_pixel = pandas.read_csv("../../data/y_train_smpl.csv")

all_y_smpl=glob.glob("../../data/y_train_smpl_*.csv")

for filename in all_y_smpl:
  print(filename)
  df_y = pandas.read_csv(filename)
  df_x = df_pixels.copy()
  df_x['2304'] = df_y
  print(df_x)
  print(df_x.corr())
  corr = df_x.corr()
  actual_file = filename.split("/")
  corr.to_csv("../../data/corr_" + actual_file[len(actual_file)-1])
