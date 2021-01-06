# Bayesian classification running on a dataframe with the top 20 pixels of an image
# In order to reduce over-fitting

import pandas
import matplotlib.pyplot as plt
import numpy as np
import glob

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

all_corr=glob.glob("../../data/corr_y_train_smpl_*.csv")

# Array with top 20, 10, and 5 pixels correlations on each class
arr20 = []
arr10 = []
arr5  = []

# Function to return an array of the top pixels of a sorted pandas dataframe
def top_pixel_val(size,df):
  arr = []
  df = df[0:size]
  
  for i in df.to_numpy():
    arr.append(i)
  
  return arr


# Iterate through all the correlation dataframes created in task 4
for filename in all_corr:
  print(filename)
  df = pandas.read_csv(filename)
  df['2304'] = abs(df['2304'])
  df = df.drop(2304, axis=0)
  # Sort the dataframes by their absolute correlation values on each class
  df = df.sort_values(by=["2304"], ascending=False)
  df = df["Unnamed: 0"]

  # Take top 20 pixel values for each class and add them to its corresponding array
  top20 = top_pixel_val(20,df)
  for i in top20:
    arr20.append(str(i))

  top10 = top_pixel_val(10,df)
  for i in top10:
    arr10.append(str(i))

  top5 = top_pixel_val(5,df)
  for i in top5:
    arr5.append(str(i))

print("\n")
# Remove duplication of pixel values 
arr20 = list(dict.fromkeys(arr20))
print("Top", len(arr20) ,"pixels:",arr20,"\n")

arr10 = list(dict.fromkeys(arr10))
print("Top", len(arr10) ,"pixels:",arr10,"\n")

arr5  = list(dict.fromkeys(arr5))
print("Top", len(arr5) ,"pixels:",arr5,"\n")

df_X = pandas.read_csv("../../data/x_train_gr_smpl.csv")
df_y = pandas.read_csv("../../data/y_train_smpl.csv")
# Creating data set using
# top 20 values
df20_X = df_X[arr20]
print(df20_X)
# Creating a joint dataframe using the x of df 20 and y 
x20 = df20_X.join(df_y)
# Exporting it to a csv to use in Task 7
x20.to_csv("../../data/top_20.csv")

# top 10 values
df10_X = df_X[arr10]
print(df10_X)
# Creating a joint dataframe using the x of df 10 and y 
x10 = df10_X.join(df_y)
# Exporting it to a csv to use in Task 7
x10.to_csv("../../data/top_10.csv")

# top  5 values
df5_X = df_X[arr5]
print(df5_X)
# Creating a joint dataframe using the x of df 5 and y 
x5 = df5_X.join(df_y)
# Exporting it to a csv to use in Task 7
x5.to_csv("../../data/top_5.csv")

# Running the Naive Bayes Classifier on the dataframes created in task 5
def run_classifier(size,dfx,dfy):
  # Shuffle the order of the data (keeping the X and y rows in sync)
  dfx, dfy = shuffle(dfx, dfy)

  # Split dataset into training and testing set, 90% and 10%, respectively
  X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.1, random_state=0)

  naive_bayes = CategoricalNB()
  classifier = naive_bayes.fit(X_train, y_train)
  y_predicted = naive_bayes.predict(X_test)
  print("\n Top",size,"Naive Bayes: ", round(metrics.accuracy_score(y_test, y_predicted)*100,2), "%\n")

run_classifier(20,df20_X,df_y)
run_classifier(10,df10_X,df_y)
run_classifier(5,df5_X,df_y)
