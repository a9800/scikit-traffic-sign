import pandas
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# Read pixel values into X, read class values into y
df_X = pandas.read_csv("../../data/x_train_gr_smpl.csv")
df_y = pandas.read_csv("../../data/y_train_smpl.csv")

# Shuffle the order of the data (keeping the X and y rows in sync)
df_X, df_y = shuffle(df_X, df_y)

# Split dataset into training and testing set, 90% and 10%, respectively
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.1, random_state=0)

naive_bayes = MultinomialNB()
classifier = naive_bayes.fit(X_train, y_train)
y_predicted = naive_bayes.predict(X_test)
print("\nNaive Bayes accuracy score: ", round(metrics.accuracy_score(y_test, y_predicted)*100,2), "%\n")

# Plot non-normalized confusion matrix
labels=["0","1","2","3","4","5","6","7","8","9"]

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()