import pandas
from sklearn.linear_model import SGDClassifier
from shared import test

print("Loading data")
# Read pixel values into X, read class values into y
df_X = pandas.read_csv("../../data/x_train_gr_smpl.csv")
df_y = pandas.read_csv("../../data/y_train_smpl.csv")

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

df_X_test = pandas.read_csv("../../data/x_test_gr_smpl.csv")
df_y_test = pandas.read_csv("../../data/y_test_smpl.csv")

test_reg = []
test_4k = []
test_9k = []

for i in range(0, 10):
    test_reg.append(pandas.read_csv("../../data/y_test_smpl_"+str(i)+".csv"))
    test_4k.append(pandas.read_csv("../../data/4000_y_test_"+str(i)+".csv"))
    test_9k.append(pandas.read_csv("../../data/9000_y_test_"+str(i)+".csv"))

# Test model on data set
print("Running test on full dataset")

test(df_X, df_y, df_X_test, df_y_test, model, test_reg)

# Alter dataset by moving 4000 examples from training set to test set
# Adding 4000 instances to a temporary x and y dataframe
# Removing 4000 instances from df_X and df_y
X_train = pandas.read_csv("../../data/4000_x_train_gr_smpl.csv")
X_test = pandas.read_csv("../../data/4000_x_test_gr_smpl.csv")
y_test = pandas.read_csv("../../data/4000_y_test_smpl.csv")
y_train = pandas.read_csv("../../data/4000_y_train_smpl.csv")

print("Running test on 4k dataset")
test(X_train, y_train, X_test, y_test, model, test_4k)

# Adding 9000 instances to a temporary x and y dataframe
# Removing 9000 instances from df_X and df_y
X_train = pandas.read_csv("../../data/9000_x_train_gr_smpl.csv")
X_test = pandas.read_csv("../../data/9000_x_test_gr_smpl.csv")
y_test = pandas.read_csv("../../data/9000_y_test_smpl.csv")
y_train = pandas.read_csv("../../data/9000_y_train_smpl.csv")

print("Running test on 9k dataset")
test(X_train, y_train, X_test, y_test, model, test_9k)
