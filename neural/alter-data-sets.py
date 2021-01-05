Simport pandas
from sklearn.utils import shuffle

def move(training, testing, number):
    train = training
    test = testing
    
    test = test.append(train[0:number])
    return test


def train_remove_samples(training, number):
    train = training
    
    train = train[number:]
    return train



df_X_train = pandas.read_csv("../data/x_train_gr_smpl.csv")
df_X_test  = pandas.read_csv("../data/x_test_gr_smpl.csv")

print("Original X train Size:",len(df_X_train))
print("Original X test Size:",len(df_X_test))

df_y_train = pandas.read_csv("../data/y_train_smpl.csv")
df_y_test  = pandas.read_csv("../data/y_test_smpl.csv")

print("Original y train Size:",len(df_y_train))
print("Original y test Size:",len(df_y_test),"\n")

#np.random.seed(3)
#np.random.shuffle(np.ravel(df_X_train))
#np.random.seed(3)
#np.random.shuffle(np.ravel(df_y_train))
df_X_train, df_y_train = shuffle(df_X_train,df_y_train)

y_test_4k = move(df_y_train, df_y_test,4000)
y_test_4k.to_csv("../data/4000_y_test_smpl.csv",index=False)

print("4k y test:", len(y_test_4k))

y_train_4k = train_remove_samples(df_y_train,4000)
y_train_4k.to_csv("../data/4000_y_train_smpl.csv",index=False)

print("4k y train:", len(y_train_4k))

y_test_9k = move(df_y_train,df_y_test,9000)
y_test_9k.to_csv("../data/9000_y_test_smpl.csv",index=False)

print("9k y test:", len(y_test_9k))

y_train_9k = train_remove_samples(df_y_train,9000)
y_train_9k.to_csv("../data/9000_y_train_smpl.csv",index=False)

print("9k y train:", len(y_train_9k),"\n")

X_test_4k = move(df_X_train,df_X_test,4000)
X_test_4k.to_csv("../data/4000_x_test_gr_smpl.csv",index=False)

print("4k X test:", len(X_test_4k))

X_train_4k = train_remove_samples(df_X_train,4000)
X_train_4k.to_csv("../data/4000_x_train_gr_smpl.csv",index=False)

print("4k X train:", len(X_train_4k))

X_test_9k = move(df_X_train,df_X_test,9000)
X_test_9k.to_csv("../data/9000_x_test_gr_smpl.csv",index=False)

print("9k X test:", len(X_test_9k))

X_train_9k = train_remove_samples(df_X_train,9000)
X_train_9k.to_csv("../data/9000_x_train_gr_smpl.csv",index=False)

print("9k X train:", len(X_train_9k))


for i in range(0,10):
    y_train = pandas.read_csv("../data/y_train_smpl_"+str(i)+".csv")
    y_test = pandas.read_csv("../data/y_test_smpl_"+str(i)+".csv")

    y_train = shuffle(y_train, random_state = 0)
    
    y_test_4k = move(df_y_train,df_y_test,4000)
    y_test_4k.to_csv("../data/4000_y_test_"+str(i)+".csv",index=False)

    y_test_9k = move(df_y_train, df_y_test,9000)
    y_test_9k.to_csv("../data/9000_y_test_"+str(i)+".csv",index=False)

