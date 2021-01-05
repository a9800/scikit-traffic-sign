import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import plot_confusion_matrix, classification_report
from statistics import mean
from sklearn.utils import shuffle


def plot_roc_curve(fpr, tpr, label=None):
    """
    This function plots a ROC Curve, adding a dashed diagonal line.

    Keyword arguments:
    fpr -- False Positive Rate
    tpr -- True Positive Rate
    label -- Label used for the plot

    Reference/Credits:
    Function from Tutorial 2
    URL: https://colab.research.google.com/drive/14sNilBGFICZV-93HBLyWv3zLTO0P0kMs?usp=sharing
    """
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)


def test(X_train, y_train, X_test, y_test, model, roc_test):
    """
    This runs the classifier, shows the confusion matrix, and other metrics.

    Keyword arguments:
    X_train -- The data to fit.
    y_train -- Target to predict.
    X_test -- The input data for the classifier to predict.
    y_test -- Target values for the estimator.
    model -- The estimator object.
    roc_test -- The target scores used for computing ROC score/graph, as an
        array.
    """
    X_train, y_train = shuffle(X_train, y_train)
    # 10-fold cross validation
    print("Running 10-fold cross validation ...")
    y_train_pred = cross_val_predict(model, X_train, np.ravel(y_train), cv=10)
    print("10-fold cross validation Average Accuracy: ",
          metrics.accuracy_score(y_train, y_train_pred))

    classifier = model.fit(X_train, np.ravel(y_train))

    y_predicted = classifier.predict(X_test)
    accuracy_score = round(metrics.accuracy_score(y_test, y_predicted)*100, 2)
    print("\nNumber of examples in training set:", len(X_train))
    print("Number of examples in testing:", len(X_test))
    print("Accuracy on supplied test:", accuracy_score, "%\n")

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]

    # Confusion Matrix
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=labels,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        fp = disp.confusion_matrix.sum(axis=0) - np.diag(disp.confusion_matrix)
        fn = disp.confusion_matrix.sum(axis=1) - np.diag(disp.confusion_matrix)
        tp = np.diag(disp.confusion_matrix)
        tn = disp.confusion_matrix.sum() - (fp + fn + tp)
        plt.show()

        fp = fp.astype(float)
        fn = fn.astype(float)
        tp = fp.astype(float)
        tn = tn.astype(float)

        if normalize is None:
            print("Detailed info:\n")
            clas_rep = classification_report(y_train, y_train_pred)
            print(clas_rep)
            print(disp.confusion_matrix.ravel())
            for i in range(0, len(roc_test)):
                print(" == class " + str(i) + " ==")
                binar = (y_predicted == i)
                fpr, tpr, thresholds = roc_curve(binar, roc_test[i])
                print("TP Rate:", mean(tpr))
                print("FP Rate:", mean(fpr))
                print("Threshold:", mean(thresholds))
                print("ROC:", 1-roc_auc_score(binar, roc_test[i]))
                plot_roc_curve(tpr, fpr, label="class"+str(i))
                plt.show()
