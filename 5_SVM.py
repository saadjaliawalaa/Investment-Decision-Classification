# 1
# Getting Required Packages and tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 2
# Uploading the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

# 3
# Dimensionality reduction
# Normalization
Scale = StandardScaler()
XT = Scale.fit_transform(X)
XT = pd.DataFrame.from_records(XT)

x_train, x_test, y_train, y_test = train_test_split(XT, Y, test_size=0.2, random_state=0)

# 4
# Model Building
# Support Vector Machine
# Linear
clf = SVC(kernel='rbf')
clf.fit(x_train, y_train.values.ravel())
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
# 0.8706502992796997

# Sigmoid
# clf = SVC(kernel='sigmoid')
# clf.fit(x_train,y_train.values.ravel())
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_test,y_pred))
# 0.8551283351932637

# classification_report
print(classification_report(y_test, y_pred))

# Confusion
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


def roc_multiclass_cruve_svm(y_test_class, y_pred_class):
    lb = LabelBinarizer()
    lb.fit(y_test_class)
    y_test_b = lb.transform(y_test_class)
    y_pred_b = lb.transform(y_pred_class)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_pred_b[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_b.ravel(), y_pred_b.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 1
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
                                               ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For Support Vector Machine')
    plt.legend(loc="lower right")
    plt.savefig('ROC For SVM')
    return plt.show()


# RoC
print(roc_multiclass_cruve_svm(y_test, y_pred))
