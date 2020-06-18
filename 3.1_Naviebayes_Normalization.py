# 1
# Getting Required Packages and tools
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 2
# Uploading the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

# 3
# Dimensionality reduction
# Feature Selection by Annova
selection = SelectKBest(score_func=f_classif)
selection_fit = selection.fit(X, Y.values.ravel())
print(selection_fit.scores_)
# print scores attained by different attributes#
XT = selection_fit.transform(X)
XT = pd.DataFrame.from_records(XT)

# Normalization
Scale = StandardScaler()
x_Scale = Scale.fit(XT)
XTT = Scale.transform(XT)
XTT = pd.DataFrame.from_records(XTT)
x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(XTT, Y, test_size=0.2, random_state=0)

# Navie Bayes with Normalization
nb = GaussianNB()
# Model Fitting without Normalization
nb.fit(x_train_n, y_train_n.values.ravel())
# Model Test without Normalization
y_pred_n = nb.predict(x_test_n)


def multi_class_roc_auc_score_confusion_matrix(y_test, y_pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)), \
           roc_auc_score(y_test, y_pred, average=average)


# results
print('Model Performance with Normalization')
print(accuracy_score(y_test_n, y_pred_n))  # 0.8350410875519935
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_n, y_pred_n))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_n, y_pred_n))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_n, y_pred_n)))
print('multi_class_roc_auc_score_Confusion_matrix:')
print(multi_class_roc_auc_score_confusion_matrix(y_test_n, y_pred_n))  # 0.5173702881298704

# classification_report
print('Classification Report')
print(classification_report(y_test_n, y_pred_n))


def roc_multiclass_cruve_naive_n(y_test_class, y_pred_class):
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
    plt.title('ROC For Navie Bayes with Normalization')
    plt.legend(loc="lower right")
    plt.savefig('ROC For Navie-Normal')
    return plt.show()


# RoC
print(roc_multiclass_cruve_naive_n(y_test_n, y_pred_n))
