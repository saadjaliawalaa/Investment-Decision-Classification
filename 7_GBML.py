# 1
# Getting Required Packages and tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# 2
# Uploading the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 4
# Model Building 1
# evaluate the Gradient Boosting Classifier model
model_1 = GradientBoostingClassifier()
cv_1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_1, x_train, y_train.values.ravel(), scoring='accuracy', cv=cv_1, n_jobs=-1,
                           error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# Accuracy: 0.873 (0.000)
# fit the model
model_1 = GradientBoostingClassifier()
model_1.fit(x_train, y_train)

# Prediction
y_hat_1 = model_1.predict(x_test)

# classification_report
print(classification_report(y_test, y_hat_1))

# Confusion
confusion_matrix_1 = confusion_matrix(y_test, y_hat_1)
print(confusion_matrix_1)

# Model Building 2
model_2 = XGBClassifier()
cv_2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_2,  x_train, y_train.values.ravel(), scoring='accuracy', cv=cv_2, n_jobs=-1,
                           error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# Accuracy: 0.871 (0.001)

# fit the model on the whole dataset
model_2.fit(x_train, y_train.values.ravel())

# Prediction
y_hat_2 = model_2.predict(x_test)

# classification_report
print(classification_report(y_test, y_hat_2))

# Confusion
confusion_matrix_2 = confusion_matrix(y_test, y_hat_2)
print(confusion_matrix_2)


def roc_multiclass_cruve_xgb(y_test_class, y_pred_class):
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
    plt.title('ROC For Extreme Gradient Boosting')
    plt.legend(loc="lower right")
    plt.savefig('ROC For XGB')
    return plt.show()


# RoC
print(roc_multiclass_cruve_xgb(y_test, y_hat_2))
