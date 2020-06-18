# 1
# Getting Required Packages and tools
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

# 2
# Uploading the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

# Combining "operating and Acquired"
Y = Y.replace(2, 0)

# 3
# Dimensionality reduction
# Feature Selection by Annova
selection = SelectKBest(score_func=f_classif)
selection_fit = selection.fit(X, Y.values.ravel())
print(selection_fit.scores_)
# print scores attained by different attributes
XT = selection_fit.transform(X)
XT = pd.DataFrame.from_records(XT)
x_train, x_test, y_train, y_test = train_test_split(XT, Y, test_size=0.2, random_state=0)

# 4
# Model Building
# Random Forest
rf = RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=42)
rf.fit(x_train, y_train.values.ravel())

y_train_preds = rf.predict(x_train)
y_test_preds = rf.predict(x_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_preds))
# 0.052450035507760985
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_preds))
# 0.052450035507760985
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_preds)))
# 0.22901972733317316
print('accuracy_score:', accuracy_score(y_test, y_test_preds))
# 0.947549964492239
print('Confusion Matrix')
print(confusion_matrix(y_test, y_test_preds))

#  Plot For CM
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized Confusion Matrix For Random Forest", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rf, x_test, y_test,
                                 display_labels=['Operating', 'Closed'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.savefig('CM For Random Forest Classifier')
plt.show()

# classification_report
print('Classification Report')
print(classification_report(y_test, y_test_preds))

# ROC Score
ROC = roc_auc_score(y_test, y_test_preds)
print('ROC Score:', ROC)


# evaluate the results
def roc_multiclass_cruve_rf(y_test_class, y_pred_class):
    lb = LabelBinarizer()
    lb.fit(y_test_class)
    y_test_b = lb.transform(y_test_class)
    y_pred_b = lb.transform(y_pred_class)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_test_b[:, 0], y_pred_b[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])
    # Compute micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_b.ravel(), y_pred_b.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 1
    # First aggregate all false positive rates
    all_fpr = fpr[0]

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    mean_tpr += np.interp(all_fpr, fpr[0], tpr[0])

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
    list = [0]
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(list, colors):
        plt.plot(fpr[0], tpr[0], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(0, roc_auc[0]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For Random Forest')
    plt.legend(loc="lower right")
    plt.savefig('ROC For RF')
    return plt.show()


# RoC
print(roc_multiclass_cruve_rf(y_test, y_test_preds))
