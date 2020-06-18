# 1
# Getting Required Packages and tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, roc_auc_score
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
# Feature Selection by Regression
rfe_selection = RFE(estimator=LogisticRegression(), step=10, verbose=5)
rfe_selection.fit(X, Y.values.ravel())
rfe_support = rfe_selection.get_support()
sel_features = X.loc[:, rfe_support].columns.tolist()
print(sel_features)

# Creating new Dataframe on selected variables
XT = X[sel_features]

# 4
# Model Building

logit_model = sm.Logit(Y, XT)
result = logit_model.fit()
print(result.summary2())

# Removing ['equity_crowdfunding','product_crowdfunding','convertible_note'] as it has p value > 0.05
XT = XT.drop(columns=['equity_crowdfunding', 'product_crowdfunding', 'convertible_note'])
XTT = XT
# x_train, x_test, y_train, y_test = train_test_split(XTT, Y, test_size=0.2, random_state=0)

# Normalization
Scale = StandardScaler()
XTTT = Scale.fit_transform(XTT)
XTTT = pd.DataFrame.from_records(XTTT)
x_train, x_test, y_train, y_test = train_test_split(XTTT, Y, test_size=0.2, random_state=0)

# Model Fitting
logreg = LogisticRegression()
logreg.fit(x_train, y_train.values.ravel())

# Predicting the test set results and calculating the accuracy
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
# Accuracy of logistic regression classifier on test set: 0.95

# Confusion matrix
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

# Ploy
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized Confusion Matrix For Random Forest", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(logreg, x_test, y_test,
                                 display_labels=['Operating', 'Closed', 'Acquired'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.show()

# classification_report
print('Classification Report')
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC For Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('ROC For Logistic Regression')
plt.show()


def roc_multiclass_cruve_lg(y_test_class, y_pred_class):
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
    plt.title('ROC For Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig('ROC For Losgistic')
    return plt.show()


# RoC
print(roc_multiclass_cruve_lg(y_test, y_pred))
