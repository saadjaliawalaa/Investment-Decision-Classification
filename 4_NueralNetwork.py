# 1
# Getting Required Packages and tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 2
# Uploading the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

# Normalization
Scale = StandardScaler()
XT = Scale.fit_transform(X)
XT = pd.DataFrame.from_records(XT)

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(XT, Y, test_size=0.2, random_state=0)

# 4
# Model Building
# Neural Model
# def neural_model():
#    model = Sequential()
#    model.add(Dense(8, input_dim=26, activation='relu'))
#    model.add(Dense(3, activation='softmax'))
#    model.compile(loss='sparse_ccategorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    return nueral_model()
# ONEHOTENCODE  #Y = np_utils.to_categorical(Y)
# THIS IS NOT WORKING DUE TO RAM PROBLEM
# estimator = KerasClassifier(build_fn=nueral_model(), epochs=20, batch_size=30, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, copy_data, data_Y) #cv=kfold
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Neural Network
# Model Building
model = Sequential()
model.add(Dense(8, input_dim=26, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Model Topology(Not working here due to GraphViz error, made it on Google Colab)
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# Model Fitting
Trained = model.fit(x_train, y_train, epochs=15, batch_size=100, validation_split=0.2)

# Loss Vs Epochs
Trained_dict = Trained.history
epochs = range(1, 16)
training_loss = Trained_dict['loss']
validation_loss = Trained_dict['val_loss']

plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show

# Accuracy Vs Epochs
Trained_dict1 = Trained.history
epochs1 = range(1, 16)
training_acc1 = Trained_dict1['accuracy']
validation_acc1 = Trained_dict1['val_accuracy']

plt.plot(epochs1, training_acc1, label='Training Accuracy')
plt.plot(epochs1, validation_acc1, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy & Loss')
plt.legend()
plt.savefig('AccuracyVsEpochs')
plt.show

# evaluate the model
accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Model Loss & Accuracy', )

# Model Testing
y_hat = model.predict_classes(x_test, batch_size=10)

# classification_report
print('Classification Report')
print(classification_report(y_test, y_hat))

# Confusion
print('Confusion Matrix')
print(confusion_matrix(y_test, y_hat))


def roc_multiclass_cruve_nn(y_test_class, y_pred_class):
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
    plt.title('ROC For Neural Network')
    plt.legend(loc="lower right")
    plt.savefig('ROC For Neural Network')
    return plt.show()


# RoC
print(roc_multiclass_cruve_nn(y_test, y_hat))
