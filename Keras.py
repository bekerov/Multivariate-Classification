import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras import optimizers
from itertools import cycle
import itertools


def read_data(file_path):
    data = pd.read_excel(file_path, header=0)
    return data


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size - 1
        start += (window_size)


def extract_segments(data, window_size):
    segments = None
    labels = np.empty((0))
    for (start, end) in windows(data, window_size):
        if (len(data.ix[start:end]) == (window_size)):
            signal = data.ix[start:end][range(2, 16)]
            if segments is None:
                segments = signal
            else:
                segments = np.vstack([segments, signal])
            labels = np.append(labels, data.ix[start:end]["goal"][start])
    return segments, labels


win_size = 89
num_var = 14

data = read_data("Multi-variate-Time-series-Data.xlsx")
segments, labels = extract_segments(data, win_size)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
reshaped_segments = segments.reshape(
    [len(segments) / (win_size), (win_size), num_var])


train_test_split = np.random.rand(len(reshaped_segments)) < 0.80
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_all_x = reshaped_segments[~train_test_split]
test_all_y = labels[~train_test_split]


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(64, input_dim=14, input_length=89, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                      epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


model.fit(train_x, train_y, nb_epoch=50, batch_size=64,
          verbose=2, validation_split=0.1)


pred_y = model.predict(test_x, batch_size=64, verbose=2)
plotroc(test_y, pred_y, n_classes=3)

class_names = ['Class 0', 'Class 1', 'Class 3']
# Compute confusion matrix
cnf_matrix = confusion_matrix(
    np.argmax(train_y, axis=1), np.argmax(pred_y, axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig("test_conf.png")
# Plot normalized confusion matrix
plt.figure()


pred_y = model.predict(train_x, batch_size=64, verbose=2)
plotroc(train_y, pred_y, n_classes=3)

class_names = ['Class 0', 'Class 1', 'Class 3']
# Compute confusion matrix
cnf_matrix = confusion_matrix(
    np.argmax(train_y, axis=1), np.argmax(pred_y, axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig("train_conf.png")
# Plot normalized confusion matrix
plt.figure()


print("ROC AUC Score: ", (roc_auc_score(test_y[:, 0], pred_y[:, 0]) + roc_auc_score(
    test_y[:, 1], pred_y[:, 1]) + roc_auc_score(test_y[:, 2], pred_y[:, 2])) / 3.0)


def plotroc(test_y, pred_y, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(test_y[:, i], pred_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot all ROC curves
    plt.figure()
    lw = 1
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LSTM Performance')
    plt.legend(loc="lower right")
    plt.savefig("test_roc.png")
    plt.show()
    return


conf_mat = np.argmax(pred_y, axis=1)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
