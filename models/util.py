import pathlib

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd
def custom_plot_confusion_matrix(cm, classes,
                                 normalize=False,
                                 title='Confusion matrix',
                                 cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_pretrain_data(number_samples=100000):
    # Prepare data
    pretrain_data = pd.read_json('./data/pretrain/snli_1.0_train.jsonl', lines=True)
    pretrain_data = pretrain_data[:number_samples]
    pretrain_data = pretrain_data[['sentence1', 'sentence2', 'gold_label']].rename(
        columns={'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})
    pretrain_data['id'] = list(range(len(pretrain_data)))
    pretrain_data['label'] = pretrain_data['label'].replace('entailment', 0)
    pretrain_data['label'] = pretrain_data['label'].replace('neutral', 1)
    pretrain_data['label'] = pretrain_data['label'].replace('contradiction', 2)
    pretrain_data = pretrain_data[pretrain_data['label'] != '-']
    return pretrain_data


def evaluate_model(model, X_test, Y_test):
    print(model.evaluate(X_test, Y_test))
    Y_pred = model.predict(X_test)
    cnf_matrix = confusion_matrix(np.argmax(np.array(Y_test), 1), np.argmax(Y_pred, 1))
    plt.figure()
    custom_plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                                 title='Confusion matrix')
    plt.show()
    print(classification_report(np.argmax(np.array(Y_test), 1), np.argmax(Y_pred, 1)))


def prepare_log_callbacks(batch_size, log_directory):
    tensorboard_log_dir = log_directory + "tensorboard_logs/"

    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=False,
        write_graph=True)

    checkpoint_log_dir = log_directory + "model_checkpoints/"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_log_dir,
        verbose=1,
        monitor="val_loss",
        save_weights_only=False,
        save_best_only=True,
        save_freq=5 * batch_size)

    return hist_callback, cp_callback