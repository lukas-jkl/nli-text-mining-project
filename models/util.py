import pathlib

import matplotlib.pyplot as plt
import itertools, json
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


def evaluate_model(model, X_test, Y_test, log_dir):
    log_dir += "/final_model/"

    print(model.evaluate(X_test, Y_test))
    tf.keras.models.save_model(model, log_dir + "final_model")
    Y_pred = model.predict(X_test)
    cnf_matrix = confusion_matrix(np.argmax(np.array(Y_test), 1), np.argmax(Y_pred, 1))
    plt.figure()
    custom_plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                                 title='Confusion matrix')
    plt.savefig(log_dir + "/confusion_matrix.png")
    plt.show()
    class_report = classification_report(np.argmax(np.array(Y_test), 1), np.argmax(Y_pred, 1))
    with open(log_dir + '/classification_report.txt', 'w') as file:
        file.write(class_report)
    print(class_report)


def get_log_directory(model_name, title, pretraining=False):
    if pretraining:
        train_type = "pretraining"
    else:
        train_type = "training"
    log_dir = "logs/" + model_name + "/" + train_type + "/" + title + "/"
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    return log_dir


def save_final_weights(model, log_directory):
    final_weights_path = log_directory + "final_weights/weights"
    model.save_weights(final_weights_path)
    print("pretraining done, final weights stored to: ", final_weights_path)
    return final_weights_path


def train_model(X_train, Y_train, model, log_directory, batch_size, epochs,
                additional_callbacks, restore_checkpoint):
    hist_callback, cp_callback = prepare_log_callbacks(batch_size, log_directory)
    callbacks = additional_callbacks + [hist_callback, cp_callback]

    if restore_checkpoint:
        print("restoring weights from checkpoint: ", cp_callback.filepath)
        model.load_weights(cp_callback.filepath)
        print("done")

    history = model.fit(X_train, Y_train,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              callbacks=callbacks,
              batch_size=batch_size)
    with open(log_directory + "history.txt", "w") as f:
        f.write(json.dumps(history.history))
    return model, log_directory


def prepare_log_callbacks(batch_size, log_directory):
    tensorboard_log_dir = log_directory + "tensorboard_logs/"

    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=False,
        write_graph=True,
        profile_batch=2)

    checkpoint_log_dir = log_directory + "model_checkpoints/"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_log_dir,
        verbose=1,
        monitor="accuracy",
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch')

    return hist_callback, cp_callback
