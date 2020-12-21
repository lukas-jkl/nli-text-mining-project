import pathlib

import kerastuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from data import calculate_embeddings_and_pos_tag, test_training_calculate_embeddings_and_pos_tags
from models.util import custom_plot_confusion_matrix, get_pretrain_data, evaluate_model


class CrossUnigramsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    # Transform a single row of the dataframe.
    def _transform(self, row):
        current_row = ' '.join(row.hypothesis_words)
        for premise_word, premise_pos_tag in zip(row.premises_words, row.premises_pos_tags):
            for hypothesis_word, hypothesis_pos_tag in zip(row.hypothesis_words, row.hypothesis_pos_tags):
                if premise_pos_tag == hypothesis_pos_tag:
                    current_row += ' ' + premise_word + ' ' + hypothesis_word
        return current_row

    def transform(self, X):
        return [self._transform(row) for row in X.itertuples()]


class HypothesisUnigramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    # Transform a single row of the dataframe.
    def _transform(self, row):
        current_row = ' '.join(row.hypothesis_words)
        return current_row

    def transform(self, X):
        return [self._transform(row) for row in X.itertuples()]


class BothSentencesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    # Transform a single row of the dataframe.
    def _transform(self, row):
        current_row = ' '.join(row.hypothesis_words)
        current_row += ' ' + ' '.join(row.premises_words)
        return current_row

    def transform(self, X):
        return [self._transform(row) for row in X.itertuples()]


def sum_embeddings(data):
    hypothesis_embeddings = []
    premises_embeddings = []
    for index, row in data.iterrows():
        hypothesis_embeddings.append(np.sum(np.array(row.hypothesis_word_vectors), 0))
        premises_embeddings.append(np.sum(np.array(row.premises_word_vectors), 0))
    return hypothesis_embeddings, premises_embeddings


class EmbeddingClassifier(HyperModel):
    def __init__(self, log_dir, embedding_size=300, num_outputs=3):
        super().__init__()
        self.log_dir = log_dir
        self.output_size = num_outputs
        self.embedding_size = embedding_size

    def build(self, hp):
        separate_hidden_neurons = hp.Choice("separate_hidden_layer_neurons", [32, 64, 128])
        separate_hidden_layers = hp.Choice("separate_hidden_layer_num_layers", [1])
        separate_hidden_layers = [separate_hidden_neurons] * separate_hidden_layers
        # concatenated_hidden_layers = [hp.Fixed("hidden_layer_concat"), 64]

        layer_hypothesis = input_hypothesis = tf.keras.Input(shape=self.embedding_size,
                                                             name="hypothesis_input_layer")
        layer_premises = input_premises = tf.keras.Input(shape=self.embedding_size,
                                                         name="premises_input_layer")
        inputs = [input_premises, input_hypothesis]
        for i, neurons in enumerate(separate_hidden_layers):
            layer_premises = tf.keras.layers.Dense(neurons, activation='relu',
                                                   name="premises_dense_hidden_layer_" + str(i))(layer_premises)
            layer_hypothesis = tf.keras.layers.Dense(neurons, activation='relu',
                                                     name="hypothesis_dense_hidden_layer_" + str(i))(layer_hypothesis)

        concat = tf.keras.layers.concatenate([layer_premises, layer_hypothesis], axis=1, name="concatenation_layer")

        concat_hidden_neurons = hp.Choice("concat_hidden_layer_neurons", [32, 64, 128])
        concat_hidden_layers = hp.Choice("concat_hidden_layer_num_layers", [1, 2, 3])
        concat_hidden_layers = [concat_hidden_neurons] * concat_hidden_layers
        for i, neurons in enumerate(concat_hidden_layers):
            concat = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                           name="concat_dense_hidden_layer_" + str(i))(concat)

        final_dense_neurons = self.output_size
        output = tf.keras.layers.Dense(final_dense_neurons, activation='softmax',
                                       name="output_layer")(concat)

        model = tf.keras.Model(inputs=inputs,
                               outputs=[output],
                               name=self.name)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        tf.keras.utils.plot_model(
            model, to_file=self.log_dir + 'embedded_model.png', show_shapes=False, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        return model


def get_prepared_embeddings_model(log_dir):
    hp = kt.HyperParameters()
    parameters = [
        hp.Fixed("separate_hidden_layer_neurons", 512),
        hp.Fixed("separate_hidden_layer_num_layers", 2),
        hp.Fixed("concat_hidden_layer_neurons", 1024),
        hp.Fixed("concat_hidden_layer_num_layers", 4)]

    classifier = EmbeddingClassifier(log_dir)
    model = classifier.build(hp)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def pretrain_word_embedding_model():
    pretrain_data = get_pretrain_data()
    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.pkl')
    hypothesis_embeddings_training, premises_embeddings_training = sum_embeddings(pretrain_feature_data)

    X_train = [np.array(hypothesis_embeddings_training), np.array(premises_embeddings_training)]
    Y_train = tf.one_hot(pretrain_data.label.values, 3)

    # Callbacks
    title = "try1"
    log_directory = "logs/embedded_classifier/pretraining/" + title + "/"
    tensorboard_log_dir = log_directory + "tensorboard_logs/"
    pathlib.Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=False)
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=True,
        write_graph=True)

    # Create a callback that saves the model's weights every 5 epochs
    batch_size = 128
    checkpoint_log_dir = log_directory + "model_checkpoints/"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_log_dir,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * batch_size)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model = get_prepared_embeddings_model(log_directory)
    model.fit(X_train, Y_train,
              epochs=100,
              validation_split=0.2,
              callbacks=[early_stopping, hist_callback, cp_callback],
              batch_size=batch_size)

    final_weights_path = log_directory + "final_weights/weights"
    model.save_weights(final_weights_path)
    print("pretraining done, final weights stored to: ", final_weights_path)


def run_word_embedding_model(train, test, load_weighs_from_pretraining=False):
    test_feature_data, train_feature_data = test_training_calculate_embeddings_and_pos_tags(test, train)

    hypothesis_embeddings_test, premises_embeddings_test = sum_embeddings(test_feature_data)
    hypothesis_embeddings_training, premises_embeddings_training = sum_embeddings(train_feature_data)

    X_train = [np.array(hypothesis_embeddings_training), np.array(premises_embeddings_training)]
    Y_train = tf.one_hot(train.label.values, 3)
    X_test = [np.array(hypothesis_embeddings_test), np.array(premises_embeddings_test)]
    Y_test = tf.one_hot(test.label.values, 3)

    title = "try1"
    log_directory = "logs/embedded_classifier/training/" + title + "/"
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    model = get_prepared_embeddings_model(log_directory)
    if load_weighs_from_pretraining:
        model.load_weights("logs/embedded_classifier/pretraining/try1/final_weights/weights")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    model.fit(X_train, Y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])
    evaluate_model(model, X_test, Y_test)


def hyperparameter_search(X_train, Y_train):
    log_directory = "logs/embedded_classifier/"

    classifier = EmbeddingClassifier()
    tuner = kt.Hyperband(classifier,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=2,
                         directory=log_directory + 'hyperparams',
                         )

    search_title = "try"
    tensorboard_log_dir = log_directory + "tensorboard_logs/" + search_title
    pathlib.Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=False,
        write_graph=True)
    stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3)

    tuner.search(X_train, Y_train, validation_split=0.2, callbacks=[hist_callback, stop_callback])
    tuner.results_summary(5)
    print("done")


def run_manual_feature_model(train, test):
    test_feature_data, train_feature_data = test_training_calculate_embeddings_and_pos_tags(test, train)
    mlp_classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(20, 20), max_iter=2000, verbose=True,
                                   early_stopping=True, n_iter_no_change=8)
    random_forest = RandomForestClassifier()
    logreg = LogisticRegression(max_iter=2000, tol=1e-4, verbose=2)

    pipeline_hypothesis_only = Pipeline(
        [
            ('transformer', HypothesisUnigramTransformer()),
            ('vectorizer', CountVectorizer()),
            ('classifier', mlp_classifier)
        ])

    mlp_classifier_big = MLPClassifier(random_state=1, hidden_layer_sizes=(8, 4), max_iter=2000, verbose=True,
                                       early_stopping=True, n_iter_no_change=8)
    pipeline_hypothesis_and_premise = Pipeline(
        [
            ('transformer', BothSentencesTransformer()),
            ('vectorizer', CountVectorizer()),
            ('classifier', mlp_classifier_big)
        ]
    )
    pipeline = pipeline_hypothesis_only

    pipeline.fit(train_feature_data, train_feature_data['label'])
    y_pred = pipeline.predict(test_feature_data)
    plot_confusion_matrix(pipeline, test_feature_data, test_feature_data['label'])
    plt.show()
    print(classification_report(test_feature_data['label'], y_pred))
