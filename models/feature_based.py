import time

import kerastuner as kt
from kerastuner import HyperModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime


from data import calculate_embeddings_and_pos_tag, test_training_calculate_embeddings_and_pos_tags
from models.util import *


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

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        with open(self.log_dir + 'hyperparameters.txt', 'w') as f:
            f.write(json.dumps(hp.values))

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
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def pretrain_word_embedding_model(title=None, restore_checkpoint=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    pretrain_data = get_pretrain_data()
    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.feather')
    hypothesis_embeddings_training, premises_embeddings_training = sum_embeddings(pretrain_feature_data)

    X_train = [np.array(hypothesis_embeddings_training), np.array(premises_embeddings_training)]
    Y_train = np.array(pretrain_data.label.values, dtype='int32')

    model_name = "embedded_classifier"
    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_prepared_embeddings_model(log_directory)

    model = train_model(X_train, Y_train,
                                     model=model,
                                     log_directory=log_directory,
                                     batch_size=batch_size,
                                     epochs=100,
                                     additional_callbacks=[early_stopping],
                                     restore_checkpoint=restore_checkpoint)

    final_weights_path = save_final_weights(model, log_directory)
    print("done")


def run_word_embedding_model(train, test, data_name, title=None, restore_checkpoint=False, load_weights_from_pretraining=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    test_feature_data, train_feature_data = test_training_calculate_embeddings_and_pos_tags(test, train, data_name)

    hypothesis_embeddings_test, premises_embeddings_test = sum_embeddings(test_feature_data)
    hypothesis_embeddings_training, premises_embeddings_training = sum_embeddings(train_feature_data)

    X_train = [np.array(hypothesis_embeddings_training), np.array(premises_embeddings_training)]
    Y_train = train.label.values
    X_test = [np.array(hypothesis_embeddings_test), np.array(premises_embeddings_test)]
    Y_test = test.label.values

    model_name = "embedded_classifier"
    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title)
    model = get_prepared_embeddings_model(log_directory)

    if load_weights_from_pretraining:
        pretrain_log_directory = get_log_directory(model_name, title, True)
        load_final_weights(model, pretrain_log_directory)

    model = train_model(X_train, Y_train,
                                     model=model,
                                     log_directory=log_directory,
                                     batch_size=batch_size,
                                     epochs=40,
                                     additional_callbacks=[early_stopping],
                                     restore_checkpoint=restore_checkpoint)
    evaluate_model(model, X_test, Y_test, log_directory)
    print("done")

#TODO: fix and rewrite if needed
# def hyperparameter_search(X_train, Y_train):
#     log_directory = "logs/embedded_classifier/"
#
#     classifier = EmbeddingClassifier()
#     tuner = kt.Hyperband(classifier,
#                          objective='val_accuracy',
#                          max_epochs=50,
#                          factor=2,
#                          directory=log_directory + 'hyperparams',
#                          )
#
#     search_title = "try"
#     tensorboard_log_dir = log_directory + "tensorboard_logs/" + search_title
#     pathlib.Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
#     hist_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=tensorboard_log_dir,
#         histogram_freq=1,
#         write_images=False,
#         write_graph=True)
#     stop_callback = tf.keras.callbacks.EarlyStopping(
#         monitor='val_accuracy', patience=3)
#
#     tuner.search(X_train, Y_train, validation_split=0.2, callbacks=[hist_callback, stop_callback])
#     tuner.results_summary(5)
#     print("done")


def run_manual_feature_model(train, test, data_name, title=None):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")
    test_feature_data, train_feature_data = test_training_calculate_embeddings_and_pos_tags(test, train, data_name)
    mlp_classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(20, 20), max_iter=2000, verbose=True,
                                   early_stopping=True, n_iter_no_change=5)
    # random_forest = RandomForestClassifier()
    # logreg = LogisticRegression(max_iter=2000, tol=1e-4, verbose=2)

    pipeline_hypothesis_only = Pipeline(
        [
            ('transformer', HypothesisUnigramTransformer()),
            ('vectorizer', CountVectorizer()),
            ('classifier', mlp_classifier)
        ])

    # mlp_classifier_big = MLPClassifier(random_state=1, hidden_layer_sizes=(8, 4), max_iter=2000, verbose=True,
    #                                    early_stopping=True, n_iter_no_change=4)
    # pipeline_hypothesis_and_premise = Pipeline(
    #     [
    #         ('transformer', BothSentencesTransformer()),
    #         ('vectorizer', CountVectorizer()),
    #         ('classifier', mlp_classifier_big)
    #     ]
    # )
    pipeline = pipeline_hypothesis_only

    pipeline.fit(train_feature_data, train_feature_data['label'])
    y_pred = pipeline.predict(test_feature_data)

    model_name = "manual_feature_classifier"
    log_dir = get_log_directory(model_name, title, True)

    cnf_matrix = confusion_matrix(test_feature_data.label.values, y_pred)
    plt.figure()
    custom_plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                                 title='Confusion matrix')
    plt.savefig(log_dir + "/confusion_matrix.png")
    plt.show()

    class_report = classification_report(test_feature_data.label.values, y_pred)
    with open(log_dir + '/classification_report.txt', 'w') as file:
        file.write(class_report)
    print(class_report)
