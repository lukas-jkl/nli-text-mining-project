import os
import pathlib
import zipfile

import gensim
import gensim.downloader
import kerastuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wget
from kerastuner import HyperModel
from sklearn.metrics import classification_report, confusion_matrix

from data import calculate_embeddings_and_pos_tag, test_training_calculate_embeddings_and_pos_tags
from models.util import custom_plot_confusion_matrix, get_pretrain_data


class LSTMClassifier(HyperModel):
    def __init__(self, embedding_weights, log_dir, sentence_length=300, num_outputs=3):
        super().__init__()
        self.embedding_weights = embedding_weights
        self.log_dir = log_dir
        self.output_size = num_outputs
        self.sentence_length = sentence_length

    def build(self, hp):
        # word2vec_embeddings = hub.KerasLayer("./downloaded_models/word2vec250",
        #                                            trainable=False, dtype=tf.string)

        # seq_mod = tf.keras.Sequential([
        #     word2vec_embeddings,
        #     tf.keras.layers.LSTM(128),
        #     tf.keras.layers.Dense(4)
        # ])

        # model = hub.load("./downloaded_models/word2vec250")

        lstm_units = hp.Choice("lstm_units", [32, 64, 128])
        layer_hypothesis = input_hypothesis = tf.keras.Input(dtype=tf.int32, shape=(self.sentence_length,),
                                                             name="hypothesis_input_layer")
        layer_premises = input_premises = tf.keras.Input(dtype=tf.int32, shape=(self.sentence_length,),
                                                         name="premises_input_layer")
        inputs = [input_premises, input_hypothesis]

        embedding_layer_hypothesis = tf.keras.layers.Embedding(
            weights=[self.embedding_weights],
            trainable=False,
            input_dim=self.embedding_weights.shape[0],
            output_dim=self.embedding_weights.shape[1],
            mask_zero=True
        )
        embedding_layer_premises = tf.keras.layers.Embedding(
            weights=[self.embedding_weights],
            trainable=False,
            input_dim=self.embedding_weights.shape[0],
            output_dim=self.embedding_weights.shape[1],
            mask_zero=True
        )

        layer_hypothesis = embedding_layer_hypothesis(layer_hypothesis)
        layer_premises = embedding_layer_premises(layer_premises)
        concat = tf.keras.layers.concatenate([layer_premises, layer_hypothesis], axis=1, name="concatenation_layer")

        concat_hidden_neurons = hp.Choice("concat_hidden_layer_neurons", [32, 64, 128])
        concat_hidden_layers = hp.Choice("concat_hidden_layer_num_layers", [1, 2, 3])
        concat_hidden_layers = [concat_hidden_neurons] * concat_hidden_layers
        for i, neurons in enumerate(concat_hidden_layers):
            concat = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                           name="concat_dense_hidden_layer_" + str(i))(concat)

        concat = tf.keras.layers.LSTM(lstm_units)(concat)

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
            model, to_file=self.log_dir + 'lstm_model.png', show_shapes=False, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        return model


def get_prepared_LSTM_model(embedding_weights, log_dir, max_len):
    hp = kt.HyperParameters()
    parameters = [
        hp.Fixed("lstm_units", 64),
        hp.Fixed("concat_hidden_layer_num_layers", 2),
        hp.Fixed("concat_hidden_layer_neurons", 128),
    ]

    classifier = LSTMClassifier(embedding_weights, log_dir, sentence_length=max_len)
    model = classifier.build(hp)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def convert_to_padded_string_sequence(data, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences([tf.constant(sentence) for sentence in data],
                                                         padding='post', value='', dtype=object, maxlen=maxlen)


def encode_strings_with_dict(data, vocab, maxlen):
    num_out_of_vocab = 0
    num_in_vocab = 0
    stentences_indices = []
    for sentence in data:
        sentence_indices = []
        for word in sentence:
            try:
                sentence_indices.append(tf.constant(vocab[word.strip().lower()].index + 1))  # 0 is out of vocab
                num_in_vocab += 1
            except KeyError:
                sentence_indices.append(tf.constant(0))  # 0 is out of vocab
                num_out_of_vocab += 1
        stentences_indices.append(sentence_indices)
    print(num_out_of_vocab, "/", num_in_vocab, " words out of vocabulary")
    return tf.keras.preprocessing.sequence.pad_sequences(stentences_indices,
                                                         padding='post', value=0, maxlen=maxlen)


def download_embeddings():
    EMBEDDING_DIMENSION = 200  # Available dimensions for 6B data is 50, 100, 200, 300
    data_directory = './downloaded_models/glove'

    if not os.path.isdir(data_directory):
        pathlib.Path(data_directory).mkdir(parents=True, exist_ok=True)

    glove_weights_file_path = os.path.join(data_directory, f'glove.6B.{EMBEDDING_DIMENSION}d.txt')

    if not os.path.isfile(glove_weights_file_path):
        # Glove embedding weights can be downloaded from https://nlp.stanford.edu/projects/glove/
        glove_fallback_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        local_zip_file_path = os.path.join(data_directory, os.path.basename(glove_fallback_url))
        if not os.path.isfile(local_zip_file_path):
            print(f'Retreiving glove weights from {glove_fallback_url}')
            wget.download(glove_fallback_url, local_zip_file_path)
        with zipfile.ZipFile(local_zip_file_path, 'r') as z:
            print(f'Extracting glove weights from {local_zip_file_path}')
            z.extractall(path=data_directory)

    print("done")


def pretrain_LSTM_model():
    # Prepare data
    pretrain_data = get_pretrain_data()

    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.pkl')

    max_length = max([max([len(a) for a in pretrain_feature_data.premises_words]),
                      max([len(b) for b in pretrain_feature_data.hypothesis_words])])

    glove_vectors = gensim.downloader.load('glove-twitter-100')
    embedding_weights = glove_vectors.vectors

    X_train = [
        encode_strings_with_dict(pretrain_feature_data.hypothesis_words.values, glove_vectors.vocab, max_length),
        encode_strings_with_dict(pretrain_feature_data.premises_words.values, glove_vectors.vocab, max_length)
    ]
    Y_train = tf.one_hot(pretrain_data.label.values, 3)

    # Callbacks
    title = "try1"
    log_directory = "logs/lstm_classifier/training/" + title + "/"
    tensorboard_log_dir = log_directory + "tensorboard_logs/"
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=False,
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

    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_length)
    model.fit(X_train, Y_train,
              epochs=100,
              validation_split=0.2,
              callbacks=[early_stopping, hist_callback, cp_callback],
              batch_size=128)

    final_weights_path = log_directory + "final_weights/weights"
    model.save_weights(final_weights_path)
    print("pretraining done, final weights stored to: ", final_weights_path)


def run_LSTM_model(train, test, load_weighs_from_pretraining=False):
    # download_embeddings()
    glove_vectors = gensim.downloader.load('glove-twitter-100')
    embedding_weights = glove_vectors.vectors

    print("done")
    test_feature_data, train_feature_data = test_training_calculate_embeddings_and_pos_tags(test, train)
    max_length_test = max([max([len(a) for a in test_feature_data.premises_words]),
                           max([len(b) for b in test_feature_data.hypothesis_words])])
    max_length_train = max([max([len(a) for a in train_feature_data.premises_words]),
                            max([len(b) for b in train_feature_data.hypothesis_words])])
    max_length = max([max_length_test, max_length_train])

    X_train = [
        encode_strings_with_dict(train_feature_data.hypothesis_words.values, glove_vectors.vocab, max_length),
        encode_strings_with_dict(train_feature_data.premises_words.values, glove_vectors.vocab, max_length)
    ]
    Y_train = tf.one_hot(train.label.values, 3)

    X_test = [
        encode_strings_with_dict(test_feature_data.hypothesis_words.values, glove_vectors.vocab, max_length),
        encode_strings_with_dict(test_feature_data.premises_words.values, glove_vectors.vocab, max_length)
    ]
    Y_test = tf.one_hot(test.label.values, 3)

    title = "try1"
    log_directory = "logs/lstm_classifier/training/" + title + "/"
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_length)
    if load_weighs_from_pretraining:
        model.load_weights("logs/lstm_classifier/pretraining/try1/final_weights/weights")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(X_train, Y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=128)
    print(model.evaluate(X_test, Y_test))
    Y_pred = model.predict(X_test)
    cnf_matrix = confusion_matrix(np.argmax(np.array(Y_test), 1), np.argmax(Y_pred, 1))
    plt.figure()
    custom_plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                                 title='Confusion matrix')
    plt.show()
    print(classification_report(np.argmax(np.array(Y_test), 1), np.argmax(Y_pred, 1)))

    print("done")
