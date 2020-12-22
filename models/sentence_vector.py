import os
import time
import zipfile

import gensim
import gensim.downloader
import kerastuner as kt
import wget
from kerastuner import HyperModel

from data import calculate_embeddings_and_pos_tag, test_training_calculate_embeddings_and_pos_tags
from models.util import *


class LSTMClassifier(HyperModel):
    def __init__(self, embedding_weights, log_dir, sentence_length=300, num_outputs=3):
        super().__init__()
        self.embedding_weights = embedding_weights
        self.log_dir = log_dir
        self.output_size = num_outputs
        self.sentence_length = sentence_length

    def build(self, hp):
        lstm_units = hp.Choice("lstm_units", [32, 64, 128])
        lstm_dropout = hp.Choice("lstm_dropout", [0.0, 0.01, 0.05, 0.1])
        concat_hidden_neurons = hp.Choice("concat_hidden_layer_neurons", [32, 64, 128])
        concat_hidden_layers = hp.Choice("concat_hidden_layer_num_layers", [1, 2, 3])

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

        concat_hidden_layers = [concat_hidden_neurons] * concat_hidden_layers
        for i, neurons in enumerate(concat_hidden_layers):
            concat = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                           name="concat_dense_hidden_layer_" + str(i))(concat)

        concat = tf.keras.layers.LSTM(lstm_units,
                                      activation='tanh',
                                      dropout=lstm_dropout)(concat)

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
        hp.Fixed("lstm_dropout", 0.05),
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


def pretrain_LSTM_model(title=None, restore_checkpoint=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    # Prepare data
    pretrain_data = get_pretrain_data()

    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.feather')

    max_length = max([max([len(a) for a in pretrain_feature_data.premises_words]),
                      max([len(b) for b in pretrain_feature_data.hypothesis_words])])

    print("loading embedding vectors")
    glove_vectors = gensim.downloader.load('glove-twitter-100')
    embedding_weights = glove_vectors.vectors
    print("done")

    X_train = [
        encode_strings_with_dict(pretrain_feature_data.hypothesis_words.values, glove_vectors.vocab, max_length),
        encode_strings_with_dict(pretrain_feature_data.premises_words.values, glove_vectors.vocab, max_length)
    ]
    Y_train = tf.one_hot(pretrain_data.label.values, 3)

    model_name = "lstm_classifier"
    batch_size = 64
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_length)
    model = train_model(X_train, Y_train,
                                     model=model,
                                     log_directory=log_directory,
                                     batch_size=batch_size,
                                     epochs=100,
                                     additional_callbacks=[early_stopping],
                                     restore_checkpoint=restore_checkpoint)

    final_weights_path = save_final_weights(model, log_directory)
    print("done")
    return final_weights_path


def run_LSTM_model(train, test, title=None, restore_checkpoint=False, load_weights_from_pretraining=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    print("loading embedding vectors")
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

    batch_size = 32
    model_name = "lstm_classifier"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title)
    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_length)

    if load_weights_from_pretraining:
        pretrain_log_directory = get_log_directory(model_name, title, True)
        model.load_weights(pretrain_log_directory)

    model = train_model(X_train, Y_train,
                                     model=model,
                                     log_directory=log_directory,
                                     batch_size=batch_size,
                                     epochs=100,
                                     additional_callbacks=[early_stopping],
                                     restore_checkpoint=restore_checkpoint)
    evaluate_model(model, X_test, Y_test, log_directory)
    print("done")
