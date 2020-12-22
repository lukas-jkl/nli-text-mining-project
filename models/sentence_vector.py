import os
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
        lstm_dropout = hp.Choice("lstm_dropout", [0, 0.01, 0.05, 0.1])
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


def pretrain_LSTM_model(title="try1", restore_checkpoint=False):
    # Prepare data
    pretrain_data = get_pretrain_data()

    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.feather')

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
    model_name = "lstm_classifier"
    log_directory = "logs/" + model_name + "/pretraining/" + title + "/"
    batch_size = 64
    hist_callback, cp_callback = prepare_log_callbacks(batch_size, log_directory)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    callbacks = [hist_callback, cp_callback, early_stopping]

    if restore_checkpoint:
        model = tf.keras.models.load_model(cp_callback.filepath)
    else:
        model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_length)
    model.fit(X_train, Y_train,
              epochs=100,
              validation_split=0.2,
              callbacks=callbacks,
              batch_size=128)

    final_weights_path = log_directory + "final_weights/weights"
    model.save_weights(final_weights_path)
    print("pretraining done, final weights stored to: ", final_weights_path)
    print("done")
    return final_weights_path


def run_LSTM_model(train, test, title="try1", restore_checkpoint=False, load_weighs_from_pretraining=False):
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

    # Callbacks
    model_name = "lstm_classifier"
    log_directory = "logs/" + model_name + "/training/" + title + "/"
    batch_size = 64
    hist_callback, cp_callback = prepare_log_callbacks(batch_size, log_directory)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    callbacks = [hist_callback, cp_callback, early_stopping]

    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_length)

    if restore_checkpoint:
        model = tf.keras.models.load_model(cp_callback.filepath)
    if load_weighs_from_pretraining:
        pretrain_log_directory = "logs/" + model_name + "/pretraining/" + title + "/" + "final_weights/weights"
        model.load_weights(pretrain_log_directory)

    model.fit(X_train, Y_train,
              epochs=100,
              verbose=1,
              validation_split=0.2,
              callbacks=callbacks,
              batch_size=128)
    evaluate_model(model, X_test, Y_test)
    print("done")
