import os
import time
import zipfile
import pickle as pkl
import gensim
import gensim.downloader
import kerastuner as kt
import wget
from kerastuner import HyperModel

from data import calculate_embeddings_and_pos_tag, test_training_calculate_embeddings_and_pos_tags
from models.util import *


class LSTMClassifier(HyperModel):
    def __init__(self, embedding_weights, log_dir, sentence_length, num_outputs=3):
        super().__init__()
        self.embedding_weights = embedding_weights
        self.log_dir = log_dir
        self.output_size = num_outputs
        self.sentence_length = sentence_length

    def build(self, hp):
        separate_hidden_layer_neurons = hp.Choice("separate_hidden_layer_neurons", [32, 64, 128])
        separate_hidden_layer_num_layers = hp.Choice("separate_hidden_layer_num_layers", [0, 1, 2])
        concat_hidden_neurons = hp.Choice("concat_hidden_layer_neurons", [64, 128])
        concat_hidden_layers = hp.Choice("concat_hidden_layer_num_layers", [0, 1])
        lstm_units = hp.Choice("lstm_units", [32, 64, 128])
        lstm_dropout = hp.Choice("lstm_dropout", [0.0, 0.01, 0.1])
        final_hidden_layer_num_layers = hp.Choice("final_hidden_layer_num_layers", [1, 2])
        final_hidden_layer_neurons = hp.Choice("final_hidden_layer_neurons", [32, 64, 128])

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
        separate_hidden_layers = [separate_hidden_layer_neurons] * separate_hidden_layer_num_layers
        for i, neurons in enumerate(separate_hidden_layers):
            layer_hypothesis = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                                     name="hypothesis_dense_hidden_layer_" + str(i))(layer_hypothesis)
            layer_premises = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                                   name="premise_dense_hidden_layer_" + str(i))(layer_premises)

        concat = tf.keras.layers.concatenate([layer_premises, layer_hypothesis], axis=1, name="concatenation_layer")

        concat_hidden_layers = [concat_hidden_neurons] * concat_hidden_layers
        for i, neurons in enumerate(concat_hidden_layers):
            concat = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                           name="concat_dense_hidden_layer_" + str(i))(concat)

        concat = tf.keras.layers.LSTM(lstm_units,
                                      activation='tanh',
                                      dropout=lstm_dropout)(concat)

        final_hidden_layers = [final_hidden_layer_neurons] * final_hidden_layer_num_layers
        for i, neurons in enumerate(final_hidden_layers):
            concat = tf.keras.layers.Dense(concat_hidden_neurons, activation='relu',
                                           name="final_dense_hidden_layer_" + str(i))(concat)

        final_dense_neurons = self.output_size
        output = tf.keras.layers.Dense(final_dense_neurons, activation='softmax',
                                       name="output_layer")(concat)

        model = tf.keras.Model(inputs=inputs,
                               outputs=[output],
                               name=self.name)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        with open(self.log_dir + 'hyperparameters.txt', 'w') as f:
            f.write(json.dumps(hp.values))

        tf.keras.utils.plot_model(
            model, to_file=self.log_dir + 'lstm_model.png', show_shapes=False, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        return model


def get_prepared_LSTM_model(embedding_weights, log_dir, max_len):
    hp = kt.HyperParameters()
    parameters = [
        hp.Fixed("lstm_units", 65),
        hp.Fixed("lstm_dropout", 0.1),

        hp.Fixed("concat_hidden_layer_num_layers", 0),
        hp.Fixed("concat_hidden_layer_neurons", 128),

        hp.Fixed("separate_hidden_layer_num_layers", 2),
        hp.Fixed("separate_hidden_layer_neurons", 32),

        hp.Fixed("final_hidden_layer_num_layers", 1),
        hp.Fixed("final_hidden_layer_neurons", 128),
    ]

    classifier = LSTMClassifier(embedding_weights, log_dir, sentence_length=max_len)
    model = classifier.build(hp)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
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


def hyperparameter_search(title=None, gensim_embeddings="glove-twitter-100", max_len=50):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    # Prepare data - use pretraining data for hyperparameter tuning
    pretrain_data = get_pretrain_data()
    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.feather')
    vocab, embedding_weights = load_and_cache_embeddings(gensim_embeddings)
    X_train = [
        encode_strings_with_dict(pretrain_feature_data.hypothesis_words.values, vocab, max_len),
        encode_strings_with_dict(pretrain_feature_data.premises_words.values, vocab, max_len)
    ]
    Y_train = np.array(pretrain_data.label.values, dtype='int32')

    model_name = "lstm_classifier"

    log_directory = get_log_directory(model_name, title=title, pretraining=True)
    vocab, embedding_weights = load_and_cache_embeddings(gensim_embeddings)
    classifier = LSTMClassifier(embedding_weights=embedding_weights, log_dir=log_directory, sentence_length=max_len)

    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_directory,
        histogram_freq=1,
        write_images=False,
        write_graph=True,
        profile_batch=2)

    tuner = kt.Hyperband(classifier,
                         objective='val_accuracy',
                         max_epochs=30,
                         directory=log_directory,
                         project_name="lstm_tuning"
                         )
    tuner.search(X_train, Y_train, validation_split=0.2, callbacks=[hist_callback, stop_callback])
    tuner.results_summary(10)
    print("done")


def pretrain_LSTM_model(title=None, restore_checkpoint=False, gensim_embeddings="glove-twitter-100", max_len=50):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    # Prepare data
    pretrain_data = get_pretrain_data()

    pretrain_feature_data = calculate_embeddings_and_pos_tag(pretrain_data, './cache/pretrain_features.feather')

    # max_length = max([max([len(a) for a in pretrain_feature_data.premises_words]),
    #                   max([len(b) for b in pretrain_feature_data.hypothesis_words])])

    print("loading embedding vectors...")
    vocab, embedding_weights = load_and_cache_embeddings(gensim_embeddings)
    print("done")

    X_train = [
        encode_strings_with_dict(pretrain_feature_data.hypothesis_words.values, vocab, max_len),
        encode_strings_with_dict(pretrain_feature_data.premises_words.values, vocab, max_len)
    ]
    Y_train = np.array(pretrain_data.label.values, dtype='int32')

    model_name = "lstm_classifier"
    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_len)
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


def load_and_cache_embeddings(embedding_vectors):
    vocab_file = './cache/embeddings_' + embedding_vectors + '_vocab.pkl'
    vector_file = './cache/embeddings_' + embedding_vectors + '_vectors.pkl'

    try:
        with open(vocab_file, 'rb') as f:
            vocab = pkl.load(f)
        with open(vector_file, 'rb') as f:
            vector = pkl.load(f)
    except:
        glove_vectors = gensim.downloader.load(embedding_vectors)
        vocab = glove_vectors.vocab
        vector = glove_vectors.vectors
        with open(vocab_file, 'wb') as f:
            pkl.dump(vocab, f)
        with open(vector_file, 'wb') as f:
            pkl.dump(vector, f)
    return vocab, vector


def run_LSTM_model(train, test, data_name, title=None, restore_checkpoint=False, load_weights_from_pretraining=False,
                   max_len=50,
                   gensim_embeddings="glove-twitter-100"):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    print("loading embedding vectors")
    vocab, embedding_weights = load_and_cache_embeddings(gensim_embeddings)
    print("done")

    test_feature_data, train_feature_data = test_training_calculate_embeddings_and_pos_tags(test, train, data_name)
    # max_length_test = max([max([len(a) for a in test_feature_data.premises_words]),
    #                        max([len(b) for b in test_feature_data.hypothesis_words])])
    # max_length_train = max([max([len(a) for a in train_feature_data.premises_words]),
    #                         max([len(b) for b in train_feature_data.hypothesis_words])])
    # max_len = max([max_length_test, max_length_train])

    X_train = [
        encode_strings_with_dict(train_feature_data.hypothesis_words.values, vocab, max_len),
        encode_strings_with_dict(train_feature_data.premises_words.values, vocab, max_len)
    ]
    Y_train = train.label.values

    X_test = [
        encode_strings_with_dict(test_feature_data.hypothesis_words.values, vocab, max_len),
        encode_strings_with_dict(test_feature_data.premises_words.values, vocab, max_len)
    ]
    Y_test = test.label.values

    batch_size = 32
    model_name = "lstm_classifier"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title)
    model = get_prepared_LSTM_model(embedding_weights, log_directory, max_len=max_len)

    if load_weights_from_pretraining:
        pretrain_log_directory = get_log_directory(model_name, title, True)
        load_final_weights(model, pretrain_log_directory)

    model = train_model(X_train, Y_train,
                        model=model,
                        log_directory=log_directory,
                        batch_size=batch_size,
                        epochs=100,
                        additional_callbacks=[early_stopping],
                        restore_checkpoint=restore_checkpoint)
    evaluate_model(model, X_test, Y_test, log_directory)
    print("done")
