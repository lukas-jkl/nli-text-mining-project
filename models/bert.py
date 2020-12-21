import pathlib

from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.util import custom_plot_confusion_matrix, evaluate_model


def encode_sentences(tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence) + ['[SEP]']
    return tokenizer.convert_tokens_to_ids(tokens)


def load_model():
    # model_name = 'bert-base-multilingual-cased'
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)


def prepare_bert_input(data, tokenizer):
    hypothesis_encoded = tf.ragged.constant(
        [encode_sentences(tokenizer, sentence) for sentence in data.hypothesis])
    premise_encoded = tf.ragged.constant(
        [encode_sentences(tokenizer, sentence) for sentence in data.hypothesis])
    cls = tf.ragged.constant(
        [tokenizer.convert_tokens_to_ids(['[CLS]'])] * premise_encoded.shape[0])
    input_ids = tf.concat([cls, hypothesis_encoded, premise_encoded], axis=1)

    input_mask = tf.ones_like(input_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_hypothesis = tf.zeros_like(hypothesis_encoded)
    type_premise = tf.ones_like(premise_encoded)
    input_type_ids = tf.concat(
        [type_cls, type_hypothesis, type_premise], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }
    return inputs


def get_bert_base_model(model_name, log_dir):
    bert_model = TFBertModel.from_pretrained(model_name)
    max_len = 111
    input_word_ids = tf.keras.Input(shape=(max_len,) , dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,) , dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,) , dtype=tf.int32, name="input_type_ids")

    bert = bert_model([input_word_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation='softmax')(bert[:,0,:])

    model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids],
        outputs=[output])
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file=log_dir + 'bert_base_model.png', show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    return model


def run_bert_base_model(train, test):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    X_train = prepare_bert_input(train, tokenizer)
    Y_train = tf.one_hot(train.label.values, 3, axis=1)

    X_test = prepare_bert_input(test, tokenizer)
    Y_test = tf.one_hot(test.label.values, 3, axis=1)

    max_length = max([
        max([a.shape[0] for a in X_train['input_word_ids']]),
        max([a.shape[0] for a in X_test['input_word_ids']])
    ])

    # pad to same length
    X_train['input_word_ids'] = X_train['input_word_ids'].to_tensor(default_value=0, shape=[None, max_length])
    X_test['input_word_ids'] = X_test['input_word_ids'].to_tensor(default_value=0, shape=[None, max_length])

    # Callbacks
    title = "try1"
    log_directory = "logs/bert_base_classifier/training/" + title + "/"
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

    model = get_bert_base_model(model_name, log_directory)
    model.fit(x=X_train, y=Y_train,
              epochs=10,
              verbose=1,
              validation_split=0.2,
              callbacks=[early_stopping, hist_callback, cp_callback],
              batch_size=batch_size,
              )

    evaluate_model(model, X_test, Y_test)
    print("done")
