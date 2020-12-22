import pathlib

from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.util import *


def encode_sentences(tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence) + ['[SEP]']
    return tokenizer.convert_tokens_to_ids(tokens)


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
        'input_word_ids': input_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }
    return inputs


def get_bert_base_model(model_name, max_len):
    bert_model = TFBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    bert = bert_model([input_word_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation='softmax')(bert[:, 0, :])

    model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids],
        outputs=[output])
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file="logs/" + model_name + "/model", show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    return model


def pretrain_bert_base_model(model_name='bert-base-uncased', title="try1", restore_checkpoint=False):
    pretrain_data = get_pretrain_data()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    X_train = prepare_bert_input(pretrain_data, tokenizer)
    Y_train = tf.one_hot(pretrain_data.label.values, 3, axis=1)

    # Callbacks
    log_directory = "logs/" + model_name + "/pretraining/" + title + "/"
    batch_size = 32
    hist_callback, cp_callback = prepare_log_callbacks(batch_size, log_directory)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
    callbacks = [hist_callback, cp_callback, early_stopping]

    if restore_checkpoint:
        model = tf.keras.models.load_model(cp_callback.filepath)
    else:
        model = get_bert_base_model(model_name, list(X_train.values())[0].shape[1])

    model.fit(x=X_train, y=Y_train,
              epochs=50,
              verbose=1,
              validation_split=0.2,
              callbacks=callbacks,
              batch_size=batch_size,
              )

    final_weights_path = log_directory + "final_weights/weights"
    model.save_weights(final_weights_path)
    print("pretraining done, final weights stored to: ", final_weights_path)
    print("done")
    return final_weights_path


def run_bert_base_model(train, test, model_name='bert-base-uncased', title="try1", restore_checkpoint=False, load_weighs_from_pretraining=False):
    tokenizer = BertTokenizer.from_pretrained(model_name)

    test = test.assign(test=True)
    train = train.assign(test=False)
    data = train.append(test)
    X = prepare_bert_input(data, tokenizer)

    X_train, X_test = dict(), dict()
    for key in X.keys():
        X_train[key] = X[key][data.test.values == False]
        X_test[key] = X[key][data.test.values == True]

    Y_train = tf.one_hot(train.label.values, 3, axis=1)
    Y_test = tf.one_hot(test.label.values, 3, axis=1)

    # Callbacks
    log_directory = "logs/" + model_name + "/training/" + title + "/"
    batch_size = 32
    hist_callback, cp_callback = prepare_log_callbacks(batch_size, log_directory)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
    callbacks = [hist_callback, cp_callback, early_stopping]

    model = get_bert_base_model(model_name, list(X_train.values())[0].shape[1])

    if restore_checkpoint:
        model = tf.keras.models.load_model(cp_callback.filepath)
    if load_weighs_from_pretraining:
        pretrain_log_directory = "logs/" + model_name + "/pretraining/" + title + "/" + "final_weights/weights"
        model.load_weights(pretrain_log_directory)

    model.fit(x=X_train, y=Y_train,
              epochs=50,
              verbose=1,
              validation_split=0.2,
              callbacks=callbacks,
              batch_size=batch_size,
              )

    evaluate_model(model, X_test, Y_test)
    print("done")
