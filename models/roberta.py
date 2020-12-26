import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModel, TFXLMRobertaModel
import tensorflow as tf

from models.util import *


def pretrain_roberta_model(model_name='bert-base-cased', title=None, restore_checkpoint=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    pretrain_data = get_pretrain_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, use_fast=True)
    max_len = 50
    X_train, Y_train = prepare_transformer_pretrain_data(pretrain_data, tokenizer, max_len)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_roberta_model(model_name, list(X_train.values())[0].shape[1], log_directory, list(X_train.keys()))
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


def get_roberta_model(model_name, max_len, log_directory, inputs):
    roberta = TFXLMRobertaModel.from_pretrained(model_name)
    layer_inputs = []
    for input in inputs:
        layer_inputs.append(tf.keras.Input(shape=(max_len,), dtype=tf.int32, name=input))

    roberta_layer = roberta(layer_inputs)[0]#[:, 0, :]
    hidden_layer = tf.keras.layers.GlobalAveragePooling1D()(roberta_layer)
    hidden_layer = tf.keras.layers.Dropout(0.25)(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(32, activation='relu')(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(16, activation='relu')(hidden_layer)
    output = tf.keras.layers.Dense(3, activation='softmax')(hidden_layer)

    model = tf.keras.Model(
        inputs=layer_inputs,
        outputs=[output])
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file=log_directory + "/roberta_model.png", show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    return model


def run_roberta(train, test, model_name="jplu/tf-xlm-roberta-base", title=None, restore_checkpoint=False,
                        load_weights_from_pretraining=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, use_fast=True)
    max_len = 50
    X_train, Y_train, X_test, Y_test = prepare_transformer_training_test_data(train, test, tokenizer, max_len)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )

    log_directory = get_log_directory(model_name, title)
    model = get_roberta_model(model_name, max_len, log_directory, list(X_train.keys()))

    if load_weights_from_pretraining:
        pretrain_log_directory = get_log_directory(model_name, title, True)
        load_final_weights(model, pretrain_log_directory)

    model = train_model(X_train, Y_train,
                        model=model,
                        log_directory=log_directory,
                        batch_size=batch_size,
                        epochs=50,
                        additional_callbacks=[early_stopping],
                        restore_checkpoint=restore_checkpoint)
    evaluate_model(model, X_test, Y_test, log_directory)
    print("done")
