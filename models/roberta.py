import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModel, TFXLMRobertaModel, \
    TFRobertaModel, TFAutoModelForSequenceClassification
import tensorflow as tf

from models.util import *
import tensorflow_datasets as tfds


def full_pretrain_roberta_model(model_name="roberta-base", max_len=50, max_pool=False, title=None,
                                restore_checkpoint=False, dropout=None):
    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True
    )

    pretrain_data = get_pretrain_data(number_samples=None)  # get all samples
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, use_fast=True)
    X_train, Y_train = prepare_transformer_pretrain_data(pretrain_data, tokenizer, max_len)

    # Pretrain snli
    ds = tfds.load('multi_nli', split='train', shuffle_files=True)  # .take(1000)
    labels = []
    input_ids = []
    attention_masks = []
    i = 0
    for element in ds.as_numpy_iterator():
        i += 1
        if i % 1000 == 0:
            print(i)
        prepared_data = tokenizer(str(element['hypothesis']), str(element['premise']),
                                  max_length=max_len, truncation=True, padding="max_length").data
        input_ids.append(prepared_data['input_ids'])
        attention_masks.append(prepared_data['attention_mask'])
        labels.append(element['label'])

    Y_train = tf.concat([Y_train, labels], axis=0)
    X_train['input_ids'] = tf.concat([X_train['input_ids'], input_ids], axis=0)
    X_train['attention_mask'] = tf.concat([X_train['attention_mask'], attention_masks], axis=0)

    log_directory = get_log_directory(model_name, title, True)
    model = get_roberta_model(model_name,
                              max_len, log_directory, list(X_train.keys()), max_pool, dropout)

    model = train_model(X_train, Y_train,
                        model=model,
                        log_directory=log_directory,
                        batch_size=batch_size,
                        epochs=40,
                        additional_callbacks=[early_stopping],
                        restore_checkpoint=restore_checkpoint)

    final_weights_path = save_final_weights(model, log_directory)


def pretrain_roberta_model(model_name="jplu/tf-xlm-roberta-base", max_len=50, max_pool=False, title=None,
                           restore_checkpoint=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    pretrain_data = get_pretrain_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, use_fast=True)
    X_train, Y_train = prepare_transformer_pretrain_data(pretrain_data, tokenizer, max_len)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_roberta_model(model_name,
                              max_len, log_directory, list(X_train.keys()), max_pool)
    model = train_model(X_train, Y_train,
                        model=model,
                        log_directory=log_directory,
                        batch_size=batch_size,
                        epochs=40,
                        additional_callbacks=[early_stopping],
                        restore_checkpoint=restore_checkpoint)

    final_weights_path = save_final_weights(model, log_directory)
    print("done")
    return final_weights_path


def get_roberta_model(model_name, max_len, log_directory, inputs, max_pool, dropout=None):
    if "xlm" in model_name:
       roberta_model = TFXLMRobertaModel.from_pretrained(model_name)
    else:
       roberta_model = TFRobertaModel.from_pretrained(model_name)
    layer_inputs = []

    for input in inputs:
       layer_inputs.append(tf.keras.Input(shape=(max_len,), dtype=tf.int32, name=input))

    roberta_layer = roberta_model(layer_inputs)[0]
    if not max_pool:
       roberta_layer = roberta_layer[:, 0, :]
       if dropout:
           roberta_layer = tf.keras.layers.Dropout(roberta_layer)
       output = tf.keras.layers.Dense(3, activation='softmax')(roberta_layer)
    else:
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


def inference(test, model_name, max_len, title, dropout=None, max_pool=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, use_fast=True)
    X = encode_transformer_input(test, tokenizer, max_len).data
    X_train, X_test = dict(), dict()
    for key in X.keys():
        X_test[key] = np.array(X[key])

    log_directory = get_log_directory(model_name, title)
    model = get_roberta_model(model_name, max_len, log_directory, list(X_test.keys()), max_pool=max_pool,
                              dropout=dropout)
    model.load_weights(log_directory + "final_model/weights")
    Y_pred = model.predict(X_test)
    print(model.evaluate(X_test, test.label.values))
    print("done")


def run_roberta_model(train, test, model_name="jplu/tf-xlm-roberta-base", max_len=50, max_pool=False, title=None,
                      restore_checkpoint=False, load_weights_from_pretraining=False, max_epochs=40, dropout=None,
                      early_stopping=True):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, use_fast=True)
    X_train, Y_train, X_test, Y_test = prepare_transformer_training_test_data(train, test, tokenizer, max_len)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )

    log_directory = get_log_directory(model_name, title)
    model = get_roberta_model(model_name, max_len, log_directory, list(X_train.keys()), max_pool=max_pool,
                              dropout=dropout)

    if load_weights_from_pretraining:
        pretrain_log_directory = get_log_directory(model_name, title, True)
        load_final_weights(model, pretrain_log_directory)

    if early_stopping:
        callbacks = [early_stopping]
    else:
        callbacks = []

    model = train_model(X_train, Y_train,
                        model=model,
                        log_directory=log_directory,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        additional_callbacks=callbacks,
                        restore_checkpoint=restore_checkpoint)
    evaluate_model(model, X_test, Y_test, log_directory)
    print("done")
