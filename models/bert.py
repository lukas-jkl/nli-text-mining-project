import time

from transformers import BertTokenizer, TFBertModel

from models.util import *


def encode_sentences(tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence) + [tokenizer.sep_token]
    return tokenizer.convert_tokens_to_ids(tokens)


#TODO: no longer used
def prepare_bert_input(data, tokenizer):
    hypothesis_encoded = tf.ragged.constant(
        [encode_sentences(tokenizer, sentence) for sentence in data.hypothesis])
    premise_encoded = tf.ragged.constant(
        [encode_sentences(tokenizer, sentence) for sentence in data.hypothesis])
    cls = tf.ragged.constant(
        [tokenizer.convert_tokens_to_ids([tokenizer.cls_token])] * premise_encoded.shape[0])
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


def get_bert_base_model(model_name, max_len, log_directory, inputs):
    bert_model = TFBertModel.from_pretrained(model_name)
    layer_inputs = []
    for input in inputs:
        layer_inputs.append(tf.keras.Input(shape=(max_len,), dtype=tf.int32, name=input))

    bert = bert_model(layer_inputs)[0]
    output = tf.keras.layers.Dense(3, activation='softmax')(bert[:, 0, :])
    # output = tf.keras.layers.Dense(3, activation='softmax')(hidden)

    model = tf.keras.Model(
        inputs=layer_inputs,
        outputs=[output])
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file=log_directory + "/bert_model.png", show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    return model


def pretrain_bert_base_model(model_name='bert-base-cased', title=None, restore_checkpoint=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    pretrain_data = get_pretrain_data()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    prepared_data = prepare_transformer_input(pretrain_data, tokenizer)
    X_train = prepared_data.data
    for key in list(X_train.keys()):
        X_train[key] = np.array(X_train[key])
    Y_train = tf.constant(pretrain_data.label.values.astype('int32'))

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_bert_base_model(model_name, list(X_train.values())[0].shape[1], log_directory, list(X_train.keys()))
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


def run_bert_base_model(train, test, model_name='bert-base-uncased', title=None, restore_checkpoint=False,
                        load_weights_from_pretraining=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    test = test.assign(test=False)
    train = train.assign(test=True)
    data = train.append(test)
    prepared_data = prepare_transformer_input(data, tokenizer)
    X = prepared_data.data

    X_train, X_test = dict(), dict()
    for key in X.keys():
        X_train[key] = np.array(X[key][data.test.values == False])
        X_test[key] = np.array(X[key][data.test.values == True])

    Y_train = train.label.values # tf.one_hot(train.label.values, 3, axis=1)
    Y_test = test.label.values # tf.one_hot(test.label.values, 3, axis=1)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )

    log_directory = get_log_directory(model_name, title)
    model = get_bert_base_model(model_name, list(X_train.values())[0].shape[1], log_directory, list(X_train.keys()))

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
