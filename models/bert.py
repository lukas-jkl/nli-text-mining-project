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


def get_bert_base_model(model_name, max_len, log_directory, inputs, max_pool):
    bert_model = TFBertModel.from_pretrained(model_name)
    layer_inputs = []
    for input in inputs:
        layer_inputs.append(tf.keras.Input(shape=(max_len,), dtype=tf.int32, name=input))

    bert_layer = bert_model(layer_inputs)[0]
    if not max_pool:
        output = tf.keras.layers.Dense(3, activation='softmax')(bert_layer[:, 0, :])
    else:
        hidden_layer = tf.keras.layers.GlobalAveragePooling1D()(bert_layer)
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
        model, to_file=log_directory + "/bert_model.png", show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    return model


def pretrain_bert_base_model(model_name='bert-base-cased', max_pool=False, title=None, restore_checkpoint=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    pretrain_data = get_pretrain_data()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    max_len = 50
    X_train, Y_train = prepare_transformer_pretrain_data(pretrain_data, tokenizer, max_len)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )
    log_directory = get_log_directory(model_name, title, True)
    model = get_bert_base_model(model_name,
                                max_len, log_directory, list(X_train.keys()), max_pool)
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


def run_bert_base_model(train, test, model_name='bert-base-uncased', max_pool=False, title=None, restore_checkpoint=False,
                        load_weights_from_pretraining=False):
    if title is None:
        title = time.strftime("%Y%m%d-%H%M%S")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    max_len = 50
    X_train, Y_train, X_test, Y_test = prepare_transformer_training_test_data(train, test, tokenizer, max_len)

    batch_size = 32
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )

    log_directory = get_log_directory(model_name, title)
    model = get_bert_base_model(model_name, max_len, log_directory, list(X_train.keys()), max_pool)

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
