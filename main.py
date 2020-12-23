import time

import data
import models.random_guessing
import models.feature_based
import models.sentence_vector
import models.bert
import tensorflow as tf
import numpy as np

from models.roberta import run_roberta


def main():
    labeled_data, unlabeled_data = data.load_data()
    train, test = get_english_labled_data(labeled_data)

    translated_data = data.translate_data(labeled_data, "./cache/labeled_data_translated.pkl")
    translated_train, translated_test = split_test_training(translated_data)
    # Run dummy classifier
    # models.random_guessing.run_dummy_model(train, test)
    # models.feature_based.run_manual_feature_model(translated_train, translated_test, "translated_english")
    # models.feature_based.pretrain_word_embedding_model(title="try2", restore_checkpoint=True)
    # models.feature_based.run_word_embedding_model(train, test)
    # models.sentence_vector.pretrain_LSTM_model(title="try3", restore_checkpoint=False)
    models.sentence_vector.run_LSTM_model(translated_train, translated_test, "translated_english", title="translated_english_no_pretrain")
    # models.bert.pretrain_bert_base_model(title="some22")

    time_date = time.strftime("%Y%m%d-%H%M%S")
    # models.bert.run_bert_base_model(train, test, model_name="distilbert-base-uncased", title=time_date + "_no_pretrain")
    print("done")



def get_english_labled_data(data):
    english_data = data[data.lang_abv == "en"]
    return split_test_training(data)

def split_test_training(data):
    train, test = \
        np.split(data.sample(frac=1, random_state=42),
                 [int(.85 * len(data))])
    return train, test

if __name__ == '__main__':
    main()
