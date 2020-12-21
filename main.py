import data
import models.random_guessing
import models.feature_based
import models.sentence_vector
import models.bert
import tensorflow as tf
import numpy as np

def main():
    labeled_data, unlabeled_data = data.load_data()
    train, test = get_english_labled_data(labeled_data)

    # Run dummy classifier
    # models.random_guessing.run_dummy_model(train, test)
    # models.feature_based.run_manual_feature_model(train, test)
    # models.feature_based.pretrain_word_embedding_model()
    # models.feature_based.run_word_embedding_model(train, test)
    # models.sentence_vector.run_LSTM_model(train, test)
    # models.sentence_vector.pretrain_LSTM_model()
    models.bert.run_bert_base_model(train, test)
    print("done")



def get_english_labled_data(data):
    english_data = data[data.lang_abv == "en"]
    train, test = \
        np.split(english_data.sample(frac=1, random_state=42),
                 [int(.85 * len(english_data))])
    return train, test


if __name__ == '__main__':
    main()
