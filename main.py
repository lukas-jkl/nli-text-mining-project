import time

import data
import models.random_guessing
import models.feature_based
import models.sentence_vector
import models.bert
import tensorflow as tf
import numpy as np

import models.roberta

def main():
    labeled_data, unlabeled_data = data.load_data()
    train_multilang, test_multilang = data.split_test_training(labeled_data)

    train_english, test_english = data.get_english_labled_data()
    train_translated, test_translated = data.get_translated_labled_data()

    # Run dummy classifier
    # models.random_guessing.run_dummy_model(train, test)
    # models.feature_based.run_manual_feature_model(train_translated, test_translated, "translated_english")
    # models.feature_based.pretrain_word_embedding_model(title="try2", restore_checkpoint=True)
    # models.feature_based.run_word_embedding_model(train, test)
    # models.sentence_vector.pretrain_LSTM_model(title="try3", restore_checkpoint=False)
    # models.sentence_vector.run_LSTM_model(train_translated, test_translated, "translated_english", title="translated_english_no_pretrain")

    time_date = time.strftime("%Y-%m-%d_%H_")
    title = time_date + "english_with_pretrain"
    model_name = "distilbert-base-cased"
    # models.roberta.pretrain_roberta_model(model_name=model_name, title=title)
    # models.bert.pretrain_bert_base_model(model_name=model_name, title=title)
    # models.roberta.run_roberta(train_multilang, test_multilang, model_name="jplu/tf-xlm-roberta-base", title="try2")
    # models.bert.run_bert_base_model(train_translated, test_translated, model_name=model_name, title=title)
    print("done")


if __name__ == '__main__':
    main()
