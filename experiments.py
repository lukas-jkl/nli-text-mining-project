import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import data
import configurations
import models.feature_based
import models.roberta


def run_bert_maxlen_maxpool_experiments():
    title_prefix = "maxlen_maxpool_experiment"
    data_name, train_english, test_english = data.get_english_labeled_data()
    model_name = "bert-base-cased"

    max_pool = False
    max_lens = [10, 30, 50, 70, 100]
    max_epochs = 8
    for max_len in max_lens:
        configurations.run_bert_config(train_english, test_english,
                                       model_name=model_name,
                                       max_pool=max_pool,
                                       max_len=max_len,
                                       pretrain_snli=False,
                                       data_name=data_name,
                                       title_prefix=title_prefix,
                                       max_epochs=max_epochs)

    max_pool = True
    for max_len in max_lens:
        configurations.run_bert_config(train_english, test_english,
                                       model_name=model_name,
                                       max_pool=max_pool,
                                       max_len=max_len,
                                       pretrain_snli=False,
                                       data_name=data_name,
                                       title_prefix=title_prefix,
                                       max_epochs=max_epochs)


def run_bert_english_model_experiments():
    title_prefix = "bert_experiments"

    data_name, train_english, test_english = data.get_translated_labeled_data()
    max_epochs = 40
    max_len = 50

    for model_name in ["distilbert-base-cased", "bert-base-cased", "bert-large-cased"]:
        for pretrain in [True, False]:
            for max_pool in [True, False]:
                configurations.run_bert_config(train_english, test_english,
                                               model_name=model_name,
                                               max_pool=max_pool,
                                               max_len=max_len,
                                               pretrain_snli=pretrain,
                                               data_name=data_name,
                                               title_prefix=title_prefix,
                                               max_epochs=max_epochs)


def run_multi_lingual_model_experiments():
    title_prefix = "multilang_experiments"

    data_name, train_multilang, test_multilang = data.get_multilang_labeled_data()
    max_epochs = 40
    max_len = 50

    for model_name in ["bert-base-multilingual-cased"]:
        for max_pool in [True, False]:
            configurations.run_bert_config(train_multilang, test_multilang,
                                           model_name=model_name,
                                           max_pool=max_pool,
                                           max_len=max_len,
                                           pretrain_snli=False,
                                           data_name=data_name,
                                           title_prefix=title_prefix,
                                           max_epochs=max_epochs)

    for model_name in ["jplu/tf-xlm-roberta-base"]:
        for max_pool in [True, False]:
            configurations.run_roberta_config(train_multilang, test_multilang,
                                              model_name=model_name,
                                              max_pool=max_pool,
                                              max_len=max_len,
                                              pretrain_snli=False,
                                              data_name=data_name,
                                              title_prefix=title_prefix,
                                              max_epochs=max_epochs)


def run_embedding_model_experiments():
    time_date = time.strftime("%Y-%m-%d_%H_")
    title = time_date + "translated_with_pretrain"
    data_name, train_translated, test_translated = data.get_translated_labeled_data()

    print("Training with pretrain")
    models.feature_based.pretrain_word_embedding_model(title=title)
    models.feature_based.run_word_embedding_model(train_translated, test_translated, data_name, title=title,
                                                  load_weights_from_pretraining=True)

    print("Training without pretrain")
    title = time_date + "translated_no_pretrain"
    models.feature_based.run_word_embedding_model(train_translated, test_translated, data_name, title=title,
                                                  load_weights_from_pretraining=False)


def test_models_translated_non_translated_data():
    LOG_DIR = "D:/text_mining_project_logs"
    _, _, test_translated = data.get_translated_labeled_data()
    _, _, test_multilang = data.get_multilang_labeled_data()

    log_dir_roberta_base = LOG_DIR + "/" + "roberta-base" + "/training/" + "roberta_final_training" + "/"
    log_dir_roberta_distilled = LOG_DIR + "/" + "distilroberta-base" + "/training/" + "2021-01-02_08_roberta_final_training" + "/"
    log_dir_ml_roberta = LOG_DIR + "/" + "tf-xlm-roberta-base" + "/training/" + "roberta_final_training" + "/"

    # English roberta models
    for model_name, log_dir, eval_data in zip(["jplu/tf-xlm-roberta-base", "roberta-base", "distilroberta-base"],
                                              [log_dir_ml_roberta, log_dir_roberta_distilled, log_dir_roberta_base],
                                              [test_multilang, test_translated, test_translated, ]):
        predictions = models.roberta.evaluate_roberta_model(eval_data,
                                                            model_name=model_name,
                                                            max_len=100,
                                                            log_directory=log_dir,
                                                            dropout=None,
                                                            max_pool=False)

        y_true_en = eval_data[eval_data.lang_abv == "en"].label.values
        y_pred_en = np.argmax(predictions, 1)[eval_data.lang_abv == "en"]
        class_report_en = classification_report(y_true_en, y_pred_en)
        with open(log_dir + '/classification_report_en.txt', 'w') as file:
            file.write(class_report_en)

        y_true_translated = eval_data[eval_data.lang_abv != "en"].label.values
        y_pred_translated = np.argmax(predictions, 1)[eval_data.lang_abv != "en"]
        class_report_translated = classification_report(y_true_translated, y_pred_translated)
        with open(log_dir + '/classification_report_non_en.txt', 'w') as file:
            file.write(class_report_translated)

    print("done")
