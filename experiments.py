import data
import configurations


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
