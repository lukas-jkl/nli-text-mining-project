import data
import configurations


def run_bert_maxlen_maxpool_experiments():
    title_prefix = "maxlen_maxpool_experiment"
    data_name, train_english, test_english = data.get_english_labeled_data()
    model_name = "distilbert-base-cased"

    max_pool = False
    for max_len in [20, 50, 70]:
        configurations.run_bert_config(train_english, test_english,
                                       model_name=model_name,
                                       max_pool=max_pool,
                                       max_len=max_len,
                                       pretrain_snli=False,
                                       data_name=data_name,
                                       title_prefix=title_prefix,
                                       max_epochs=4)

    max_pool=True
    for max_len in [20, 50, 70]:
        configurations.run_bert_config(train_english, test_english,
                                       model_name=model_name,
                                       max_pool=max_pool,
                                       max_len=max_len,
                                       pretrain_snli=False,
                                       data_name=data_name,
                                       title_prefix=title_prefix,
                                       max_epochs=4)
