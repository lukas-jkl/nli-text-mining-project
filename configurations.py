import time

import data
import models.random_guessing
import models.feature_based
import models.sentence_vector
import models.bert
import models.roberta


def get_config_title(model_name, max_pool, max_len, pretrain_snli, data_name, title_prefix):
    time_date = time.strftime("%Y-%m-%d_%H_")
    title = time_date
    if title_prefix:
        title += title_prefix + "_"
    title += model_name + "_"
    if max_pool:
        title += "maxpool_"

    title += data_name + "_"
    title += "max-len_" + str(max_len) + "_"

    if pretrain_snli:
        title += "with-snli-pretrain"
    else:
        title += "without-pretrain"
    return title


def run_bert_config(train, test, model_name, max_pool, max_len, pretrain_snli, data_name, title_prefix=None, max_epochs=40):
    title = get_config_title(model_name, max_pool, max_len, pretrain_snli, data_name, title_prefix)

    if pretrain_snli:
        models.bert.pretrain_bert_model(model_name=model_name,
                                        max_len=max_len,
                                        title=title,
                                        max_pool=max_pool,
                                        restore_checkpoint=False)
        models.bert.run_bert_model(train, test,
                                   model_name=model_name,
                                   max_len=max_len,
                                   title=title,
                                   max_pool=max_pool,
                                   restore_checkpoint=False,
                                   load_weights_from_pretraining=True,
                                   max_epochs=max_epochs)
    else:
        models.bert.run_bert_model(train, test,
                                   model_name=model_name,
                                   max_len=max_len,
                                   title=title,
                                   max_pool=max_pool,
                                   restore_checkpoint=False,
                                   load_weights_from_pretraining=False,
                                   max_epochs=max_epochs)


def run_roberta_config(train, test, model_name, max_pool, max_len, pretrain_snli, data_name, title_prefix=None, max_epochs=40):
    title = get_config_title(model_name, max_pool, max_len, pretrain_snli, data_name, title_prefix)

    if pretrain_snli:
        models.roberta.pretrain_roberta_model(model_name=model_name,
                                              max_len=max_len,
                                              title=title,
                                              max_pool=max_pool,
                                              restore_checkpoint=False)
        models.roberta.run_roberta_model(train, test,
                                         model_name=model_name,
                                         max_len=max_len,
                                         title=title,
                                         max_pool=max_pool,
                                         restore_checkpoint=False,
                                         load_weights_from_pretraining=True,
                                         max_epochs=max_epochs)
    else:
        models.roberta.run_roberta_model(train, test,
                                         model_name=model_name,
                                         max_len=max_len,
                                         title=title,
                                         max_pool=max_pool,
                                         restore_checkpoint=False,
                                         load_weights_from_pretraining=False,
                                         max_epochs=max_epochs)
