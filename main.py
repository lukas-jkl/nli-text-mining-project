import time

import data
import models.random_guessing
import models.feature_based
import models.sentence_vector
import models.bert
import models.roberta
import tensorflow as tf
import numpy as np
import configurations
import experiments


def main():
    max_len = 100
    title = "roberta_final_training"
    dropout = None
    model_name = "distilroberta-base"

    models.roberta.pretrain_roberta_model(model_name=model_name,
                                          title=title,
                                          max_len=max_len,
                                          max_pool=False,
                                          restore_checkpoint=False)
    data_name, train, test = data.get_translated_labeled_data()
    max_epochs = 10
    models.roberta.run_roberta_model(train, test,
                                     model_name=model_name,
                                     max_epochs=max_epochs,
                                     title=title,
                                     max_len=max_len,
                                     dropout=dropout,
                                     max_pool=False,
                                     load_weights_from_pretraining=True,
                                     restore_checkpoint=False,
                                     early_stopping=True)


if __name__ == '__main__':
    main()
