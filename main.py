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
    time_date = time.strftime("%Y-%m-%d_%H_")
    title = time_date + "translated_no_pretrain"
    labeled_data, _ = data.load_data()
    data_name, train_translated, test_translated = data.get_translated_labeled_data()

    model_name = "roberta-base"
    title = "roberta_final_training_early_stopping"
    models.roberta.inference(test_translated, model_name, 100, title)


if __name__ == '__main__':
    main()
