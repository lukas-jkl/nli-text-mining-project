import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = "D:/text_mining_project_logs/"

embedded_classifier_model_name = 'embedded_classifier'
embedded_classifier_pretrain = "2020-12-29_15_translated_with_pretrain"
embedded_classifier = "2020-12-29_18_translated_no_pretrain"

lstm_classifier_model_name = 'lstm_classifier'
lstm_classifier_pretrain = "2020-12-29_12_translated_with_pretrain"
lstm_classifier = "2020-12-29_12_translated_no_pretrain"

bert_base_model_name = "bert-base-cased"
bert_base_maxpool_pretrain = "2020-12-27_18_bert_english_experiments_bert-base-cased_translated_english_max-len_50_with-snli-pretrain"
bert_base_pretrain = "2020-12-27_18_bert_english_experiments_bert-base-cased_maxpool_translated_english_max-len_50_with-snli-pretrain"
bert_base = "2020-12-27_15_maxlen_maxpool_experiment_bert-base-cased_only_english_max-len_50_without-pretrain"

distilbert_model_name = 'distilbert-base-cased'
distilbert_maxpool_pretrain = "2020-12-27_16_bert_english_experiments_distilbert-base-cased_maxpool_translated_english_max-len_50_with-snli-pretrain"
distilbert_pretrain = "2020-12-27_17_bert_english_experiments_distilbert-base-cased_translated_english_max-len_50_with-snli-pretrain"
distilbert = "2020-12-27_18_bert_english_experiments_distilbert-base-cased_translated_english_max-len_50_without-pretrain"

roberta_base_model_name = 'roberta-base'
roberta_base_maxpool_pretrain = "2020-12-28_17_roberta_english_experiments_roberta-base_maxpool_translated_english_max-len_50_with-snli-pretrain"
roberta_base_pretrain = "2020-12-28_18_roberta_english_experiments_roberta-base_translated_english_max-len_50_with-snli-pretrain"
roberta_base = "2020-12-28_19_roberta_english_experiments_roberta-base_translated_english_max-len_50_without-pretrain"

distilroberta_model_name = 'distilroberta-base'
distilroberta_maxpool_pretrain = "2020-12-28_16_roberta_english_experiments_distilroberta-base_maxpool_translated_english_max-len_50_with-snli-pretrain"
distilroberta_pretrain = "2020-12-28_17_roberta_english_experiments_distilroberta-base_translated_english_max-len_50_with-snli-pretrain"
distilroberta = "2020-12-28_17_roberta_english_experiments_distilroberta-base_translated_english_max-len_50_without-pretrain"

multilingual_bert_model_name = "bert-base-multilingual-cased"
multilingual_bert_pretrain = "2020-12-30_00_multilang_experiments_with_pretraining_bert-base-multilingual-cased_multilang_max-len_50_with-snli-pretrain"
multilingual_bert = "2020-12-29_22_multilang_experiments_bert-base-multilingual-cased_multilang_max-len_50_without-pretrain"

multilingual_roberta_model_name = "tf-xlm-roberta-base"
multilingual_roberta_pretrain = "tf-xlm-roberta-base_multilang_max-len_50_with-snli-pretrain"
multilingual_roberta = "tf-xlm-roberta-base_multilang_max-len_50_without-pretrain"

roberta_final_model_name = "roberta-base"
roberta_final_model_pretrain = "roberta_final_training"
roberta_final_model = "roberta_final_training"

distilroberta_final_model_name = "distilroberta-base"
distilroberta_final_model_pretrain = "2021-01-02_08_roberta_final_training"
distilroberta_final_model = "2021-01-02_08_roberta_final_training"



def get_pretrain_dir(title, model_name):
    return DATA_DIR + model_name + '/pretraining/' + title


def get_training_dir(title, model_name):
    return DATA_DIR + model_name + '/training/' + title


def load_history(title, model_name):
    try:
        with open(
                get_pretrain_dir(title, model_name) + '/history.txt') as json_file:
            pretraining_history = json.load(json_file)
    except FileNotFoundError:
        print("No pretraining history found")
        pretraining_history = None
    with open(
            get_training_dir(title, model_name) + '/history.txt') as json_file:
        training_history = json.load(json_file)
    return pretraining_history, training_history


def plot_accuracy(history, file_path):
    plt.plot(history['accuracy'], color='blue', label='Accuracy')
    plt.plot(history['val_accuracy'], color='green', label='Validation accuracy')
    min_epoch = np.argmin(history['val_loss'])
    plt.scatter([min_epoch], [history['val_accuracy'][min_epoch]], color='green')  # , label='min val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    axes = plt.gca()
    axes.set_ylim([0.25, 1])
    x_max = max(5, len(history['val_accuracy']))
    axes.set_xlim([0, x_max])
    plt.legend()
    plt.savefig(file_path)
    plt.show()


def plot_comparison(history_1, name_1, history_2, name_2, file_path):
    plt.plot(history_1['accuracy'], color='blue', label=name_1 + ' accuracy')
    plt.plot(history_1['val_accuracy'], color='green', label=name_1 + ' validation accuracy')
    min_epoch_1 = np.argmin(history_1['val_loss'])
    plt.scatter([min_epoch_1], [history_1['val_accuracy'][min_epoch_1]],
                color='green')  # , label=name_1 + ' min val_loss')

    plt.plot(history_2['accuracy'], color='lightblue', label=name_2 + ' accuracy')
    plt.plot(history_2['val_accuracy'], color='lightgreen', label=name_2 + ' validation accuracy')
    min_epoch_2 = np.argmin(history_2['val_loss'])
    plt.scatter([min_epoch_2], [history_2['val_accuracy'][min_epoch_2]],
                color='lightgreen')  # , label=name_2 + ' min val_loss')

    axes = plt.gca()
    axes.set_ylim([0.25, 1])
    x_max = max(5, len(history_1['val_accuracy']), len(history_2['val_accuracy']))
    axes.set_xlim([0, x_max])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(file_path)
    plt.show()


def plot_loss(history, file_path):
    plt.plot(history['loss'], color='blue', label='Loss')
    plt.plot(history['val_loss'], color='green', label='Validation loss')
    axes = plt.gca()
    axes.set_ylim([0, 1.5])
    x_max = max(5, len(history['val_accuracy']))
    axes.set_xlim([0, x_max])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path)
    plt.show()


def plot_history(model_name, title):
    pretraining_history, training_history = load_history(title, model_name)
    if pretraining_history:
        plot_accuracy(pretraining_history, get_pretrain_dir(title, model_name) + '/' + model_name + '_accuracy.png')
        plot_loss(pretraining_history, get_pretrain_dir(title, model_name) + '/' + model_name + '_loss.png')
    plot_accuracy(training_history, get_training_dir(title, model_name) + '/' + model_name + '_accuracy.png')
    plot_loss(training_history, get_training_dir(title, model_name) + '/' + model_name + '_loss.png')


def compare_history(model_name, title_1, name_1, title_2, name_2):
    pretraining_history_1, training_history_1 = load_history(title_1, model_name)
    pretraining_history_2, training_history_2 = load_history(title_2, model_name)
    plot_comparison(training_history_1, name_1, training_history_2, name_2,
                    DATA_DIR + model_name + "/" + model_name + "_comparison_" + name_1 + "_" + name_2 + ".png")

    if pretraining_history_1 and pretraining_history_2:
        plot_comparison(pretraining_history_1, name_1, pretraining_history_2, name_2,
                        DATA_DIR + model_name + "/" + model_name + "_pretrain_comparison_" + name_1 + "_" + name_2 + ".png")


def plot_bert_max_length_comparison():
    val_accuracies = {}
    for max_len in [30, 50, 50, 70, 100]:
        title = "2020-12-27_15_maxlen_maxpool_experiment_bert-base-cased_only_english_max-len_" + \
                str(max_len) + "_without-pretrain"

        _, train_history = load_history(title, "bert-base-cased")
        val_accuracies[max_len] = max(train_history['val_accuracy'])
    plt.scatter(x=val_accuracies.keys(), y=val_accuracies.values())
    plt.plot(val_accuracies.keys(), val_accuracies.values())
    plt.xlabel('Maximal sentence length in tokens')
    plt.ylabel('Validation accuracy')
    axes = plt.gca()
    axes.set_ylim([0.25, 1])
    plt.show()
    print("done")


def plot_bert_classifier():
    plot_bert_max_length_comparison()

    # bert base
    plot_history(model_name='bert-base-cased', title=bert_base_maxpool_pretrain)
    plot_history(model_name='bert-base-cased', title=bert_base_pretrain)
    compare_history(bert_base_model_name,
                    bert_base_pretrain, "without_maxpool",
                    bert_base_maxpool_pretrain, "with_maxpool",
                    )
    compare_history(bert_base_model_name,
                    bert_base_pretrain, "with_pretrain",
                    bert_base, "no_pretrain")

    # distilbert
    plot_history(model_name=distilbert_model_name,
                 title=distilbert_maxpool_pretrain)
    plot_history(model_name='distilbert-base-cased',
                 title=distilbert_pretrain)
    compare_history(distilbert_model_name,
                    distilbert_pretrain, "without_maxpool",
                    distilbert_maxpool_pretrain, "with_maxpool"
                    )
    compare_history(distilbert_model_name,
                    distilbert_pretrain, "with_pretrain",
                    distilbert, "no_pretrain")

    model_names = [
        bert_base_model_name,
        bert_base_model_name,
        distilbert_model_name,
        distilbert_model_name
    ]
    titles = [
        bert_base_pretrain,
        bert_base,
        distilbert_pretrain,
        distilbert
    ]
    plot_names = [
        "bert base with pretrain",
        "bert base no pretrain",
        "distilbert with pretrain",
        "distilbert no pretrain"
    ]
    compare_models_max_validation(model_names, titles, plot_names, DATA_DIR + "bert_distilbert_comparison.png")


def plot_roberta_classifier():
    # robert base
    plot_history(model_name=roberta_base_model_name,
                 title=roberta_base_maxpool_pretrain)
    plot_history(model_name=roberta_base_model_name,
                 title=roberta_base_pretrain)
    compare_history(roberta_base_model_name,
                    roberta_base_pretrain, "without_maxpool",
                    roberta_base_maxpool_pretrain, "with_maxpool"
                    )
    compare_history('roberta-base',
                    roberta_base_pretrain, "with_pretrain",
                    roberta_base, "no_pretrain")

    # distilroberta
    plot_history(model_name=distilroberta_model_name,
                 title=distilroberta_maxpool_pretrain)
    plot_history(model_name=distilroberta_model_name,
                 title=distilroberta_pretrain)
    compare_history(distilroberta_model_name,
                    distilroberta_pretrain, "without_maxpool",
                    distilroberta_maxpool_pretrain, "with_maxpool"
                    )
    compare_history(distilroberta_model_name,
                    distilroberta_pretrain, "with_pretrain",
                    distilroberta, "no_pretrain")

    model_names = [
        roberta_base_model_name,
        roberta_base_model_name,
        distilroberta_model_name,
        distilroberta_model_name
    ]
    titles = [
        roberta_base_pretrain,
        roberta_base,
        distilroberta_pretrain,
        distilroberta
    ]
    plot_names = [
        "roberta base with pretrain",
        "roberta base no pretrain",
        "distilroberta with pretrain",
        "distilroberta no pretrain"
    ]
    compare_models_max_validation(model_names, titles, plot_names, DATA_DIR + "roberta_distilroberta_comparison.png")


def compare_models_max_validation(model_names, titles, plot_names, file_path):
    val_accuracies = {}
    for (model_name, title, plot_name) in zip(model_names, titles, plot_names):
        pretraining_history, training_history = load_history(title, model_name)
        val_accuracies[plot_name] = max(training_history['val_accuracy'])

    plot_data = {
        'val_accuracy': list(val_accuracies.values()),
        'model': list(val_accuracies.keys()),
        'hue': list(val_accuracies.values())
    }

    sns.barplot(x='val_accuracy', y='model', data=plot_data, palette='PuBu', hue='hue', dodge=False)
    plt.gcf().subplots_adjust(left=0.45)
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    axes.get_legend().remove()
    axes.set_xlim([0, 1])
    plt.xlabel('Maximal validation accuracy')
    plt.savefig(file_path)
    plt.show()
    print("done")


def plot_multilingual_transformers():
    # BERT
    plot_history(model_name=multilingual_bert_model_name,
                 title=multilingual_bert_pretrain)
    compare_history(multilingual_bert_model_name,
                    multilingual_bert_pretrain, "with_pretrain",
                    multilingual_bert, "no_pretrain")

    # RoBERTa
    plot_history(model_name=multilingual_roberta_model_name,
                 title=multilingual_roberta_pretrain)
    compare_history(multilingual_roberta_model_name,
                    multilingual_roberta_pretrain, "with_pretrain",
                    multilingual_roberta, "no_pretrain")


def plot_embedded_classifier():
    plot_history(model_name=embedded_classifier_model_name, title=embedded_classifier_pretrain)
    plot_history(model_name=embedded_classifier_model_name, title=embedded_classifier)
    compare_history(embedded_classifier_model_name, embedded_classifier_pretrain, "with_pretrain",
                    embedded_classifier, "no_pretrain")


def plot_lstm_classifier():
    plot_history(model_name=lstm_classifier_model_name, title=lstm_classifier_pretrain)
    plot_history(model_name=lstm_classifier_model_name, title=lstm_classifier)
    compare_history(lstm_classifier_model_name, lstm_classifier_pretrain, "with_pretrain",
                    lstm_classifier, "no_pretrain")


def compare_transformer_models():
    model_names = [
        bert_base_model_name,
        bert_base_model_name,
        distilbert_model_name,
        distilbert_model_name,
        roberta_base_model_name,
        roberta_base_model_name,
        distilroberta_model_name,
        distilroberta_model_name,
        multilingual_bert_model_name,
        multilingual_bert_model_name,
        multilingual_roberta_model_name,
        multilingual_roberta_model_name
    ]
    titles = [
        bert_base_pretrain,
        bert_base,
        distilbert_pretrain,
        distilbert,
        roberta_base_pretrain,
        roberta_base,
        distilroberta_pretrain,
        distilroberta,
        multilingual_bert_pretrain,
        multilingual_bert,
        multilingual_roberta_pretrain,
        multilingual_roberta
    ]
    plot_names = [
        "BERT with pretrain",
        "BERT no pretrain",
        "DistilBERT with pretrain",
        "DistilBERT no pretrain",
        "RoBERTa with pretrain",
        "RoBERTa no pretrain",
        "DistilRoBERTa with pretrain",
        "DistilRoBERTa no pretrain",
        "Multilingual BERT with pretrain",
        "Multilingual BERT no pretrain",
        "Multilingual RoBERTa with pretrain",
        "Multilingual RoBERTa no pretrain",
    ]
    compare_models_max_validation(model_names, titles, plot_names, DATA_DIR + "bert_roberta_comparison.png")


def plot_model_performance_comparison():
    test_accuracy = {
        "word embedding": 0.48,
        "LSTM model": 0.5,
        "DistilRoBERTa": 0.73,
        "RoBERTa": 0.78,
        "ml-RoBERTa": 0.73
    }
    test_accuracy_full_pretrain = {
        "DistilRoBERTa": 0.8,
        "RoBERTa": 0.83,
    }
    model_size_full_pretrain_kb = {
        "DistilRoBERTa": 958000,
        "RoBERTa": 1456000,
    }
    model_size_kb = {
        "word embedding": 6000,
        "lstm model": 934000,
        "DistilRoBERTa": 958000,
        "RoBERTa": 1456000,
        "ml-RoBERTa": 3254000
    }
    trainable_parameters = {
        "word embedding": 506000,
        "LSTM": 118155,  # 238,820,955 total
        "DistilRoBERTa": 82121000,
        "RoBERTa": 124648000,
        "ml-RoBERTa": 278046000
    }
    trainable_parameters_full_pretrain = {
        "DistilRoBERTa": 82121000,
        "RoBERTa": 124648000,
    }

    # Plot model size KB
    X = np.array(list(model_size_kb.values())) / 1000
    Y = list(test_accuracy.values())
    plt.scatter(X, Y, label="Pretrain 100,000", color="royalblue")
    for i, label in enumerate(test_accuracy.keys()):
        y = 0.007
        x = 50
        if i % 2 == 0:
            y = - 0.03
        if i == 4:
            x -= 200
            y -= 0.025
        plt.annotate(label, (X[i] + x, Y[i] + y), color="royalblue")

    X = np.array(list(model_size_full_pretrain_kb.values())) / 1000
    Y = list(test_accuracy_full_pretrain.values())
    plt.scatter(X, Y, label="Full Pretrain", color="darkred")

    for i, label in enumerate(test_accuracy_full_pretrain.keys()):
        y = 0.012
        x = 550
        if i == 0:
            x += 300
        print("here")
        plt.annotate(label, (X[i] - x, Y[i] + y), color='darkred')

    plt.xlabel("Model size in MB")
    plt.ylabel("Test Accuracy")
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend()
    plt.savefig(DATA_DIR + "test_accuracy_vs_model_size.png")
    plt.show()

    # Plot model size Parameters
    X = np.array(list(trainable_parameters.values()))
    Y = list(test_accuracy.values())
    plt.scatter(X, Y, label="Pretrain 100,000", color="royalblue")
    for i, label in enumerate(test_accuracy.keys()):
        y = 0.007
        x = 5000000
        if i % 2 == 0:
            y = - 0.03
        if i == 4:
            x -= 40000000
            y -= 0.025
        plt.annotate(label, (X[i] + x, Y[i] + y), color="royalblue")
    X = np.array(list(trainable_parameters_full_pretrain.values()))
    Y = list(test_accuracy_full_pretrain.values())
    plt.scatter(X, Y, label="Full Pretrain", color="darkred")
    for i, label in enumerate(test_accuracy_full_pretrain.keys()):
        y = 0.015
        x = 45000000
        if i == 0:
            y = 0.01
            x = 70000000
        print("here")
        plt.annotate(label, (X[i] - x, Y[i] + y), color='darkred')
    plt.xlabel("Model size in trainable parameters")
    plt.ylabel("Test Accuracy")
    # plt.xscale('log')
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend()
    plt.savefig(DATA_DIR + "test_accuracy_vs_trainable_parameters.png")
    plt.show()


def plot_final_roberta():
    plot_history(model_name=roberta_final_model_name, title=roberta_final_model_pretrain)
    plot_history(model_name=distilroberta_final_model_name, title=distilroberta_final_model_pretrain)




if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    plot_model_performance_comparison()
    compare_transformer_models()
    plot_final_roberta()
    plot_multilingual_transformers()
    plot_bert_classifier()
    plot_roberta_classifier()
    plot_embedded_classifier()
    plot_lstm_classifier()
