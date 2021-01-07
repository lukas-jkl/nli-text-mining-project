import os

import numpy as np
import pyarrow
from google_trans_new import google_translator
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from transformers import AutoTokenizer
import seaborn as sns


def load_data():
    training_data = pd.read_csv('data/train.csv')
    unlabeled_test_data = pd.read_csv('data/test.csv')
    return training_data, unlabeled_test_data


def plot_label_histogram():
    data, _ = load_data()
    sns.set_palette("deep")
    data['label'] = data['label'].replace([0], 'entailment')
    data['label'] = data['label'].replace([1], 'neutral')
    data['label'] = data['label'].replace([2], 'contradiction')
    data[['language', 'label']].groupby(by='label')['language'].value_counts() \
        .unstack(0).plot(kind='bar')
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.xlabel('Language')
    plt.ylabel('Number of samples')
    plt.savefig('plots/label_hist.png')
    plt.show()


def plot_sentence_length_histogram():
    labeled_data, _ = load_data()
    tokenizer = AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-base', padding=True, use_fast=True)
    premise_tokens = [tokenizer.tokenize(a) for a in labeled_data.premise.values.tolist()]
    hypothesis_tokens = [tokenizer.tokenize(a) for a in labeled_data.hypothesis.values.tolist()]
    premise_sentence_lengths = [len(a) for a in premise_tokens]
    hypothesis_sentence_lengths = [len(a) for a in hypothesis_tokens]
    sentence_lengths = {'premises': premise_sentence_lengths, 'hypothesis': hypothesis_sentence_lengths}

    # Plot
    sns.set_palette("deep")
    ax = sns.histplot(sentence_lengths, bins=120, kde=False, element='step')
    ax.set(xlabel="Number of tokens in sentence", ylabel="Number of samples")
    plt.savefig('plots/length_hist.png')
    plt.show()


def get_word_features(docs):
    pos_tags = []
    words = []
    word_vectors = []

    for doc in docs:
        pos_tag = []
        current_words = []
        current_word_vectors = []
        for token in doc:
            pos_tag.append(token.pos_)
            current_words.append(token.string)
            current_word_vectors.append(token.vector)

        pos_tags.append(pos_tag)
        words.append(current_words)
        word_vectors.append(current_word_vectors)
    return pos_tags, words, word_vectors


def calculate_embeddings_and_pos_tag(data, cache_file):
    root, ext = os.path.splitext(cache_file)
    try:
        if ext == ".pkl":
            features = pd.read_pickle(cache_file)
        elif ext == ".feather":
            features = pd.read_feather(cache_file)
        else:
            raise NotImplementedError()
        print("Loaded embeddings and pos tags from cache file: ", cache_file)

    except FileNotFoundError or pyarrow.ArrowIOError:
        print("cache file ", cache_file, "not found, calculating embeddings and pos tags...")
        print("loading spacy model...")
        nlp = spacy.load('en_core_web_lg')
        print("done")
        print("calculating premises...")
        premises = [nlp(sentence) for sentence in data["premise"]]
        print("done")
        print("calculating hypothesises...")
        hypothesis = [nlp(sentence) for sentence in data["hypothesis"]]
        print("done")

        premises_pos_tags, premises_words, premises_word_vectors = get_word_features(premises)
        hypothesis_pos_tags, hypothesis_words, hypothesis_word_vectors = get_word_features(hypothesis)
        features = pd.DataFrame(index=data["id"], data={
            'premises_pos_tags': premises_pos_tags,
            'premises_words': premises_words,
            'premises_word_vectors': premises_word_vectors,
            'hypothesis_pos_tags': hypothesis_pos_tags,
            'hypothesis_words': hypothesis_words,
            'hypothesis_word_vectors': hypothesis_word_vectors,
            'label': data["label"].values
        })

        print("saving to cache file ", cache_file)
        if ext == ".pkl":
            features.to_pickle(cache_file)
        elif ext == ".feather":
            features.reset_index().to_feather(cache_file)
        else:
            raise NotImplementedError()
        print("done")
    return features


def test_training_calculate_embeddings_and_pos_tags(test, train, data_name):
    test_feature_data = calculate_embeddings_and_pos_tag(test, './cache/test_' + data_name + '_features.pkl')
    train_feature_data = calculate_embeddings_and_pos_tag(train, './cache/train_' + data_name + '_features.pkl')
    return test_feature_data, train_feature_data


def translate_data(data, file):
    try:
        return pd.read_pickle(file)
    except FileNotFoundError:
        translator = google_translator()
        for i, row in data[data.language != "English"].iterrows():
            print("translating ", i, "/", data.size)
            row.premise = translator.translate(row.premise, lang_tgt='en', lang_src=row.lang_abv)
            # If we have multiple options we choose the first one
            if type(row.premise) == list:
                row.premise = row.premise[0]

            row.hypothesis = translator.translate(row.hypothesis, lang_tgt='en', lang_src=row.lang_abv)
            # If we have multiple options we choose the first one
            if type(row.hypothesis) == list:
                row.hypothesis = row.hypothesis[0]

            data.iloc[i] = row
        data.to_pickle(file)
        return data


def get_english_labeled_data():
    labeled_data, _ = load_data()
    english_data = labeled_data[labeled_data.lang_abv == "en"]
    train, test = split_test_training(english_data)
    return "only_english", train, test


def get_translated_labeled_data():
    labeled_data, _ = load_data()
    translated_data = translate_data(labeled_data, "./cache/labeled_data_translated.pkl")
    train, test = split_test_training(translated_data)
    return "translated_english", train, test


def get_multilang_labeled_data():
    labeled_data, _ = load_data()
    train, test = split_test_training(labeled_data)
    return "multilang", train, test


def split_test_training(data):
    train, test = \
        np.split(data.sample(frac=1, random_state=42),
                 [int(.85 * len(data))])
    return train, test


if __name__ == '__main__':
    plot_label_histogram()
    plot_sentence_length_histogram()
