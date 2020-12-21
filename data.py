import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

def load_data():
    training_data = pd.read_csv('data/train.csv')
    unlabeled_test_data = pd.read_csv('data/test.csv')
    return training_data, unlabeled_test_data


def explore_data(data, name):
    data['label'] = data['label'].replace([0], 'entailment')
    data['label'] = data['label'].replace([1], 'neutral')
    data['label'] = data['label'].replace([2], 'contradiction')
    data[['language', 'label']].groupby(by='label')['language'].value_counts()\
        .unstack(0).plot(kind='bar')
    plt.title(name)
    plt.xlabel('Language')
    plt.ylabel('Number of samples')
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
    except FileNotFoundError:
        nlp = spacy.load('en_core_web_lg')

        premises = [nlp(sentence) for sentence in data["premise"]]
        hypothesis = [nlp(sentence) for sentence in data["hypothesis"]]

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
        if ext == ".pkl":
            features.to_pickle(cache_file)
        elif ext == ".feather":
            features.reset_index().to_feather(cache_file)
        else:
            raise NotImplementedError()
    return features

def test_training_calculate_embeddings_and_pos_tags(test, train):
    test_feature_data = calculate_embeddings_and_pos_tag(test, './cache/test_features.pkl')
    train_feature_data = calculate_embeddings_and_pos_tag(train, './cache/train_features.pkl')
    return test_feature_data, train_feature_data


if __name__ == '__main__':
    data, unlabeled_data = load_data()
    explore_data(data, 'Data')
