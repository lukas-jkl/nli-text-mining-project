import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    training_data = pd.read_csv('data/train.csv')
    unlabeled_test_data = pd.read_csv('data/test.csv')
    return training_data, unlabeled_test_data


def explore_data(data, name):
    data[['language', 'label']].groupby(by='label')['language'].value_counts().unstack(0).plot(kind='bar')
    plt.title(name)
    plt.xlabel('Language')
    plt.ylabel('Number of samples')
    plt.show()


if __name__ == '__main__':
    data, unlabeled_data = load_data()
    explore_data(data, 'Data')
