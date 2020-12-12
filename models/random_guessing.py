import tensorflow as tf
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


def run_dummy_model(train, test):
    dummy_clf = DummyClassifier(
        strategy="stratified")  # generates predictions by respecting the training set’s class distribution.
    dummy_clf.fit(train['premise'], train['label'])
    y_pred = dummy_clf.predict(test['premise'])
    y_true = test['label']
    print(classification_report(y_true, y_pred))
