import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

from mowgli import model, datasets


def test_should_return_vectorizer():
    given_dataset = datasets.load_dataset('tests/resources/dataset.csv')

    actual = model.train_vectorizer(given_dataset)
    expected_type = CountVectorizer
    expected_features = ['bar', 'foo', 'foobar', 'spaghetti']

    assert expected_type == type(actual)
    assert expected_features == actual.get_feature_names()


def test_should_train_vectorizer():
    given_dataset = datasets.load_dataset('tests/resources/dataset.csv')

    vectorizer = model.train_vectorizer(given_dataset)
    actual = vectorizer.transform(tf.constant(['foo'])).toarray()
    expected = [0, 1, 0, 0]

    assert (expected == actual).all
