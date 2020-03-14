import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import Model

from mowgli import model


def test_should_return_vectorizer():
    given_dataset = tf.data.Dataset.from_tensor_slices((
        ['foo bar', 'foobar', 'spaghetti foo', 'bar'],
        [2, 1, 0, 0]
    ))

    actual = model.train_vectorizer(given_dataset, 1337)
    expected_type = CountVectorizer
    expected_features = ['bar', 'foo', 'foobar', 'spaghetti']

    assert expected_type == type(actual)
    assert expected_features == actual.get_feature_names()


def test_should_train_vectorizer():
    given_dataset = tf.data.Dataset.from_tensor_slices((
        ['foo bar', 'foobar', 'spaghetti foo', 'bar'],
        [2, 1, 0, 0]
    ))

    vectorizer = model.train_vectorizer(given_dataset, 4)
    actual = vectorizer.transform(tf.constant(['foo'])).toarray()
    expected = [0, 1, 0, 0]

    assert (expected == actual).all


def test_should_return_model():
    actual = type(model.classification_model(1000, 16, 10))
    expected = Model

    assert expected == actual
