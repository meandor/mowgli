import numpy.testing as npt
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

    assert (expected == actual).all()


def test_should_return_model():
    actual = type(model.classification_model(1000, 16, 10))
    expected = Model

    assert expected == actual


def test_should_calculate_confusion_matrix_with_3_classes(mocker):
    given_dataset = tf.data.Dataset.from_tensor_slices((
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
    ))
    mock_model = mocker.Mock()
    mock_model.predict.return_value = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
    model_metrics_mock = mocker.patch('mowgli.model._model_metrics')
    model_metrics_mock.return_value = 'foo'

    actual = model.evaluate_classification_model(
        mock_model,
        given_dataset,
        {0: 'foo', 1: 'bar', 2: 'foobar'}
    )
    actual_metrics, actual_confusion_matrix, actual_classification_report = actual
    assert 'foo' == actual_metrics
    expected_confusion_matrix = [
        [1, 0, 1],
        [1, 0, 0],
        [0, 0, 1]
    ]
    expected_classification_report = {
        'foo': {
            'precision': 0.5,
            'recall': 0.5,
            'f1-score': 0.5,
            'support': 2
        },
        'bar': {
            'precision': 0.0,
            'recall': 0.0,
            'f1-score': 0.0,
            'support': 1
        },
        'foobar': {
            'precision': 0.5,
            'recall': 1.0,
            'f1-score': 2 / 3,
            'support': 1},
        'accuracy': 0.5,
        'macro avg': {
            'precision': 1 / 3, 'recall': 0.5,
            'f1-score': 0.38888888888888884, 'support': 4},
        'weighted avg': {'precision': 0.375, 'recall': 0.5,
                         'f1-score': 0.41666666666666663, 'support': 4}}

    npt.assert_array_equal(expected_confusion_matrix, actual_confusion_matrix)
    assert expected_classification_report == actual_classification_report
