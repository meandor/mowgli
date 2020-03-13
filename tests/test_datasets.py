import numpy as np
import numpy.testing as npt
from pytest import raises

from mowgli import datasets


def test_should_load_1_label():
    actual = datasets.labels("tests/resources/one-label.csv")
    expected = [[1337, "foobar"]]

    assert expected == actual


def test_should_load_2_labels():
    actual = datasets.labels("tests/resources/labels.csv")
    expected = [[0, "foo"], [1, "bar"]]

    assert expected == actual


def test_should_throw_error_for_non_existent_file():
    with raises(FileNotFoundError):
        datasets.labels("foo.csv")


def test_should_load_dataset():
    actual_dataset = datasets.load_dataset("tests/resources/dataset.csv")
    actual_labels, actual_features = next(iter(actual_dataset.batch(3)))

    npt.assert_array_equal(actual_labels, np.array([2, 1, 0], dtype=int))
    npt.assert_array_equal(actual_features, np.array([b'foo bar', b'foobar', b'spaghetti foo'], dtype=object))
