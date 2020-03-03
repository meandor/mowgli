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
