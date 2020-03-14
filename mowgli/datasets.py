import csv
import logging
import os
from functools import partial
from itertools import repeat

import tensorflow as tf

LOG = logging.getLogger(__name__)


def labels(csv_file_path):
    with open(os.path.realpath(csv_file_path), 'r') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=',')
        labels_list = list(labels_reader)
        return [[int(label_row[0]), label_row[1]] for label_row in labels_list]


def parse_line(labels_count, line):
    split_line = tf.strings.split(line, sep=',')
    parsed_label = tf.strings.to_number(split_line[0], out_type=tf.dtypes.int32)
    return split_line[1], tf.one_hot(parsed_label, labels_count)


def load_dataset(dataset_path, labels_count):
    absolute_dataset_path = os.path.realpath(dataset_path)
    raw_lines_dataset = tf.data.TextLineDataset(absolute_dataset_path)
    return raw_lines_dataset.map(partial(parse_line, labels_count))


def _vectorize_feature(vectorizer, vocabulary_size, feature):
    [vectorized_feature] = vectorizer.transform(feature).toarray()
    zeros = list(repeat(0, vocabulary_size - len(vectorized_feature)))
    return tf.concat([vectorized_feature, zeros], 0)


def _apply_vectorizer(vectorizer, vocabulary_size, feature, label):
    feature = tf.py_function(partial(_vectorize_feature, vectorizer, vocabulary_size), [[feature]], tf.int64)
    return tf.reshape(feature, [vocabulary_size]), label


def vectorize(vectorizer, vocabulary_size, dataset):
    return dataset.map(partial(_apply_vectorizer, vectorizer, vocabulary_size))
