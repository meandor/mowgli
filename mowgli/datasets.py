import csv
import logging
import os
from functools import partial

import tensorflow as tf

LOG = logging.getLogger(__name__)


def labels(csv_file_path):
    with open(os.path.realpath(csv_file_path), 'r') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=',')
        labels_list = list(labels_reader)
        return [[int(label_row[0]), label_row[1]] for label_row in labels_list]


def parse_line(line):
    split_line = tf.strings.split(line, sep=',')
    parsed_label = tf.strings.to_number(split_line[0], out_type=tf.dtypes.int32)
    return split_line[1], parsed_label


def load_dataset(dataset_path):
    absolute_dataset_path = os.path.realpath(dataset_path)
    raw_lines_dataset = tf.data.TextLineDataset(absolute_dataset_path)
    return raw_lines_dataset.map(parse_line)


def _vectorize_feature(vectorizer, feature):
    [vectorized_feature] = vectorizer.transform(feature).toarray()
    LOG.info('%s',vectorized_feature)
    return vectorized_feature


def _apply_vectorizer(vectorizer, feature, label):
    feature = tf.py_function(partial(_vectorize_feature, vectorizer), [[feature]], tf.int64)
    return feature, label


def vectorize(vectorizer, dataset):
    return dataset.map(partial(_apply_vectorizer, vectorizer))
