import csv
import os

import tensorflow as tf


def labels(csv_file_path):
    with open(os.path.realpath(csv_file_path), 'r') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=',')
        labels_list = list(labels_reader)
        return [[int(label_row[0]), label_row[1]] for label_row in labels_list]


def parse_line(line):
    split_line = tf.strings.split(line, sep=',')
    parsed_label = tf.strings.to_number(split_line[0], out_type=tf.dtypes.int32)
    return parsed_label, split_line[1]


def load_dataset(dataset_path):
    absolute_dataset_path = os.path.realpath(dataset_path)
    raw_lines_dataset = tf.data.TextLineDataset(absolute_dataset_path)
    return raw_lines_dataset.map(lambda line: parse_line(line))
