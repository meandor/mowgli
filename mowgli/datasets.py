import csv
import os


def labels(csv_file_path):
    with open(os.path.realpath(csv_file_path), 'r') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=',')
        labels_list = list(labels_reader)
        return [[int(label_row[0]), label_row[1]] for label_row in labels_list]
