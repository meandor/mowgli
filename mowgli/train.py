#!/usr/bin/env python3
import logging

from mowgli import datasets, model

LOG = logging.getLogger(__name__)


def run():
    LOG.info('Start loading datasets')
    labels = datasets.labels('resources/labels.csv')
    labels_count = len(labels)
    train_dataset = datasets.load_dataset('resources/train.csv', labels_count)
    test_dataset = datasets.load_dataset('resources/test.csv', labels_count)
    LOG.info('Done loading datasets')

    vocabulary_size = 300
    embedding_dimensions = 125

    LOG.info('Start training vectorizer')
    vectorizer = model.train_vectorizer(train_dataset, vocabulary_size)
    LOG.info('Done training vectorizer')

    LOG.info('Start building model')
    classification_model = model.classification_model(
        vocabulary_size,
        embedding_dimensions,
        labels_count
    )
    classification_model.summary(print_fn=LOG.info)
    LOG.info('Done building model')

    LOG.info('Start training model')
    epochs = 10
    batch_size = 32
    model.train_classification_model(
        classification_model,
        batch_size,
        epochs,
        datasets.vectorize(vectorizer, vocabulary_size, train_dataset),
        datasets.vectorize(vectorizer, vocabulary_size, test_dataset),
    )
    LOG.info('Done training model')


if __name__ == '__main__':
    run()
