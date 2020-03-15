#!/usr/bin/env python3
import logging
import pickle

from mowgli import datasets, models

LOG = logging.getLogger(__name__)


def run():
    LOG.info('Start loading datasets')
    labels = datasets.labels('resources/labels.csv')
    labels_count = len(labels)
    train_dataset = datasets.load_dataset('resources/train.csv', labels_count)
    test_dataset = datasets.load_dataset('resources/test.csv', labels_count)
    LOG.info('Done loading datasets')

    vocabulary_size = 500
    embedding_dimensions = 50

    LOG.info('Start training vectorizer')
    vectorizer = models.train_vectorizer(train_dataset, vocabulary_size)
    LOG.info('Done training vectorizer')

    LOG.info('Start building model')
    classification_model = models.classification_model(
        vocabulary_size,
        embedding_dimensions,
        labels_count
    )
    classification_model.summary(print_fn=LOG.info)
    LOG.info('Done building model')

    LOG.info('Start training model')
    epochs = 36
    batch_size = 64
    vectorized_train_dataset = datasets.vectorize(vectorizer, vocabulary_size, train_dataset)
    vectorized_test_dataset = datasets.vectorize(vectorizer, vocabulary_size, test_dataset)
    models.train_classification_model(
        classification_model,
        batch_size,
        epochs,
        vectorized_train_dataset,
        vectorized_test_dataset,
    )
    LOG.info('Done training model')

    LOG.info('Start evaluating model')
    model_metrics, confusion_matrix, classification_report = models.evaluate_classification_model(
        classification_model,
        vectorized_test_dataset,
        labels
    )
    models.save_evaluation_results(
        model_metrics,
        confusion_matrix,
        classification_report,
        list(labels.values())
    )
    LOG.info('Done evaluating model')

    LOG.info('Start persisting model')
    classification_model.save('resources/models/classification_model.h5')
    pickle.dump(vectorizer, open('resources/models/word_vectorizer.pickle', 'wb'))
    LOG.info('Done persisting model')


if __name__ == '__main__':
    run()
