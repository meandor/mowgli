import json
import logging
import os
from functools import reduce, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

LOG = logging.getLogger(__name__)


def _extract_features(features, _labels):
    return features


def _extract_labels(_features, labels):
    return labels


def train_vectorizer(dataset, vocabulary_size):
    vectorizer = CountVectorizer(
        max_features=vocabulary_size,
        preprocessor=lambda x: str(x.numpy())
    )
    vectorizer.fit(dataset.map(_extract_features))
    return vectorizer


def classification_model(vocabulary_size, embedding_dimension, labels_count):
    input_layer = tf.keras.Input(shape=(vocabulary_size,), dtype='int32', name='word_vector_input')
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        name='embedding_layer'
    )(input_layer)

    layer1 = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
    layer2 = tf.keras.layers.Dense(100, activation='relu')(layer1)
    layer3 = tf.keras.layers.Dense(25, activation='relu')(layer2)

    output_layer = tf.keras.layers.Dense(
        labels_count,
        activation='softmax',
        name='classification_output'
    )(layer3)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def _count_dataset_size(dataset):
    return dataset.map(_extract_features).reduce(tf.constant(0), lambda x, _: x + 1).numpy()


def train_classification_model(model, batch_size, epochs, train_dataset, test_dataset):
    train_dataset_size = _count_dataset_size(train_dataset)
    train_dataset_batches = int(train_dataset_size / batch_size)
    batched_train_dataset = train_dataset.batch(batch_size).repeat()

    test_dataset_size = _count_dataset_size(test_dataset)
    test_dataset_batches = int(test_dataset_size / batch_size)
    batched_test_dataset = test_dataset.batch(batch_size).repeat()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='resources/tensorboard')
    ]

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.Precision()]
    )

    return model.fit(
        batched_train_dataset,
        epochs=epochs,
        steps_per_epoch=train_dataset_batches,
        validation_data=batched_test_dataset,
        validation_steps=test_dataset_batches,
        shuffle=False,
        callbacks=callbacks,
    )


def _model_metrics(model, batched_test_dataset):
    result = model.evaluate(batched_test_dataset)
    return dict(zip(model.metrics_names, result))


def evaluate_classification_model(model, test_dataset, labels):
    test_dataset_size = _count_dataset_size(test_dataset)
    model_metrics = _model_metrics(model, test_dataset.batch(test_dataset_size))

    predictions_one_hot = model.predict(test_dataset.batch(test_dataset_size))
    predicted_labels = list(map(np.argmax, predictions_one_hot))
    predicted_labels_with_names = list(map(labels.get, predicted_labels))

    actual_labels_one_hot = next(iter(test_dataset.map(_extract_labels).batch(test_dataset_size))).numpy()
    actual_labels = list(map(np.argmax, actual_labels_one_hot))
    actual_labels_with_names = list(map(labels.get, actual_labels))

    label_names = list(labels.values())

    return (
        model_metrics,
        confusion_matrix(
            actual_labels_with_names,
            predicted_labels_with_names,
            label_names
        ),
        classification_report(
            actual_labels_with_names,
            predicted_labels_with_names,
            label_names,
            output_dict=True
        )
    )


def _fill_acc(metrics_names, acc, intent_metrics):
    intent, metrics = intent_metrics
    return {
        'intent': acc['intent'] + ([intent] * len(metrics_names)),
        'metric': acc['metric'] + metrics_names,
        'metric_value': acc['metric_value'] + list(map(lambda metric_name: metrics[metric_name], metrics_names))
    }


def _to_columns(classification_report):
    metrics = ['precision', 'recall', 'f1-score']
    initial_state = {
        'intent': [],
        'metric': [],
        'metric_value': []
    }
    return reduce(partial(_fill_acc, metrics), classification_report.items(), initial_state)


def _plot_classification_report(base_path, classification_report_data):
    sns.set()
    df = pd.DataFrame(_to_columns(classification_report_data))
    pivot_df = df.pivot(index='intent', columns='metric', values='metric_value')
    f, ax = plt.subplots(figsize=(9, 6))
    sns_heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        fmt="f",
        linewidths=.5,
        ax=ax,
        cmap="YlGnBu"
    )
    sns_heatmap.set_yticklabels(
        sns_heatmap.get_yticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plot = sns_heatmap.get_figure()
    plot.savefig(base_path + 'classification_report.png')


def save_evaluation_results(model_metrics, confusion_matrix, classification_report):
    base_path = 'resources/evaluation/'
    with open(os.path.realpath(base_path + 'metrics.json'), 'w') as metrics_file:
        formatted_metrics = {k: '%.2f' % v for k, v in model_metrics.items()}
        metrics_file.write(json.dumps(formatted_metrics))
        metrics_file.close()
    _plot_classification_report(base_path, classification_report)
    return None
