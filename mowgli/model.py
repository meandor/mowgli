import logging

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras

LOG = logging.getLogger(__name__)


def _extract_features(features, _labels):
    return features


def train_vectorizer(dataset, vocabulary_size):
    vectorizer = CountVectorizer(
        max_features=vocabulary_size,
        preprocessor=lambda x: str(x.numpy())
    )
    vectorizer.fit(dataset.map(_extract_features))
    return vectorizer


def classification_model(vocabulary_size, embedding_dimension, labels_count):
    input_layer = keras.Input(shape=(vocabulary_size,), dtype='int32', name='word_vector_input')
    embedding_layer = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        name='embedding_layer'
    )(input_layer)

    layer1 = keras.layers.GlobalAveragePooling1D()(embedding_layer)
    layer2 = keras.layers.Dense(100, activation='relu')(layer1)
    layer3 = keras.layers.Dense(25, activation='relu')(layer2)

    output_layer = keras.layers.Dense(
        labels_count,
        activation='softmax',
        name='classification_output'
    )(layer3)
    return keras.Model(inputs=input_layer, outputs=output_layer)


def train_classification_model(model, batch_size, epochs, train_dataset, test_dataset):
    # test_dataset_size = test_dataset.map(_extract_features).reduce(tf.constant(0), lambda x, _: x + 1).numpy()
    train_dataset_size = train_dataset.map(_extract_features).reduce(tf.constant(0), lambda x, _: x + 1).numpy()
    number_of_batches = int(train_dataset_size / batch_size)
    train_dataset_batches = train_dataset.batch(1).repeat()
    # train_dataset_batch_generator = iter(train_dataset_batches)
    #
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    # for epoch in range(epochs):
    #     for batch in range(number_of_batches):
    #         LOG.info('epoch: %s, batch: %s', epoch, batch)
    #         features, labels = next(train_dataset_batch_generator)
    #         model.fit(features, labels)
    #
    # return model
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    # LOG.info("train: %s, test: %s", train, test)

    images, labels = train
    images = images / 255.0
    labels = labels.astype(np.int32)
    LOG.info("labels: %s", labels)

    fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(1)
    asd = next(iter(fmnist_train_ds))
    foo = next(iter(train_dataset_batches))
    LOG.info("sample fmnist: %s", asd)
    LOG.info("sample intents: %s", foo)
    LOG.info(" fmnist: %s", fmnist_train_ds)
    LOG.info(" intents: %s", train_dataset_batches)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(10)
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    model.fit(train_dataset_batches, epochs=2, steps_per_epoch=number_of_batches)
