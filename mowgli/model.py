from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras


def _extract_features(_labels, features):
    return features


def train_vectorizer(dataset, vocabulary_size):
    vectorizer = CountVectorizer(max_features=vocabulary_size, preprocessor=lambda x: str(x.numpy()))
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
