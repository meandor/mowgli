from sklearn.feature_extraction.text import CountVectorizer


def _extract_features(_labels, features):
    return features


def train_vectorizer(dataset):
    vectorizer = CountVectorizer(preprocessor=lambda x: str(x.numpy()))
    vectorizer.fit(dataset.map(_extract_features))
    return vectorizer
