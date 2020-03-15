from mowgli import endpoints

endpoints.APP.testing = True

with endpoints.APP.test_client() as client:
    def test_should_return_classified_hello_intent(mocker):
        classify_intent_mock = mocker.patch('mowgli.models.classify_intent')
        classify_intent_mock.return_value = ('greet', 1.0)
        get_classifier_mock = mocker.patch('mowgli.models.get_classifier')
        classifier_mock = mocker.Mock()
        get_classifier_mock.return_value = classifier_mock
        get_vectorizer_mock = mocker.patch('mowgli.models.get_vectorizer')
        vectorizer_mock = mocker.Mock()
        get_vectorizer_mock.return_value = vectorizer_mock
        load_labels_mock = mocker.patch('mowgli.datasets.labels')
        labels_mock = mocker.Mock()
        load_labels_mock.return_value = labels_mock

        response = client.get('/intent?message=hello%20there')
        actual = response.get_json()
        expected = {'intent': {'name': 'greet', 'probability': 1.0}}

        assert 200 == response.status_code
        assert expected == actual
        classify_intent_mock.assert_called_with(classifier_mock, vectorizer_mock, labels_mock, 'hello there')


    def test_should_return_400():
        response = client.get('/intent?message=')

        assert 400 == response.status_code
