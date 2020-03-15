import logging

from flask import Flask, request

from mowgli import models, datasets

LOG = logging.getLogger(__name__)
APP = Flask(__name__)


@APP.route('/health')
def ping():
    return {'status': 'ok'}


def is_valid(incoming_request):
    return incoming_request.is_json and 'message' in incoming_request.get_json()


@APP.route('/intent', methods=['GET'])
def classify_intent():
    message = request.args.get('message')
    if not message:
        return {'error': 'message is not present'}, 400

    intent, probability = models.classify_intent(
        models.get_classifier(),
        models.get_vectorizer(),
        datasets.labels('resources/labels.csv'),
        message
    )
    return {'intent': {'name': intent, 'probability': probability}}
