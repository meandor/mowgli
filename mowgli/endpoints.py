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
    LOG.info('Got intent classification request')
    message = request.args.get('message')
    if not message:
        LOG.info('No message param present')
        return {'error': 'message is not present'}, 400

    intent, probability = models.classify_intent(
        models.get_classifier(),
        models.get_vectorizer(),
        datasets.labels('resources/labels.csv'),
        message
    )
    LOG.info('Classified intent: (%s, %f)', intent, probability)
    return {'intent': {'name': intent, 'probability': probability}}
