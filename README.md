# Mowgli
[![CircleCI](https://circleci.com/gh/meandor/mowgli.svg?style=svg)](https://circleci.com/gh/meandor/mowgli)

A _Python_ service to train and serve a neural network that classifies intents.
Mowgli uses [pipenv](https://github.com/pypa/pipenv).

## Install
To install locally:
```bash
pipenv install --dev
``` 

## Test
```bash
pipenv run pytest 
```

## Train and evaluate
To train and evaluate run:
```bash
pipenv run train
```

This will train the model, persist it and run evaluations on the model.

The trained model will be stored in the directory `resources/models`.
The evaluation results will be save in the directory `resources/evaluation`.
This includes confusion matrix and classification diagrams. The raw metrics of precision
accuracy and loss are save in the `metrics.json` file.

## Run tensorboard
To validate the model training you can observe the training with the [tensorboard](https://www.tensorflow.org/tensorboard):
```bash
pipenv run tensorboard
```
This will the tensorboard on [http://localhost:6006](http://localhost:6006).

## Serve locally
To run serve a trained model locally you have to first train it. Afterwards you can start a local
server with:
```bash
pipenv run serve
```

## Run linter
```bash
pipenv run lint
```
