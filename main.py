import json
from util import *
from model import *

def TestModel(numIters: int, eta: float):
    trainExamples = readExamples('data/data_rt.train')
    testExamples = readExamples('data/data_rt.test')

    featureExtractor = lambda x: extractFeatures(x, 'ngram') # n-gram

    weights, bias = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=numIters, eta=eta)
    json.dump(weights, open('weights.json', 'w+'))
    trainError = evaluatePredictor(trainExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) + bias >= 0 else -1))
    testError = evaluatePredictor(testExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) + bias >= 0 else -1))
    print (f"train error = {trainError}, test error = {testError}")
    

if __name__ == "__main__":
    TestModel(100000, 0.01)