import json
from util import *
from model import *

def TestModel(numIters, eta):
    trainExamples = readExamples('data/data_rt.train')
    testExamples = readExamples('data/data_rt.test')

    # featureExtractor = extractFeatures # bag of words

    featureExtractor = lambda x: extractFeatures(x, 'ngram') # n-gram

    # wv = json.load(open('word_vector_lite.json')) # word vectors
    # featureExtractor = lambda x: extractFeatures(x, 'wv', wv = wv)

    weights, bias = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=numIters, eta=eta)
    # json.dump(weights, open('weights.json', 'w+')) # save the weight
    trainError = evaluatePredictor(trainExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) + bias >= 0 else -1))
    testError = evaluatePredictor(testExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) + bias >= 0 else -1))
    print (f"train error = {trainError}, test error = {testError}")
    

if __name__ == "__main__":
    TestModel(100000, 0.01)