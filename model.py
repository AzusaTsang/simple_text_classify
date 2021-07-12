import random
from util import *
import collections
import matplotlib.pyplot as plt


def extractFeatures(
    x: str, 
    model_type: Literal['bow', 'ngram', 'wv'] = 'bow', 
    n: int = 3, 
    wv: dict[str, list[float]] = None
) -> Counter:
    """
    从字符串x中提取特征
    `x`: 输入字符串
    `model_type`: 模型类型，`'bow'`词袋模型，`'ngram'`n-gram模型，`'wv'`词向量模型。
    `n`: `model_type == 'ngram'` 时的 `n`。
    `wv`: `model_type == 'wv'` 时储存词向量的字典。
    """
    ls = x.strip().split()
    if model_type == 'bow':
        c = collections.Counter(ls)
    if model_type == 'ngram':
        c = collections.Counter()
        for i in range(n):
            for j in range(i, len(ls)):
                c[' '.join(ls[j - i: j + 1])] += 1
    if model_type == 'wv':
        c = collections.Counter()
        for word in ls:
            if word in wv:
                for i, v in enumerate(wv[word]):
                    c[i] += v
    return c

def learnPredictor(
    trainExamples: list[tuple[str, int]], 
    testExamples: list[tuple[str, int]], 
    featureExtractor: Callable[..., Counter], 
    numIters: int, 
    eta: float
) -> tuple[Counter, float]:
    '''
    给定训练数据和测试数据，特征提取器`featureExtractor`、训练轮数`numIters`和学习率`eta`，
    返回学习后的权重weights
    你需要实现随机梯度下降优化权重
    '''
    losses = []
    evaltrain = []
    evaltest = []
    weights = Counter()
    bias = 0.
    for epoch in range(numIters):
        X, y = random.choice(trainExamples)
        X = featureExtractor(X)
        y_pred = dotProduct(weights, X) + bias
        loss = 1 - y_pred * y
        if loss > 0:
            increment(weights, eta*y, X)
            bias += eta * y
        print(f'epoch: {epoch}, loss: {loss:.5f}.', end = '\r', flush = True)
        if (epoch + 1) % 1000 == 0:
            losses.append(loss)
            # evaltrain.append(evaluatePredictor(trainExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) + bias >= 0 else -1)))
            # evaltest.append(evaluatePredictor(testExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) + bias >= 0 else -1)))
    print()
    plt.plot(losses)
    # ax = plt.subplot(1, 1, 1)
    # ax.plot(evaltrain, label = 'evaltrain')
    # ax.plot(evaltest, label = 'evaltest')
    # plt.legend()
    plt.show()
    return weights, bias
