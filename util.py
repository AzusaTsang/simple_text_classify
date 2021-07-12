from typing import *

def dotProduct(d1: Counter, d2: Counter) -> float:
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1[f] * v for f, v in d2.items())

def increment(d1: Counter, scale: float, d2: Counter):
    for f, v in d2.items():
        d1[f] = d1[f] + v * scale

def readExamples(path:str, return_X_y: bool = False) -> Union[list[tuple[str, int]], tuple[list[str], list[int]]]:
    if return_X_y:
        X, y = [], []
    else:
        examples: list[tuple[str, int]] = []
    for line in open(path, encoding = 'utf8'):
        # Format of each line: <output label (+1 or -1)> <input sentence>
        _y, _x = line.split(' ', 1)
        if return_X_y:
            X.append(_x.strip())
            y.append(int(_y))
        else:
            examples.append((_x.strip(), int(_y)))
    print(f'Read {len(X if return_X_y else examples)} examples from {path}')
    return (X, y) if return_X_y else examples

def evaluatePredictor(examples: list[tuple], predictor: Callable) -> float:
    '''
    在`examples`上测试`predictor`的性能，返回错误率
    '''
    error = 0
    with open('errors.txt', 'w+', encoding='utf8') as f:
        for x, y in examples:
            if predictor(x) != y:
                error += 1
                f.write(f'{y:+d} {x}\n')
    return error / len(examples)