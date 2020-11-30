import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def RMSE(x, y):
    x, y = np.array(x).reshape(-1), np.array(y).reshape(-1)
    return mean_squared_error(x, y) ** 0.5


def MAPE(x, y):
    x, y = np.array(x).reshape(-1), np.array(y).reshape(-1)
    return np.mean(np.abs((y - x) / y)) * 100


def CORR(x, y):
    x, y = np.array(x).reshape(-1), np.array(y).reshape(-1)
    return stats.pearsonr(x, y)[0]


"""
 3가지 평가 방식을 모두 사용하여 평가
 truth 는 [1, 2, 3, 4...] 형태 혹은 [[1], [2], [3], [4], ...]
 prd 는 [[1, 2, 3, 4, ...],
         [20, 30, 40, 50, ...], ...] 형태. 
 0번째 index의 [1, 2, 3, 4, ...] 를 보자면 
 1은 1일 예측결과, 2는 2일 예측결과 ... 
"""

def evaluate(truth, prd, look_ahead):
    rmse, mape, corr = [], [], []
    for ahead in range(look_ahead):
        rmse.append(RMSE(truth[ahead:len(prd)+ahead], prd[:, ahead]))
        mape.append(MAPE(truth[ahead:len(prd)+ahead], prd[:, ahead]))
        corr.append(CORR(truth[ahead:len(prd)+ahead], prd[:, ahead]))
    return rmse, mape, corr


