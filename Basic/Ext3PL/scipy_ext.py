import numpy as np
import scipy.stats as ss


def normal_test(series, flag = ''):
    series = series[(series == series) & (series != np.inf)]
    t, p = ss.normaltest(series, nan_policy = 'omit')
    # mu,sigma=plt_norm(df[col])
    mu = series.mean()
    sigma = series.std()
    # pNum(col, val_count, p, mu, sigma)
    if p > 0.001:
        print('%s   got p-value %s indicate non-normal distribution' % (flag, p))
    return mu, sigma
