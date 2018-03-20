import scipy.stats as ss


def normal_test(series, flag = '', idx = ''):
    t, p = ss.normaltest(series, nan_policy = 'omit')
    # mu,sigma=plt_norm(df[col])
    mu = series.mean()
    sigma = series.std()
    # pNum(col, val_count, p, mu, sigma)
    if p > 0.001:
        print('%s at %s got p-value %s indicate non-normal distribution' % (flag, idx, p))
    return mu, sigma
