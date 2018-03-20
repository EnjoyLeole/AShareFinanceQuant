import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

mu, sigma = 100, 15
xr = mu + sigma * np.random.randn(10000)


def plt_hist(x, segment = 100):
    x = x[(x == x) & (x != np.inf)]
    # the histogram of the data
    n, bins, patches = plt.hist(x, segment, facecolor = 'green')
    mu = x.mean()
    sigma = x.std()
    # print(mu,sigma)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth = 1)

    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()
    return mu,sigma
