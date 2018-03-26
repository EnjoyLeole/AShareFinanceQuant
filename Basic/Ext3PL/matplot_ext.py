import matplotlib.mlab as matplot
import matplotlib.pyplot as plt
import numpy as np


# mu, sigma = 100, 15
# xr = mu + sigma * np.random.randn(10000)


def plt_line(df):
    plt.plot(df)
    plt.show()


def plt_hist(series, segment = 100):
    series = series[(series == series) & (series != np.inf)]
    # the histogram of the data
    n, bins, patches = plt.hist(series, segment, facecolor = 'green')
    mu = series.mean()
    sigma = series.std()

    # add a 'best fit' line
    y = matplot.normpdf(bins, mu, sigma)
    les = plt.plot(bins, y, 'r--', linewidth = 1)

    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()
    return mu, sigma
