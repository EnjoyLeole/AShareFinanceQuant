import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc, plot_day_summary_ohlc, volume_overlay3


def candle_test(dataset):
    # drop the date index from the dateframe
    dataset.reset_index(inplace=True)
    # convert the datetime64 column in the dataframe to 'float days'
    dataset.date = matplotlib.dates.date2num(dataset.date.astype(datetime.datetime))
    # dataAr=dataAr
    dataAr = [tuple(x) for x in
              dataset.ix[:, ['date', 'open', 'high', 'low', 'close', 'volume']].to_records(index=False)]
    pos = dataset.apply(lambda x: x['open'] - x['close'] < 0, axis=1)
    neg = dataset.apply(lambda x: x['open'] - x['close'] > 0, axis=1)
    fig, ax = plt.subplots()  # squeeze=False)

    candlestick_ohlc(ax, dataAr, width=0.6, colorup='r', colordown='g')

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, len(dataAr) - 1)
        return dataAr[0][thisind].strftime('%Y-%m-%d')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    # fig.subplots_adjust(bottom=0.2)

    # mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    # alldays = DayLocator()  # minor ticks on the days
    # weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    # dayFormatter = DateFormatter('%d')  # e.g., 12
    # ax.xaxis.set_major_locator(mondays)
    # ax.xaxis.set_minor_locator(alldays)
    # ax.xaxis.set_major_formatter(weekFormatter)
    # ax.xaxis.set_minor_formatter(dayFormatter)

    # get data from candlesticks for a bar plot
    dates = [x[0] for x in dataAr]
    dates = np.asarray(dates)
    volume = [x[5] for x in dataAr]
    volume = np.asarray(volume)
    # print(volume)
    ax2 = ax.twinx()
    # make bar plots and color differently depending on up/down for the day
    ax2.bar(dates * pos, volume * pos, color='red', width=1, align='center', alpha=0.4)
    ax2.bar(dates * neg, volume * neg, color='green', width=1, align='center', alpha=0.4)

    # scale the x-axis tight
    ax2.set_xlim(min(dates), max(dates))
    # the y-ticks for the bar were too dense, keep only every third one
    # yticks = ax2.get_yticks()
    # ax2.set_yticks(yticks[::3])

    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Volume', size=10)

    ax.xaxis_date()
    # ax.autoscale_view()
    # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()
