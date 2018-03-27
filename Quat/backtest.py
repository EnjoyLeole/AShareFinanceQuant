import backtrader as bt
import talib
from tinyevolver import Population

from .statistic import *


def back_test():
    for code in DMGR.code_list:
        c = BackTest()
        c.once()


class BackTest(object):
    """BackTest"""

    def __init__(self):
        self.feed_data = bt.feeds.PandasData(dataname=self.df_tradelog)
        self.df_indexes = PatternIndicator.calc_all_signal(self.df_tradelog)
        print(self.df_tradelog.shape)

    def once(self, show_detail=True):
        """run back test for once

        :param show_detail: show detail log or not"""
        self.test([1 for _ in range(self.index_rows_number)], show_detail=show_detail)

    def evolve(self, pop_size=300, n_gen=40):
        """run back test with genius algorithm to optimize

        :param pop_size: population size of genius algorithm
        :param n_gen: number of generations"""
        low = -1
        high = 1
        prototype = [np.random.randint(low, high) for _ in range(self.index_rows_number)]
        bounds = [[low, high] for _ in range(self.index_rows_number)]
        # self.test(prototype)

        p = Population(prototype=prototype, gene_bounds=bounds, fitness_func=self.test)
        p.populate(popsize=pop_size)
        p.evolve(ngen=n_gen)
        print(p.best.genes)

    def test(self, params, show_detail=False):
        """main body of back test, setup and rung

        :param params: parameter of genius algorithm
        :param show_detail: show detail log or not
        """
        brain = bt.Cerebro()
        brain.broker.setcash(10 ** 6)
        brain.broker.setcommission(commission=0.003)
        brain.adddata(self.feed_data)
        brain.addstrategy(TestStrategy, self.df_indexes, params, show_detail=show_detail)
        brain.run()
        value = self.get_pref(brain.broker)
        print('Final Portfolio Value: %.2f' % value)
        return value

    @property
    def basic_pref(self):
        return self.df_tradelog.tail(1).ix[0, "close"] / self.df_tradelog.head(1).ix[0, "open"]

    @property
    def index_rows_number(self):
        return self.df_indexes.shape[1] - 1

    def get_pref(self, broker):
        """get net profit rate of current strategy
        :param broker: brain.broker"""
        return broker.getvalue() / broker.startingcash / self.basic_pref


class TestStrategy(bt.Strategy):
    """Customized Trading Strategy"""

    def __init__(self, df_indexes, params, show_detail=False):
        # Keep a reference to the "close" line in the data[0] dataseries
        super().__init__()
        # self.bar_executed = len(self)
        self.data_close = self.datas[0].close_cad
        self.show_detail = show_detail
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buy_price = None
        self.buy_comm = None

        self.last_size = 0
        # Add a MovingAverageSimple indicator
        self.indicator = PatternIndicator(df_indexes, params)

    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        if self.show_detail:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.executed.price, order.executed.value, order.executed.comm))
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.executed.price, order.executed.value, order.executed.comm))

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f, values %.2f' % (self.data_close[0], self.broker.getvalue()))
        # print(self.indicator.df_indxs[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.indicator[0] > 0:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.data_close[0])
                # Keep track of the created order to avoid a 2nd order
                # self.sizer.setsizing()
                self.last_size = size = int(self.broker.cash * .98 / self.data_close[0] / 10) * 10
                self.order = self.buy(size=size)  # print(self.order)
        else:
            if self.indicator[0] < 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.data_close[0])
                # Keep track of the created order to avoid a 2nd order
                size = self.last_size
                self.order = self.sell(size=size)


class PatternIndicator(bt.Indicator):
    """Indicator simply by ta-lib's Recognition Functions"""
    lines = ('signal',)

    # params = dict(period=20, movav=bt.ind.MovAv.Simple)

    def __init__(self, df_indexes, params):
        super().__init__()
        self.df_indexes = df_indexes
        x = np.array(self.df_indexes.ix[:, 1:])
        self.lines.signal.array = np.dot(x, params).tolist()

    def next(self):
        pass

    @staticmethod
    def calc_all_signal(df):
        """        :param df: data frame to calculate
        """
        # func_list = talib.get_function_groups()['Pattern Recognition']
        func_list = ["CDL3OUTSIDE", "CDLDARKCLOUDCOVER", "CDLHIKKAKE"]
        indexes = pd.DataFrame(df.index)
        for func_name in func_list:
            func = getattr(talib, func_name)
            signal = func(np.array(df['open']), np.array(df['high']), np.array(df['low']),
                          np.array(df['close']))
            indexes[func_name] = signal
        return indexes
