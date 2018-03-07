# general propose
import numpy as np

# finacial-lib
import talib
import backtrader as bt
from tinyevolver import Population
import pandas as pd
from Basic.IO import *


def backtest():
    pgsql = PgSQLHandler()
    list = pgsql.table_read("stock_list")
    for stock in list["code"]:
        c = BackTest(pgsql.Engine, stock)
        c.once()

class BackTest(object):
    """BackTest"""

    def __init__(self, engine, ds_name):
        self.df_tradelog = pd.read_sql_table(ds_name, engine).set_index('date').sort_index()
        self.feed_data = bt.feeds.PandasData(dataname=self.df_tradelog)
        self.df_indxs = PatternIndicator.calc_all_signal(self.df_tradelog)
        print(self.df_tradelog.shape)

    def once(self, show_detail=True):
        """run backtest for once

        :param show_detail: show detail log or not"""
        self.test([1 for _ in range(self.index_rows_number)], show_detail=show_detail)

    def evolve(self, popsize=300, ngen=40):
        """run back test with genious algorithm to optimize

        :param popsize: population size of genious algorithm
        :param ngen: number of generations"""
        low = -1
        high = 1
        prototype = [np.random.randint(low, high) for _ in range(self.index_rows_number)]
        bounds = [[low, high] for _ in range(self.index_rows_number)]
        # self.test(prototype)

        p = Population(prototype=prototype, gene_bounds=bounds, fitness_func=self.test)
        p.populate(popsize=popsize)
        p.evolve(ngen=ngen)
        print(p.best.genes)

    def test(self, params, show_detail=False):
        """main body of backtest, setup and rung

        :param params: parameter of genious algorithm
        :param show_detail: show detail log or not
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(10 ** 6)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.adddata(self.feed_data)
        cerebro.addstrategy(TestStrategy, self.df_indxs, params, show_detail=show_detail)
        cerebro.run()
        value = self.get_pref(cerebro.broker)
        print('Final Portfolio Value: %.2f' % value)
        return value

    @property
    def basic_pref(self):
        return self.df_tradelog.tail(1).ix[0, "close"] / self.df_tradelog.head(1).ix[0, "open"]

    @property
    def index_rows_number(self):
        return self.df_indxs.shape[1] - 1

    def get_pref(self, broker):
        """get net profit rate of current strategy

        :param broker: cerebro.broker"""
        return broker.getvalue() / broker.startingcash / self.basic_pref


class TestStrategy(bt.Strategy):
    """Customized Trading Strategy"""

    def __init__(self, df_indxs, params, show_detail=False):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close_cad
        self.show_detail = show_detail
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.lastsize = 0
        # Add a MovingAverageSimple indicator
        self.indicator = PatternIndicator(df_indxs, params)

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
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
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.executed.price, order.executed.value, order.executed.comm))

            self.bar_executed = len(self)

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f, values %.2f' % (self.dataclose[0], self.broker.getvalue()))
        # print(self.indicator.df_indxs[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.indicator[0] > 0:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                # self.sizer.setsizing()
                self.lastsize = size = int(self.broker.cash * .98 / self.dataclose[0] / 10) * 10
                self.order = self.buy(size=size)
                # print(self.order)
        else:
            if self.indicator[0] < 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                size = self.lastsize
                self.order = self.sell(size=size)


class PatternIndicator(bt.Indicator):
    """Indicator simply by ta-lib's Recognization Functions"""
    lines = ('signal',)

    # params = dict(period=20, movav=bt.ind.MovAv.Simple)

    def __init__(self, df_indxs, params):
        self.df_indxs = df_indxs
        x = np.array(self.df_indxs.ix[:, 1:])
        self.lines.signal.array = sigs = np.dot(x, params).tolist()

    def next(self):
        pass

    @staticmethod
    def calc_all_signal(df):
        """
        :param df: dataframe to calculate
        CDL3OUTSIDE
        /* Proceed with the calculation for the requested range.
        * Must have:
        * - first: black (white) real body
        * - second: white (black) real body that engulfs the prior real body
        * - third: candle that closes higher (lower) than the second candle
        * outInteger is positive (1 to 100) for the three outside up or negative (-1 to -100) for the three outside down;
        * the user should consider that a three outside up must appear in a downtrend and three outside down must appear
        * in an uptrend, while this function does not consider it
        */

        CDLDARKCLOUDCOVER
        /* Proceed with the calculation for the requested range.
        * Must have:
        * - first candle: long white candle
        * - second candle: black candle that opens above previous day high and closes within previous day real body;
        * Greg Morris wants the close to be below the midpoint of the previous real body
        * The meaning of "long" is specified with TA_SetCandleSettings, the penetration of the first real body is specified
        * with optInPenetration
        * outInteger is negative (-1 to -100): dark cloud cover is always bearish
        * the user should consider that a dark cloud cover is significant when it appears in an uptrend, while
        * this function does not consider it

        CDLHIKKAKE
        /* Proceed with the calculation for the requested range.
        * Must have:
        * - first and second candle: inside bar (2nd has lower high and higher low than 1st)
        * - third candle: lower high and lower low than 2nd (higher high and higher low than 2nd)
        * outInteger[hikkakebar] is positive (1 to 100) or negative (-1 to -100) meaning bullish or bearish hikkake
        * Confirmation could come in the next 3 days with:
        * - a day that closes higher than the high (lower than the low) of the 2nd candle
        * outInteger[confirmationbar] is equal to 100 + the bullish hikkake result or -100 - the bearish hikkake result
        * Note: if confirmation and a new hikkake come at the same bar, only the new hikkake is reported (the new hikkake
        * overwrites the confirmation of the old hikkake)
        */

        """
        # func_list = talib.get_function_groups()['Pattern Recognition']
        func_list = ["CDL3OUTSIDE", "CDLDARKCLOUDCOVER", "CDLHIKKAKE"]
        indxs = pd.DataFrame(df.index)
        for func_name in func_list:
            func = getattr(talib, func_name)
            signal = func(np.array(df['open']), np.array(df['high']), np.array(df['low']), np.array(df['close']))
            indxs[func_name] = signal
        return indxs
