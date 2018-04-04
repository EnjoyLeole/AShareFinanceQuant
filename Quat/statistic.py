# -*- coding: utf-8 -*-
"""
Updater manage all raw data & calculate result behavior

"""

import matplotlib.pyplot as plt

from Basic.Util import print_num
from .formulary import *

PRICE_TAG = '_price'
NON_RISK_RETURN = 0.01


class Strategy:

    @classmethod
    def profit_measures(cls):
        sd_sets = {}
        for target in ['ROE', 'ROA', 'ROC', 'ROIC', 'RNOA']:
            mus = []
            sds = []
            df = DMGR.read('cluster_target', target)
            for code in DMGR.code_list:
                if code not in df:
                    continue
                profit_series = df[code]
                profit_series = profit_series[profit_series == profit_series]
                org_size = profit_series.size
                profit_series = profit_series[
                    (profit_series != np.inf) & (profit_series != -np.inf)]
                after_size = profit_series.size
                if after_size != org_size:
                    print(target, code, org_size - after_size)
                if profit_series.size > 3:
                    mu = profit_series.mean()
                    if mu > 10:
                        continue  # print(target, code, se)
                    sd = statistics.stdev(profit_series)
                    if sd != sd:
                        print(code, profit_series)
                    mus.append(mu)
                    sds.append(sd)
            # print(list)
            sd_sets[target] = [len(mus), statistics.mean(mus), statistics.mean(sds)]
        for key in sd_sets:
            print_num(key, sd_sets[key])

    @classmethod
    def stock_detail(cls, code_list):
        stocks = DMGR.code_table.loc[code_list]
        return stocks
        # best = pd.merge(last, DMGR.code_table, how='left', left_on='index', right_on='code')

    @classmethod
    def quarterly_price_change(cls):
        pass

    @classmethod
    def find_security(cls, policy='PkuBook'):
        price_change = DMGR.read('cluster_target', 'DercCloseChange')
        price_change.index = price_change.quarter

        df = DMGR.read('cluster_target', policy + PERCENTILE)
        df.index = df.quarter
        df = df[df.quarter > '2009Q2']
        df.drop('quarter', axis=1, inplace=True)
        share = df > 0.99
        share.replace(False, np.nan, inplace=True)
        share.dropna(axis=1, how='all', inplace=True)
        for code in share:
            if code not in price_change:
                share.drop(code, axis=1, inplace=True)

        share.fillna(0, inplace=True)
        flag_codes = share.columns.values
        quarter_count = share.count(axis=1)

        for quarter, row in share.iterrows():
            factor = quarter_count.loc[quarter]
            share.loc[quarter] = row / factor
        print(share.shape, quarter_count.shape)

        for code in share:
            if code in price_change:
                # financial report release delay in one quarter, so buy in&out delay the same
                share[code + PRICE_TAG] = price_change[code].shift(-1)

        commission = 0.0001

        def return_calc(row):
            earn = 0
            for code in flag_codes:
                if row[code] == 0:
                    continue
                code_r = row[code + PRICE_TAG]
                if code_r != code_r:
                    print(code, row.name, 'do not have earn data!')
                    code_r = 0
                earn += row[code] * (code_r - commission)
            return earn

        r = share.apply(return_calc, axis=1)
        # draw_vs_benchmark(r)


def cumulative_sharp(earns):
    mean = earns.mean()
    sd = earns.std()
    sharp = (mean - NON_RISK_RETURN) / sd
    cum_yield = (earns + 1).cumprod()
    return cum_yield, sharp


class _GraphReport:

    def _setup(self):
        plt.close('all')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.figure = plt.figure(figsize=(8, 16))
        # self.figure
        # self.data = pd.DataFrame()

    def _get_subplot(self, pos):
        ax = self.figure.add_subplot(pos)

        return ax

    def show(self):
        plt.tight_layout()
        plt.show()

    def draw_vs_benchmark(self, earn_series):
        earn_col = 'yield'
        benchmark_col = 'benchmark'
        cum_yield = 'Policy'
        cum_benchmark = 'HS300'
        earn_series.name = earn_col
        df = earn_series.reset_index()
        df.index = df.quarter

        df[benchmark_col] = Indexes.hs300.quarter_performance()
        print(df[benchmark_col])

        def _draw(fig, ax):
            def draw_cum(yield_col, cum_col):
                df[cum_col], my_sharp = cumulative_sharp(df[yield_col])
                ax.plot(df[cum_col], label=f"{cum_col}  Sharp:{my_sharp:{4}.{2}}")

            draw_cum(earn_col, cum_yield)
            draw_cum(benchmark_col, cum_benchmark)

            ax.set_xticklabels(df.index, rotation=45, fontsize=7)

    @staticmethod
    def draw_line(df, ax, col, label=None, color=None, if_percent=False):
        label = label if label else col
        line = ax.plot(df[col].values, label=label, color=color)
        color=line[0].get_color()
        x = df[col].size
        latest_val = df[col].iloc[-1]
        label = ('%.1f' % (latest_val * 100)) + '%' if if_percent else '%.2f' % latest_val
        ax.annotate(label, xy=(x - 2, latest_val),  xycoords='data',color=color)

    @staticmethod
    def draw_dict(df, ax, dict, if_percent=False):
        if 'title' in dict:
            ax.set_title(dict['title'])
        for col in dict['lines']:
            line = dict['lines'][col]
            label = line['label'] if 'label' in line else None
            color = line['color'] if 'color' in line else None
            Graph.draw_line(df, ax, col, label=label, color=color, if_percent=if_percent)

    def draw_left(self, df, seq, dict1, if_percent=True):
        ax = self._get_subplot(seq)

        self.draw_dict(df, ax, dict1, if_percent)

        ax.yaxis.grid(True, which='both', alpha=0.2)
        lb = df.quarter[::1]
        # x=[i for i in range(lb.size)]
        # print(lb.size)
        ax.set_xticks(range(lb.size))

        ax.set_xticklabels(labels=lb.values, rotation=45, fontsize=7)
        ax.legend(loc='upper left', frameon=False)
        if if_percent:
            ax.set_yticklabels(['{:.0f}%'.format(x * 100) for x in ax.get_yticks()])
        return ax

    def draw_twin(self, df, seq, dict1, dict2, if_percent=True):
        left_ax = self.draw_left(df, seq, dict1, if_percent=False)
        left_ax.yaxis.grid(False)
        ax = left_ax.twinx()
        self.draw_dict(df, ax, dict2, if_percent)
        ax.yaxis.grid(True, which='both', alpha=0.2)
        # # ax2.set_ylim([-0.5, 2])

        ax.legend(loc='upper right', frameon=False)
        if if_percent:
            ax.set_yticklabels(['{:.0f}%'.format(x * 100) for x in ax.get_yticks()])
        # ax2.set_xticks(x)
        # ax.xaxis.set_major_locator(ticker.IndexLocator(1, 0))

    def analysis(self, code):
        self._setup()

        df = DMGR.read('stock_target', code)
        df.index = df.quarter
        df = df[df.index > '2009Q4']
        name = code + u'  %s ' % DMGR.code_name(code)
        self.draw_twin(df, 411,
                       {
                           'title': name + "估值水平",
                           'lines': {
                               'PE': {'color': 'black'},
                               }
                           }, {
                           'lines': {
                               'SaleGrowth': {},
                               'PG':         {'label': 'ProfitGrowth'},
                               'PEG':        {}
                               }
                           })
        self.draw_left(df, 412,
                       {
                           'title': name + '盈利能力(杜邦分析)',
                           'lines': {
                               'ATO_Yearly':  {},
                               'ROE':         {},
                               'GrossMargin': {}
                               }
                           })
        self.draw_left(df, 413,
                       {
                           'title': name + '盈利质量',
                           'lines': {
                               'ReceivableRatio': {},
                               'PayableRatio':    {},
                               'AccrualRatio':    {}
                               }
                           })
        self.draw_twin(df, 414,
                       {
                           'title': name + '财务风险',
                           'lines': {
                               'MscoreRegular': {},
                               'ZscoreRegular': {}
                               }
                           },
                       {
                           'lines': {
                               'PFD_percentile': {'color': 'pink'}
                               }
                           })
        self.show()


Graph = _GraphReport()
