# -*- coding: utf-8 -*-
"""
Updater manage all raw data & calculate result behavior

"""
from Basic.Util import print_num
from .formulary import *


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
        df = DMGR.read('cluster_target', policy + PERCENTILE)
        df.index = df.quarter
        df.drop('quarter', axis=1, inplace=True)
        flag = df > 0.99
        flag.replace(False, np.nan, inplace=True)
        flag.dropna(axis=1, how='all', inplace=True)
        quarter_count = flag.count(axis=1)

        for key, row in flag.iterrows():
            factor = quarter_count.loc[key]
            flag.loc[key] = row / factor
        print(flag.shape, quarter_count.shape)

        print(flag)
        for code in df:
            pass
        # last = df.iloc[-1]
        # last = last.dropna()
        # last = last.reset_index()

        # print(df.shape[1], last.size, best)
