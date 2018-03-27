# -*- coding: utf-8 -*-
"""
Updater manage all raw data & calculate result behavior

"""
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
    def find_security(cls):
        df = DMGR.read('cluster_target', 'Policy' + PERCENTILE)
        df.index = df.quarter
        df.drop('quarter', axis=1, inplace=True)
        df = df[df > 0.99]
        df.dropna(axis=1, how='all', inplace=True)
        last = df.iloc[-1]
        last = last.dropna()
        last = last.reset_index()
        best = pd.merge(last, DMGR.code_table, how='left', left_on='index', right_on='code')
        print(df.shape[1], last.size, best)
