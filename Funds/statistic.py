from .formulary import *


class Updater:
    @classmethod
    def everything(cls):
        # raw data fetch
        cls.crawler_idx()
        cls.crawler_stock()
        cls.crawler_finance()
        cls.crawler_macro()

    # region raw data
    @classmethod
    def crawler_macro(cls):
        Tuget.override_macros()
        Tuget.override_margins()
        Tuget.override_shibor()
        Tuget.override_category()

    @classmethod
    def crawler_finance(cls):
        DMgr.loop_stocks(N163.override_finance, 'financial_override', num_process = 12)

    @classmethod
    def crawler_stock(cls):
        DMgr.loop_stocks(N163.update_stock_hist, 'stock_update', num_process = 12)

    @classmethod
    def crawler_idx(cls):
        DMgr.loop_index(N163.update_idx_hist, 'index_update', num_process = 12)

    # endregion

    # region target
    @classmethod
    def targets_calculate(cls, target_list):
        def calc(code):
            # if os.path.getmtime(DMgr.csv_path('stock_target', code)) >= datetime(2018, 3, 18, 23,
            #         0).timestamp():
            #     print('%s jumped' % code)
            #     return
            stk = Stocks(code)
            stk.calc_list(target_list)
            stk.save_targets()

        DMgr.loop_stocks(calc, 'target_calc', show_seq = True, num_process = 7)

    @classmethod
    def targets_stock2cluster(cls, target_list):
        all_quarters = quarter_range()

        sort_method = {
            'max'        : lambda series: series.sort_values(ascending = True).index,
            'min'        : lambda series: series.sort_values(ascending = False).index,
            'minus_verse': minus_verse}

        def combine_stocks(code_list):
            comb = {}
            for code in code_list:
                df = DMgr.read('stock_target', code)
                if df is None:
                    continue
                df.index = df.quarter
                for target in target_list:
                    if target in df:
                        if target not in comb:
                            comb[target] = pd.DataFrame(index = all_quarters)
                        comb[target][code] = df[target]
            return comb

        def cluster_stat(target):
            def combine_dict(target):
                dfs = [dic[target] for dic in results]
                merged = None
                for df in dfs:
                    if merged is None:
                        merged = df
                    else:
                        merged = pd.merge(merged, df, left_index = True, right_index = True,
                            how = 'outer')
                return merged

            func_sort_idx = sort_method[Formula.target_trend[target]]

            df = combine_dict(target)
            df.dropna(axis = 0, how = 'all', inplace = True)
            percentile = pd.DataFrame(columns = df.columns + PERCENTILE)
            std_dis = pd.DataFrame(columns = df.columns + STD_DISTANCE)

            def row_normal_stat(row):
                series = row
                quarter = row.name
                val_count = (series == series).sum()
                if val_count < row.size / 10 or val_count < 20:
                    return
                # print(val_count)
                mu, sigma = normal_test(series,
                    '%s at %s with %s points' % (target, quarter, val_count))
                ids = [i for i in range(val_count)]
                na = [np.nan for i in range(series.size - val_count)]

                sorted_index = func_sort_idx(series)
                ids = pd.Series(ids + na, index = sorted_index + PERCENTILE)
                percentile.loc[quarter] = ids / val_count
                sd = (series - mu) / sigma
                sd.index = series.index + STD_DISTANCE
                std_dis.loc[quarter] = sd

            df.apply(row_normal_stat, axis = 1)
            df = pd.merge(df, percentile, how = 'left', left_index = True, right_index = True)
            df = pd.merge(df, std_dis, how = 'left', left_index = True, right_index = True)
            DMgr.save(df, 'cluster_target', target)

        n = 7
        arr_list = np.array_split(DMgr.code_list, n)
        results = loop(combine_stocks, arr_list, flag = 'stock_target_combine', num_process = n)

        m = min(len(target_list), 7)
        loop(cluster_stat, target_list, num_process = m, flag = 'cluster_target_stat')

    @classmethod
    def targets_cluster2stock(cls, target_list):
        """save market-wide statistic result back to stocks' target table
        :return:
        """

        target_dfs = DMgr.read('cluster_target', target_list, DWash.idx_by_quarter)

        def cluster_separate_by_code(code):
            comb = pd.DataFrame()
            for target in target_dfs:
                for suf in [PERCENTILE, STD_DISTANCE]:
                    source_col = code + suf
                    dest_col = target + suf
                    if source_col not in target_dfs[target]:
                        continue
                    comb[dest_col] = target_dfs[target][source_col]

            comb.dropna(axis = 0, how = 'all', inplace = True)
            comb['quarter'] = comb.index
            df = DMgr.read('stock_target', code)
            df = pd.merge(df, comb, on = 'quarter', how = 'left')
            DMgr.save(df, 'stock_target', code)

        DMgr.loop_stocks(cluster_separate_by_code, 'cluster_separate', show_seq = True,
            num_process = 7)

    # endregion


class Stat:
    @classmethod
    def test(cls):
        sd_sets = {}
        for target in ['ROE', 'ROA', 'ROC', 'ROIC', 'RNOA']:
            mus = []
            sds = []
            df = DMgr.read('cluster_target', target)
            for code in DMgr.code_list:
                if code in df:
                    se = df[code]
                    se = se[se == se]
                    os = se.size
                    se = se[(se != np.inf) & (se != -np.inf)]
                    afs = se.size
                    if afs != os:
                        print(target, code, os - afs)
                    if se.size > 3:
                        mu = se.mean()
                        if mu > 10:
                            continue
                            print(target, code, se)
                        sd = statistics.stdev(se)
                        if sd != sd:
                            print(code, se)
                        mus.append(mu)
                        sds.append(sd)
            # print(list)
            sd_sets[target] = [len(mus), statistics.mean(mus), statistics.mean(sds)]
        for key in sd_sets:
            pNum(key, sd_sets[key])


class Analysis:
    def __init__(self, code):
        self.code = code
        self.stat = DMgr.read('main_select', code)

    def plot(self):
        cols = [x for x in self.stat if PERCENTILE in x]
        self.stat.plot(kind = 'line', y = cols)
        plt.show()


class Strategy:
    pass
