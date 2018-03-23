from .formulary import *


class Updater:
    @classmethod
    def everything(cls):
        # raw data fetch
        cls.all_idx()
        cls.all_stock()
        cls.all_finance()
        cls.all_macro()

        # target calc
        cls.all_target_update()
        cls.all_target_cluster()
        cls.all_cluster_separate()

    # region raw data
    @classmethod
    def all_macro(cls):
        Tuget.override_macros()
        Tuget.override_margins()
        Tuget.override_shibor()
        Tuget.override_category()

    @classmethod
    def all_finance(cls):
        DMgr.loop_stocks(N163.override_finance, 'financial_override', num_process = 12)

    @classmethod
    def all_stock(cls):
        DMgr.loop_stocks(N163.update_stock_hist, 'stock_update', num_process = 12)

    @classmethod
    def all_idx(cls):
        DMgr.loop_index(N163.update_idx_hist, 'index_update', num_process = 12)

    # endregion

    # region target
    @classmethod
    def all_target_update(cls):
        def calc(code):
            # if os.path.getmtime(DMgr.csv_path('stock_target', code)) >= datetime(2018, 3, 18, 23,
            #         0).timestamp():
            #     print('%s jumped' % code)
            #     return
            # print(code + ' start')
            stk = Stocks(code)
            res = stk.calc_all_vector()
            if res is not None:
                stk.save_targets()
            # print(code + ' saved')

        DMgr.loop_stocks(calc, 'target_calc', show_seq = True, num_process = 7)

    @classmethod
    def all_target_cluster(cls):
        clusters = {}
        all_quarters = quarter_range()

        for code in DMgr.code_list:
            print(code)
            # if code >= '000109':
            #     break

            df = DMgr.read_csv('stock_target', code)
            if df is None:
                continue
            df.index = df.quarter
            for target in Formula.key_targets:
                if target in df:
                    if target not in clusters:
                        clusters[target] = pd.DataFrame(index = all_quarters)
                    clusters[target][code] = df[target]



        sort_method = {
            'max'        : lambda series: series.sort_values(ascending = True).index,
            'min'        : lambda series: series.sort_values(ascending = False).index,
            'minus_verse': minus_verse}

        def cluster_stat(target):
            df = clusters[target]
            df.dropna(axis = 0, how = 'all', inplace = True)
            percentile = pd.DataFrame(columns = df.columns + PERCENTILE)
            std_dis = pd.DataFrame(columns = df.columns + STD_DISTANCE)

            func_sort_idx = sort_method[Formula.key_targets[target]]

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
            df['quarter'] = df.index
            DMgr.save_csv(df, 'cluster_target', target)

        loop(cluster_stat, list(Formula.key_targets.keys()), num_process = 7,
            flag = 'cluster_target_stat', show_seq = True)

    @classmethod
    def all_cluster_separate(cls):
        targets = cls.fetch_all_cluster_target_stat()

        def cluster_separate_by_code(code):
            comb = pd.DataFrame()
            for target in Formula.key_targets:
                for suf in ['', PERCENTILE, STD_DISTANCE]:
                    source_col = code + suf
                    dest_col = target + suf
                    if source_col not in targets[target]:
                        continue
                    comb[dest_col] = targets[target][source_col]

            comb.dropna(axis = 0, how = 'all', inplace = True)
            comb['quarter'] = comb.index
            DMgr.save_csv(comb, 'target_stock', code)
            # return comb

        DMgr.loop_stocks(cluster_separate_by_code, 'cluster_separate', show_seq = True,
            num_process = 7)

    # endregion

    @classmethod
    def fetch_all_cluster_target_stat(cls):
        targets = {}
        for target in Formula.key_targets:
            df = DMgr.read_csv('cluster_target', target)
            df.index = df.quarter
            targets[target] = df
        return targets


class Stat:
    @classmethod
    def test(cls):
        sd_sets = {}
        for target in ['ROE', 'ROA', 'ROC', 'ROIC', 'RNOA']:
            mus = []
            sds = []
            df = DMgr.read_csv('cluster_target', target)
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
        self.stat = DMgr.read_csv('main_select', code)

    def plot(self):
        cols = [x for x in self.stat if PERCENTILE in x]
        self.stat.plot(kind = 'line', y = cols)
        plt.show()


class Strategy:
    pass
