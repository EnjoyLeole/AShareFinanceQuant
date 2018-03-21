from .formulary import *

PERCENTILE = '_percentile'
STD_DISTANCE = '_std_distance'
TRANSPOSE = '_T'


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
        cls.all_cluster_target_stat()

    # region raw data
    @classmethod
    def all_macro(cls):
        Tuget.override_macros()
        Tuget.override_margins()
        Tuget.override_shibor()
        Tuget.override_category()

    @classmethod
    def all_finance(cls):
        DMgr.iter_stocks(N163.override_finance, 'financial_override', num_process = 12)

    @classmethod
    def all_stock(cls):
        DMgr.iter_stocks(N163.update_stock_hist, 'stock_update', num_process = 12)

    @classmethod
    def all_idx(cls):
        DMgr.iter_index(N163.update_idx_hist, 'index_update', num_process = 12)

    # endregion

    @classmethod
    def all_target_update(cls):
        def calc(code):
            if os.path.getmtime(DMgr.csv_path('stock_target', code)) >= datetime(2018, 3, 18, 23,
                    0).timestamp():
                print('%s jumped' % code)
                return
            # print(code + ' start')
            stk = Stocks(code)
            res = stk.calc_all_vector()
            if res is not None:
                stk.save_targets()
            # print(code + ' saved')

        DMgr.iter_stocks(calc, 'target_calc', show_seq = True, num_process = 7)

    @classmethod
    def all_target_cluster(cls, if_by_row = True):
        axis = 0 if if_by_row else 1
        clusters = {}
        all_quarters = quarter_range()

        for code in DMgr.code_list:
            print(code)
            df = DMgr.read_csv('stock_target', code)
            if df is None:
                continue
            df.index = df.quarter
            for target in Formula.key_targets:
                if target in df:
                    if axis == 0:
                        if target not in clusters:
                            clusters[target] = pd.DataFrame(index = all_quarters)
                        clusters[target][code] = df[target]
                    else:
                        def get_selected():
                            selected = df[[target]].T
                            selected.index = [code]
                            return selected

                        if target not in clusters:
                            # clusters[target] = pd.DataFrame()

                            clusters[target] = get_selected()
                            # print(clusters)
                        else:
                            clusters[target] = pd.concat(
                                [clusters[target], get_selected()])
                            print(clusters)

        for target in clusters:
            df = clusters[target]
            df.dropna(axis = axis, how = 'all', inplace = True)
            if axis == 0:
                clusters[target]['quarter'] = clusters[target].index
            else:
                clusters[target]['code'] = clusters[target].index
            DMgr.save_csv(df, 'cluster_target', target)

    @classmethod
    def all_cluster_target_stat(cls):
        loop(cls.cluster_target_stat, Formula.key_targets, num_process = 6,
            flag = 'cluster_target_stat', show_seq = True)

    @classmethod
    def cluster_target_stat(self, target):
        df = DMgr.read_csv('cluster_target', target)
        df.index = df['quarter']
        df.drop('quarter', axis = 1, inplace = True)
        # df = df.T
        df.dropna(axis = 0, how = 'all', inplace = True)

        def row_sub(row):
            series = row
            quarter = row.name
            val_count = (series == series).sum()
            if val_count < row.size / 10:
                return

            mu, sigma = normal_test(series,
                '%s at %s with %s points' % (target, quarter, val_count))
            ids = [i for i in range(val_count)]
            na = [np.nan for i in range(series.size - val_count)]

            sorted = series.sort_values()
            ids = pd.Series(ids + na, index = sorted.index)
            df.loc[quarter + PERCENTILE] = ids / val_count
            df.loc[quarter + STD_DISTANCE] = (series - mu) / sigma

        df.apply(row_sub, axis = 1)
        df['quarter'] = df.index
        DMgr.save_csv(df, 'cluster_target', target + TRANSPOSE)

    @classmethod
    def cluster_target_stat_transpose(self, target):
        df = DMgr.read_csv('cluster_target', target)
        df.index = df['quarter']
        df.drop('quarter', axis = 1, inplace = True)
        df = df.T
        df.dropna(axis = 1, how = 'all', inplace = True)
        for quarter in df:
            series = df[quarter]
            val_count = (series == series).sum()
            if val_count < df.shape[0] / 10:
                continue

            mu, sigma = normal_test(series,
                '%s at %s with %s points' % (target, quarter, val_count))
            ids = [i for i in range(val_count)]
            na = [np.nan for i in range(series.size - val_count)]

            sorted = series.sort_values()
            ids = pd.Series(ids + na, index = sorted.index)
            df[quarter + PERCENTILE] = ids / val_count
            df[quarter + STD_DISTANCE] = (series - mu) / sigma

        df['code'] = df.index
        DMgr.save_csv(df, 'cluster_target', target + TRANSPOSE)


class _Analysis:
    def __init__(self):
        # self.fetch_all_cluster_target_stat()
        pass

    def fetch_all_cluster_target_stat(self):
        targets = {}
        for target in Formula.key_targets:
            df = DMgr.read_csv('cluster_target', target + TRANSPOSE)
            df.index = df.quarter
            targets[target] = df
        return targets

    def cluster_separate_by_code(self, code, quarter = None):
        comb = pd.DataFrame()
        targets=self.fetch_all_cluster_target_stat()
        for target in Formula.key_targets:
            if quarter is None:
                comb[target] = targets[target][code]
            else:
                for idx in [quarter, quarter + PERCENTILE, quarter + STD_DISTANCE]:
                    comb.loc[idx, target] = self.targets[target].at[idx, code]
        comb['quarter'] = comb.index
        DMgr.save_csv(comb, 'main_select', code)
        return comb


Analysis = _Analysis()
