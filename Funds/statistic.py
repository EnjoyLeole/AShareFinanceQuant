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
        cls.all_stock_target_update()
        cls.all_stock_target_cluster()
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
    def all_stock_target_update(cls):
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
    def all_stock_target_cluster(cls, if_by_row = True):
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
            DMgr.save_csv(df, 'target', target)

    @classmethod
    def all_cluster_target_stat(cls):
        loop(cls.cluster_target_stat, Formula.key_targets, num_process = 4,
            flag = 'cluster_target_stat')

    @classmethod
    def cluster_target_stat(self, target, quarter = None):
        df = DMgr.read_csv('target', target)
        df.index = df['quarter']
        df.drop('quarter', axis = 1, inplace = True)
        df = df.T
        df.dropna(axis = 1, how = 'all', inplace = True)
        for col in df:
            series = df[col]
            val_count = (series == series).sum()
            if val_count < df.shape[0] / 10:
                continue

            mu, sigma = normal_test(series, target, col)
            ids = [i for i in range(val_count)]
            na = [np.nan for i in range(series.size - val_count)]

            sorted = series.sort_values()
            ids = pd.Series(ids + na, index = sorted.index)
            # sorted[PERCENTILE] = 1
            # sorted_ids=sorted.reset_index()
            df[col + PERCENTILE] = ids / val_count
            df[col + STD_DISTANCE] = (series - mu) / sigma

        df['code'] = df.index
        DMgr.save_csv(df, 'target', target + TRANSPOSE)


class Analysis:
    def __init__(self):
        pass


class _Cluster(metaclass = SingletonMeta):

    def __init__(self, code_list):
        self.code_list = code_list

        # print(df)


Cluster = _Cluster(DMgr.code_list)
