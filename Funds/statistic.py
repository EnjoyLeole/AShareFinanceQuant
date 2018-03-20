from .formulary import *

PERCENTILE = '_percentile'
STD_DISTANCE = '_std_distance'
TRANSPOSE = '_T'


class _Cluster(metaclass = SingletonMeta):

    def __init__(self, code_list):
        self.code_list = code_list

    def stock2target(self, if_by_row = True):
        axis = 0 if if_by_row else 1
        clusters = {}
        all_quarters = quarter_range()

        for code in self.code_list:
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

    def stat(self, target, quarter = None):
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
            sorted = series.sort_values()
            # df[col+PERCENTILE]=
            df[col + STD_DISTANCE] = (series - mu) / sigma

        df['code'] = df.index
        DMgr.save_csv(df, 'target', target + TRANSPOSE)

        # print(df)


Cluster = _Cluster(DMgr.code_list)
