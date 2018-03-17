from .formulary import *


class _Cluster(metaclass = SingletonMeta):

    def __init__(self, code_list):
        self.code_list = code_list

    def stock2target(self):
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
                    if target not in clusters:
                        clusters[target] = pd.DataFrame(index = all_quarters)
                        clusters[target]['quarter'] = clusters[target].index
                    clusters[target][code] = df[target]

        for target in clusters:
            df = clusters[target]
            df.dropna(axis = 0, how = 'all', inplace = True)
            DMgr.save_csv(df, 'target', target)

    # def stat(self,target,quarter=None):


Cluster = _Cluster(DMgr.code_list)
