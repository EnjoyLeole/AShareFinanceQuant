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
    def target_pipeline(cls, target_list):
        cls.targets_calculate(target_list)
        cls.targets_stock2cluster(target_list)
        cls.targets_cluster2stock(target_list)

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
            'max':         lambda series: series.sort_values(ascending = True).index,
            'min':         lambda series: series.sort_values(ascending = False).index,
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
            def combine_dict(tar):
                dfs = [dic[tar] for dic in results]
                merged = None
                for sub_df in dfs:
                    if merged is None:
                        merged = sub_df
                    else:
                        merged = pd.merge(merged, sub_df, left_index = True, right_index = True,
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
                val_count = np.count_nonzero(series == series)
                if val_count < row.size / 10 or val_count < 20:
                    return
                # print(val_count)
                notice = '%s at %s with %s points' % (target, quarter, val_count)
                mu, sigma = normal_test(series, notice)
                ids = [i for i in range(val_count)]
                na = [np.nan for _ in range(series.size - val_count)]

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

        loop(cluster_stat, target_list, num_process = n, flag = 'cluster_target_stat')

    @classmethod
    def targets_cluster2stock(cls, target_list):
        """save market-wide statistic result back to stocks' target table
        :return:
        """

        target_dfs = DMgr.read('cluster_target', target_list, idx_by_quarter)

        def cluster_separate_by_code(code):
            comb = pd.DataFrame()
            for target in target_dfs:
                for suf in [PERCENTILE, STD_DISTANCE]:
                    source_col = code + suf
                    destination_col = target + suf
                    if source_col not in target_dfs[target]:
                        continue
                    comb[destination_col] = target_dfs[target][source_col]

            comb.dropna(axis = 0, how = 'all', inplace = True)
            comb['quarter'] = comb.index
            df = DMgr.read('stock_target', code)
            df = pd.merge(df, comb, on = 'quarter', how = 'left')
            DMgr.save(df, 'stock_target', code)

        DMgr.loop_stocks(cluster_separate_by_code, 'cluster_separate', num_process = 7)

    @classmethod
    def cluster_spread(cls, target_list):
        def spread(target):
            df = DMgr.read('cluster_target', target)
            df.index = df.quarter
            tail_dict = {}
            for col in df:
                for tail in [PERCENTILE, STD_DISTANCE]:
                    if col.endswith(tail):
                        if tail not in tail_dict:
                            tail_dict[tail] = []
                        tail_dict[tail].append(col)
            for tail in tail_dict:
                sub = df[tail_dict[tail]]
                nc = {}
                for col in sub:
                    nc[col] = col.replace(PERCENTILE, '')
                sub.rename(columns = nc, inplace = True)
                DMgr.save(sub, 'cluster_target', target + tail)

        loop(spread, target_list, num_process = 4, flag = 'cluster_target_spread')  # endregion


class Strategy:

    @classmethod
    def profit_measures(cls):
        sd_sets = {}
        for target in ['ROE', 'ROA', 'ROC', 'ROIC', 'RNOA']:
            mus = []
            sds = []
            df = DMgr.read('cluster_target', target)
            for code in DMgr.code_list:
                if code in df:
                    se = df[code]
                    se = se[se == se]
                    org_size = se.size
                    se = se[(se != np.inf) & (se != -np.inf)]
                    after_size = se.size
                    if after_size != org_size:
                        print(target, code, org_size - after_size)
                    if se.size > 3:
                        mu = se.mean()
                        if mu > 10:
                            continue  # print(target, code, se)
                        sd = statistics.stdev(se)
                        if sd != sd:
                            print(code, se)
                        mus.append(mu)
                        sds.append(sd)
            # print(list)
            sd_sets[target] = [len(mus), statistics.mean(mus), statistics.mean(sds)]
        for key in sd_sets:
            print_num(key, sd_sets[key])

    @classmethod
    def find_security(cls):
        df = DMgr.read('cluster_target', 'Policy' + PERCENTILE)
        df.index = df.quarter
        df.drop('quarter', axis = 1, inplace = True)
        df = df[df > 0.99]
        df.dropna(axis = 1, how = 'all', inplace = True)
        last = df.iloc[-1]
        last = last.dropna()
        last = last.reset_index()
        best = pd.merge(last, DMgr.code_table, how = 'left', left_on = 'index', right_on = 'code')
        print(df.shape[1], last.size, best)
