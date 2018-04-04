# -*- coding: utf-8 -*-
"""
Formula calculate the indicators( targets in code) defined in formula.csv

"""

import math
import statistics
from functools import reduce

import numba

from Basic.Ext3PL import brief_detail_merge, column_duplicate_remove_inplace, fill_miss, \
    idx_by_quarter, \
    minus_verse, normal_test, reduce2brief, truncate_period
from Basic.Util import ClassProperty, INTERVAL_ORDER, SingletonMeta, now, quarter_range, \
    std_date_str
from .dataio import *

KPI = 'kpi'
POLICY = 'policy'
TICK = 'ticks_daily'
REPORT = 'financial_report_quarterly'
FORMULA_TARGET = 'calculate_from_formula'
PERCENTILE = '_percentile'
STD_DISTANCE = '_sd'
CAPITAL_COST = .08
LONG_TERM_RETURN = .12
INDEX_FLAG = 'Index_'
RESERVED_KEYWORDS = ['math', 'log', 'log10']
INDEX_DICT = {
    'ALL':   ['all', '000001'],
    'HS300': ['hs300', '399300'],
    'SZ50':  ['sz50', '000016'],
    'ZZ500': ['zz500', '000905']}


# todo Dupont detail into ATO
def discount(year):
    return 1 / math.pow(1 + CAPITAL_COST, year)


class Macro(object):
    table_periods = {
        'money_supply':         'month',
        'gdp':                  'year',
        'social_finance_scale': 'year',
        'shibor':               'date'}

    @classmethod
    def macro_table(cls, interval='quarter'):
        result = None
        for key in cls.table_periods:
            print(key)
            df_interval = cls.table_periods[key]
            df = DMGR.read('macro', key)
            if result is None:
                result = df
            else:
                if INTERVAL_ORDER[interval] < INTERVAL_ORDER[df_interval]:
                    brief = result
                    brief_col = interval
                    detail = df
                    detail_col = df_interval
                    if_reduce2brief = True
                else:
                    brief = df
                    brief_col = df_interval
                    detail = result
                    detail_col = interval
                    if_reduce2brief = False
                result = brief_detail_merge(brief, detail, if_reduce2brief=if_reduce2brief,
                                            brief_col=brief_col, detail_col=detail_col)
        return result


# region reduce methods

@numba.jit
def increase(arr):
    return arr[-1] - arr[0]


@numba.jit
def change(arr):
    return arr[-1] / arr[0] - 1 if arr[0] != 0 else 0


@numba.jit
def vix_yearly(arr):
    return (sum(arr ** 2) * 252 / len(arr)) ** 0.5


@numba.jit
def percent_geometric(arr):
    return -1 + reduce(lambda accumulated, x: accumulated * (1 + x), arr, 1) ** (1 / len(arr))


@numba.jit
def decline_avg(arr):
    return reduce(lambda accumulated, x: (accumulated + x) * 2 ** (-1 / 3), arr, 0)


# endregion

class _FinancialFormula(metaclass=SingletonMeta):
    REDUCE_METHODS = {
        'sum':               sum,
        'sd':                statistics.stdev,
        'mean':              np.mean,
        'vix_yearly':        vix_yearly,
        'percent_geometric': percent_geometric,
        'change':            change,
        'inc':               increase,
        'decline_avg':       decline_avg}

    STD_DURATION = {
        'year':    5,
        'quarter': 2}
    If_Debug = False

    @staticmethod
    def drop_tick_target(df):
        for idx in FORMULA.sub_targets[TICK]:
            if idx in df:
                df = df.drop(idx, axis=1)
        return df

    @property
    def cluster_targets(self):
        return self.sub_targets[KPI] + self.sub_targets[POLICY]

    def __init__(self):
        # get table-field relation pairs for further use, table of *_indicator excluded
        self._table_fields = {row.field: row.table for i, row in DWASH.mapper.iterrows() if
                              '_in' not in row.table}

        def _organise_table():
            table = get_lib('formula')
            table.index = table.target
            eqt_paras = {}
            sub_targets = {
                KPI:    [],
                POLICY: [],
                TICK:   []}
            target_favor_trend = {}

            paras = r'([A-Za-z_]\w*)'
            reg_factor = re.compile(paras)
            for target, formula in table.iterrows():
                if formula.source == TICK:
                    sub_targets[TICK].append(target)
                fields = reg_factor.findall(formula.equation)
                fields = [x for x in fields if x not in RESERVED_KEYWORDS]
                # get all fields included in equation
                eqt_paras[target] = list(set(fields))

                if formula.trend == formula.trend:
                    policy_flag = sum([1 for x in fields if x.endswith(PERCENTILE)])
                    if policy_flag > 1:
                        sub_targets[POLICY].append(target)
                    else:
                        sub_targets[KPI].append(target)
                    target_favor_trend[target] = formula.trend

                # tables of equation's fields
                tables = [self._table_fields[x] for x in fields if x in self._table_fields]
                tables = list(set(tables))
                source = FORMULA_TARGET
                # judge tables refered by equation
                if tables:
                    if 'history' in tables:
                        if len(tables) == 1:
                            source = TICK
                        else:
                            source = 'mixed'
                    else:
                        source = REPORT
                table.ix[target, 'source'] = source

            return table, eqt_paras, sub_targets, target_favor_trend

        self.table, self.eqt_paras, self.sub_targets, self.target_favor_trend = _organise_table()

        self.formula_by_type = {
            'stock': self.table[self.table.target.apply(lambda val: INDEX_FLAG not in val)],
            'index': self.table[self.table.target.apply(lambda val: INDEX_FLAG in val)]}

        stock_non_value_targets = [tar for tar in self.formula_by_type['stock'].target if
                                   tar not in ['E', 'FI']]
        percentile_targets = [p + PERCENTILE for p in self.sub_targets[KPI]]
        sd_targets = [s + STD_DISTANCE for s in self.sub_targets[KPI]]
        self.target_savings = stock_non_value_targets + percentile_targets + sd_targets

    def table_of(self, field):
        """return the name of the table contains field"""
        if field in self._table_fields:
            return self._table_fields[field]
        return FORMULA_TARGET

    def calculate(self, data, formula, prelude, target_calculator):
        """calculate formula based on data, either ticks or quarter report
        :param data: data frame used to calculate formula
        :param formula:
        :param prelude:
        :param target_calculator: callback func to get the target from upper level object
        :return: series contains formula result
        """

        target = formula.target

        def _prelude():
            """ prepare fields and corresponding columns from formula
            :return: updated equation and fields
            """
            pre_eqt = formula.equation
            pre_fields = self.eqt_paras[formula.target]
            data_fields = [x for x in pre_fields if FORMULA.table_of(x) != FORMULA_TARGET]
            numeric_inplace(data, include=data_fields)

            if self.If_Debug:
                print(formula.target)
            if prelude != prelude:
                for field in pre_fields:
                    if self.table_of(field) == FORMULA_TARGET:
                        if not field.endswith(PERCENTILE):
                            target_calculator(field)
                        elif field not in data:
                            print(target, field, 'percentile field not offered!')
                            return None, None
                return pre_eqt, pre_fields

            def target_calc(df, field_name, new_field_name, periods):
                """calculate target first, then ttm as required"""
                df[new_field_name] = target_calculator(field_name, prelude)
                # ttm(df, field_name, new_field_name, periods)

            def balance(df, field_name, new_field_name, periods):
                df[field_name].fillna(0, inplace=True)
                df[new_field_name] = df[field_name].rolling(periods).apply(
                        lambda li: (li[0] + li[-1]) / 2)

            def ttm(df, field_name, new_field_name, periods):
                df[new_field_name] = DWASH.ttm_column(df, field_name, new_column=new_field_name,
                                                      n=periods)

            tb2prelude = {
                FORMULA_TARGET: target_calc,
                'history':      None,
                'balance':      balance,
                'income':       ttm,
                'cash_flow':    ttm}
            num_quarters = self.STD_DURATION[prelude]
            new_fields = []
            # print(target, pre_eqt, pre_fields, data_source.shape)
            for field in pre_fields:
                if field in new_fields:
                    continue
                table = self.table_of(field)
                if table == 'history':
                    new_fields.append(field)
                else:
                    new_field = field + TTM_SUFFIX + str(num_quarters)
                    # for target, force re-calculate
                    if new_field not in data:
                        tb2prelude[table](data, field, new_field, num_quarters)
                    new_fields.append(new_field)
                    pre_eqt = pre_eqt.replace(field, new_field)
            if self.If_Debug:
                print(data[[*pre_fields, *new_fields]])
            return pre_eqt, new_fields

        eqt, fields = _prelude()
        if eqt is None:
            return None

        eqt_series = data.eval(eqt, parser='pandas')
        if formula.finale != formula.finale:
            result = eqt_series
        else:
            window_size = int(formula.quarter) if formula.source != TICK else int(
                    formula.quarter * 60)
            result = eqt_series.rolling(window_size).apply(self.REDUCE_METHODS[formula.finale])
        return result


FORMULA = _FinancialFormula()


class Ticks(object):
    DATA_SOURCE = ''

    @property
    def target_source(self):
        return self.DATA_SOURCE + '_target'

    @property
    def formulas(self):
        return FORMULA.formula_by_type[self.DATA_SOURCE]

    @property
    def last_quarter(self):
        return self.report['quarter'].iloc[-1]

    def __init__(self, code, table_getter: callable, refer_index=None):
        def raw_data_retrieve():
            report, tick = table_getter()

            if report is None or report.empty or tick is None or tick.empty:
                # print('%s %s do not have data!' % (self.DATA_SOURCE, code))
                return None, None, True
            tick.date = tick.date.apply(date_str2std)
            tick.index = tick.date
            tick['quarter'] = tick.date.apply(to_quarter)

            return report, tick, False

        self.code = code

        self.refer_index = refer_index

        self.report, self.tick, self.non_data = raw_data_retrieve()
        if self.non_data:
            return

        reduced = reduce2brief(self.tick)
        reduced = FORMULA.drop_tick_target(reduced)
        reduced = truncate_period(reduced, self.last_quarter)
        self.major = pd.merge(self.report, reduced, on='quarter', how='left')
        self.major.index = self.major.quarter

        targets = DMGR.read(self.target_source, code)
        if targets is not None and not targets.empty:
            self.major = pd.merge(self.major, targets, on='quarter', how='left')
            self.major.index = self.major.quarter

    def calc_list(self, target_list=None):
        if self.non_data:
            return
        target_list = target_list if target_list else FORMULA.sub_targets[KPI]
        for target in target_list:
            self.calc_target(target=target)

    def calc_target(self, formula=None, target: str = None, prelude=np.nan, if_renew=True):

        if self.non_data:
            return None
        if formula is None and target is None:
            raise Exception(' not specified at %s %s' % (self.DATA_SOURCE, self.code))
        if formula is not None:
            target = formula.target

        if target in self.major and not if_renew:
            return self.major[target]
        if target.endswith(PERCENTILE) and target not in self.major:
            print(self.code, 'no percentile data', target)
            return None
        formula = formula if formula is not None else FORMULA.table.loc[target]
        prelude = formula.prelude if prelude != prelude else prelude
        target_calculator = lambda tar, prelude=np.nan: \
            self.calc_target(target=tar, prelude=prelude, if_renew=True)
        if formula.source == TICK:
            # for daily tick, insert into self.tick first, then reduce to self.data
            if target not in self.formulas.index:
                self.refer_index.calc_target(target=target, prelude=prelude)
                tick_result = self.refer_index.tick[target]
                result = self.refer_index.major[target]
            else:
                tick_result = FORMULA.calculate(self.tick, formula, prelude, target_calculator)
                df = pd.DataFrame({
                    'date': tick_result.index,
                    target: tick_result.values})
                df = reduce2brief(df)
                df = fill_miss(df, self.major.index)
                df = truncate_period(df, self.major.index[-1])
                result = df[target]
            self.tick[target] = tick_result

        else:
            if target not in self.formulas.index:
                self.refer_index.calc_target(target=target, prelude=prelude)
                ref = self.refer_index.data[['quarter', target]]
                # df = fill_miss(ref, self.major.index, ifLabelingFilled = False)
                result = ref[target]
            else:

                result = FORMULA.calculate(self.major, formula, prelude, target_calculator)

        self.major[target] = result
        return result

    def save_targets(self, if_tick=True):
        if self.non_data:
            return
        cols2save = [x for x in FORMULA.target_savings if x in self.major]
        targets = self.major[cols2save]
        DMGR.save(targets, self.target_source, self.code)
        if if_tick:
            self.save_ticks()

    def save_ticks(self):
        DMGR.save(self.tick, self.DATA_SOURCE, self.code)

    def quarter_performance(self, start=None, end=None):
        return self.tick


class Indexes(Ticks):
    _hs300 = None
    DATA_SOURCE = 'index'

    @ClassProperty
    def hs300(self):
        if self._hs300 is None:
            self._hs300 = Indexes(*INDEX_DICT['HS300'])
        return self._hs300

    def __init__(self, label, code, if_update=False):
        def __get_table():
            tick = DMGR.read('index', self.code)
            column_duplicate_remove_inplace(tick)
            for col in ['market_cap', 'circulating_market_cap', 'net_profit']:
                if col in tick:
                    tick.drop(col, axis=1, inplace=True)
            tick.sort_values('date', inplace=True)
            tick.date = tick.date.apply(date_str2std)
            if if_update:
                report = DMGR.update_file('index', self.elements_quarter,
                                          self._fetch_element_report, index='quarter')

                daily = DMGR.update_file('index', self.elements_daily, self._fetch_element_tick)
            else:
                report = DMGR.read('index', self.elements_quarter)
                daily = DMGR.read('index', self.elements_daily)
            tick = pd.merge(tick, daily, on='date', how='left')
            return report, tick

        self.label = label
        if label == 'all':
            self.code_list = DMGR.code_list
        else:
            self.elements = DMGR.read('category', label)
            self.code_list = self.elements['code'].astype(str)
            self.code_list = self.code_list.apply(lambda val: val.zfill(6))

        self.elements_daily = code + '_daily'
        self.elements_quarter = code + '_quarterly'

        super().__init__(code, __get_table)

    def _fetch_element_report(self, start_date):
        all_fina = DMGR.category_concat(self.code_list, 'income', ['net_profit'], start_date)
        if all_fina.shape[0] == 0:
            return None
        all_fina['quarter'] = all_fina['date'].apply(to_quarter)
        all_fina = numeric_inplace(all_fina)
        profit_group = all_fina.groupby('quarter').agg({
            'quarter':    'max',
            'net_profit': 'sum'})
        profit_group['net_profit'] = DWASH.ttm_column(profit_group, 'net_profit')
        return profit_group

    def _fetch_element_tick(self, start_date):
        all_ticks = DMGR.category_concat(self.code_list, 'stock',
                                         ['market_cap', 'circulating_market_cap'], start_date)
        if all_ticks.empty:
            return None
        all_ticks = numeric_inplace(all_ticks)
        all_ticks = all_ticks.groupby('date').agg({
            'date':                   'max',
            "market_cap":             'sum',
            'circulating_market_cap': 'sum'})
        return all_ticks

    def calc_list(self, target_list=None):
        for target, formula in self.formulas.iterrows():
            self.calc_target(target=target)

    def quarter_performance(self, start=None, end=None):
        t = self.major['Index_ReturnQuarterly']
        return t
        # print(t)


class Stocks(Ticks):
    DATA_SOURCE = 'stock'
    TICK_SOURCE = 'lines'
    SIMPLE_COLUMNS = ['date', 'quarter', 'open', 'high', 'low', 'close', 'factor', 'derc_close',
                      'volume', 'market_cap', 'i_close', 'i_high', 'i_low', 'i_open']
    ACTIVE_TIME_LINE = std_date_str(today().year - 2, 1, 1)

    @classmethod
    def simplify_line(cls, code):
        df = DMGR.read('stock', code)
        df = DWASH.fill_derc(df)
        DMGR.save_csv(df, 'stock', code)
        if df is None or df.empty:
            return
        if date_str2std(df.iloc[-1].date) < cls.ACTIVE_TIME_LINE:
            print(code, f'last tick older than {cls.ACTIVE_TIME_LINE}, abort')
            return
        ticks = [x for x in FORMULA.sub_targets[TICK] if x in df]
        # if 'derc_close' not in df:

        df = df[cls.SIMPLE_COLUMNS + ticks]
        DMGR.save(df, 'lines', code)

    @classmethod
    def simplify_all_lines(cls, code_list=None):
        loop(cls.simplify_line, code_list, num_process=7)
        # DMGR.loop_stocks(cls.simplify_line, 'simple_line')

    @classmethod
    def target_pipeline(cls, target_list=None):
        target_list = target_list if target_list else FORMULA.sub_targets[KPI]
        cls.targets_calculate(target_list)
        cls.targets_stock2cluster(target_list)
        cls.targets_cluster_spread(target_list)
        cls.targets_cluster2stock(target_list)
        cls.targets_cluster2csv(target_list)

    # region target

    @classmethod
    def targets_calculate(cls, target_list=None, code_list=None):
        target_list = target_list if target_list else FORMULA.sub_targets[KPI]
        code_list = code_list if code_list else DMGR.code_list

        def calc(code):
            stk = Stocks(code)
            stk.calc_list(target_list)
            stk.save_targets()

        loop(calc, code_list, num_process=5)

    @classmethod
    def targets_stock2cluster(cls, target_list=None, code_list=None):
        target_list = target_list if target_list else FORMULA.sub_targets[KPI]
        if code_list:
            code_list = code_list
            list_tail = f'_{len(code_list)}'
        else:
            code_list = DMGR.code_list
            list_tail = ''

        all_quarters = quarter_range()

        sort_method = {
            'max':         lambda series: series.sort_values(ascending=True).index,
            'min':         lambda series: series.sort_values(ascending=False).index,
            'minus_verse': minus_verse}

        def combine_stocks(codes):
            comb = {}
            for code in codes:
                df = DMGR.read('stock_target', code)
                if df is None:
                    continue
                df.index = df.quarter
                for target in target_list:
                    if target in df:
                        if target not in comb:
                            comb[target] = pd.DataFrame(index=all_quarters)
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
                        merged = pd.merge(merged, sub_df, left_index=True, right_index=True,
                                          how='outer')
                return merged

            func_sort_idx = sort_method[FORMULA.target_favor_trend[target]]

            df = combine_dict(target)
            df.index.name = 'quarter'
            df.dropna(axis=0, how='all', inplace=True)
            percentile = pd.DataFrame(columns=df.columns + PERCENTILE)
            std_dis = pd.DataFrame(columns=df.columns + STD_DISTANCE)

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
                nan_ids = [np.nan for _ in range(series.size - val_count)]

                sorted_index = func_sort_idx(series)
                ids = pd.Series(ids + nan_ids, index=sorted_index + PERCENTILE)
                percentile.loc[quarter] = ids / val_count
                sd = (series - mu) / sigma
                sd.index = series.index + STD_DISTANCE
                std_dis.loc[quarter] = sd

            df.apply(row_normal_stat, axis=1)
            df = pd.merge(df, percentile, how='left', left_index=True, right_index=True)
            df = pd.merge(df, std_dis, how='left', left_index=True, right_index=True)
            DMGR.save(df, 'cluster_target' + list_tail, target)

        n = 5
        arr_list = np.array_split(code_list, n)
        results = loop(combine_stocks, arr_list, num_process=n)

        loop(cluster_stat, target_list, num_process=n, flag='_stat')

    @classmethod
    def targets_cluster2stock(cls, target_list=None, code_list=None):
        """save market-wide statistic result back to stocks' target table
        :return:
        """
        target_list = target_list if target_list else FORMULA.cluster_targets
        code_list = DMGR.code_list if code_list is None else code_list
        target_dfs = DMGR.read2dict('cluster_target', target_list, idx_by_quarter)

        def cluster_separate_by_code(code):
            df = DMGR.read('stock_target', code)
            if df is None or df.empty:
                print(code, 'does not have targets and abort!')
                return
            df.index = df.quarter
            # comb = pd.DataFrame()
            for target in target_dfs:
                for suf in [PERCENTILE, STD_DISTANCE]:
                    source_col = code + suf
                    destination_col = target + suf
                    if source_col not in target_dfs[target]:
                        continue
                    df[destination_col] = target_dfs[target][source_col]

            # comb.dropna(axis=0, how='all', inplace=True)
            # if comb.shape[0] == 0:
            #     print('cluster_separate: not data to separate', code)
            #     return
            # comb['quarter'] = comb.index
            # df = pd.merge(df, comb, on='quarter', how='left')
            DMGR.save(df, 'stock_target', code)

        loop(cluster_separate_by_code, code_list, num_process=5)

    @classmethod
    def targets_cluster_spread(cls, target_list=None):
        target_list = target_list if target_list else FORMULA.cluster_targets

        def spread(target):
            df = DMGR.read('cluster_target', target)
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
                column_transactions = {col: col.replace(PERCENTILE, '') for col in sub}
                # for col in sub:
                #     nc[col] = col.replace(PERCENTILE, '')
                sub.rename(columns=column_transactions, inplace=True)
                DMGR.save(sub, 'cluster_target', target + tail)

        loop(spread, target_list, num_process=4)

    @classmethod
    def targets_cluster2csv(cls, target_list=None):
        target_list = target_list if target_list else FORMULA.cluster_targets

        def transpose(code):
            for tail in ['', PERCENTILE, STD_DISTANCE]:
                df = DMGR.read('cluster_target', code + tail)
                df.index = df.quarter
                df.sort_index(ascending=False, inplace=True)
                df.drop('quarter', axis=1, inplace=True)
                df = df.T
                # df.reset_index(inplace=True)
                # DWASH.std_code_col_inplace(df)
                df = pd.merge(DMGR.code_details, df, left_index=True, right_index=True, how='inner')
                DMGR.save_csv(df, 'cluster_target', code + tail)

        loop(transpose, target_list, num_process=5)

    # endregion

    def __init__(self, code):
        def __get_table():
            def __no_read(flag):
                print(self.DATA_SOURCE, code, flag, 'is empty')
                return None, None

            report = None
            for table in ['balance', 'cash_flow', 'income']:
                df = DMGR.read(table, self.code)
                if df is not None and not df.empty:
                    df['quarter'] = df['date'].apply(to_quarter)
                    dup = df[df['quarter'].duplicated()]
                    if not dup.empty:
                        print(self.code, table, 'has duplicate index! Dropped!')
                        df.drop_duplicates('quarter', inplace=True)
                    # noinspection PyComparisonWithNone
                    df = df[df.quarter != None]
                    if report is None:
                        report = df
                    else:
                        report = pd.merge(report, df, on='quarter', suffixes=(
                            '', DWASH.DUPLICATE_SEPARATOR + table + DWASH.DUPLICATE_FLAG))
                else:
                    return __no_read(table)
            if report is not None and not report.empty:
                report = DWASH.column_select(report, code)
                report.index = report.quarter
                report.sort_index(inplace=True)
                # some column's dtype is read as object, change to number
                report = numeric_inplace(report)

            tick = DMGR.read(self.TICK_SOURCE, code)
            if tick is not None and not tick.empty:
                # hopefully only once, but actually have to keep
                tick = tick.drop_duplicates(subset='date', keep='first')
                tick = tick[tick.close != 0]
                tick = DWASH.calc_change(tick)
                tick.sort_values('date', inplace=True)
            else:
                return __no_read('tick')

            return report, tick

        super().__init__(code, __get_table, Indexes.hs300)

    def save_ticks(self):
        DMGR.save(self.tick, self.TICK_SOURCE, self.code)

    def _financial_compare(self):
        if self.non_data:
            return None
        self.calc_target(target='FinanceExpense')
        res = column_compare_choose_inplace(self.major, 'FinanceExpense', 'financial_expense')
        print(res)
        return res

    def rim(self, quarter=None):
        quarter = to_quarter(now()) if quarter is None else quarter
        quarter = min(self.last_quarter, quarter)
        fin = self.major
        quart = self.report.loc[quarter]

        def _roe_estimate():
            hist = fin[fin.index <= quarter].tail(5 * 4)
            print(hist.ROE)
            factor = DWASH.percentage_factor_by_values(hist.ROE)
            roe_local = hist.ROE.mean()
            return roe_local / factor

        def _value_observe(roe_in, equity):
            whole = 12
            high_speed = 5
            # low_speed = whole - high_speed
            for i in range(whole):
                if i < high_speed:
                    earn = roe_in
                else:
                    earn = LONG_TERM_RETURN

                dis = discount(i + 1)
                equity += equity * (earn - CAPITAL_COST) * dis
            return equity

        roe = _roe_estimate()
        val = _value_observe(roe, quart.total_owner_equities)
        return val, val / quart.total_owner_equities
