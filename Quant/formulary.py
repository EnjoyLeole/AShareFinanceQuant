import math
import statistics
import warnings
from functools import reduce

import numba

from .webio import *

CAPITAL_COST = .08
LONG_TERM_RETURN = .12
INDEX_FLAG = 'Index_'
RESERVED_KEYWORDS = ['math', 'log', 'log10']
INDEX_DICT = {
    'ALL':   ['all', '000001'],
    'HS300': ['hs300', '399300'],
    'SZ50':  ['sz50', '000016'],
    'ZZ500': ['zz500', '000905']}
HS300 = None


# todo Dupt  detail into ATO
def discount(year):
    return 1 / math.pow(1 + CAPITAL_COST, year)


class Macro(object):
    table_periods = {
        'money_supply':         'month',
        'gdp':                  'year',
        'social_finance_scale': 'year',
        'shibor':               'date'}

    @classmethod
    def macro_table(cls, interval = 'quarter'):
        result = None
        for key in cls.table_periods:
            print(key)
            df_interval = cls.table_periods[key]
            df = DMgr.read('macro', key)
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
                result = brief_detail_merge(brief, detail, if_reduce2brief = if_reduce2brief,
                                            brief_col = brief_col, detail_col = detail_col)
        return result

    # @classmethod  # def cap_base(cls):  #     cap = DMgr.read_csv('temp', 'cap')  #     cap[  #
    #  'year'] = cap['date'].apply(lambda x: str2date(x).strftime('%Y'))  #  #     gdp =   #  #
    #  DMgr.read_csv('macro', 'gdp').ix[:, ['year', 'gdp']]  #     gdp['year'] = gdp[  #  #  #
    # 'year'].astype('str')  #  #     sfs = DMgr.read_csv('macro', 'social_finance_scale')  #  #
    #  sfs['year'] = sfs['year'].astype('str')  #     print(sfs.dtypes)  #     merged = pd.merge(
    #  cap, sfs, on = 'year')  #     merged['flow_cap_sfs'] = merged['circulating_market_cap'] /
    #  merged['social_finance_scale']  #     DMgr.save_csv(merged, 'temp', 'cap_sfs')

    # @classmethod  # def market_pe(cls):  #     mp = DMgr.read_csv('temp', 'agg_profit')  #  #
    #  cap = DMgr.read_csv('temp', 'cap')  #  #     merged = pd.merge(cap, mp, on = 'quarter')  #
    #     print(merged)  #     shibor = DMgr.read_csv('macro', 'shibor')  #     # shibor[  #  #
    #  'quarter'] = shibor['date'].apply(to_quarter)  #  #     merged = pd.merge(shibor, merged,
    #  on = 'date')  #     merged['profit/cap'] = merged['net_profit_ttm'] * 10000 / merged[  #
    #  'market_cap']  #     merged['interest/shibor'] = merged['profit/cap'] / merged['1Y'] * 100
    #     DMgr.save_csv(merged, 'temp', 'market_pe')


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

class _FinancialFormula(metaclass = SingletonMeta):
    REDUCE_METHODS = {
        'sum':               sum,
        'sd':                statistics.stdev,
        'mean':              lambda arr: arr.mean(),
        'vix_yearly':        vix_yearly,
        'percent_geometric': percent_geometric,
        'change':            change,
        'inc':               increase,
        'decline_avg':       decline_avg}

    STD_DURATION = {
        'year':    5,
        'quarter': 2}

    def __init__(self):
        # get table-field relation pairs for further use, table of *_indicator excluded
        self._table_fields = {row.field: row.table for i, row in DWash.mapper.iterrows() if
                              '_in' not in row.table}

        def _organise_table():
            table = get_lib('formula')
            table.index = table.target
            factor_dict = {}
            key_targets = []
            tick_targets = []
            policy_targets = []
            target_trend = {}
            paras = '([A-Za-z_]\w*)'
            reg_factor = re.compile(paras)
            for target, formula in table.iterrows():
                if formula.source == TICK:
                    tick_targets.append(target)
                fs = reg_factor.findall(formula.equation)
                fs = [x for x in fs if x not in RESERVED_KEYWORDS]
                # get all fields included in equation
                factor_dict[target] = list(set(fs))

                if formula.trend == formula.trend:
                    policy_flag = sum([1 for x in fs if x.endswith(PERCENTILE)])
                    if policy_flag > 1:
                        policy_targets.append(target)
                    else:
                        key_targets.append(target)
                    target_trend[target] = formula.trend

                # tables of equation's fields
                tbs = [self._table_fields[x] for x in fs if x in self._table_fields]
                tbs = list(set(tbs))
                source = 'NotInData'
                # judge tables refered by equation
                if len(tbs) > 0:
                    if 'history' in tbs:
                        if len(tbs) == 1:
                            source = TICK
                        else:
                            source = 'mixed'
                    else:
                        source = REPORT
                table.ix[target, 'source'] = source

            return table, factor_dict, key_targets, policy_targets, tick_targets, target_trend

        self.table, self.equation_parameters, self.key_targets, self.polices, self.tick_targets, \
        self.target_trend = _organise_table()

        self.formula_by_type = {
            'stock': self.table[self.table.target.apply(lambda val: INDEX_FLAG not in val)],
            'index': self.table[self.table.target.apply(lambda val: INDEX_FLAG in val)]}

        self.target_savings = [tar for tar in self.formula_by_type['stock'].target if
                               tar not in ['E', 'FI']] + [p + PERCENTILE for p in
                                                          self.key_targets] + [s + STD_DISTANCE for
                                                                               s in
                                                                               self.key_targets]

    def table_of(self, field):
        """return the name of the table contains field"""
        if field in self._table_fields:
            return self._table_fields[field]
        return 'NotInData'

    def calculate(self, data, formula, prelude, target_calculator):
        """calculate formula based on data, either ticks or quarter report
        :param data: data frame used to calculate formula
        :param formula:
        :param prelude:
        :param target_calculator: callback func to get the target from upper level object
        :return: series contains formula result
        """
        if_debug = False

        target = formula.target
        if target in data:
            return data[target]
        else:
            def _prelude():
                """ prepare fields and corresponding columns from formula
                :return: updated equation and fields
                """
                pre_eqt = formula.equation
                pre_fields = self.equation_parameters[formula.target]
                if if_debug:
                    print(formula.target)
                # print(data[fields])
                if prelude != prelude:
                    for field in pre_fields:
                        if self.table_of(field) == 'NotInData':
                            if field.endswith(PERCENTILE):
                                if field not in data:
                                    print(target, field, 'percentile field not offered!')
                                    return None, None
                            else:
                                target_calculator(field)
                    return pre_eqt, pre_fields
                else:
                    def target_calc(df, field_name, new_field_name, n):
                        _ = df, new_field_name, n
                        target_calculator(field_name)

                    def balance(df, field_name, new_field_name, n):
                        df[new_field_name] = df[field_name].rolling(n).apply(
                                lambda li: (li[0] + li[-1]) / 2)

                    def ttm(df, field_name, new_field_name, n):
                        df[new_field_name] = DWash.ttm_column(df, field_name,
                                                              new_column = new_field_name, n = n)

                    tb2prelude = {
                        'NotInData': target_calc,
                        'history':   None,
                        'balance':   balance,
                        'income':    ttm,
                        'cash_flow': ttm}
                    num_quarters = self.STD_DURATION[prelude]
                    new_fields = []
                    for field in pre_fields:
                        if field in new_fields:
                            continue
                        tb = self.table_of(field)
                        func = tb2prelude[tb]
                        if func is None:
                            new_fields.append(field)
                        else:
                            if tb == 'NotInData':
                                new_field = field
                            else:
                                new_field = field + TTM_SUFFIX + str(num_quarters)
                            if new_field not in data:
                                func(data, field, new_field, num_quarters)
                            new_fields.append(new_field)
                            pre_eqt = pre_eqt.replace(field, new_field)
                    return pre_eqt, new_fields

            eqt, fields = _prelude()
            if eqt is None:
                return
            if if_debug:
                print(data[fields])
            eqt_series = data.eval(eqt, parser = 'pandas')
            if formula.finale != formula.finale:
                result = eqt_series
            else:
                window_size = int(formula.quarter) if formula.source != TICK else int(
                        formula.quarter * 90)
                result = eqt_series.rolling(window_size).apply(self.REDUCE_METHODS[formula.finale])
        return result

    @staticmethod
    def drop_tick_target(df):
        for idx in Formula.tick_targets:
            if idx in df:
                df = df.drop(idx, axis = 1)
        return df


Formula = _FinancialFormula()


class Ticks(object):
    If_Renew = True

    def __init__(self, code, table_getter: callable, security_type: str, refer_index = None):
        def raw_data_retrieve():
            report, tick = table_getter()
            if report is None or report.shape[0] == 0 or tick is None or tick.shape[0] == 0:
                self.formulas = None
                warnings.warn('%s %s do not have data!' % (security_type, code))
                self.NonData = True
                return None, None
            tick.date = tick.date.apply(date_str2std)
            tick.index = tick.date
            tick['quarter'] = tick.date.apply(to_quarter)

            return report, tick

        def saved_target_retrieve():
            targets_table = DMgr.read(self.target_category, code)
            if targets_table is None or self.If_Renew:
                targets_table = pd.DataFrame(index = self.report.index)
                targets_table['quarter'] = targets_table.index
            else:
                targets_table.index = targets_table.quarter
            return targets_table

        self.code = code
        self.type = security_type
        self.formulas = Formula.formula_by_type[self.type]
        self.refer_index = refer_index

        self.NonData = False
        self.report, self.tick = raw_data_retrieve()
        if self.NonData:
            return

        self.lastQuarter = self.report['quarter'].iloc[-1]
        self.target_category = self.type + '_target'

        reduced = reduce2brief(self.tick)
        reduced = Formula.drop_tick_target(reduced)
        reduced = truncate_period(reduced, self.lastQuarter)
        self.major = pd.merge(self.report, reduced, on = 'quarter', how = 'left')
        self.major.index = self.major.quarter

        if self.If_Renew:
            self.tick = Formula.drop_tick_target(self.tick)
        else:
            targets = saved_target_retrieve()
            self.major = pd.merge(self.major, targets, on = 'quarter', how = 'left')
            self.major.index = self.major.quarter  # def stat_retrieve():  #     stat =   #  #  #
            #  DMgr.read_csv('main_select', code)  #     if stat is None:  #         return None
            #     stat.index = stat['quarter']  #     fields = []  #     for col in stat:  #  #
            #    if col.endswith(PERCENTILE):  #             fields.append(col)  #  #     return
            #  stat[fields]  # stat = stat_retrieve()  # if stat is not None:  #     self.major =
            #  pd.merge(self.major, stat, left_index = True, right_index = True,  #         how =
            #  'left')  # self.major.index = self.major.quarter

    def calc_list(self, target_list = None):
        if self.NonData:
            return None

        target_list = Formula.key_targets if target_list is None else target_list
        cols = []
        for target in target_list:
            cols.append(target)
            self.calc_target(target = target)

    def calc_target(self, formula = None, target: str = None, prelude = np.nan):
        if self.NonData:
            return None
        if formula is None and target is None:
            raise Exception(' not specified at %s %s' % (self.type, self.code))
        if formula is not None:
            target = formula.target

        if not self.If_Renew and target in self.major:
            return
        if target.endswith(PERCENTILE) and target not in self.major:
            print(self.code, 'no percentile data', target)
            return
        formula = Formula.table.loc[target] if formula is None else formula
        prelude = formula.prelude if prelude != prelude else prelude
        target_calculator = lambda tar: self.calc_target(target = tar)
        if formula.source == TICK:
            # for daily tick, insert into self.tick first, then reduce to self.data
            if target not in self.formulas.index:
                self.refer_index.calc_target(target = target, prelude = prelude)
                tick_result = self.refer_index.tick[target]
                result = self.refer_index.major[
                    target]  # fill_missI(ref, self.tick.index, 'date', ifLabelingFilled = False)
                #  df = ref
            else:
                tick_result = Formula.calculate(self.tick, formula, prelude, target_calculator)
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
                self.refer_index.calc_target(target = target, prelude = prelude)
                ref = self.refer_index.data[['quarter', target]]
                # df = fill_miss(ref, self.major.index, ifLabelingFilled = False)
                result = ref[target]
            else:
                result = Formula.calculate(self.major, formula, prelude, target_calculator)

        self.major[target] = result
        return result

    def save_targets(self, if_tick = True):
        if not self.NonData:
            cols = [x for x in Formula.target_savings if x in self.major]
            targets = self.major[cols]
            DMgr.save(targets, self.target_category, self.code)
            if if_tick:
                DMgr.save(self.tick, self.type, self.code)


class Indexes(Ticks):

    def __init__(self, label, code, if_update = False):
        def __get_table():
            tick = DMgr.read('index', self.code)
            column_duplicate_remove_i(tick)
            for col in ['market_cap', 'circulating_market_cap', 'net_profit']:
                if col in tick:
                    tick.drop(col, axis = 1, inplace = True)
            tick.sort_values('date', inplace = True)
            tick.date = tick.date.apply(date_str2std)
            if if_update:
                report = DMgr.update_file('index', self.elements_quarter,
                                          lambda start: self._fetch_element_report(start),
                                          index = 'quarter')

                daily = DMgr.update_file('index', self.elements_daily,
                                         lambda start: self._fetch_element_tick(start))
            else:
                report = DMgr.read('index', self.elements_quarter)
                daily = DMgr.read('index', self.elements_daily)
            tick = pd.merge(tick, daily, on = 'date', how = 'left')
            return report, tick

        self.label = label
        if label == 'all':
            self.code_list = DMgr.code_list
        else:
            self.elements = DMgr.read('category', label)
            self.code_list = self.elements['code']

        self.elements_daily = code + '_daily'
        self.elements_quarter = code + '_quarterly'

        super().__init__(code, __get_table, 'index')

    def _fetch_element_report(self, start_date):
        all_fina = DMgr.category_concat(self.code_list, 'income', ['net_profit'], start_date,
                                        show_seq = True)
        if all_fina.shape[0] == 0:
            return None
        all_fina['quarter'] = all_fina['date'].apply(to_quarter)
        mp = all_fina.groupby('quarter').agg({
            'quarter':    'max',
            'net_profit': 'sum'})
        mp = DWash.ttm_column(mp, 'net_profit')
        return mp

    def _fetch_element_tick(self, start_date):
        all_ticks = DMgr.category_concat(self.code_list, 'stock',
                                         ['market_cap', 'circulating_market_cap'], start_date,
                                         show_seq = True)
        if all_ticks.shape[0] == 0:
            return None
        all_ticks = all_ticks.groupby('date').agg({
            'date':                   'max',
            "market_cap":             'sum',
            'circulating_market_cap': 'sum'})
        return all_ticks

    # def __concat_element(self, category, columns):  #     if sum([1 for x in columns if x not
    #  in self.ticks]) == 0:  #         test_col = columns[0]  #         nans = self.ticks[  #  #
    #  self.ticks[test_col] != self.ticks[test_col]]  #         if nans.shape[0] == 0:  #  #  #
    # return  #         start_date = nans.ix[0, 'date']  #     else:  #         start_date =  #
    #  ''  #     concat = DMgr.category_concat(self.code_list, category, columns, start_date)  #
    #    return concat


class Stocks(Ticks):

    def __init__(self, code):
        def __get_table():
            report = None
            for tb in ['balance', 'cash_flow', 'income']:
                df = DMgr.read(tb, self.code)
                if df is not None:
                    df['quarter'] = df['date'].apply(to_quarter)
                    dup = df[df['quarter'].duplicated()]
                    if dup.shape[0] > 0:
                        print(self.code, tb, 'has duplicate index! Dropped!')
                        df.drop_duplicates('quarter', inplace = True)
                    # noinspection PyComparisonWithNone
                    df = df[df.quarter != None]
                    if report is None:
                        report = df
                    else:
                        report = pd.merge(report, df, on = 'quarter', suffixes = (
                            '', DWash.DUPLICATE_SEPARATOR + tb + DWash.DUPLICATE_FLAG))
            if report is not None and report.shape[0] > 0:
                DWash.column_select_i(report)
                report.index = report.quarter
                report.sort_index(inplace = True)

            tick = DMgr.read('stock', code)
            if tick is not None:
                tick = tick.drop_duplicates(subset = 'date', keep = 'first')  # todo only once
                tick = tick[tick.close != 0]
                DWash.calc_change_i(tick)
                tick.sort_values('date', inplace = True)

            return report, tick

        global HS300
        if HS300 is None:
            HS300 = Indexes(*INDEX_DICT['HS300'])
        super().__init__(code, __get_table, 'stock', HS300)

    def _financial_compare(self):
        if self.NonData:
            return
        self.calc_target(target = 'FinanceExpense')
        res = column_compare_choose_i(self.major, 'FinanceExpense', 'financial_expense')
        print(res)
        return res

    def rim(self, quarter = None):
        quarter = to_quarter(now()) if quarter is None else quarter
        quarter = min(self.lastQuarter, quarter)
        fin = self.major
        quart = self.report.loc[quarter]
        equity = quart.total_owner_equities

        def _roe_estimate():
            hist = fin[fin.index <= quarter].tail(5 * 4)
            print(hist.ROE)
            factor = DWash.percentage_factor_by_values(hist.ROE)
            roe_local = hist.ROE.mean()
            return roe_local / factor

        def _value_observe(roe_in, b):
            whole = 12
            high_speed = 5
            # low_speed = whole - high_speed
            for i in range(whole):
                if i < high_speed:
                    en = roe_in
                else:
                    en = LONG_TERM_RETURN

                dis = discount(i + 1)
                b += b * (en - CAPITAL_COST) * dis
            return b

        roe = _roe_estimate()
        val = _value_observe(roe, equity)
        return val, val / equity
