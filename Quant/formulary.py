import math
import statistics
import warnings
from functools import reduce

import numba

from .webio import *

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
        if_debug = False
        target = formula.target
        if target in data:
            return data[target]

        def _prelude():
            """ prepare fields and corresponding columns from formula
            :return: updated equation and fields
            """
            pre_eqt = formula.equation
            pre_fields = self.eqt_paras[formula.target]
            if if_debug:
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
                _ = df, new_field_name, periods
                target_calculator(field_name)

            def balance(df, field_name, new_field_name, periods):
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
            for field in pre_fields:
                if field in new_fields:
                    continue
                table = self.table_of(field)
                if table == 'history':
                    new_fields.append(field)
                else:
                    new_field = field if table == FORMULA_TARGET else field + TTM_SUFFIX + str(
                            num_quarters)
                    if new_field not in data:
                        tb2prelude[table](data, field, new_field, num_quarters)
                    new_fields.append(new_field)
                    pre_eqt = pre_eqt.replace(field, new_field)
            return pre_eqt, new_fields

        eqt, fields = _prelude()
        if eqt is None:
            return None
        if if_debug:
            print(data[fields])
        eqt_series = data.eval(eqt, parser='pandas')
        if formula.finale != formula.finale:
            result = eqt_series
        else:
            window_size = int(formula.quarter) if formula.source != TICK else int(
                    formula.quarter * 90)
            result = eqt_series.rolling(window_size).apply(self.REDUCE_METHODS[formula.finale])
        return result

    @staticmethod
    def drop_tick_target(df):
        for idx in FORMULA.sub_targets[TICK]:
            if idx in df:
                df = df.drop(idx, axis=1)
        return df


FORMULA = _FinancialFormula()


class Ticks(object):
    If_Renew = True

    @property
    def target_category(self):
        return self.type + '_target'

    @property
    def formulas(self):
        return FORMULA.formula_by_type[self.type]

    @property
    def last_quarter(self):
        return self.report['quarter'].iloc[-1]

    def __init__(self, code, table_getter: callable, security_type: str, refer_index=None):
        def raw_data_retrieve():
            report, tick = table_getter()
            if report is None or report.shape[0] == 0 or tick is None or tick.shape[0] == 0:
                warnings.warn('%s %s do not have data!' % (security_type, code))
                self.non_data = True
                return None, None
            tick.date = tick.date.apply(date_str2std)
            tick.index = tick.date
            tick['quarter'] = tick.date.apply(to_quarter)

            return report, tick

        def saved_target_retrieve():
            targets_table = DMGR.read(self.target_category, code)
            if targets_table is None or self.If_Renew:
                targets_table = pd.DataFrame(index=self.report.index)
                targets_table['quarter'] = targets_table.index
            else:
                targets_table.index = targets_table.quarter
            return targets_table

        self.code = code
        self.type = security_type
        self.refer_index = refer_index

        self.non_data = False
        self.report, self.tick = raw_data_retrieve()
        if self.non_data:
            return

        reduced = reduce2brief(self.tick)
        reduced = FORMULA.drop_tick_target(reduced)
        reduced = truncate_period(reduced, self.last_quarter)
        self.major = pd.merge(self.report, reduced, on='quarter', how='left')
        self.major.index = self.major.quarter

        if self.If_Renew:
            self.tick = FORMULA.drop_tick_target(self.tick)
        else:
            targets = saved_target_retrieve()
            self.major = pd.merge(self.major, targets, on='quarter', how='left')
            self.major.index = self.major.quarter

    def calc_list(self, target_list=None):
        if self.non_data:
            return

        target_list = FORMULA.sub_targets[KPI] if target_list is None else target_list
        cols = []
        for target in target_list:
            cols.append(target)
            self.calc_target(target=target)

    def calc_target(self, formula=None, target: str = None, prelude=np.nan):
        if self.non_data:
            return None
        if formula is None and target is None:
            raise Exception(' not specified at %s %s' % (self.type, self.code))
        if formula is not None:
            target = formula.target

        if not self.If_Renew and target in self.major:
            return None
        if target.endswith(PERCENTILE) and target not in self.major:
            print(self.code, 'no percentile data', target)
            return None
        formula = FORMULA.table.loc[target] if formula is None else formula
        prelude = formula.prelude if prelude != prelude else prelude
        target_calculator = lambda tar: self.calc_target(target=tar)
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
        if not self.non_data:
            cols = [x for x in FORMULA.target_savings if x in self.major]
            targets = self.major[cols]
            DMGR.save(targets, self.target_category, self.code)
            if if_tick:
                DMGR.save(self.tick, self.type, self.code)


class Indexes(Ticks):
    _hs300 = None

    @ClassProperty
    def hs300(self):
        if self._hs300 is None:
            self._hs300 = Indexes(*INDEX_DICT['HS300'])
        return self._hs300

    def __init__(self, label, code, if_update=False):
        def __get_table():
            tick = DMGR.read('index', self.code)
            column_duplicate_remove_i(tick)
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
            self.code_list = self.elements['code']

        self.elements_daily = code + '_daily'
        self.elements_quarter = code + '_quarterly'

        super().__init__(code, __get_table, 'index')

    def _fetch_element_report(self, start_date):
        all_fina = DMGR.category_concat(self.code_list, 'income', ['net_profit'], start_date)
        if all_fina.shape[0] == 0:
            return None
        all_fina['quarter'] = all_fina['date'].apply(to_quarter)
        profit_group = all_fina.groupby('quarter').agg({
            'quarter':    'max',
            'net_profit': 'sum'})
        profit_group = DWASH.ttm_column(profit_group, 'net_profit')
        return profit_group

    def _fetch_element_tick(self, start_date):
        all_ticks = DMGR.category_concat(self.code_list, 'stock',
                                         ['market_cap', 'circulating_market_cap'], start_date)
        if all_ticks.shape[0] == 0:
            return None
        all_ticks = all_ticks.groupby('date').agg({
            'date':                   'max',
            "market_cap":             'sum',
            'circulating_market_cap': 'sum'})
        return all_ticks


class Stocks(Ticks):

    def __init__(self, code):
        def __get_table():
            report = None
            for table in ['balance', 'cash_flow', 'income']:
                df = DMGR.read(table, self.code)
                if df is not None:
                    df['quarter'] = df['date'].apply(to_quarter)
                    dup = df[df['quarter'].duplicated()]
                    if dup.shape[0] > 0:
                        print(self.code, table, 'has duplicate index! Dropped!')
                        df.drop_duplicates('quarter', inplace=True)
                    # noinspection PyComparisonWithNone
                    df = df[df.quarter != None]
                    if report is None:
                        report = df
                    else:
                        report = pd.merge(report, df, on='quarter', suffixes=(
                            '', DWASH.DUPLICATE_SEPARATOR + table + DWASH.DUPLICATE_FLAG))
            if report is not None and report.shape[0] > 0:
                DWASH.column_select_i(report)
                report.index = report.quarter
                report.sort_index(inplace=True)

            tick = DMGR.read('stock', code)
            if tick is not None:
                tick = tick.drop_duplicates(subset='date', keep='first')  # todo only once
                tick = tick[tick.close != 0]
                DWASH.calc_change_i(tick)
                tick.sort_values('date', inplace=True)

            return report, tick

        super().__init__(code, __get_table, 'stock', Indexes.hs300)

    def _financial_compare(self):
        if self.non_data:
            return None
        self.calc_target(target='FinanceExpense')
        res = column_compare_choose_i(self.major, 'FinanceExpense', 'financial_expense')
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
