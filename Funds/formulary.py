import math
import statistics
import warnings
from functools import reduce
from .webio import *
import numba

CAPITAL_COST = .08
LONG_TERM_RETURN = .12
INDEX_FLAG = 'Index_'
RESERVED_KEYWORDS = ['math', 'log', 'log10']

Idx_dict = {
    'ALL'  : ['all', '000001'],
    'HS300': ['hs300', '399300'],
    'SZ50' : ['sz50', '000016'],
    'ZZ500': ['zz500', '000905']}

HS300 = None


# todo Dupt  detail into ATO
# todo compare MG & MS cross the market
def discount(year):
    return 1 / math.pow(1 + CAPITAL_COST, year)


class Macro(object):
    tbs = {
        'money_supply'        : 'month',
        'gdp'                 : 'year',
        'social_finance_scale': 'year',
        'shibor'              : 'date'}

    @classmethod
    def macro_table(cls, interval = 'quarter'):
        result = None
        for key in cls.tbs:
            print(key)
            df_interval = cls.tbs[key]
            df = DMgr.read_csv('macro', key)
            if result is None:
                result = df
            else:
                if INTERVAL_ORDER[interval] < INTERVAL_ORDER[df_interval]:
                    brief = result
                    brief_col = interval
                    detail = df
                    detail_col = df_interval
                    ifReduce2brief = True
                else:
                    brief = df
                    brief_col = df_interval
                    detail = result
                    detail_col = interval
                    ifReduce2brief = False
                result = brief_detail_merge(brief, detail, ifReduce2brief = ifReduce2brief,
                    brief_col = brief_col, detail_col = detail_col)
        return result

    # @classmethod
    # def cap_base(cls):
    #     cap = DMgr.read_csv('temp', 'cap')
    #     cap['year'] = cap['date'].apply(lambda x: str2date(x).strftime('%Y'))
    #
    #     gdp = DMgr.read_csv('macro', 'gdp').ix[:, ['year', 'gdp']]
    #     gdp['year'] = gdp['year'].astype('str')
    #
    #     sfs = DMgr.read_csv('macro', 'social_finance_scale')
    #     sfs['year'] = sfs['year'].astype('str')
    #     print(sfs.dtypes)
    #     merged = pd.merge(cap, sfs, on = 'year')
    #     merged['flow_cap_sfs'] = merged['circulating_market_cap'] / merged['social_finance_scale']
    #     DMgr.save_csv(merged, 'temp', 'cap_sfs')

    # @classmethod
    # def market_pe(cls):
    #     mp = DMgr.read_csv('temp', 'agg_profit')
    #     cap = DMgr.read_csv('temp', 'cap')
    #
    #     merged = pd.merge(cap, mp, on = 'quarter')
    #
    #     print(merged)
    #     shibor = DMgr.read_csv('macro', 'shibor')
    #     # shibor['quarter'] = shibor['date'].apply(to_quarter)
    #
    #     merged = pd.merge(shibor, merged, on = 'date')
    #     merged['profit/cap'] = merged['net_profit_ttm'] * 10000 / merged['market_cap']
    #     merged['interest/shibor'] = merged['profit/cap'] / merged['1Y'] * 100
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
    return -1 + reduce(lambda accum, x: accum * (1 + x), arr, 1) ** (1 / len(arr))


@numba.jit
def decline_avg(arr):
    return reduce(lambda accum, x: (accum + x) * 2 ** (-1 / 3), arr, 0)


# endregion

class _FinancialFormula(metaclass = SingletonMeta):
    reduce_methods = {
        'sum'              : sum,
        'sd'               : statistics.stdev,
        'mean'             : lambda arr: arr.mean(),
        'vix_yearly'       : vix_yearly,
        'percent_geometric': percent_geometric,
        'change'           : change,
        'inc'              : increase,
        'decline_avg'      : decline_avg}

    std_duration = {
        'year'   : 5,
        'quarter': 2}

    def __init__(self):
        # get table-field relation pairs for further use, table of *_indicator excluded
        self._table_fields = {row.field: row.table for i, row in DWash.mapper.iterrows() if
                              '_in' not in row.table}

        def _organise_table():
            table = get_lib('formula')
            factor_dict = {}
            key_targets = {}
            table.index = table.target

            paras = '([A-Za-z_]\w*)'
            reg_factor = re.compile(paras)
            for target, formula in table.iterrows():
                if formula.indicia == formula.indicia:
                    key_targets[formula.target] = formula.indicia

                fs = reg_factor.findall(formula.equation)
                fs = [x for x in fs if x not in RESERVED_KEYWORDS]
                # get all fields included in equation
                factor_dict[target] = list(set(fs))
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

            return table, factor_dict, key_targets

        self.table, self.target_factors, self.key_targets = _organise_table()

    def table_of(self, field):
        '''return the name of the table contains field'''
        if field in self._table_fields:
            return self._table_fields[field]
        return 'NotInData'

    def calculate(self, data, formula, prelude, target_calculator):
        '''calculate formula based on data, either ticks or quarter report
        :param data: data frame used to calculate formula
        :param formula:
        :param prelude:
        :param target_calculator: callback func to get the target from upper level object
        :return: series contains formula result
        '''
        if_debug = False

        def _prelude(data, formula, prelude):
            '''
            prepare fields and corresponding columns from formula
            :param data: data frame contains/will contain fields
            :param formula:
            :param prelude:
            :return: updated equation and fields
            '''
            eqt = formula.equation
            fields = Formula.target_factors[formula.target]
            if if_debug:                print(formula.target)
            # print(data[fields])
            if prelude != prelude:
                for field in fields:
                    if Formula.table_of(field) == 'NotInData':
                        target_calculator(field)
                return eqt, fields
            else:
                def target(df, field, new_field, n):
                    target_calculator(field)

                def balance(df, field, new_field, n):
                    df[new_field] = df[field].rolling(n).apply(
                        lambda li: (li[0] + li[-1]) / 2)

                def ttm(df, field, new_field, n):
                    df[new_field] = DWash.ttm_column(df, field, new_column = new_field, n = n)

                tb2prelude = {
                    'NotInData': target,
                    'history'  : None,
                    'balance'  : balance,
                    'income'   : ttm,
                    'cash_flow': ttm}
                n = Formula.std_duration[prelude]
                new_fields = []
                for field in fields:
                    if field in new_fields:
                        continue
                    tb = Formula.table_of(field)
                    func = tb2prelude[tb]
                    if func is None:
                        new_fields.append(field)
                    else:
                        if tb == 'NotInData':
                            new_field = field
                        else:
                            new_field = field + TTM_SUFFIX + str(n)
                        if new_field not in data:
                            func(data, field, new_field, n)
                        new_fields.append(new_field)
                        eqt = eqt.replace(field, new_field)
                return eqt, new_fields

        target = formula.target
        if target in data:
            return data[target]
        else:
            eqt, fields = _prelude(data, formula, prelude)
            if if_debug:            print(data[fields])
            eqt_series = data.eval(eqt, parser = 'pandas')
            if formula.finale != formula.finale:
                result = eqt_series
            else:
                n = int(formula.quarter) if formula.source != TICK else int(formula.quarter * 90)
                result = eqt_series.rolling(n).apply(Formula.reduce_methods[formula.finale])
        return result


Formula = _FinancialFormula()


class Ticks(object):
    If_Renew = False

    def __init__(self, code, formula_selector, table_getter: callable, security_type: str,
                 refer_index = None):
        self.formulas = Formula.table[Formula.table.target.apply(formula_selector)]
        self.refer_index = refer_index

        self.code = code
        self.type = security_type

        def raw_data_fetch():
            report, tick = table_getter()
            if report is None or report.shape[0] == 0 or tick is None:
                self.formulas = None
                warnings.warn('%s %s dont have data!' % (security_type, code))
                return None, None
            tick.date = tick.date.apply(date_str2std)
            tick.index = tick.date
            tick['quarter'] = tick.date.apply(to_quarter)

            # self.lastQuarter = tick.iloc[-1].quarter
            # if self.lastQuarter != report.iloc[-1].quarter:
            #     report = fill_miss(report, [self.lastQuarter], 'quarter')
            return report, tick

        self.report, self.tick = raw_data_fetch()

        if self.report is None or self.tick is None or self.tick.shape[0] == 0:
            self.NonData = True
            return
        else:
            self.NonData = False
        self.lastQuarter = self.report['quarter'].iloc[-1]

        self.target_category = self.type + '_target'

        reduced = reduce2brief(self.tick)
        reduced = truncate_period(reduced, self.lastQuarter)
        self.major = pd.merge(self.report, reduced, on = 'quarter', how = 'left')
        self.major.index = self.major.quarter

        def saved_target_retrieve():
            targets = DMgr.read_csv(self.target_category, code)
            if targets is None or self.If_Renew:
                targets = pd.DataFrame(index = self.report.index)
                targets['quarter'] = targets.index
            else:
                targets.index = targets.quarter
            return targets

        self.targets = saved_target_retrieve()

        def stat_retrieve():
            stat = DMgr.read_csv('main_select', code)
            if stat is None:
                return None
            stat.index = stat['quarter']
            fields = []
            for col in stat:
                if col.endswith(PERCENTILE):
                    fields.append(col)

            return stat[fields]

        if self.If_Renew:
            for idx, formula in Formula.table[Formula.table.source == TICK].iterrows():
                if idx in self.tick:
                    self.tick = self.tick.drop(idx, axis = 1)
        else:
            self.major = pd.merge(self.major, self.targets, on = 'quarter', how = 'left')
            stat = stat_retrieve()
            if stat is not None:
                self.major = pd.merge(self.major, stat, left_index = True, right_index = True,
                    how = 'left')
            self.major.index = self.major.quarter

    def calc_all_vector(self):
        if self.NonData:
            return None
        cols = []
        for i, formula in self.formulas.iterrows():
            # print('vector',i)
            if formula.indicia == 'policy':
                continue
            cols.append(i)
            # if formula.source != TICK:
            self.calc_target_vector(formula)
        self.targets = self.major[['quarter', *cols]]
        return self.targets

    def calc_target_vector(self, formula = None, target: str = None, prelude = np.nan):
        if self.NonData:
            return None
        if formula is None and target is None:
            raise Exception(' not specified at %s %s' % (self.type, self.code))
        if formula is not None:
            target = formula.target

        formula = Formula.table.loc[target] if formula is None else formula
        prelude = formula.prelude if prelude != prelude else prelude
        target_calculator = lambda target: self.calc_target_vector(target = target)
        if formula.source == TICK:
            # for daily tick, insert into self.tick first, then reduce to self.data
            if target not in self.formulas.index:
                self.refer_index.calc_target_vector(target = target, prelude = prelude)
                tick_result = self.refer_index.tick[target]
                result = self.refer_index.major[target]
                # fill_missI(ref, self.tick.index, 'date', ifLabelingFilled = False)
                # df = ref
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
                self.refer_index.calc_target_vector(target = target, prelude = prelude)
                ref = self.refer_index.data[['quarter', target]]
                df = fill_miss(ref, self.major.index, ifLabelingFilled = False)
                result = ref[target]
            else:
                result = Formula.calculate(self.major, formula, prelude, target_calculator)

        # print(self.code,target, result.shape)
        self.major[target] = result

    def save_targets(self):
        if not self.NonData:
            DMgr.save_csv(self.targets, self.target_category, self.code)
            DMgr.save_csv(self.tick, self.type, self.code)


class Indexs(Ticks):

    def __init__(self, label, code, ifUpdate = False):
        def __get_table():
            tick = DMgr.read_csv('index', self.code)
            column_duplicate_removeI(tick)
            for col in ['market_cap', 'circulating_market_cap', 'net_profit']:
                if col in tick:
                    tick.drop(col, axis = 1, inplace = True)
            tick.sort_values('date', inplace = True)
            tick.date = tick.date.apply(date_str2std)
            if ifUpdate:
                report = DMgr.update_csv('index', self.elements_quarter,
                    lambda start: self._fetch_element_report(start), index = 'quarter')

                daily = DMgr.update_csv('index', self.elements_daily,
                    lambda start: self._fetch_element_tick(start))
            else:
                report = DMgr.read_csv('index', self.elements_quarter)
                daily = DMgr.read_csv('index', self.elements_daily)
            tick = pd.merge(tick, daily, on = 'date', how = 'left')
            return report, tick

        self.label = label
        if label == 'all':
            self.code_list = DMgr.code_list
        else:
            self.elements = DMgr.read_csv('category', label)
            self.code_list = self.elements['code']

        self.elements_daily = code + '_daily'
        self.elements_quarter = code + '_quarterly'

        super().__init__(code, lambda target: INDEX_FLAG in target, __get_table, 'index')

    def _fetch_element_report(self, start_date):
        allFina = DMgr.category_concat(self.code_list, 'income', ['net_profit'], start_date,
            show_seq = True)
        if allFina.shape[0] == 0:
            return None
        allFina['quarter'] = allFina['date'].apply(to_quarter)
        mp = allFina.groupby('quarter').agg({
            'quarter'   : 'max',
            'net_profit': 'sum'})
        mp = DWash.ttm_column(mp, 'net_profit')
        return mp

    def _fetch_element_tick(self, start_date):
        allTicks = DMgr.category_concat(self.code_list, 'stock',
            ['market_cap', 'circulating_market_cap'], start_date, show_seq = True)
        if allTicks.shape[0] == 0:
            return None
        allTicks = allTicks.groupby('date').agg({
            'date'                  : 'max',
            "market_cap"            : 'sum',
            'circulating_market_cap': 'sum'})
        return allTicks

    # def __concat_element(self, category, columns):
    #     if sum([1 for x in columns if x not in self.ticks]) == 0:
    #         test_col = columns[0]
    #         nans = self.ticks[self.ticks[test_col] != self.ticks[test_col]]
    #         if nans.shape[0] == 0:
    #             return
    #         start_date = nans.ix[0, 'date']
    #     else:
    #         start_date = ''
    #     concat = DMgr.category_concat(self.code_list, category, columns, start_date)
    #     return concat


class Stocks(Ticks):

    def __init__(self, code):
        def __get_table():
            report = None
            for tb in ['balance', 'cash_flow', 'income']:
                df = DMgr.read_csv(tb, self.code)
                if df is not None:
                    df['quarter'] = df['date'].apply(to_quarter)
                    dup = df[df['quarter'].duplicated()]
                    if dup.shape[0] > 0:
                        print(self.code, tb, 'has duplicate index! Dropped!')
                        df.drop_duplicates('quarter', inplace = True)
                    df = df[df.quarter != None]
                    if report is None:
                        report = df
                    else:
                        report = pd.merge(report, df, on = 'quarter',
                            suffixes = ('', DWash.DUPLICATE_SEPARATOR + tb + DWash.DUPLICATE_FLAG))
            if report is not None and report.shape[0] > 0:
                DWash.column_selectI(report)
                report.index = report.quarter
                report.sort_index(inplace = True)

            tick = DMgr.read_csv('stock', code)
            if tick is not None:
                tick = tick.drop_duplicates(subset = 'date', keep = 'first')  # todo only once
                tick = tick[tick.close != 0]
                DWash.get_changeI(tick)
                tick.sort_values('date', inplace = True)

            return report, tick

        global HS300
        if HS300 is None:
            HS300 = Indexs(*Idx_dict['HS300'])
        super().__init__(code, lambda target: INDEX_FLAG not in target, __get_table, 'stock', HS300)

    def financial_compare(self):
        if self.NonData:
            return
        self.calc_target_vector(target = 'FinanceExpense')
        res = column_compare_chooseI(self.major, 'FinanceExpense', 'financial_expense')
        print(res)
        return res

    def RIM(self, quarter = None):
        quarter = to_quarter(now()) if quarter is None else quarter
        quarter = min(self.lastQuarter, quarter)
        fin = self.targets
        quart = self.report.loc[quarter]
        equity = quart.total_owner_equities

        def _roe_estimate():
            hist = fin[fin.index <= quarter].tail(5 * 4)
            print(hist.ROE)
            factor = DWash.percentage_factor_by_values(hist.ROE)
            roe = hist.ROE.mean()
            return roe / factor

        def _value_observe(roe, B):
            whole = 12
            highSpeed = 5
            lowSpeed = whole - highSpeed
            for i in range(whole):
                if i < highSpeed:
                    en = roe
                else:
                    en = LONG_TERM_RETURN

                dis = discount(i + 1)
                B += B * (en - CAPITAL_COST) * dis
            return B

        roe = _roe_estimate()
        val = _value_observe(roe, equity)
        return val, val / equity
