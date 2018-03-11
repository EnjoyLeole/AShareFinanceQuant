import math
import statistics
import warnings
from functools import reduce
from Basic.Util import *
from .webio import *

CAPITAL_COST = .08
LONG_TERM_RETURN = .12
INDEX_FLAG = 'Index_'
RESERVED_KEYWORDS = ['math', 'log', 'log10', 'e']


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
    #
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


class _FinancialFormula(metaclass = SingletonMeta):

    def __init__(self):
        # get table-field relation pairs for further use, table of *_indicator excluded
        self._table_fields = {row.field: row.table for i, row in DWash.mapper.iterrows() if
                              '_in' not in row.table}

        def _organise_table():
            table = get_lib('formula')
            factor_dict = {}
            table.index = table.target

            paras = '([A-Za-z_]\w*)'
            reg_factor = re.compile(paras)
            for target, formula in table.iterrows():
                fs = reg_factor.findall(formula.equation)
                fs = [x for x in fs if x not in RESERVED_KEYWORDS]
                # get all fields included in equation
                factor_dict[target] = fs
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

            return table, factor_dict

        self.table, self.target_factors = _organise_table()

    def table_of(self, field):
        '''return the name of the table contains field'''
        if field in self._table_fields:
            return self._table_fields[field]
        return 'NotInData'


Formula = _FinancialFormula()


# class TimePoint:


class Ticks(object):

    def __init__(self, code, filter, table_getter: callable, type: str, refer_index = None):
        self.code = code
        self.type = type
        self.symbol = code2symbol(code)
        self.report, self.tick = table_getter()
        if self.report is None or self.report.shape[0] == 0 or self.tick is None:
            self.formulas = None
            warnings.warn('%s %s dont have data!' % (type, code))
            return
        self.tick.date = self.tick.date.apply(std_date_str)
        self.tick.index = self.tick.date
        self.tick['quarter'] = self.tick.date.apply(to_quarter)
        # if self.report.shape[0]>0:
        self.lastQuarter = self.report['quarter'].iloc[-1]

        self.target_category = self.type + '_target'
        self.targets = DMgr.read_csv(self.target_category, code)

        if self.targets is None:
            self.targets = pd.DataFrame(index = self.report.index)
            self.targets['quarter'] = self.targets.index

        else:
            self.targets.index = self.targets.quarter

        self.formulas = Formula.table[Formula.table.target.apply(filter)]
        self.refer_index = refer_index

        self.idx_map = {}

    def save_targets(self):
        DMgr.save_csv(self.targets, self.target_category, self.code)
        DMgr.save_csv(self.tick, self.type, self.code)

    def all_target(self):
        if self.formulas is None:
            return None
        for i, formula in self.formulas.iterrows():
            # print(i)
            if formula.source != TICK:
                self.target_series(formula.target)
        self.save_targets()
        return self.targets

    def target_series(self, target: str):
        for idx in range(-self.report.shape[0], 0, 1):
            self.target_calc(target, idx)
        if target in self.targets:
            return self.targets[target]

    def target_calc(self, target: str, idx = -1, prelude = np.nan):
        '''
        using data from self(Ticks) to fill formula's equation, consider prelude & finale procedure to get final value
        :param formula: formula to be calculated
        :param idx: offset from the bottom of tables(ticks or financial report)
        :param prelude: procedure before evaluate equation
        :return: value result from formula
        '''
        if target not in self.formulas.index.values:
            # print(self.code,target,self.formulas.index.values)
            val = self.refer_index.target_calc(target, idx, prelude)
            return val
        formula = self.formulas.loc[target]

        if formula.source == TICK:
            return self._tick_formula_val(formula, idx, prelude)
        else:
            return self._report_formula_calc(formula, idx, prelude)

    def _report_formula_calc(self, formula, idx = -1, prelude = np.nan):
        target = formula.target
        if abs(idx) <= self.targets.shape[0] and target in self.targets:
            fixes_val = self.targets.ix[idx,target]
            if fixes_val == fixes_val:
                return fixes_val

        prelude = formula.prelude if prelude != prelude else prelude

        if formula.finale != formula.finale:
            val = self.__equation_val(formula, idx, prelude)
        else:
            if formula.source == 'mixed':
                raise Exception('Source of finale_calc %s not specified!' % formula.target)
            n = int(formula.duration)
            start = idx - n + 1
            chosen = range(start, start + n, 1)
            val = self.__series_reduce(chosen, formula, prelude)

        # print('report val:', idx, formula.target, formula.source, val)
        if abs(idx) <= self.targets.shape[0]:
            self.targets.ix[idx, formula.target] = val
        # else:
        #     print(idx)
        #     self.targets.iloc[idx] = pd.Series()
        #     self.targets.ix[idx, formula.target] = val
        return val

    def _tick_formula_val(self, formula, idx, prelude):
        tick = self.tick
        idx_date, idx_quart = self.__idx2date(idx)
        if idx_date is None:
            return None
        if formula.target in tick and idx_date in tick.index:
            val = tick.ix[idx_date, formula.target]
            if val == val:
                return val
        # print(self,idx)

        elif formula.finale != formula.finale:
            val = self.__equation_val(formula, idx_date, prelude)
        else:
            n = int(formula.duration)
            start_quarter = quarter_add(idx_quart, -n)
            _, start_date = quarter_dates(start_quarter)
            chosen = tick[(tick.index > start_date) & (tick.index <= idx_date)].index.values
            # print(start_date,idx_date,len(chosen),chosen)
            val = self.__series_reduce(chosen, formula, prelude)
        # print('tick_target',idx,idx_date,formula.target,val)
        # if val is not None:
        self.tick.ix[idx_date, formula.target] = val
        return val

    def __series_reduce(self, idx_list, formula, prelude):
        fi = 2 ** (-1 / 3)
        iter = {
            'sum'              : lambda list: sum(list),
            'mean'             : lambda list: sum(list) / len(list),
            'sd'               : lambda list: statistics.stdev(list),
            'vix_yearly'       : lambda list: (sum([
                x ** 2 for x in list]) * 252 / len(list)) ** 0.5,
            'percent_geometric': lambda list: -1 + reduce(lambda accum, x: accum * (1 + x),
                list, 1) ** (1 / len(list)),
            'change'           : lambda list: list[-1] / list[0] - 1 if list[0] != 0 else 0,
            'inc'              : lambda list: list[-1] - list[0],
            'decline_avg'      : lambda list: reduce(lambda accum, x: (accum + x) * fi, list,
                0)}

        valS = [self.__equation_val(formula, idx, prelude) for idx in idx_list]
        # pNum(formula.target, idx_list, valS, prelude,len(valS))
        if None in valS or len(valS) == 0:
            val = None
        else:
            val = iter[formula.finale](valS)
            # print(len(valS),formula.target,val,valS)
        return val

    def __equation_val(self, formula, idx, prelude = np.nan):
        eqt = formula.equation
        fields = Formula.target_factors[formula.target]
        prelude = formula.prelude if prelude != prelude else prelude
        havePrelude = prelude == prelude

        tick = self.tick
        report = self.report

        def _hist(field):
            if idx in tick.index:
                # print(idx,field)
                val = tick.ix[idx, field]
            else:
                idx_date, _ = self.__idx2date(idx)
                if idx_date is None:
                    val = None
                else:
                    val = tick.at[idx_date, field]
                # raise Exception('%s not in %s for %s %s' % (idx, self.code, formula.target, field))
            return val

        def _target(field):
            if havePrelude:
                if Formula.table.at[field,'flag'] != 'addOnly':
                    raise Exception(
                        'Formula %s %s contains calculating factor while requiring prelude '
                        'could cause un-predicted '
                        'problem!' % (formula.target, fields))
            return self.target_calc(field, idx, prelude)

        def _fina(field):
            if not havePrelude:
                if idx >= report.shape[0] or idx < -report.shape[0]:
                    return None
                return report.ix[idx, field]
                # print(idx,field,report.shape,val)
            else:
                std_duration = {
                    'year'   : 5,
                    'quarter': 2}
                n = std_duration[prelude]
                if idx < 0:
                    start = report.shape[0] + idx - n + 1
                else:
                    start = idx - n + 1
                end = start + n
                if start < 0 or end > report.shape[0]:
                    return None

                slice = report.iloc[start:end]

                if slice.shape[0] < n:
                    return None
                else:
                    table = Formula.table_of(field)

                    if table in ['cash_flow', 'income']:
                        dic = slice[field].to_dict()
                        val = DWash.ttm_dict(dic)
                    else:
                        val = (slice.ix[0, field] + slice.ix[-1, field]) / 2
                    # print(idx,table,val)
                    return val

        tb_funcs = {
            'NotInData': _target,
            'history'  : _hist,
            'balance'  : _fina,
            'income'   : _fina,
            'cash_flow': _fina}

        for field in fields:
            tb = Formula.table_of(field)
            val = tb_funcs[tb](field)
            # print('Field fetch',idx, tb, field, val)
            if val is None or val != val:
                return None
            eqt = eqt.replace(field, '%s' % val)

        try:
            # todo make it safer
            # print(idx, formula.target, eqt)
            result = eval(eqt)
            # print(idx, formula.target, eqt,result)
        except ZeroDivisionError:
            self.__warn_zero_division(formula, eqt)
            result = None
        except Exception as e:
            print(idx, formula.target, eqt, e)
            raise Exception('Unpredict')
        return result

    def __idx2date(self, idx):
        if idx in self.idx_map:
            return self.idx_map[idx]

        if isinstance(idx, str) and DATE_SEPARATOR in idx:
            idx_date = idx
            idx_quart = to_quarter(idx_date)
        else:
            if (isinstance(idx, str) and idx.isdigit()) or isinstance(idx, int):
                idx_quart = quarter_add(to_quarter(today()), idx + 1)
            elif QUARTER_SEPARATOR in idx:
                idx_quart = idx
            elif DATE_SEPARATOR not in idx:
                raise Exception('Un expected idx %s for tick' % (idx))
            tick = self.tick
            idx_cands = tick[tick.quarter <= idx_quart]
            if idx_cands.shape[0] == 0:
                return None, None
            idx_date = std_date_str(idx_cands.iloc[-1]['date'])
        self.idx_map[idx] = [idx_date, idx_quart]
        return idx_date, idx_quart

    def __warn_insufficient_data(self, formula, idx, num_req, num_actual):
        warnings.warn('%s @%s: only %s data point while %s required, evaluation aborted!' % (
            formula.target, idx, num_actual, num_req))

    def __warn_zero_division(self, formula, eqt):
        return
        warnings.warn('Zero divided! %s->%s' % (formula.equation, eqt))


class Indexs(Ticks):

    def __init__(self, label, code, ifUpdate = False):
        def __get_table():
            tick = DMgr.read_csv('index', self.code)
            for col in ['market_cap', 'circulating_market_cap', 'net_profit']:
                if col in tick:
                    tick.drop(col, axis = 1, inplace = True)
            tick.sort_values('date', inplace = True)
            tick.date = tick.date.apply(std_date_str)
            if ifUpdate:
                report = DMgr.update_csv('index', self.elements_quarter,
                    lambda start: self._fetch_element_report(start), index = 'quarter')

                daily = DMgr.update_csv('index', self.elements_daily,
                    lambda start: self._fetch_element_tick(start))
            else:
                report = DMgr.read_csv('index', self.elements_quarter)
                daily = DMgr.read_csv('index', self.elements_daily)
            df = pd.merge(tick, daily, on = 'date', how = 'left')
            tick = brief_detail_merge(report, df, ifReduce2brief = False,
                brief_col = 'quarter',
                detail_col = 'date')
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
            showSeq = True)
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
            ['market_cap', 'circulating_market_cap'], start_date, showSeq = True)
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


Idx_dict = {
    'ALL'  : ['all', '000001'],
    'HS300': ['hs300', '399300'],
    'SZ50' : ['sz50', '000016'],
    'ZZ500': ['zz500', '000905']}

HS300 = Indexs(*Idx_dict['HS300'])


# todo Dupt  detail into ATO
# todo compare MG & MS cross the market

class Stocks(Ticks):

    def __init__(self, code):
        def __get_table():
            report = None
            for tb in ['balance', 'cash_flow', 'income']:
                df = DMgr.read_csv(tb, self.code, ifRegular = False)
                if df is not None:
                    df['quarter'] = df['date'].apply(to_quarter)
                    df = df[df.quarter != None]
                    if report is None:
                        report = df
                    else:
                        report = pd.merge(report, df, on = 'quarter',
                            suffixes = ('', DWash.DuplicatedFlag))
            if report.shape[0] > 0:
                DWash.column_regularI(report, 'financial_set')
                report.index = report.quarter
                report.sort_index(inplace = True)

            tick = DMgr.read_csv('stock', code)
            if tick is not None:
                tick = tick[tick.close != 0]
                DWash.get_changeI(tick)
                tick.sort_values('date', inplace = True)

            return report, tick

        super().__init__(code, lambda target: INDEX_FLAG not in target, __get_table, 'stock', HS300)

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
