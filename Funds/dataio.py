from Meta import *
from Basic.Util import *
from Basic.IO import *
import time, os, re
import numpy as np
import pandas as pd
from urllib3 import PoolManager

OLDEST_DATE = '1988-01-01'
STOCK_ROOT = 'D:\\StockData\\'
DATA_FOLDERS = {
    'indicator'   : 'financial_indicator',
    'balance'     : 'financial_balance',
    'cash_flow'   : 'financial_cash_flow',
    'income'      : 'financial_income',
    'index'       : 'market_index',
    'macro'       : 'macro',
    'stock'       : 'market_stock',
    'temp'        : 'temp',
    'stock_target': 'target_stock',
    'index_target': 'target_index',
    'category'    : 'category'}
TICK = 'ticks_daily'
REPORT = 'financial_report_quarterly'
DUPLICATE_FLAG = '_d'
FORCE_FILLED = 'force_filled'
TTM_SUFFIX = '_ttm'


def get_url(url, encoding = ''):
    if encoding == '':
        encoding = GBK
    pm = PoolManager()
    res = pm.request('get', url)
    dict = res.data.decode(encoding)
    return dict


def code2symbol(code):
    if '.' in code:
        return code
    tail = 'SH' if code[0] == '6' else 'SZ'
    return code + '.' + tail


def remove_duplicateI(df):
    dups = [x for x in df if x.endswith(DUPLICATE_FLAG)]
    df.drop(dups, axis = 1, inplace = True)


def fill_miss(df, refer_idxs, brief_col = 'quarter', ifLabelingFilled = True):
    # brief_col = 'index' if brief_col is None else brief_col
    missing = [x for x in refer_idxs if x not in df[brief_col].values]
    missing = list(set(missing))
    if ifLabelingFilled:
        df[FORCE_FILLED] = False
    for qt in missing:
        # if qt is None:
        #     continue
        # cands = df[df[brief_col] > qt][brief_col]
        # if cands.shape[0] > 0:
        #     chosen = cands[0]
        # else:
        #     chosen = df[brief_col][-1]
        # df.loc[qt] =None

        df.loc[qt, brief_col] = qt
        if ifLabelingFilled:
            df.loc[qt, FORCE_FILLED] = True
    df = df.fillna(method = 'pad')
    return df


def reduce2brief(df, brief_col = 'quarter', detail_col = 'date'):
    if brief_col not in df:
        detail2brief_func = INTERVAL_TRANSFER[(detail_col, brief_col)]
        df[brief_col] = df[detail_col].apply(detail2brief_func)
    group = df.groupby(brief_col).agg({
        detail_col: 'max'})[detail_col].tolist()
    df.index = df[detail_col]
    # flag = df[detail_col].apply(lambda x: True if x in group.values else False)
    df_reduced = df.ix[group].copy()
    df_reduced.index = df_reduced[brief_col]
    return df_reduced


def brief_detail_merge(brief, detail, ifReduce2brief = False, brief_col = 'quarter',
                       detail_col = 'date'):
    '''将周期不同的两张表做合并，在两表的index不完全重合时，会使用相邻项进行填充
    :param ifReduce2brief
        True:以汇总表为基础合并，明细表中取detail_col最大项
        False：以明细表为基础合并，汇总表分散对应至明细表中各个brief_col的相同项'''
    detail2brief_func = INTERVAL_TRANSFER[(detail_col, brief_col)]
    if brief_col not in detail:
        detail[brief_col] = detail[detail_col].apply(detail2brief_func)
    if brief_col not in brief:
        brief[brief_col] = brief[detail_col].apply(detail2brief_func)
    brief.sort_values(brief_col, inplace = True)
    brief.index = brief[brief_col]
    brief = brief[brief[brief_col] > '']

    if ifReduce2brief:
        df_reduced = reduce2brief(detail, brief_col, detail_col)
        if df_reduced.shape[0] < brief.shape[0]:
            df_reduced = fill_miss(df_reduced, brief.index, brief_col)
        detail = df_reduced
        method = 'left'
        valid = '1:m'
    else:
        brief = fill_miss(brief, detail[brief_col], brief_col)
        method = 'right'
        valid = '1:m'
    df = pd.merge(brief, detail, on = brief_col, how = method, validate = valid,
        suffixes = ['', DUPLICATE_FLAG])
    df.sort_values([brief_col, detail_col], inplace = True)
    # cols=sorted(df.columns.values)
    # pco=sorted(set(cols))
    # print(cols)
    # print(pco)
    # print(brief.shape, detail.shape, df.shape)
    return df


class _DataManager(metaclass = SingletonMeta):
    cl_path = STOCK_ROOT + 'code_list.csv'

    def __init__(self):
        self.code_table = pd.read_csv(self.cl_path, encoding = GBK)
        self.active_table = self.code_table[self.code_table.stop == False]
        self.code_list = self.code_table['code']
        folder = STOCK_ROOT + DATA_FOLDERS['index']
        codes = re.compile('(\d+)')
        self.idx_list = []
        for file in os.listdir(folder):
            # print(1)
            code = codes.search(file)[0]
            self.idx_list.append(code)

    def __financial_report_sustaining_check(self):
        def __if_stop(code):
            df = DMgr.read_csv('balance', code)
            if df is not None and df.shape[0] > 0:
                df['quarter'] = df.date.apply(to_quarter)
                curQuarter = to_quarter()
                for i in range(4):
                    qt = quarter_add(curQuarter, -i - 1)
                    if (df.quarter == qt).any():
                        return False
            return True

        DMgr.code_table['stop'] = DMgr.code_table.code.apply(__if_stop)
        DMgr.code_table.to_csv(DMgr.cl_path, encoding = GBK, index = False)

    def _create_all_folders(self):
        for key in DATA_FOLDERS:
            folder = STOCK_ROOT + DATA_FOLDERS[key]
            if not os.path.exists(folder):
                os.makedirs(folder)

    def __csv_path(self, category, code):
        folder = STOCK_ROOT + DATA_FOLDERS[category] + '\\'
        return folder + '%s.csv' % code

    def read_csv(self, category, code, ifRegular = True):
        path = self.__csv_path(category, code)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, encoding = GBK)
        DWash.value_scale_by_column_name(df)
        # print(df)
        if ifRegular:
            DWash.column_regularI(df, category)
        return df

    def save_csv(self, df: pd.DataFrame, category, code, encode = GBK, index = False,
                 ifRegular = True):
        path = self.__csv_path(category, code)
        DWash.value_scale_by_column_name(df)
        if ifRegular:
            DWash.column_regularI(df, category)
        df.to_csv(path, index = index, encoding = encode)

    def update_csv(self, category, code, fetcher, index = 'date'):
        def __msg(*txt):
            print(category, code, *txt)

        exist = DMgr.read_csv(category, code)
        # DWash.reform_tick(exist)
        if exist is None:
            exist = pd.DataFrame()
            start = OLDEST_DATE
        else:
            idx = exist[index]
            if idx[0] > idx[1]:
                exist.sort_values(index, inplace = True)
            start = exist[index].iloc[-1]
        if index == 'date':
            dist = today() - str2date(start).date()
            if dist.days <= 3:
                __msg('Last update in %s days, assume filled!' % dist.days)
                return exist
            start = str2date(start, 1).date()
            start = date_str(start)

        new = fetcher(start)
        if new is None:
            __msg('Non Data')
            return exist
        else:
            __msg(start, len(new))
            new = pd.concat([exist, new])
            DMgr.save_csv(new, category, code)
            return new

    def loop(self, func, code_list, flag = '', times = 1, delay = 0, showSeq = False, limit = -1):
        fails = []
        count = 0
        for code in code_list:
            if count > limit and limit > 0:  break
            if showSeq:
                print(count, flag, code)
            for i in range(times):
                try:
                    func(code)
                except TimeoutError as e:
                    if i == times - 1:
                        print(e)
                        fails.append([flag, code])
                    if delay > 0:
                        time.sleep(delay)
                except Exception as e:
                    print(e)
                    fails.append([flag, code])
            count += 1

        if len(fails) > 0:
            path = get_error_path(flag + '.txt')
            obj2file(path, fails)  # return fails

    def iter_stocks(self, func, flag, limit = -1):
        self.loop(func, self.code_list, flag, limit = limit)

    def iter_index(self, func, flag, limit = -1):
        self.loop(func, self.idx_list, flag, limit = limit)

    def category_concat(self, code_list, category, columns, start_date, showSeq = False):
        tid = category + '_concat' + uid()
        setattr(self, tid, pd.DataFrame())

        def _concat_code(code):
            df = self.read_csv(category, code)
            if df is not None:
                df = df[df.date >= start_date]
                if df.shape[0] > 0:
                    df = df[['date', *columns]]
                    temp = getattr(self, tid)
                    temp = pd.concat([temp, df])
                    setattr(self, tid, temp)

        self.loop(_concat_code, code_list, tid, showSeq = showSeq)

        return getattr(self, tid)


DMgr = _DataManager()


class _DataWasher(metaclass = SingletonMeta):
    # region column name regular
    COLUMN_UPDATE = True
    WARNING_LEVEL = 0.5

    def __init__(self):
        self.mapper = get_lib('mapper')
        self.mapper['alias'] = self.mapper['alias'].apply(self._simplify_name)
        if self.COLUMN_UPDATE:
            self.mapper['matchCount'] = 0
            self.mapper['matches'] = ''

        self.match_path = get_lib_path('matched')
        if not os.path.exists(self.match_path):
            with open(self.match_path, 'w') as f:
                f.write('{}')
        self.matched = file2obj(self.match_path)

    def _simplify_name(cls, name):
        swap_pair = [['所有者', '股东'], ['的', ''], ['所', '']]
        for pair in swap_pair:
            name = name.replace(*pair)
        return name

    DuplicatedFlag = '*2'

    def _column_match(self, df: pd.DataFrame, category = ''):
        matches = {}
        if category != '' and category in self.matched:
            matches = self.matched[category]
        else:
            for col_name in df:
                col = self._simplify_name(col_name)
                candidates = []

                for key, row in self.mapper.iterrows():
                    alias = row['alias']
                    if isinstance(alias, str) and alias in col:
                        candidates.append([key, row])

                n = len(candidates)
                if n > 0:
                    if n == 1:
                        key, chosen = candidates[0]
                    else:
                        key, chosen = max_at(candidates, lambda cand: len(cand[1]['alias']))
                    new_name = chosen['field']

                    matches[col_name] = new_name
                    if self.COLUMN_UPDATE:
                        self.mapper.ix[key, 'matchCount'] += 1
                        self.mapper.ix[key, 'matches'] += '%s %s ' % (col, category)
                        self.mapper.to_csv('D:/field_mapper.csv', encoding = GBK)
                else:
                    judge = col == self.mapper['field']
                    if self.mapper[judge].shape[0] == 0:
                        print(category, col_name)

            duplicate = {}
            for col in matches:
                field = matches[col]
                duplicate[field] = flag = 0 if field not in duplicate else duplicate[field] + 1
                if flag > 0 and col != field:
                    print('Duplicated column %s: %s -> %s' % (flag, col, field))
                    matches[col] += '~%s%s' % (flag, self.DuplicatedFlag)
            if category != '':
                self.matched[category] = matches
                obj2file(self.match_path, self.matched)
        return matches

    def _column_select(self, df, matches):
        rename = lambda x: df.rename(columns = x, inplace = True)

        def __compareI(origin, dual_key):
            dual = df[dual_key]
            if isinstance(dual, pd.DataFrame):
                raise Exception('%s has multiple columns, Should handle before!' % dual_key)

            self._numericI(df, [origin, dual_key])
            orgSum = df[origin].sum()
            dualSum = df[dual_key].sum()
            diff = orgSum - dualSum
            maxSum = max(orgSum, dualSum)
            diff_rate = diff / maxSum if maxSum > 0 else 0

            if abs(diff_rate) < 0.001 or dualSum == 0:
                rename({
                    dual_key: dual_key + 'Done'})
            elif orgSum == 0 and dualSum != 0:
                rename({
                    origin: origin + 'Done'})
                rename({
                    dual_key: origin})
            else:
                def validate(series):
                    if series[origin] != series[dual_key] and series[origin] != 0 and series[
                        dual_key] != 0:
                        # print('Inconsistent of value between %s(%s) and %s(%s)' % (
                        #     origin, series[origin], dual_key, series[dual_key]))
                        return 1
                    return 0

                check = df.apply(validate, axis = 1).sum()
                if check / df.shape[0] > self.WARNING_LEVEL:
                    print('Inconsistent for %s bigger than %s with %s in %s / %s records' % (
                        origin, dual_key, diff_rate, check, df.shape[0]))

                if diff < 0:
                    rename({
                        origin: origin + 'Done'})
                    rename({
                        dual_key: origin})  # print('Prefer %s than %s / diff:%s' % (  #  #  #  #  #  dual_key,  #  origin, diff))
                else:
                    rename({
                        dual_key: dual_key + 'Done'})

        keys = {}
        cols_id = [[x.split('~')[0], x] for x in df.columns]
        for id, col in cols_id:
            if id not in keys:
                keys[id] = [0, [col]]
            else:
                keys[id] = [keys[id][0] + 1, keys[id].append(col)]

        for id in keys:
            if keys[id][0] > 1:
                for field in keys[id][1]:
                    if id == 'date' and field != 'date':
                        df.drop(field, inplace = True)
                    elif id != field:
                        __compareI(id, field)

    def column_regularI(self, df: pd.DataFrame, category = ''):
        matches = self._column_match(df, category)

        df.rename(columns = matches, inplace = True)

        # print([col for col in df.columns.values if 'long_d' in col])
        self._column_select(df, matches)

    # endregion

    # region one time active for files

    def simplify_dirs(cls, category):
        folder = STOCK_ROOT + DATA_FOLDERS[category] + '\\'
        prefix = re.compile('([0-9]*.csv)')
        for file in os.listdir(folder):
            new_file = prefix.search(file)[0]
            # print(file,nf)
            os.rename(folder + file, folder + new_file)

    def simplify_mapper(cls):
        pattern = re.compile(r'(.*)(\s*\(|（)')
        for row in cls.mapper.iterrows():
            alias = row[1]['alias']
            # print(alias)
            if isinstance(alias, str) and ('(' in alias or '（' in alias):
                match = pattern.match(alias)
                new_alias = match.group(1)
                cls.mapper.ix[row[0], 'alias'] = new_alias
                print(new_alias)

    # endregion

    # region number & scale
    def replaceI(self, df: pd.DataFrame, old = '--', new = 0):
        df.replace(old, new, inplace = True)

    def __to_num(self, val):
        try:
            return float(val)
        except:
            return 0

    def _numericI(self, df: pd.DataFrame, include = [], exclude = []):
        include = df if len(include) == 0 else include
        include = [include] if not isinstance(include, list) else include
        for col in include:
            if col not in exclude and df[col].dtype in [object, str]:
                try:
                    df[col] = df[col].astype(np.float64)  # print('simple')
                except Exception as e:
                    df[col] = df[col].apply(self.__to_num)

                    # print('%s has a shape %s which is incorrect! %s' % (col, df[col].shape, e))

    def value_scale_by_column_name(self, df: pd.DataFrame):
        self.replaceI(df)
        for col in df:
            if '万元' in col:
                self._numericI(df, col)
                df[col] *= 10000
            elif '率' in col or '%' in col:
                self._numericI(df, col)
                df[col] /= 100
            elif '元' in col:
                self._numericI(df, col)

    def percentage_factor_by_values(self, series: pd.Series):
        series.dropna(inplace = True)
        if (series < 1).all():
            return 1
        if (series > 2).all():
            return 100
        raise Exception('Can not determing percentage factor!')

    # endregion

    def ttm_column(self, df, column, new_column = None, n = 4):
        newCol = column + TTM_SUFFIX if new_column is None else new_column
        df[newCol] = 0
        if 'quarter' not in df:
            print('no quarter given, aborted!')
            return

        df = df.sort_values('quarter')
        if n == 2:
            col = df[column].fillna(method = 'pad')
            shift = col.shift(1)
            df['quart'] = df['quarter'].apply(lambda x: x.split(QUARTER_SEPARATOR)[1])
            df[new_column] = col - shift
            ttms = df.apply(lambda row: row[column] if row['quart'] == '1' else row[newCol],
                axis = 1)
        else:
            last = [0, 0, 0, 0, 0]
            ttms = []
            for key, row in df.iterrows():
                cur = row[column]
                qs = row['quarter']

                year, qt = qs.split(QUARTER_SEPARATOR)
                qt = int(qt)
                ttm = cur + last[4] - last[qt]
                last[qt] = row[column]
                ttms.append(ttm)
            # df[newCol] = li
        return ttms

    def ttm_dict(self, dict):
        if len(dict) == 2:
            qts = []
            vals = []
            for key, value in dict.items():
                _, qt = key.split(QUARTER_SEPARATOR)
                qts.append(qt)
                vals.append(value)
            if qts[1] == '1':
                return vals[1] if vals[1] > 0 else vals[0]
            else:
                return vals[1] - vals[0] if vals[1] > 0 else vals[0]
        else:
            last = [0, 0, 0, 0, 0]
            for key in dict:
                val = dict[key]
                year, qt = key.split(QUARTER_SEPARATOR)
                qt = int(qt)
                ttm = val + last[4] - last[qt]
                last[qt] = val
            return ttm

    def get_changeI(self, df: pd.DataFrame):
        preClose = df['close'].values
        preClose = np.insert(preClose, 0, np.nan)
        preClose = preClose[0:len(preClose) - 1]
        df['pre_close'] = preClose
        df['change_amount'] = df['close'] - df['pre_close']

        if 'change' in df:
            df.drop('change', axis = 1, inplace = True)  # print(df)

        def ratio(f):
            return f[1] / f[0] - 1

        pr = 'derc_close' if 'derc_close' in df else 'close'
        df['change_rate'] = df[pr].rolling(2).apply(ratio)

        # print(df.loc[:,['close','pre_close','change_amount']])


DWash = _DataWasher()
