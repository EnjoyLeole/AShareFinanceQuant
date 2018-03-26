# -*- coding: utf-8 -*-
""" data file io
save & retrieve of data to feather format files
data frame wash, including  ttm and columns rename
"""

import re

from Basic import *
from Basic.IO.file import get_direct_files
from Meta import *

OLDEST_DATE = '1988-01-01'


TTM_SUFFIX = '_ttm'



def code2symbol(code):
    """change """
    if '.' in code:
        return code
    tail = 'SH' if code[0] == '6' else 'SZ'
    return code + '.' + tail


class _DataManager:
    CODE_LIST_PATH = DATA_ROOT + 'code_list.csv'
    DATA_FOLDERS = {
        'indicator':      'financial_indicator',
        'balance':        'financial_balance',
        'cash_flow':      'financial_cash_flow',
        'income':         'financial_income',
        'index':          'market_index',
        'macro':          'macro',
        'stock':          'market_stock',
        'temp':           'temp',
        'category':       'category',
        'stock_target':   'target_stock',
        'index_target':   'target_index',
        'cluster_target': 'target_cluster'}

    def __init__(self):
        put_failure_path(get_error_path)
        self._create_all_folders()
        self.code_table = pd.read_csv(self.CODE_LIST_PATH, encoding=GBK)
        self.active_table = self.code_table[self.code_table.stop == False]
        self.code_list = self.code_table['code']
        folder = DATA_ROOT + self.DATA_FOLDERS['index']
        codes = re.compile(r'(\d+)')
        self.idx_list = []
        for file_name in os.listdir(folder):
            # print(1)
            code = codes.search(file_name)[0]
            self.idx_list.append(code)

    @staticmethod
    def __financial_report_sustaining_check():
        def __if_stop(code):
            df = DMGR.read('balance', code)
            if df is not None and df.shape[0] > 0:
                df['quarter'] = df.date.apply(to_quarter)
                cur_quarter = to_quarter()
                for i in range(4):
                    quarter = quarter_add(cur_quarter, -i - 1)
                    if np.any(df.quarter == quarter):
                        return False
            return True

        DMGR.code_table['stop'] = DMGR.code_table.code.apply(__if_stop)
        DMGR.code_table.to_csv(DMGR.CODE_LIST_PATH, encoding=GBK, index=False)

    def _create_all_folders(self):
        for key in self.DATA_FOLDERS:
            folder = DATA_ROOT + self.DATA_FOLDERS[key]
            if not os.path.exists(folder):
                os.makedirs(folder)

    def csv_path(self, category, code):
        folder = DATA_ROOT + self.DATA_FOLDERS[category] + '\\'
        return folder + '%s.csv' % code

    def feather_path(self, category, code):
        folder = DATA_ROOT + self.DATA_FOLDERS[category] + '\\'
        return folder + '%s.feather' % code

    def read(self, category, code):
        path = self.feather_path(category, code)
        if not os.path.exists(path):
            return None
        df = pd.read_feather(path)
        return df

    def read2dict(self, category, code_list, df_transfer=None):
        dic = {}
        for code in code_list:
            df = self.read(category, code)
            if df_transfer is not None:
                df = df_transfer(df)
            dic[code] = df
        return dic

    def save(self, df: pd.DataFrame, category, code):
        if df.index.name is not None:
            if_drop = True if df.index.name in df else False
        else:
            if_drop = True
        df = df.reset_index(drop=if_drop)
        path = self.feather_path(category, code)
        df.to_feather(path)

    @staticmethod
    def update_file(category, code, fetcher, index='date'):
        def __msg(*txt):
            print(category, code, *txt)

        exist = DMGR.read(category, code)
        # DWash.reform_tick(exist)
        if exist is None:
            exist = pd.DataFrame()
            start = OLDEST_DATE
        else:
            # todo simple in future
            exist[index] = exist[index].apply(date_str2std)
            exist = exist[exist[index] == exist[index]]

            # exist = exist[exist['derc_close'] == exist['derc_close']]

            idx = exist[index]
            if idx[0] > idx[1]:
                exist.sort_values(index, inplace=True)
            start = exist[index].iloc[-1]
        if index == 'date':
            dist = today() - str2date(start).date()
            if dist.days <= 3:
                __msg('Last update in %s days, assume filled!' % dist.days)
                return exist
            start = str2date(start, 1).date()
            start = date2str(start)

        new = fetcher(start)
        if new is None:
            __msg('Non Data')
            return exist

        __msg(start, len(new))
        new = pd.concat([exist, new])
        DMGR.save(new, category, code)
        return new

    def loop_stocks(self, func, flag, num_process=4, limit=-1):
        return loop(func, self.code_list, num_process=num_process, flag=flag, limit=limit)

    def loop_index(self, func, flag, num_process=4, limit=-1):
        return loop(func, self.idx_list, num_process=num_process, flag=flag, limit=limit)

    def category_concat(self, code_list, category, columns, start_date):
        tid = category + '_concat' + uid()
        setattr(self, tid, pd.DataFrame())

        def _concat_code(code):
            df = self.read(category, code)
            if df is not None:
                df = df[df.date >= start_date]
                if df.shape[0] > 0:
                    df = df[['date', *columns]]
                    temp = getattr(self, tid)
                    temp = pd.concat([temp, df])
                    setattr(self, tid, temp)

        loop(_concat_code, code_list, tid, show_seq=True)

        return getattr(self, tid)

    def all_csv2feather(self):
        for key in self.DATA_FOLDERS:
            folder = DATA_ROOT + self.DATA_FOLDERS[key] + '\\'
            file_list = get_direct_files(folder)
            for file_name in file_list:
                if ext(file_name) != 'csv':
                    continue  # print(folder,len(files))

    def csv2feather(self, category, code):
        old_file = self.csv_path(category, code)
        new_file = self.feather_path(category, code)
        print(category, code, old_file, new_file)
        df = pd.read_csv(old_file, encoding=GBK)
        if df is None or df.shape[0] == 0:
            return
        df.to_feather(new_file)

    def feather2csv(self, category, code):
        new_file = self.csv_path(category, code)
        old_file = self.feather_path(category, code)
        print(category, code, old_file, new_file)
        df = pd.read_feather(old_file)
        if df is None or df.shape[0] == 0:
            return
        df.to_csv(new_file, index=False, encoding=GBK)


DMGR = _DataManager()


class _DataWasher:
    COLUMN_UPDATE = True
    DUPLICATE_SEPARATOR = '~'
    DUPLICATE_FLAG = '*2'

    def __init__(self):
        self.mapper = get_lib('mapper')
        self.mapper['alias'] = self.mapper['alias'].apply(self._simplify_name)
        if self.COLUMN_UPDATE:
            self.mapper['matchCount'] = 0
            self.mapper['matches'] = ''

        self.match_path = lib_path['matched']
        if not os.path.exists(self.match_path):
            with open(self.match_path, 'w') as match_file:
                match_file.write('{}')
        self.matched = file2obj(self.match_path)

    def raw_regular_i(self, df: pd.DataFrame, category=''):
        df.replace('--', 0, inplace=True)

        self.value_scale_by_column_name(df)

        matches = self._column_match(df, category)
        df.rename(columns=matches, inplace=True)
        self.column_select_i(df)

    # region column name regular
    @staticmethod
    def _simplify_name(name):
        swap_pair = [['所有者', '股东'], ['的', ''], ['所', '']]
        for pair in swap_pair:
            name = name.replace(*pair)
        return name

    def _column_match(self, df: pd.DataFrame, category=''):
        """match columns of df to table<field_mapper.csv>'s standard name"""
        if category != '' and category in self.matched:
            return self.matched[category]
        matches = {}
        for col_name in df:
            col = self._simplify_name(col_name)

            candidates = [[key, row] for key, row in self.mapper.iterrows() if
                          isinstance(row['alias'], str) and row['alias'] in col]

            if candidates:
                if len(candidates) == 1:
                    key, chosen = candidates[0]
                else:
                    key, chosen = max_at(candidates, lambda candidate: len(candidate[1]['alias']))
                matches[col_name] = chosen['field']
                if self.COLUMN_UPDATE:
                    self.mapper.ix[key, 'matchCount'] += 1
                    self.mapper.ix[key, 'matches'] += '%s %s ' % (col, category)
                    self.mapper.to_csv('D:/field_mapper.csv', encoding=GBK)
            else:
                if np.all(col != self.mapper['field']):
                    print(category, col_name, 'column has no match')

        duplicate = {}
        for col in matches:
            field = matches[col]
            duplicate[field] = count = 0 if field not in duplicate else duplicate[field] + 1
            if count > 0 and col != field:
                print('Duplicated column %s: %s -> %s' % (count, col, field))
                matches[col] += '%s%s%s' % (self.DUPLICATE_SEPARATOR, count, self.DUPLICATE_FLAG)
        if category != '':
            self.matched[category] = matches
            obj2file(self.match_path, self.matched)
        return matches

    def column_select_i(self, df):
        keys = {}
        cols_id = [[x.split(self.DUPLICATE_SEPARATOR)[0], x] for x in df.columns]
        for idx, col in cols_id:
            if idx not in keys:
                keys[idx] = [0, [col]]
            else:
                keys[idx][0] += 1
                keys[idx][1].append(col)
        dup_list = [[key, *keys[key]] for key in keys if keys[key][0] > 0]
        for idx, _, fields in dup_list:
            # if keys[id][0] > 1:
            for field in fields:
                if idx == 'date' and field != 'date':
                    if field in df:
                        df.drop(field, axis=1, inplace=True)
                elif idx != field:
                    column_compare_choose_i(df, idx, field)

    # endregion

    # region one time active for files
    @staticmethod
    def simplify_dirs(category):
        folder = DATA_ROOT + _DataManager.DATA_FOLDERS[category] + '\\'
        prefix = re.compile('([0-9]*.csv)')
        for file_name in os.listdir(folder):
            new_file = prefix.search(file_name)[0]
            # print(file,nf)
            os.rename(folder + file_name, folder + new_file)

    def simplify_mapper(self):
        pattern = re.compile(r'(.*)(\s*\(|（)')
        for row in self.mapper.iterrows():
            alias = row[1]['alias']
            # print(alias)
            if isinstance(alias, str) and ('(' in alias or '（' in alias):
                match = pattern.match(alias)
                new_alias = match.group(1)
                self.mapper.ix[row[0], 'alias'] = new_alias
                print(new_alias)

    # endregion

    @staticmethod
    def value_scale_by_column_name(df: pd.DataFrame):
        for col in df:
            if '万元' in col:
                numeric_i(df, col)
                df[col] *= 10000
            elif '率' in col or '%' in col:
                numeric_i(df, col)
                df[col] /= 100
            elif '元' in col:
                numeric_i(df, col)

    # noinspection PyTypeChecker
    @staticmethod
    def percentage_factor_by_values(series: pd.Series):
        series.dropna(inplace=True)
        if np.all(series < 1):
            return 1
        if np.all(series > 2):
            return 100
        raise Exception('Can not determine percentage factor!')

    @staticmethod
    def ttm_column(df, column, new_column=None, n=4):
        new_col = column + TTM_SUFFIX if new_column is None else new_column
        df[new_col] = 0
        if 'quarter' not in df:
            print('no quarter given, aborted!')
            return None

        df = df.sort_values('quarter')
        if n == 2:
            col = df[column].fillna(method='pad')  # todo better fill
            df['quart'] = df['quarter'].apply(lambda x: x.split(QUARTER_SEPARATOR)[1])
            last_col = 'last' + column
            df[last_col] = col.shift(1)

            def ttm2(row):
                cur = row[column]
                last_val = row[last_col]
                res = cur if row['quart'] == '1' else cur - last_val
                return res if cur > 0 else last_val if last_val > 0 else np.nan

            ttm_list = df.apply(ttm2, axis=1)
            ttm_list.fillna(method='pad', inplace=True)
        else:
            last = [0, 0, 0, 0, 0]

            def get_ttm(row, last_val_list):
                cur = row[column]
                year_quarter = row['quarter']
                _, quarter = year_quarter.split(QUARTER_SEPARATOR)
                quarter = int(quarter)
                ttm = cur + last_val_list[4] - last_val_list[quarter]
                last_val_list[quarter] = cur
                return ttm

            ttm_list = df.apply(lambda row: get_ttm(row, last), axis=1)
        return ttm_list

    @staticmethod
    def ttm_dict(val_dict):
        if len(val_dict) == 2:
            quarters = []
            values = []
            for key, value in val_dict.items():
                _, quarter = key.split(QUARTER_SEPARATOR)
                quarters.append(quarter)
                values.append(value)
            if quarters[1] == '1':
                return values[1] if values[1] > 0 else values[0]
            return values[1] - values[0] if values[1] > 0 else values[0]

        last = [0, 0, 0, 0, 0]
        ttm = 0
        for key in val_dict:
            val = val_dict[key]
            _, quarter = key.split(QUARTER_SEPARATOR)
            quarter = int(quarter)
            ttm = val + last[4] - last[quarter]
            last[quarter] = val
        return ttm

    @staticmethod
    def calc_change_i(df: pd.DataFrame):
        pre_close = df['close'].values
        pre_close = np.insert(pre_close, 0, np.nan)
        pre_close = pre_close[0:len(pre_close) - 1]
        df['pre_close'] = pre_close
        df['change_amount'] = df['close'] - df['pre_close']

        if 'change' in df:
            df.drop('change', axis=1, inplace=True)  # print(df)

        def ratio(series):
            return series[1] / series[0] - 1

        pre_close = 'derc_close' if 'derc_close' in df else 'close'
        df['change_rate'] = df[pre_close].rolling(2).apply(ratio)


DWASH = _DataWasher()
