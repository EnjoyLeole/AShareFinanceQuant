# -*- coding: utf-8 -*-
""" data file io
save & retrieve of data to feather format files
data frame wash, including  ttm and columns rename
"""

import os
import re

import numpy as np
import pandas as pd

from Basic.Ext3PL import column_compare_choose_inplace, numeric_inplace
from Basic.Ext3PL.multipro import loop, put_failure_path
from Basic.IO.file import ext, get_direct_files, obj2file
from Basic.Util import QUARTER_SEPARATOR, date2str, date_str2std, max_at, quarter_add, str2date, \
    to_quarter, today, uid
from Meta import DATA_ROOT, GBK, get_error_path, get_lib, get_line_backup, lib_path

OLDEST_DATE = '1988-01-01'

TTM_SUFFIX = '_ttm'


def code2symbol(code):
    """change """
    if '.' in code:
        return code
    tail = 'SH' if code[0] == '6' else 'SZ'
    return code + '.' + tail


def std_code_col_inplace(df):
    if 'code' not in df:
        print('code not in df!')
        return
    df['code'] = df['code'].apply(lambda val: str(val).zfill(6))


class _DataManager:
    CODE_LIST_PATH = DATA_ROOT + 'my_code_list.csv'
    DATA_FOLDERS = {
        '':               '',
        'indicator':      'financial_indicator',
        'balance':        'financial_balance',
        'cash_flow':      'financial_cash_flow',
        'income':         'financial_income',
        'index':          'lines_index',
        'stock':          'lines_stock',
        'lines':          'lines_stock_simple',
        'macro':          'macro',
        'temp':           'temp',
        'category':       'category',
        'stock_target':   'target_stock',
        'index_target':   'target_index',
        'cluster_target': 'target_cluster'}

    @property
    def code_details(self):
        return self.code_table[['code', 'secShortName', 'industry']]

    def __init__(self):
        put_failure_path(get_error_path)
        self._create_all_folders()
        self.code_table = pd.read_csv(self.CODE_LIST_PATH, encoding=GBK)
        std_code_col_inplace(self.code_table)
        self.code_table.index = self.code_table.code
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

    def code_name(self, code):
        return self.code_table.loc[code, 'secShortName']

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
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            return None
        df = pd.read_feather(path)
        return df

    def read2dict(self, category, code_list, df_transfer=None):
        dic = {}
        for code in code_list:
            df = self.read(category, code)
            if df is None:
                print(code, 'not exist!')
                continue
            if df_transfer is not None:
                df = df_transfer(df)
            dic[code] = df
        return dic

    def save(self, df: pd.DataFrame, category, code, if_object2str=False):
        if df.empty or df.shape[0] == 0:
            print(category, code, 'df empty, saving abort!')
            return
        if df.index.name is not None:
            if_drop = True if df.index.name in df else False
        else:
            if_drop = True
        df = df.reset_index(drop=if_drop)
        path = self.feather_path(category, code)
        if if_object2str:
            for col in df:
                if isinstance(df[col].dtype, object):
                    df[col] = df[col].astype(str)

        df.to_feather(path)

    def read_csv(self, category, code):
        csv_path = self.csv_path(category, code)
        return pd.read_csv(csv_path, encoding=GBK)

    def save_csv(self, df, category, code, index=False):
        csv_path = self.csv_path(category, code)
        df.to_csv(csv_path, index=index, encoding=GBK)

    @staticmethod
    def update_file(category, code, fetcher, index='date'):
        def __msg(*txt):
            print(category, code, *txt)

        exist = DMGR.read(category, code)
        # DWash.reform_tick(exist)
        if exist is None or exist.empty:
            exist = pd.DataFrame()
            start = OLDEST_DATE
        else:
            # todo hopefully only once
            exist[index] = exist[index].apply(date_str2std)
            exist = exist[exist[index] == exist[index]]
            if exist.empty:
                exist = pd.DataFrame()
                start = OLDEST_DATE
            else:
                idx = exist[index]

                if exist.shape[0] > 1 and idx[0] > idx[1]:
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
            __msg('Non Data Crawled')
            return exist

        __msg(start, len(new))
        new = pd.concat([exist, new])
        DMGR.save(new, category, code, if_object2str=True)
        return new

    def category_concat(self, code_list, category, columns, start_date):
        tid = category + uid()
        setattr(self, tid, pd.DataFrame())

        def _concat_code(code):
            df = self.read(category, code)
            if df is not None:
                df = df[df.date >= start_date]
                if not df.empty:
                    df = df[['date', *columns]]
                    temp = getattr(self, tid)
                    temp = pd.concat([temp, df])
                    setattr(self, tid, temp)

        loop(_concat_code, code_list, flag=tid, show_seq=True)

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
        if not os.path.exists(old_file):
            return
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
        self.matched = {}
        # self.matched = file2obj(self.match_path)

    # raw data regular
    def raw_regular(self, df: pd.DataFrame, category=''):
        for invalid_str in ['--', ' --', '', ' ', '\t\t']:
            df.replace(invalid_str, np.nan, inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df = self._value_scale_by_column_name(df)
        matches = self._column_match(df, category)
        df.rename(columns=matches, inplace=True)

        df = self.column_select(df)
        return df

    @staticmethod
    def _simplify_name(name):
        swap_pair = [['所有者', '股东'], ['的', ''], ['所', '']]
        for pair in swap_pair:
            name = name.replace(*pair)
        return name

    @staticmethod
    def _value_scale_by_column_name(df: pd.DataFrame):
        for col in df:
            if '万元' in col:
                df = numeric_inplace(df, col)
                df[col] = df[col] * 10000
            elif '率' in col or '%' in col:
                df = numeric_inplace(df, col)
                df[col] = df[col] / 100
            elif '元' in col:
                df = numeric_inplace(df, col)
        return df

    def _column_match(self, df: pd.DataFrame, category=''):
        """match columns of df to table<field_mapper.csv>'s standard name"""
        if category != '' and category in self.matched and self.matched[category]:
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
                    if col_name not in ['index']:
                        print(category, col_name, 'column has no match')

        duplicate = {}
        for col in matches:
            field = matches[col]
            duplicate[field] = count = 0 if field not in duplicate else duplicate[field] + 1
            if count > 0 and col != field:
                print('Duplicated column %s: %s of %s' % (count, col, field))
                matches[col] += '%s%s%s' % (self.DUPLICATE_SEPARATOR, count, self.DUPLICATE_FLAG)
        if category != '':
            self.matched[category] = matches
            obj2file(self.match_path, self.matched)
        return matches

    def column_select(self, df, code=''):
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
                    column_compare_choose_inplace(df, idx, field, flag=f"{code} duplicate {field}")
        return df

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

    # endregion

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
        if 'quart' not in df:
            df['quart'] = df['quarter'].apply(lambda x: x.split(QUARTER_SEPARATOR)[1])
        df = df.sort_values('quarter')
        if n == 2:
            # col = df[column].fillna(method='pad')
            last_col = 'last' + column
            df[last_col] = df[column].shift(1)

            def ttm2(row):
                cur = row[column]
                cur = cur if cur == cur else 0
                last_val = row[last_col]
                last_val = last_val if last_val == last_val else 0
                res = cur if row['quart'] == '1' else cur - last_val
                if cur * last_val == 0:
                    max_val = max(cur, last_val)
                    if max_val == 0:
                        val = np.nan
                    else:
                        val = max_val / 2
                else:
                    val = res
                return val

            ttm_list = df.apply(ttm2, axis=1)
            # ttm_list.fillna(method='pad', inplace=True)
        else:
            last = [0, 0, 0, 0, 0]
            last_ttm = [0, 0, 0, 0, 0]

            def get_ttm(row, last_val_list, last_ttm_list):
                cur = row[column]
                cur = cur if cur == cur else 0
                # year_quarter = row['quarter']
                quarter = row['quart']
                quarter = int(quarter)
                if cur == 0 and last_val_list[quarter] == 0:
                    last_quarter = quarter - 1 if quarter != 1 else 4
                    ttm = last_ttm_list[last_quarter]
                    last_val_list[quarter] = 0
                else:
                    last_ttm_list[quarter] = ttm = cur + last_val_list[4] - last_val_list[quarter]
                    last_val_list[quarter] = cur
                return ttm

            ttm_list = df.apply(lambda row: get_ttm(row, last, last_ttm), axis=1)
            # print(df[column], ttm_list)
        return ttm_list

    @staticmethod
    def calc_change(df: pd.DataFrame):
        df = numeric_inplace(df, ['close'])
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
        return df

    @staticmethod
    def fill_derc(df: pd.DataFrame):
        if 'derc_close' not in df and 'backward_right_price' in df:
            df['derc_close'] = df['backward_right_price']

        cols = ['derc_close', 'close', 'high', 'low', 'open']
        numeric_inplace(df, cols)
        df['factor'] = df['derc_close'] / df['close']
        # using backward fill to complete major derc factors
        df['factor'].fillna(method='bfill', inplace=True)
        # and forward fill for recently missing
        df['factor'].fillna(method='ffill', inplace=True)
        df['i_close'] = df.close * df.factor
        df['i_high'] = df.high * df.factor
        df['i_low'] = df.low * df.factor
        df['i_open'] = df.open * df.factor
        return df

    @staticmethod
    def fill_miss_tick_from_backup(code_list=None):
        def fill(code):
            backup = get_line_backup(code)
            if backup is None:
                return
            backup = DWASH.raw_regular(backup, 'backup')
            backup.index = backup.date.apply(date_str2std)
            df = DMGR.read('stock', code)
            df.index = df['date']
            missed = [x for x in backup.index if x not in df.index]
            if not missed:
                return
            print(code, len(missed), 'missing lines')
            miss_in_backup = backup.loc[missed]
            for key, row in miss_in_backup.iterrows():
                df.loc[key] = row
            df.sort_index(inplace=True)
            df = DWASH.fill_derc(df)

            DMGR.save(df, 'stock', code)

            # raise Exception('missed')

        loop(fill, code_list, num_process=5)

    @staticmethod
    def all_lines_fill_derc(category='stock', code_list=None):
        if category not in ['stock', 'lines']:
            print(category, 'can not fill derc')
            return
        code_list = code_list if code_list else DMGR.code_list

        def fill(code):
            df = DMGR.read(category, code)
            if df is None or df.empty:
                return
            df = DWASH.fill_derc(df)
            DMGR.save(df, category, code)

        loop(fill, code_list, num_process=7)


DWASH = _DataWasher()
