import os

import pandas as pd

from Basic.IO import file2obj

GBK = 'gbk'

META_DIR = os.path.dirname(os.path.abspath(__file__)) + '\\'
DATA_ROOT = 'C:\\StockData\\'
BACKUP_DIR = META_DIR + 'Old\\lines\\'
lib_path = {
    'code_table':  META_DIR + 'code_list.csv',
    # 'index_table': META_DIR + 'index_list.csv',
    'mapper':      META_DIR + 'field_mapper.csv',
    'formula':     META_DIR + 'formula.csv',
    'matched':     DATA_ROOT + 'matched.json'}
lib = {}


def get_lib(key):
    global lib
    if key not in lib:
        lib[key] = pd.read_csv(lib_path[key], encoding=GBK)
    return lib[key]


def get_error_path(name):
    return META_DIR + '\\%s.txt' % name


def get_error_list(name):
    path = get_error_path(name)
    return file2obj(path)


def get_line_backup(code):
    prefix = 'sh' if code[0] == '6' else 'sz'
    backup_file = BACKUP_DIR + prefix + code + '.csv'
    if not os.path.exists(backup_file):
        return None
    df = pd.read_csv(backup_file, encoding=GBK)
    return df
