import pandas as pd
import os
from Basic.IO import file2obj

GBK = 'gbk'
__file_name = {
    'mapper' : 'field_mapper.csv',
    'formula': 'formula.csv',
    'matched': 'matched.json'}
lib = {}

dir = os.path.dirname(os.path.abspath(__file__))


def get_error_path(name):
    return dir + '\\' + name


def get_error_list(name):
    path = get_error_path(name+'.txt')
    return file2obj(path)


def get_lib_path(key):
    return dir + '\\' + __file_name[key]


def get_lib(key):
    lib[key] = pd.read_csv(get_lib_path(key), encoding = GBK)
    return lib[key]
