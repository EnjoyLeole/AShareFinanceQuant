# -*- coding: utf-8 -*-
import itchat
import numpy

from Basic.Ext3PL import force_debug
from Basic.IO import file2obj
from Basic.Util import date_of, save_pkg_list
from Meta import META_DIR
from Quat.statistic import *
from Quat.webio import N163, WebCrawler

simples = {
    'letv':   '300104',
    'zzd':    '002069',
    'bym':    '002570',
    'maotai': '600519'}
wk = '000002'
mt = '600519'
pingan = '601318'
hr = '600276'
letv = '300104'
bcode = '000001'
sts = '2018-01-01'
end = '2018-01-20'
quarter = '2017-3'
pt = '002500'
start = date_of(2018, 1, 1)
path = 'D:/test.csv'
dict = {
    'A': 100,
    'B': 200,
    'D': 300}
list = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
tdf = pd.DataFrame(list, columns=['A', 'B', 'B'])
se = pd.Series([np.nan, 1, np.nan, 5, np.nan])
formula = 'market_cap+total_liability-1cas3423h_at_end0~2+1200'
eqt = '21361470000.0+107340000.0+7131370000.0+976342054116.0/21361470000.0+0.0+0.0'

error_file = 'error_reshow'


# region web
def all_update():
    WebCrawler.index()


def idx_hit():
    df = N163.fetch_hist('000001', start, index=True)
    print(df)


def N163_test():
    N163.fetch_stock_combined_his(bcode, start)


def derc_test():
    df = N163._update_stock_derc(bcode)
    # df=N163.fetch_derc(bcode, 2013)
    print(df)


# endregion

# region basic func

def fin_test():
    code = "agg_profit"
    cate = "temp"
    df = DMGR.read(cate, code)
    # self.mapper.to_csv("d:/test.csv")

    df = DWASH.quarter_ttm(df, 'net_profit')
    print(df)


def reform_test():
    df = DMGR.read('stock', '000001')  # DWash.reform_tick(df,'index')


# endregion

# region equation resolve
def __byline(cls, form):
    def __get_val(obj, row):
        getter = {
            'obj': lambda row: getattr(getattr(obj, row.source), row.factor),
            'df':  lambda row: getattr(obj, row.factor)}
        if row.factor in cls.targets:
            val = cls.funcs[row.factor](obj)
        else:
            val = getter['df'](row)
        return val

    def __calculator(obj, input_type='df'):
        calc = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y}
        result = None
        legal = []
        for i, row in form.iterrows():
            if not isinstance(row.factor, str):
                continue
            val = __get_val(obj, row)
            if result is None:
                if row.operator in ['+', '-']:
                    result = 0
                    legal = ['+', '-']
                elif row.operator in ['*', '/']:
                    if val == 0:
                        raise Exception('ZERO as multiple factor!')
                    result = 1
                    legal = ['*', '/']
                else:
                    raise Exception('Operator %s is not defined' % row.operator)
            if row.operator not in legal:
                raise Exception('In-consistent operators!')
            # print(result,row.operator,val)
            result = calc[row.operator](result, val)
        return result

    return __calculator


def reg_resolve_test(formula):
    import re

    bracket = '([()])'
    reg_bracket = re.compile(bracket)
    mt = reg_bracket.split(formula)
    operator = '([+\-\*/^])'
    reg_operator = re.compile(operator)
    calc = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '^': lambda x, y: math.pow(x, y)}

    def process(list):
        bracketCount = 0
        sublist = []
        for part in list:
            if part == '(':
                bracketCount += 1
                continue
            elif part == ')':
                bracketCount -= 1
                continue
            # print(part)
            if bracketCount == 0 and len(sublist) > 0:
                val = process(sublist)
                sublist = []
            if bracketCount > 0:
                sublist.append(part)
            else:
                dv = reg_operator.split(part)
                for element in dv:
                    if element == '':
                        continue
                    print(element)

    process(mt)


# endregion

# region formula
def quarter_date_merge_test():
    quart = DMGR.read('balance', bcode)
    dat = DMGR.read('stock', bcode)


def stock_test(field):
    df = None
    for key, code in simples.items():
        # key,code='zzd','002069'
        print(key, code)
        tt = Stocks(simples[key])
        tt.target_series(field)
        # res=tt.quota

        res = tt.targets.loc[:, [field]]
        res.rename(columns={
            field: key}, inplace=True)

        if df is None:
            df = res
        else:
            df = pd.merge(res, df, left_index=True, right_index=True, how='outer')
    df.to_csv(path)
    print(df)


def code_re():
    DMGR.code_table['code'] = DMGR.code_table['code'].apply(lambda x: x.zfill(6))
    df = DMGR.code_table
    df.to_csv('D:/code.csv', encoding=GBK)


# endregion

# region numba
li = [x for x in range(100)]
li = numpy.array(li)


def python(arr=li):
    val = sum(arr)  # print(val)


@numba.jit
def numba(arr=li):
    val = sum(arr)  # print(val)


# endregion
def force():
    force_debug()
    FORMULA.If_Debug = True


def wechat(who='filehelper'):
    itchat.auto_login()
    def to(user):
        itchat.send('Hello, message from python', toUserName=user)
    itchat.send('Hello, message from python', toUserName=who)
    friends=itchat.get_friends()
    to(friends[0])
    rooms=itchat.get_chatrooms()
    to(rooms[0].UserName)
    # itchat.send('Hello, message from python', toUserName='filehelper')
    room=itchat.search_chatrooms(name='股票与基金交流')
    to(room[0].UserName)

    # itchat.send('Hello, message from python', toUserName='股票与基金交流')
def df_test():
    # d=pd.Series([np.nan,1,np.nan,4,np.nan])
    # print(d.fillna(method='bfill'))

    df = DMGR.read('stock', bcode)
    df = DWASH.fill_derc(df)
    print(df[['i_close', 'derc_close', 'close']])
    se = df['close']
    v = np.sum(se == se)
    e = (se == se).sum()
    print(v, e)  # df.to_csv(path)


def feather_test():
    df = DMGR.read('stock_target', '002004')
    df.index = df.quarter
    DMGR.save(df, 'temp', 'test_feather')


def financial_expense_compare():
    dic = {}
    for code in DMGR.code_list:
        stk = Stocks(code)
        res_li = stk._financial_compare()
        if res_li is None:
            continue
        print(code)
        res = res_li[0]
        if res not in dic:
            dic[res] = 1
        else:
            dic[res] += 1
    print(dic)


def index_target_test(code='399300'):
    idx = Indexes('hs300', '399300', if_update=False)
    idx.calc_list()
    idx.save_targets()
    r = idx.quarter_performance()


def stock_vector_test(code=hr, target=None):
    stk = Stocks(code)
    if target is None:
        df_res = stk.calc_list()

        stk.save_targets()
    else:
        res = stk.calc_target(target=target)
        # print(stk.major['depreciation'])
        stk.save_targets()
        print(res)


def get_error_codes(f):
    ef = 'error_reshow'
    path = get_error_path(f)
    li = file2obj(path)
    li = [x[1] for x in li]
    return li


def error_reshow(f):
    li = get_error_codes(f)
    # Stocks.targets_cluster2stock(target_list=FORMULA.sub_targets[POLICY], code_list=li)
    # DWASH.fill_miss_tick_from_backup(li)
    # DWASH.all_lines_fill_derc('lines',li)
    # def test(code):
    #     df = DMGR.read('stock', code)
    #     df = DWASH.fill_derc(df)
    #     return df
    #
    # for c in li:
    #     N163.update_stock_hist(c)
    # Stocks.simplify_all_lines(li)
    Stocks.targets_calculate(code_list=li)
    # Stocks.targets_cluster2stock(code_list=li)
    # loop(stock_vector_test, li, 1, flag='error_reshow', if_debug=True)


def test():
    # force()
    code = '000786'
    WebCrawler.override_finances()

    # print('root', META_DIR)
    # WebCrawler.everything()
    # save_pkg_list()
    # stock_vector_test(code)
    # Graph.analysis(code)

    # Stocks.targets_cluster2stock()
    # error_reshow('targets_calculate')

    # N163.override_finance(bcode)

    # Stocks.target_pipeline()

    # DMGR.feather2csv('stock_target', code)

    # Strategy.find_security( )

    pass
