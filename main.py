from timeit import timeit

import numpy

from Quant.statistic import *
from Meta import *

simples = {
    'letv'  : '300104',
    'zzd'   : '002069',
    'bym'   : '002570',
    'maotai': '600519'}
hr = '600276'
letv = '300104'
mt = '600519'
bcode = '000001'
code = '300104'
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
tdf = pd.DataFrame(list, columns = ['A', 'B', 'B'])

formula = 'market_cap+total_liability-1cas3423h_at_end0~2+1200'
eqt = '21361470000.0+107340000.0+7131370000.0+976342054116.0/21361470000.0+0.0+0.0'

error_file = 'error_reshow.txt'


# region web
def all_update():
    Updater.crawler_idx()


def idx_hit():
    df = N163.fetch_idx_hist('000001', start)
    print(df)


def N163_test():
    N163.fetch_stock_combined_his(code, start)


def derc_test():
    df = Updater._update_stock_derc(bcode)
    # df=N163.fetch_derc(bcode, 2013)
    print(df)


# endregion

# region basic func

def fin_test():
    code = "agg_profit"
    cate = "temp"
    df = DMgr.read(cate, code)
    # self.mapper.to_csv("d:/test.csv")

    df = DWash.quarter_ttm(df, 'net_profit')
    print(df)


def reform_test():
    df = DMgr.read('stock', '000001')  # DWash.reform_tick(df,'index')


# endregion

# region equation resolve
def __byline(cls, form):
    def __get_val(obj, row):
        getter = {
            'obj': lambda row: getattr(getattr(obj, row.source), row.factor),
            'df' : lambda row: getattr(obj, row.factor)}
        if row.factor in cls.targets:
            val = cls.funcs[row.factor](obj)
        else:
            val = getter['df'](row)
        return val

    def __calculator(obj, input_type = 'df'):
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
    quart = DMgr.read('balance', code)
    dat = DMgr.read('stock', code)
    # d1 = brief_detail_merge(quart, dat, ifReduce2brief = True)
    # d2 = brief_detail_merge(quart, dat, ifReduce2brief = False)
    # qddate=quart[['date','quarter']]
    # dq=d1.loc[:,['date','quarter']]


def stock_test(field):
    df = None
    for key, code in simples.items():
        # key,code='zzd','002069'
        print(key, code)
        tt = Stocks(simples[key])
        tt.target_series(field)
        # res=tt.quota

        res = tt.targets.loc[:, [field]]
        res.rename(columns = {
            field: key}, inplace = True)

        if df is None:
            df = res
        else:
            df = pd.merge(res, df, left_index = True, right_index = True, how = 'outer')
    df.to_csv(path)
    print(df)


def code_re():
    DMgr.code_table['code'] = DMgr.code_table['code'].apply(lambda x: x.zfill(6))
    df = DMgr.code_table
    df.to_csv('D:/code.csv', encoding = GBK)


# endregion

# region numba
li = [x for x in range(100)]
li = numpy.array(li)


def python(arr = li):
    val = sum(arr)
    # print(val)


@numba.jit
def numba(arr = li):
    val = sum(arr)
    # print(val)


# endregion

def df_test():
    df = DMgr.read('stock', letv)
    se=df['close']
    v=np.sum(se==se)
    e=(se==se).sum()
    print(v,e)
    # df.to_csv(path)


def feather_test():
    df = DMgr.read('stock_target', '002004')
    df.index = df.quarter
    DMgr.save(df, 'temp', 'test_feather')


def index_target_test(code = '399300'):
    id = Indexes('hs300', code)
    id.calc_list()


def stock_vector_test(code = hr, target = None):
    stk = Stocks(code)
    if target is None:
        df_res = stk.calc_list()
        stk.save_targets()
    else:
        res = stk.calc_target(target = target)
        stk.save_targets()
        print(res)


def error_reshow():
    path = get_error_path('')
    list = file2obj(path)
    list = [x[1] for x in list]
    res = loop(lambda code: stock_vector_test(code = code), list, 1, flag = 'error_reshow',
        if_debug = True)
    obj2file(get_error_path(error_file), res)
    print(res)


def financial_expense_compare():
    dic = {}
    for code in DMgr.code_list:
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


def war_game():
    save_folder = 'C:\\Users\\Hui Lei\\Saved Games\\EugenSystems\\WarGame3\\'
    li = get_direct_files(save_folder)
    for f in li:
        if ext(f) == 'wargamerpl2':
            with open(save_folder + f, 'r', encoding = 'latin_1') as file:
                content = file.read()
                if '[IRQ] 1st Special Operations Brigade' in content:
                    player_num = '"NbPlayer":"(\d+)"'
                    nums = re.findall(player_num, content)
                    print(f, 'players', nums[0])

                    player_name = '"PlayerName":"(.+?)"'
                    player_side = '"PlayerAlliance":"(\d)"'
                    names = re.findall(player_name, content)
                    sides = re.findall(player_side, content)
                    for i in range(len(names)):
                        if names[i] in ['[IRQ] 1st Special Operations Brigade','LongDiDi']:
                            print(names[i], sides[i])
                    # print(content)
                    # break

def test():
    df_test()
    # Updater.cluster_spread(Formula.polices)
    # Strategy.find_security()
    # Updater.all_finance()
    # DWash.csv2feather()
    # DMgr.read2dict('cluster_target',Formula.key_targets)
    # Updater.targets_stock2cluster(Formula.polices)
    # feather_test()
    # DMgr.feather2csv('cluster_target','Policy')
    # stock_vector_test(bcode)

if __name__ == '__main__':
    print('start')
    print(timeit(test, number = 1))
    times = 100
    # print(timeit(df_test, number = times))
    # print(timeit(feather_test, number = times))
