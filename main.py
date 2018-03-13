from Funds.statistic import *
from Citics.trade import *
from Meta import *

simples = {
    'letv'  : '300104',
    'zzd'   : '002069',
    'bym'   : '002570',
    'maotai': '600519'}
hr='600276'
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


# region web
def all_update():
    Updater.all_idx()


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
    df = DMgr.read_csv(cate, code)
    # self.mapper.to_csv("d:/test.csv")

    df = DWash.quarter_ttm(df, 'net_profit')
    print(df)


def reform_test():
    df = DMgr.read_csv('stock', '000001')  # DWash.reform_tick(df,'index')


def dm_save_test():
    df = pd.DataFrame()

    print(df)
    DMgr.save_csv(df, 'temp', 't02', index = True)


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
    quart = DMgr.read_csv('balance', code)
    dat = DMgr.read_csv('stock', code)
    d1 = brief_detail_merge(quart, dat, ifReduce2brief = True)
    d2 = brief_detail_merge(quart, dat, ifReduce2brief = False)
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


def error_reshow(name):
    list = file2obj(get_error_path(name))
    for code in list:
        N163.override_finance(code[1])
    # codes = ['000003']

    # Updater._update_stock_hist(codes[0])


# endregion

def stock_target_test(code = '000001', target = None, idx = 0):
    t0 = datetime.now()
    stk = Stocks(code)
    if target is None:
        stk.all_target()
        # print(stk.data)
    else:
        if idx == 0:
            vs = stk.target_series(target)
            # print(vs)
        else:
            v = stk.target_calc(target, idx)
            print(target, code, v)
        stk.save_targets()
    print(datetime.now() - t0)


def stock_vector_test(target = None, code = hr):
    t0 = datetime.now()
    stk = Stocks(code)
    if target is None:
        df_res = stk.all_vector()
    else:
        stk.target_vector(target = target)
    print(datetime.now() - t0)


def stock_calc_compare():
    stk = Stocks(simples['letv'])
    df2 = stk.all_vector()
    df1 = stk.all_target()

    df = df2['Zscore'] - df1['Zscore']
    print(df)


def index_target_test(code = '399300'):
    id = Indexs('hs300', code)
    id.all_target()


def stockgroup_test():
    all = StockGroups(DMgr.code_list)
    all.all_targets()


if __name__ == '__main__':
    # stockgroup_test()
    # stock_target_test(code = '300104')
    # stock_target_test()
    stock_vector_test()
    # stock_calc_compare()
    pass
