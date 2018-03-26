# huileitest@abc123
from Basic.IO import dbf2df, list2csv
from Basic.Util import uid
from Quant import *
from .privacy import *


def _get_path(name, suffix=''):
    if_dbf = False

    files = {
        'order':  '\InPut\\',
        'asset':  'asset',
        'report': 'order_updates'}
    root = dbf_root if if_dbf else csv_root
    ext = '.dbf' if if_dbf else '.csv'
    return root + files[name] + suffix + ext


def policy():
    return 0, 0, 0, 0


def trade_run():
    ongoing = []

    def check_report():
        report = _citics_df('report')
        for idx in ongoing:
            path = _get_path('order', idx)
            import os

            if os.path.exists(path):
                print('WCATS may not running! %s non-read!' % path)
                continue
            curr = report[report.client_id == idx]
            if len(curr) > 0:
                status = curr.order_status
                cont = True if status in [0, 1, 3] else False
                follows = ''
                if cont:
                    ongoing.remove(idx)
                    follows = 'keep watching'
                print('%s status: %s  %s' % (idx, OrderStatus[status], follows))
            else:
                print('WCATS doesnt has record of %s ' % idx)

    while True:
        # current position
        # get policy
        code, trade_cate, ord_qty, ord_price = policy()
        # delivery order
        nid = write_order(code, trade_cate, ord_qty, ord_price)
        ongoing.append(nid)
        # check report
        check_report()


def write_order(code, trade_cate, ord_qty, ord_price=0, ord_cate='self_best'):
    order_columns = ['inst_type', 'acct_type', 'acct', 'symbol', 'ord_qty', 'tradeside',
                     'ord_price', 'ord_type', 'client_id', 'datetime']
    inst_type = 'O'
    acct_type = AcctType[CurrAcctType]
    symbol = code2symbol(code)
    trade_side = TradeSide[trade_cate]
    ord_type = OrderType[ord_cate]

    client_id = uid()
    order_file = _get_path('order', client_id)
    order = [inst_type, acct_type, Acct, symbol, ord_qty, trade_side, ord_price, ord_type]
    list2csv(order_file, order)
    return client_id


def withdraw_order(ord_no):
    inst_type = 'C'
    acct_type = AcctType[CurrAcctType]
    cmd = [inst_type, acct_type, Acct, ord_no]
    order_file = _get_path('order', )
    list2csv(order_file, cmd)


def _citics_df(name):
    path = _get_path(name)
    return dbf2df(path)
