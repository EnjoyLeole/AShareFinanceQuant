# huileitest@abc123
import pandas as pd
from datetime import datetime
from Basic.Util import uid
from Basic.IO import list2csv, dbf2df
from Stock.DataApi import *

Acct = 'huileitest'
# Acct = '700497'
CurrAcctType = 'simulator'

TradeSide = {
    'buy': '1',
    'sell': '2',
    'buyETF': 'F',
    'sellETF': 'G'}
AcctType = {
    'allStock': '0',
    'szStock': 'F0',
    'shStock': 'SHF0',
    'simulator': 'S0'}
OrderStatus = {
    0: '已报',
    1: '部分成交',
    2: '全部成交',
    3: '部分撤单',
    4: '全部撤单',
    5: '交易所拒单',
    6: '柜台未接受'}
OrderType = {
    'limit': '0',
    'others_best': 'Q',
    'self_best': 'S',
    'best5_limit': 'R',
    'best5_cancel': 'U',
    'now_or_never': 'T',
    'all_or_nothing': 'V'}


def _get_path(name, suffix = ''):
    ifDbf = False
    DbfRoot = 'D:\Program Files\Wealth CATS 4.0_TestOut\scan_order\\'
    CsvRoot = 'D:\Program Files\Wealth CATS 4.0_TestOut\CSVClientTrade\\'
    File = {
        'order': '\InPut\\',
        'asset': 'asset',
        'report': 'order_updates'}
    root = DbfRoot if ifDbf else CsvRoot
    ext = '.dbf' if ifDbf else '.csv'
    return root + File[name] + suffix + ext


def policy():
    return 0, 0, 0


def trade_run():
    ongoing = []

    def check_report():
        report = _citics_df('report')
        for id in ongoing:
            path = _get_path('order', id)
            import os
            if os.path.exists(path):
                print('WCATS may not running! %s non-read!' % path)
                continue
            curr = report[report.client_id == id]
            if len(curr) > 0:
                status = curr.order_status
                cont = True if status in [0, 1, 3] else False
                follows = ''
                if cont:
                    ongoing.remove(id)
                    follows = 'keep watching'
                print('%s status: %s  %s' % (id, OrderStatus[status], follows))
            else:
                print('WCATS doesnt has record of %s ' % id)

    while (True):
        # current position
        # get policy
        code, trade_cate, ord_qty, ord_price = policy()
        # delivery order
        nid = write_order(code, trade_cate, ord_qty, ord_price)
        ongoing.append(nid)
        # check report
        check_report()


def write_order(code, trade_cate, ord_qty, ord_price = 0, ord_cate = 'self_best'):
    OrderColumns = ['inst_type', 'acct_type', 'acct', 'symbol', 'ord_qty', 'tradeside',
                    'ord_price',
                    'ord_type', 'client_id', 'datetime']
    inst_type = 'O'
    acct_type = AcctType[CurrAcctType]
    symbol = code2symbol(code)
    trade_side = TradeSide[trade_cate]
    ord_type = OrderType[ord_cate]

    client_id = uid()
    orderFile = _get_path('order', client_id)
    order = [inst_type, acct_type, Acct, symbol, ord_qty, trade_side, ord_price, ord_type]
    list2csv(orderFile, order)
    return client_id


def withdraw_order(ord_no):
    inst_type = 'C'
    acct_type = AcctType[CurrAcctType]
    cmd = [inst_type, acct_type, Acct, ord_no]
    orderFile = _get_path('order', )
    list2csv(orderFile, cmd)


def _citics_df(name):
    path = _get_path(name)
    return dbf2df(path)
