# -*- coding: utf-8 -*-
"""
main tick data & finance report from www.163.com
tushare offers macro economic data

"""

import json

import tushare as ts
import yahoo_finance as yf
from urllib3 import PoolManager

from .dataio import *


def get_url(url, encoding=''):
    if encoding == '':
        encoding = GBK
    pool = PoolManager()
    res = pool.request('get', url)
    res_dict = res.data.decode(encoding)
    return res_dict


class WebCrawler:

    @classmethod
    def everything(cls):
        # raw data fetch
        cls.index()
        cls.stock()
        cls.finance()
        cls.macro()

    # region raw data
    @classmethod
    def macro(cls):
        Tuget.override_macros()
        Tuget.override_margins()
        Tuget.override_shibor()
        Tuget.override_category()

    @classmethod
    def finance(cls):
        DMGR.loop_stocks(N163.override_finance, 'financial_override', num_process=12)

    @classmethod
    def stock(cls):
        DMGR.loop_stocks(N163.update_stock_hist, 'stock_update', num_process=12)

    @classmethod
    def index(cls):
        DMGR.loop_index(N163.update_idx_hist, 'index_update', num_process=12)

    # endregion


class N163(object):
    # 0 for index   1 for stock
    url_his = "http://quotes.money.163.com/service/chddata.html?code=%s%s&start=%s"
    url_finance = {
        "indicator": "https://quotes.money.163.com/service/zycwzb_%s.html?type=report",
        "balance":   "https://quotes.money.163.com/service/zcfzb_%s.html",
        "income":    "https://quotes.money.163.com/service/lrb_%s.html",
        "cash_flow": "https://quotes.money.163.com/service/xjllb_%s.html"}

    right = ['kline', 'klinederc']
    period = ['day', 'week', 'month']

    # "dbfx": "https://quotes.money.163.com/f10/dbfx_%s.html"}
    @classmethod
    def update_idx_hist(cls, code):
        category = 'index'
        DMGR.update_file(category, code,
                         lambda start_year: N163.fetch_hist(code, start_year, index=True))

    @classmethod
    def update_stock_hist(cls, code):
        category = 'stock'
        DMGR.update_file(category, code,
                         lambda start_year: N163.fetch_stock_combined_his(code, start_year))

    @classmethod
    def override_finance(cls, code):
        for cate in cls.url_finance:
            url = cls.url_finance[cate] % code
            # print(url)
            df = pd.read_csv(url, encoding=GBK)
            df.set_index(df.columns[0], inplace=True)
            # DWash.replaceI(df, '--', 0)
            df.dropna(1, 'all', inplace=True)
            df = df.T.reset_index()
            DWASH.raw_regular_i(df, 'raw_finance_' + cate)
            df.rename(columns={
                "index": "date"}, inplace=True)
            DMGR.save(df, cate, code)
        print("%s saved" % code)

    @classmethod
    def fetch_stock_combined_his(cls, code, start, end=None):
        his = cls.fetch_hist(code, start, end)
        if his is None:
            return None

        his = his[his['close'] != 0]
        if isinstance(start, str):
            start_year = int(start.split(DATE_SEPARATOR)[0])
        else:
            start_year = start.year
        derc = cls.fetch_derc(code, start_year, end.year if end is not None else None)
        if derc is not None:
            df = pd.merge(his, derc, on='date', how='left')
            df['factor'] = df['derc_close'] / df['close']
        else:
            df = his
        df.sort_values('date', inplace=True)
        DWASH.calc_change_i(df)
        return df

    @classmethod
    def fetch_derc(cls, code, start_year, end_year=None):
        start_year = max(start_year, 2014)
        end_year = today().year if end_year is None else end_year

        def _derc(year):
            exchange = 0 if code[0] == '6' else 1
            url = "http://img1.money.126.net/data/%s/%s/%s/history/%s/%s%s.json" % (
                'hs', cls.right[1], cls.period[0], year, exchange, code)

            res = get_url(url)
            if '404 Not Found' in res:
                return None
            data = json.loads(res)['data']
            df = pd.DataFrame(data,
                              columns=['date', 'derc_open', 'derc_close', 'derc_high', 'derc_low',
                                       'volume', 'change_rate'])

            df.drop(['volume', 'change_rate'], axis=1, inplace=True)
            df['date'] = df['date'].apply(date_str2std)
            return df

        derc = pd.DataFrame()
        curry = start_year
        while True:
            cur = _derc(curry)
            if cur is not None:
                derc = pd.concat([derc, cur])

            if end_year == curry:
                break
            curry += 1
        if derc.shape[0] <= 0:
            return None
        derc.index = derc.date
        return derc

    @classmethod
    def fetch_hist(cls, code, start=None, end=None, index=False):
        if end is not None:
            end = '&end=%s' % end
        else:
            end = ''

        start_str = date2str(start).replace(DATE_SEPARATOR, '')
        # end_str = date2str(end)
        i = (0 if code[0] == '0' else 1) if index else (0 if code[0] == '6' else 1)
        url = cls.url_his % (i, code, start_str) + end
        # print(url)
        df = pd.read_csv(url, encoding=GBK)
        if df.shape[0] == 0:
            return None
        flag = 'index_tick' if index else 'hist_tick'
        DWASH.raw_regular_i(df, flag)
        df.sort_values('date', inplace=True)
        return df

    @classmethod
    def renew_derc_values(cls, category, code_list):
        def combine(code):
            if category == 'stock':
                df = cls._update_stock_derc(category, code)
            else:
                df = DMGR.read(category, code)
            if df is None:
                return
            DWASH.calc_change_i(df)
            DMGR.save(df, category, code)

        loop(combine, code_list, category)

    @classmethod
    def _update_stock_derc(cls, category, code):
        df = DMGR.read(category, code)
        if df is None:
            return None
        df.date = df.date.apply(date_str2std)

        df.index = df.date
        chosen = df[df.date >= '2015-01-01'].date
        derc = N163.fetch_derc(code, 2015)
        if derc is None:
            return None
        last_factor = None
        for key in chosen:
            if key in derc.index:
                for col in derc:
                    if col != 'date':
                        df.loc[key, col] = derc.loc[key, col]
            else:
                if last_factor is None:
                    idx = df.index.get_loc(key)
                    if idx == 0:
                        last_factor = 1
                    else:
                        last_factor = df.ix[idx - 1, 'factor']

                df.loc[key, 'derc_close'] = df.loc[key, 'close'] * last_factor
            last_factor = df.loc[key, 'factor'] = df.loc[key, 'derc_close'] / df.loc[key, 'close']
        DWASH.calc_change_i(df)
        return df


class Tuget(object):
    MACRO = {
        "money_supply": "get_money_supply",
        "deposit_rate": "get_deposit_rate",
        "loan_rate":    "get_loan_rate",
        "rrr":          "get_rrr",
        "gdp":          "get_gdp_year",
        "gdp_quarter":  "get_gdp_quarter",
        "gdp_for":      "get_gdp_for",
        "gdp_pull":     "get_gdp_pull",
        "gdp_contrib":  "get_gdp_contrib",
        "cpi":          "get_cpi",
        "ppi":          "get_ppi"}
    CATEGORY = {
        'industry': 'get_industry_classified',
        'concept':  'get_concept_classified',
        'area':     'get_area_classified',  # 'sme': 'get_sme_classified',
        # 'gem': 'get_gem_classified',
        'hs300':    'get_hs300s',
        'sz50':     'get_sz50s',
        'zz500':    'get_zz500s'}

    @classmethod
    def warning(cls):
        ts.forecast_data(2017, 4)  # 业绩预告
        ts.profit_data()  # 分红方案
        ts.new_stocks()  # 新股
        ts.xsg_data()  # 限售解禁

    #  noinspection SpellCheckingInspection
    @classmethod
    def override_margins(cls):

        def _sh_margin():
            start = '1990-01-01'
            sh = ts.sh_margins(start)
            sh.rename(columns={
                'opDate':   'date',
                'rqylje':   'rqye',
                'rzrqjyzl': 'rzrqye'}, inplace=True)
            return sh

        def _sz_margin():
            end = today()
            start = end - timedelta(days=300)
            sz = pd.DataFrame()
            while True:
                print('\n')
                print(start, end)
                cur = ts.sz_margins(start, end)
                # print(cur)
                if cur is None or cur.empty:
                    break
                sz = pd.concat([cur, sz])
                end = start - timedelta(days=1)
                start = end - timedelta(days=300)

            sz.rename(columns={
                'opDate': 'date'}, inplace=True)
            return sz

        mar = pd.concat([_sh_margin(), _sz_margin()])
        mar = mar.groupby('date').agg({
            'rzye':   'sum',
            'rqye':   'sum',
            'rzrqye': 'sum'})
        DMGR.save(mar, 'macro', 'margin')

    @classmethod
    def override_shibor(cls):
        start = 2006
        shibor = pd.DataFrame()
        while True:
            table = ts.shibor_data(start)
            if table is None or table.empty:
                break
            shibor = pd.concat([shibor, table])
            start += 1
        DMGR.save(shibor, 'macro', 'shibor')

    @classmethod
    def override_macros(cls):
        cls._override_by_dict(cls.MACRO, 'macro')

    @classmethod
    def override_category(cls):
        cls._override_by_dict(cls.CATEGORY, 'category')

    # region obsolete
    @classmethod
    def override_share_div(cls, start):
        # start = 1990
        sd = pd.DataFrame()
        while True:
            try:
                print('\n', start)
                df = ts.profit_data(year=start, top='all')
                if df is None or df.empty:
                    break
                df.sort_values('report_date')
                sd = pd.concat([sd, df])
                start -= 1
            except Exception as e:
                print(e)
                break
        DMGR.save(sd, 'macro', 'share_div')

    @classmethod
    def fetch_code_list(cls):
        """Renew stock list including basic info
        code,代码
        name,名称
        industry,所属行业
        area,地区
        pe,市盈率
        outstanding,流通股本
        totals,总股本(万)
        totalAssets,总资产(万)
        liquidAssets,流动资产
        fixedAssets,固定资产
        reserved,公积金
        reservedPerShare,每股公积金
        eps,每股收益
        bvps,每股净资
        pb,市净率
        timeToMarket,上市日期"""
        # 股票基本信息列表
        stock_basic = ts.get_stock_basics()
        return stock_basic  # 每次必须重新覆盖  # self.db.table_save(  # stock_basic.sort(),

    #  'stock_list',   # append =  #  False)  #   #  #  print(  # '股票列表已覆盖更新')

    @classmethod
    def fetch_his_factor(cls, code, start=None, end=None, recurrent_type=None):
        df = ts.get_h_data(code, date2str(start), date2str(end), autype=recurrent_type,
                           drop_factor=False, )
        return df

    @staticmethod
    def combine_163_sina_stock_hist(code, start=None):
        start = date_of(1988, 1, 1) if start is None else start
        end = today()
        ndf = N163.fetch_stock_combined_his(code, start)
        print(ndf.columns)
        # print(DataWash.column_match(ndf))
        # DWash.column_regularI(ndf)

        tdf = Tuget.fetch_his_factor(code, start, end)
        tdf = tdf.reset_index()
        tdf.drop(['volume', 'amount'], axis=1, inplace=True)
        tdf['date'] = tdf['date'].astype('str')
        # print(tdf.dtypes)
        merged = pd.merge(ndf, tdf, on=['date', 'close', 'high', 'low', 'open'])
        merged = merged.sort_values('date')
        return merged

    # endregion

    @classmethod
    def _override_by_dict(cls, my_dict, name):
        for element in my_dict:
            method = getattr(ts, my_dict[element])
            df = method()
            DMGR.save(df, name, element)

    @classmethod
    def _money_supply_month_clean(cls):
        index = ['macro', 'money_supply']
        m2 = DMGR.read(*index)
        last = ''
        for key, row in m2.iterrows():
            year, month = str(row['month']).split('.')
            if last != '11':
                month = month.zfill(2)
            else:
                month = '10'
            last = month
            m2.ix[key, 'month'] = year + '-' + month

        print(m2['month'])
        DMGR.save(m2, *index)


class Yahoo(object):

    @classmethod
    def hist_url(cls, code, start, end):
        start_year, start_month, start_day = start.split('-')
        end_year, end_month, end_day = end.split('-')
        url_str = 'http://ichart.finance.yahoo.com/table.csv?s=%s&a=%s&b=%s&c=%s&d=%s&e=%s&f=%s&g' \
                  '=d&ignore=.csv'
        url = url_str % (code, start_month, start_day, start_year, end_month, end_day, end_year)
        return url

    @classmethod
    def get_hist(cls, code):
        portfolio = yf.Share(code)
        _ = portfolio  # portfolio.get_historical()
