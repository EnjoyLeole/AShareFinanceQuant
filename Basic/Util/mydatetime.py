# import datetime
from datetime import datetime, timedelta

import numpy

QUARTER_SEPARATOR = 'Q'

INTERVAL_ORDER = {
    'year'   : 0,
    'quarter': 1,
    'month'  : 2,
    'date'   : 3}


def today():
    return datetime.now().date()


def now():
    return datetime.now()


def str2date(str, day_delta = 0):
    if '-' in str:
        reg = '%Y-%m-%d'
    elif '/' in str:
        reg = '%Y/%m/%d'
    else:
        reg = '%Y%m%d'
    try:
        date = datetime.strptime(str, reg)
        date += timedelta(days = day_delta)
    except Exception as e:
        if not 'Unnamed' in str and str != ' ':
            print(str, e)
        date = None
    return date


def date_str(date, separate = '-'):
    return date.strftime("%Y" + separate + "%m" + separate + "%d")


def std_date_str(str):
    if '/' in str:
        sep = '/'
    else:
        sep = '-'
    year, m, d = str.split(sep)
    m = m.zfill(2)
    d = d.zfill(2)
    return '%s-%s-%s' % (year, m, d)


def date_of(year, month, day):
    return datetime(year, month, day).date()


def __get_date(date_str):
    if date_str == 'non-give':
        dt = now()
    else:
        if isinstance(date_str, str):
            dt = str2date(date_str)
            if dt is None:
                return None
        else:
            raise Exception('unpredicted!')
    return dt


def to_quarter(date_str = 'non-give'):
    dt = __get_date(date_str)
    if dt is None:
        return None
    yr = dt.year
    qrt = [[datetime(yr, 3, 31), 1], [datetime(yr, 6, 30), 2], [datetime(yr, 9, 30), 3],
           [datetime(yr, 12, 31), 4]]
    for t in qrt:
        if dt <= t[0]:
            return '%s%s%s' % (yr, QUARTER_SEPARATOR, t[1])


def to_year(date_str = 'non-give'):
    dt = __get_date(date_str)
    return dt.year


def to_month(date_str = 'non-give'):
    dt = __get_date(date_str)
    return '%s-%s' % (dt.year, dt.month)


def quarter2year(quarter):
    year, qt = quarter.split(QUARTER_SEPARATOR)
    return year


def month2quarter(month):
    for note in ['.', '-', '/']:
        if note in month:
            sep = note
    year, m = month.split(sep)
    return m


INTERVAL_TRANSFER = {
    ('quarter', 'year') : quarter2year,
    ('month', 'quarter'): month2quarter,
    ('date', 'year')    : to_year,
    ('date', 'quarter') : to_quarter,
    ('date', 'month')   : to_month}


def quarter_add(quarter, i):
    year, q = quarter.split('-')
    if len(year) > 4 or len(q) > 1:
        raise Exception('Ill format in quarter %s' % quarter)
    year = int(year)
    q = int(q)
    res = q + i
    q = (res % 4)
    year += res // 4
    if q == 0:
        q = 4
        year -= 1
    return '%s%s%s' % (year, QUARTER_SEPARATOR, q)


class Quarter(object):
    """Year/quarter class"""

    def __init__(self, init_year = 1990, init_quarter = 1, report_quarter = None):
        if report_quarter is None:
            self.year = init_year
            self.quarter = init_quarter
        else:
            temp = report_quarter.split(QUARTER_SEPARATOR)
            self.year = int(temp[0])
            self.quarter = int(temp[1])

    def __str__(self):
        return self.str_index

    @property
    def str_index(self):
        return str(self.year) + '-' + str(self.quarter)

    @property
    def if_exceed_current_quarter(self):
        now = datetime.now()
        now_q = (now.month / 4)
        if self.year > now.year:
            return True
        elif self.year == now.year and self.quarter > now_q:
            return True
        else:
            return False

    def moveto_next_quarter(self):
        if self.quarter < 4:
            self.quarter += 1
        else:
            self.year += 1
            self.quarter = 1
        return self
