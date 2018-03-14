from datetime import datetime, timedelta, date

DATE_SEPARATOR = '-'
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


def date_of(year, month, day):
    return datetime(year, month, day).date()


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


def date_str(date):
    return std_date(date.year, date.month, date.day)


def std_date(year, month, day):
    return '%s%s%s%s%s' % (year, DATE_SEPARATOR, month, DATE_SEPARATOR, day)


def __parse_str(str):
    if str != str:
        return None, None, None
    if '/' in str:
        sep = '/'
    elif '-' in str:
        sep = '-'
    else:
        return None, None, None
    # print(str)
    year, m, d = str.split(sep)
    m = m.zfill(2)
    d = d.zfill(2)
    return year, m, d


def std_date_str(str):
    year, m, d = __parse_str(str)
    return std_date(year, m, d)


def __parse_date(date_str):
    if date_str is None or date_str != date_str:
        return None, None, None
    dt = None
    if isinstance(date_str, date):
        dt = date_str
    if isinstance(date_str, datetime):
        dt = date_str.date()
    if date_str == 'non-give':
        dt = now()
    if dt is not None:
        return str(dt.year), str(dt.month).zfill(2), dt.day

    if isinstance(date_str, str):
        y, m, d = __parse_str(date_str)
        if y is None:
            return None, None, None
        else:
            return y, m, d

    raise Exception('%s unpredicted!' % date_str)


def to_quarter(date_str = 'non-give'):
    # assert isinstance(date_str, str)
    year, month, day = __parse_date(date_str)
    month_quarters = [['03', QUARTER_SEPARATOR + '1'], ['06', QUARTER_SEPARATOR + '2'],
                      ['09', QUARTER_SEPARATOR + '3'],
                      ['12', QUARTER_SEPARATOR + '4']]

    if year is None:
        return None
    for t in month_quarters:
        if month <= t[0]:
            return year + t[1]


def to_year(date_str = 'non-give'):
    dt = __parse_date(date_str)
    return dt.year


def to_month(date_str = 'non-give'):
    dt = __parse_date(date_str)
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


def quarter_dates(quarter):
    year, qt = quarter.split(QUARTER_SEPARATOR)
    dates = {
        '1': (('01', '01'), ('03', '31')),
        '2': (('04', '01'), ('06', '30')),
        '3': (('07', '01'), ('09', '30')),
        '4': (('10', '01'), ('12', '31'))}
    dts = dates[qt]

    start = std_date(year, dts[0][0], dts[0][1])
    end = std_date(year, dts[1][0], dts[1][1])
    return start, end


INTERVAL_TRANSFER = {
    ('quarter', 'year') : quarter2year,
    ('month', 'quarter'): month2quarter,
    ('date', 'year')    : to_year,
    ('date', 'quarter') : to_quarter,
    ('date', 'month')   : to_month}


def quarter_add(quarter, i):
    year, q = quarter.split(QUARTER_SEPARATOR)
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
