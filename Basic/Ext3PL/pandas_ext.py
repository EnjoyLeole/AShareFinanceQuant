import numpy as np
import pandas as pd

from Basic.Util import print_num
from Basic.Util.mydatetime import INTERVAL_TRANSFER

DUPLICATE_FLAG = '_d'
FORCE_FILLED = 'force_filled'
WARNING_LEVEL = 1
COMPARED_FLAG = 'Done'


def column_duplicate_remove_inplace(df):
    duplicates = [x for x in df if x.endswith(DUPLICATE_FLAG)]
    df.drop(duplicates, axis=1, inplace=True)


def column_compare_choose_inplace(df, origin, dual_key, flag=''):
    rename = lambda x: df.rename(columns=x, inplace=True)
    winner = []

    def __chose_origin():
        rename({
            dual_key: dual_key + COMPARED_FLAG})
        winner.append(origin)

    def __chose_dual():
        rename({
            origin: origin + COMPARED_FLAG})
        rename({
            dual_key: origin})
        winner.append(dual_key)

    dual = df[dual_key]
    if isinstance(dual, pd.DataFrame):
        raise Exception(f'{flag} %s has multiple columns, Should handle before!' % dual_key)

    numeric_inplace(df, [origin, dual_key])
    org_count = (df[origin] != 0).sum()
    dual_count = (df[dual_key] != 0).sum()

    org_sum = df[origin].apply(abs).sum()
    dual_sum = df[dual_key].apply(abs).sum()
    diff = org_sum - dual_sum
    max_sum = max(org_sum, dual_sum)
    diff_rate = diff / max_sum if max_sum > 0 else 0
    sig_level = 0.001
    if (org_count > 0 and dual_count == 0) or diff_rate < sig_level:
        __chose_origin()
    elif org_count == 0 and dual_count > 0:
        __chose_dual()
    elif org_count == dual_count:
        if diff >= 0:
            __chose_origin()
        else:
            __chose_dual()
    else:  # todo if count lower but sum much bigger
        if org_sum < sig_level:
            __chose_dual()
        elif dual_sum < sig_level:
            __chose_origin()
        else:
            def validate(series):
                if series[origin] != series[dual_key] and series[origin] != 0 and series[
                    dual_key] != 0:
                    return 1
                return 0

            unequal_num = df.apply(validate, axis=1).sum()
            if unequal_num / df.shape[0] > WARNING_LEVEL:
                print(flag, 'Inconsistent for %s bigger than %s with %s in %s / %s records' % (
                    origin, dual_key, diff_rate, unequal_num, df.shape[0]))

            if org_count > dual_count:
                if diff < 0:
                    print_num(flag, 'Warning!', org_count, org_sum, 'vs', dual_count, dual_sum)
                __chose_origin()
            else:
                if diff > 0:
                    print_num(flag, 'Warning!', org_count, org_sum, 'vs', dual_count, dual_sum)
                __chose_dual()

    return winner


def fill_miss(df, refer_indexes, brief_col='quarter', if_labeling_filled=True):
    missing = [x for x in refer_indexes if x not in df[brief_col].values]
    missing = list(set(missing))
    if if_labeling_filled:
        df[FORCE_FILLED] = False
    for qt in missing:
        df.loc[qt, brief_col] = qt
        if if_labeling_filled:
            df.loc[qt, FORCE_FILLED] = True
    df = df.fillna(method='pad')
    return df


def truncate_period(df, truncate_val, truncate_col='quarter'):
    """ set the value of truncate point to the last,(to prepare for further merge) """
    df.sort_values(truncate_col, inplace=True)
    last = df.iloc[-1]
    if last[truncate_col] != truncate_val:
        df.loc[truncate_val, :] = last
        df.loc[truncate_val, truncate_col] = truncate_val
    return df


def reduce2brief(df, brief_col='quarter', detail_col='date'):
    if brief_col not in df:
        detail2brief_func = INTERVAL_TRANSFER[(detail_col, brief_col)]
        df[brief_col] = df[detail_col].apply(detail2brief_func)
    group = df.groupby(brief_col).agg({
        detail_col: 'max'})[detail_col].tolist()
    df.index = df[detail_col]
    # flag = df[detail_col].apply(lambda x: True if x in group.values else False)
    df_reduced = df.ix[group].copy()
    df_reduced.index = df_reduced[brief_col]
    return df_reduced


def brief_detail_merge(brief, detail, if_reduce2brief=False, brief_col='quarter',
                       detail_col='date'):
    """将周期不同的两张表做合并，在两表的index不完全重合时，会使用相邻项进行填充
    :param detail:
    :param brief_col:
    :param detail_col:
    :param brief:
    :param if_reduce2brief
        True:以汇总表为基础合并，明细表中取detail_col最大项
        False：以明细表为基础合并，汇总表分散对应至明细表中各个brief_col的相同项
    """
    detail2brief_func = INTERVAL_TRANSFER[(detail_col, brief_col)]
    if brief_col not in detail:
        detail[brief_col] = detail[detail_col].apply(detail2brief_func)
    if brief_col not in brief:
        brief[brief_col] = brief[detail_col].apply(detail2brief_func)
    brief.sort_values(brief_col, inplace=True)
    brief.index = brief[brief_col]
    brief = brief[brief[brief_col] > '']

    if if_reduce2brief:
        df_reduced = reduce2brief(detail, brief_col, detail_col)
        if df_reduced.shape[0] < brief.shape[0]:
            df_reduced = fill_miss(df_reduced, brief.index, brief_col)
        detail = df_reduced
        method = 'left'  # valid = '1:m'
    else:
        brief = fill_miss(brief, detail[brief_col], brief_col)
        method = 'right'  # valid = '1:m'
    df = pd.merge(brief, detail, on=brief_col, how=method, suffixes=['', DUPLICATE_FLAG])
    df.sort_values([brief_col, detail_col], inplace=True)

    return df


def numeric_inplace(df: pd.DataFrame, include=None, exclude=None):
    exclude = exclude if exclude else []
    exclude += ['quarter', 'date', 'date_x', 'date_y']
    if include:
        if not isinstance(include, list):
            include = [include]
    else:
        include = df.columns.values
    df.dropna(axis=0, how='all', inplace=True)
    for col in include:
        if col not in exclude and isinstance(df[col], pd.Series) and df[col].dtype in [object, str]:
            try:
                df[col] = df[col].astype(np.float64)
            except ValueError as e:
                print('numeric', col, df.shape, e)
                force = df[col].apply(__to_num)
                df[col] = force
    return df


def __to_num(val):
    try:
        return float(val)
    except ValueError as _:
        return 0


def minus_verse(series):
    """ Special series sorting method, for PE
    positive ascending -》 negative descending -> nan
    :param series:
    :return:
    """
    pos = series[series > 0]
    # print(pos.index)
    pos_idx = pos.sort_values(ascending=False).index.values
    minus = series[series <= 0]
    minus_idx = minus.sort_values(ascending=True).index.values
    nan_idx = series[series != series].index.values
    comb = np.concatenate((minus_idx, pos_idx, nan_idx))
    se = pd.Series(index=comb)
    # return pos_idx + minus_idx + nan_idx
    return se.index


def idx_by_quarter(df):
    df.index = df.quarter
    return df
