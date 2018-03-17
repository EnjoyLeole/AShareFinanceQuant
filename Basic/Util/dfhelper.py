import pandas as pd
from .mydatetime import INTERVAL_TRANSFER

DUPLICATE_FLAG = '_d'
FORCE_FILLED = 'force_filled'


def remove_duplicateI(df):
    dups = [x for x in df if x.endswith(DUPLICATE_FLAG)]
    df.drop(dups, axis = 1, inplace = True)


def fill_miss(df, refer_idxs, brief_col = 'quarter', ifLabelingFilled = True):
    missing = [x for x in refer_idxs if x not in df[brief_col].values]
    missing = list(set(missing))
    if ifLabelingFilled:
        df[FORCE_FILLED] = False
    for qt in missing:
        df.loc[qt, brief_col] = qt
        if ifLabelingFilled:
            df.loc[qt, FORCE_FILLED] = True
    df = df.fillna(method = 'pad')  # todo better fill
    return df


def truncate_period(df, truncate_val, truncate_col = 'quarter'):
    df.sort_values(truncate_col, inplace = True)
    last = df.iloc[-1]
    if last[truncate_col] != truncate_val:
        df.loc[truncate_val, :] = last
        df.loc[truncate_val, truncate_col] = truncate_val
    return df


def reduce2brief(df, brief_col = 'quarter', detail_col = 'date'):
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


def brief_detail_merge(brief, detail, ifReduce2brief = False, brief_col = 'quarter',
                       detail_col = 'date'):
    '''将周期不同的两张表做合并，在两表的index不完全重合时，会使用相邻项进行填充
    :param ifReduce2brief
        True:以汇总表为基础合并，明细表中取detail_col最大项
        False：以明细表为基础合并，汇总表分散对应至明细表中各个brief_col的相同项'''
    detail2brief_func = INTERVAL_TRANSFER[(detail_col, brief_col)]
    if brief_col not in detail:
        detail[brief_col] = detail[detail_col].apply(detail2brief_func)
    if brief_col not in brief:
        brief[brief_col] = brief[detail_col].apply(detail2brief_func)
    brief.sort_values(brief_col, inplace = True)
    brief.index = brief[brief_col]
    brief = brief[brief[brief_col] > '']

    if ifReduce2brief:
        df_reduced = reduce2brief(detail, brief_col, detail_col)
        if df_reduced.shape[0] < brief.shape[0]:
            df_reduced = fill_miss(df_reduced, brief.index, brief_col)
        detail = df_reduced
        method = 'left'
        # valid = '1:m'
    else:
        brief = fill_miss(brief, detail[brief_col], brief_col)
        method = 'right'
        # valid = '1:m'
    df = pd.merge(brief, detail, on = brief_col, how = method, suffixes = ['', DUPLICATE_FLAG])
    df.sort_values([brief_col, detail_col], inplace = True)

    return df
