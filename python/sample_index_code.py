import pandas as pd
import os
import re


def sample_regen(num=3):
    '''

    :param num:
    :return:
    '''
    import random

    dirpath = os.path.dirname(__file__) + r'\..' + r'\data'
    index_nav_wgt_path = dirpath + r'\index_nav_wgt.csv'
    index_info_path = dirpath + r'\index_info.csv'

    cat_list = ['_10VOL', '_20VOL', '_30VOL', '_40VOL', '_50VOL']

    index_nav_wgt = pd.read_csv(index_nav_wgt_path)
    index_info = pd.read_csv(index_info_path)

    exchange_list = list(set(index_info.underlying_code))
    exchange_list = sorted(exchange_list)  # length = 3495

    # random sample the given num stock codes
    target = random.sample(exchange_list, num)
    target_list = [x + y for x in target for y in cat_list]

    df = pd.concat([index_nav_wgt[index_nav_wgt['index_code'] == x] for x in target_list])

    # pd.to_datetime(df['trade_date'])
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.reset_index(inplace=True)

    res = data_gen(df)

    return res


def data_gen(target):
    # target_list = sorted(list(set(target['index_code'])))

    df1 = target[target.index_code.str.contains('_10VOL')].reset_index()
    df1['10VOL'] = df1.nav

    df2 = target[target.index_code.str.contains('_20VOL')].reset_index()
    df2['20VOL'] = df2.nav

    df3 = target[target.index_code.str.contains('_30VOL')].reset_index()
    df3['30VOL'] = df3.nav

    df4 = target[target.index_code.str.contains('_40VOL')].reset_index()
    df4['40VOL'] = df4.nav

    df5 = target[target.index_code.str.contains('_50VOL')].reset_index()
    df5['50VOL'] = df5.nav

    df = pd.concat([df1, df2, df3, df4, df5], axis=1)

    # only select the not duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]
    # remove the remained duplicated columns
    df.drop(columns=['index', 'nav'], inplace=True)

    df.index_code = df.index_code.apply(lambda x: x[:9])
    df['diff'] = df['nav_base'].pct_change(periods=4)
    df['diff'].fillna(0, inplace=True)

    return df


if __name__ == "__main__":
    res = sample_regen(3)
    print(res)
