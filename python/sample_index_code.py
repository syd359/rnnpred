import pandas as pd
import os
import re
import time

CAT_LIST = ['_10VOL', '_20VOL', '_30VOL', '_40VOL', '_50VOL']
BREAK_LINE = '----------------------------------------------------------------------------------'


def data_file_path(filename):
    path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', filename)
    return path


def sample_regen(num=3):
    '''

    :param num:
    :return:
    '''
    import random

    # index_nav_wgt_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'index_nav_wgt.csv')
    index_nav_wgt_path = data_file_path('index_nav_wgt.csv')
    # index_info_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'index_info.csv')
    index_info_path = data_file_path('index_info.csv')

    index_nav_wgt = pd.read_csv(index_nav_wgt_path)
    index_info = pd.read_csv(index_info_path)

    # len(index_nav_wgt['index_code'].str[:9].unique())
    # 3470
    exchange_list = list(set(index_info.underlying_code))
    exchange_list = sorted(exchange_list)  # length = 3495

    # random sample the given num stock codes
    target = random.sample(exchange_list, num)
    target_list = [x + y for x in target for y in CAT_LIST]

    print(BREAK_LINE)
    start_time = time.time()
    print('Start timing')

    df_cat_list = [index_nav_wgt[index_nav_wgt['index_code'] == x] for x in target_list]
    df = pd.concat(df_cat_list)

    # pd.to_datetime(df['trade_date'])
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.reset_index(inplace=True)
    print("The stock dataframe gen time --- {} seconds ---".format(time.time() - start_time))

    res = data_gen(df)

    return res


def industry_gen(industry_list):
    '''

        :param num:
        :return:
        '''
    print('Start to read dataframe')
    start_time = time.time()
    print('Start timing the read df time')

    # Read in the data files
    # make a func
    # index_nav_wgt_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'index_nav_wgt.csv')
    index_nav_wgt_path = data_file_path('index_nav_wgt.csv')
    # citic_industry_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'citic_industry.csv')
    citic_industry_path = data_file_path('citic_industry.csv')

    index_nav_wgt = pd.read_csv(index_nav_wgt_path)
    # index_info = pd.read_csv(index_info_path)
    citic_industry = pd.read_csv(citic_industry_path)

    print("The read df time --- {} seconds ---".format(time.time() - start_time))

    # concat the target industry stocks code to a list
    cat_list = [citic_industry[citic_industry['citic'] == el] for el in industry_list]
    stock_in_industry = list(pd.concat(cat_list)['stock'])

    print('The stocks selected are:{}'.format(stock_in_industry))

    target_list = [x + y for x in stock_in_industry for y in CAT_LIST]

    print(BREAK_LINE)
    start_time = time.time()
    print('Start timing the concat stocks df time')

    df_cat_list = [index_nav_wgt[index_nav_wgt['index_code'] == x] for x in target_list]
    df = pd.concat(df_cat_list)

    print("The read df time --- {} seconds ---".format(time.time() - start_time))

    # pd.to_datetime(df['trade_date'])
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.reset_index(inplace=True)

    res = data_gen(df)

    return res


def whole_stocks_gen():
    '''
    Directly read dataframe from the 'reformated_whole_stock.csv'
    '''
    # index_nav_wgt_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'index_nav_wgt.csv')
    index_nav_wgt_path = data_file_path('index_nav_wgt.csv')
    # index_nav_wgt_path = r'C:\Users\syd13065\PycharmProjects\rnnnav\data\index_nav_wgt.csv'
    index_nav_wgt = pd.read_csv(index_nav_wgt_path)

    index_nav_wgt['trade_date'] = pd.to_datetime(index_nav_wgt['trade_date'], format='%Y%m%d')
    res = data_gen(index_nav_wgt)

    return res


def data_gen(target):
    # target_list = sorted(list(set(target['index_code'])))
    print('Start to reformat the dataframe')

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
    print('Dataframe concat is done')

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
    # res = industry_gen(['b101', 'b102'])
    # res = whole_stocks_gen()
    print(res)
