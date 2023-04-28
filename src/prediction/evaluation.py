#!/usr/bin/env python 
# -*- coding:utf-8 -*

import pandas as pd
import numpy as np
import sys

tick_data_file = sys.argv[1]
order_time_file = sys.argv[2]
symbol_file = 'SampleStocks.csv'

tick_data = pd.read_csv(tick_data_file, index_col=None)
order_data = pd.read_csv(order_time_file, index_col=None)
# Order Sequence Check
if not np.array_equal(order_data['dataIdx'], tick_data['COLUMN01']):
    mismatchIdx = [i for i in range(order_data.shape[0]) if order_data['dataIdx'][i] != tick_data['COLUMN01'][i]]
    print('order-tick index mismatch: {}, {}'.format(order_data['dataIdx'][mismatchIdx[0]],
                                                     tick_data['COLUMN01'][mismatchIdx[0]]))
    sys.exit()

order_data = order_data[order_data['BSflag'] != 'N']
tick_tm = tick_data['COLUMN03'].tolist()
order_data['tickTm'] = [tick_tm[i] for i in order_data['dataIdx'].tolist()]
tick_prc = tick_data['COLUMN07'].tolist()
order_data['tickPrc'] = [tick_prc[i] for i in order_data['dataIdx'].tolist()]


def tm_to_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    return ms


def check_validity(df):
    if not df.shape[0] >= 3:
        return 'number of orders of a single stock must be not less than 3'
    if not np.array_equal(df['volume'], df['volume'].astype(int)):
        return 'field VOLUME is not in integer format'
    if not (df['volume'] >= 1).all():
        return 'the volume of each transaction must be not less than 1'
    if not df['volume'].sum() <= 100:
        return 'the total volume of a single stock must be not larger than 100'
    if not df['tickTm'].apply(tm_to_ms).diff().min() > 60000:
        return 'interval between two consecutive transactions must be not less than 1 minute'
    return None


symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()
profit = []
for sym in symbol:
    sym_data = tick_data[tick_data['COLUMN02'] == sym]
    mkt_mean_prc = sym_data.iloc[-1]['COLUMN49'] / sym_data.iloc[-1]['COLUMN48']

    sym_order_buy = order_data[(order_data['symbol'] == sym) & (order_data['BSflag'] == 'B')]
    errInfo = check_validity(sym_order_buy)
    if errInfo:
        print(f'order of {sym} invalid: {errInfo}')
        sys.exit()
    buy_mean_prc = (sym_order_buy['volume'] * sym_order_buy['tickPrc']).sum() / 100

    sym_order_sell = order_data[(order_data['symbol'] == sym) & (order_data['BSflag'] == 'S')]
    errInfo = check_validity(sym_order_sell)
    if errInfo:
        print(f'order of {sym} invalid: {errInfo}')
        sys.exit()
    sell_mean_prc = (sym_order_sell['volume'] * sym_order_sell['tickPrc']).sum() / 100

    if not sym_order_buy['volume'].sum() == sym_order_sell['volume'].sum():
        print('the total buy volume must be the same as the total sell volume')
        sys.exit()
    profit.append((sell_mean_prc - buy_mean_prc) / mkt_mean_prc)

print('Earning rate is: {:.2f} bp'.format(np.mean(profit)))
