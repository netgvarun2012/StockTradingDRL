#!/usr/bin/env python 
# -*- coding:utf-8 -*
import sys
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import joblib
#from stockstats import StockDataFrame as Sdf
from stable_baselines3 import PPO
from SingleStockEnv import SingleStockEnv
filename='model_PPO_trained'
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
stock_dict = {'000009.SZ': 0,
 '000012.SZ': 1,
 '000021.SZ': 2,
 '000027.SZ': 3,
 '000031.SZ': 4,
 '000039.SZ': 5,
 '000050.SZ': 6,
 '000060.SZ': 7,
 '000062.SZ': 8,
 '000089.SZ': 9,
 '000155.SZ': 10,
 '000156.SZ': 11,
 '000400.SZ': 12,
 '000401.SZ': 13,
 '000402.SZ': 14,
 '000415.SZ': 15,
 '000513.SZ': 16,
 '000519.SZ': 17,
 '000537.SZ': 18,
 '000540.SZ': 19,
 '000547.SZ': 20,
 '000553.SZ': 21,
 '000559.SZ': 22,
 '000563.SZ': 23,
 '000581.SZ': 24,
 '000591.SZ': 25,
 '000598.SZ': 26,
 '000623.SZ': 27,
 '000629.SZ': 28,
 '000636.SZ': 29,
 '000656.SZ': 30,
 '000683.SZ': 31,
 '000703.SZ': 32,
 '000709.SZ': 33,
 '000728.SZ': 34,
 '000729.SZ': 35,
 '000738.SZ': 36,
 '000739.SZ': 37,
 '000750.SZ': 38,
 '000778.SZ': 39,
 '000783.SZ': 40,
 '000785.SZ': 41,
 '000807.SZ': 42,
 '000825.SZ': 43,
 '000830.SZ': 44,
 '000831.SZ': 45,
 '000869.SZ': 46,
 '000878.SZ': 47,
 '000883.SZ': 48,
 '000887.SZ': 49,
 '000898.SZ': 50,
 '000930.SZ': 51,
 '000932.SZ': 52,
 '000937.SZ': 53,
 '000960.SZ': 54,
 '000961.SZ': 55,
 '000967.SZ': 56,
 '000987.SZ': 57,
 '000988.SZ': 58,
 '000990.SZ': 59,
 '000997.SZ': 60,
 '000998.SZ': 61,
 '000999.SZ': 62,
 '001203.SZ': 63,
 '001872.SZ': 64,
 '001914.SZ': 65,
 '002002.SZ': 66,
 '002010.SZ': 67,
 '002019.SZ': 68,
 '002025.SZ': 69,
 '002028.SZ': 70,
 '002030.SZ': 71,
 '002048.SZ': 72,
 '002056.SZ': 73,
 '002065.SZ': 74,
 '002078.SZ': 75,
 '002080.SZ': 76,
 '002081.SZ': 77,
 '002110.SZ': 78,
 '002124.SZ': 79,
 '002127.SZ': 80,
 '002128.SZ': 81,
 '002131.SZ': 82,
 '002138.SZ': 83,
 '002146.SZ': 84,
 '002152.SZ': 85,
 '002153.SZ': 86,
 '002155.SZ': 87,
 '002156.SZ': 88,
 '002183.SZ': 89,
 '002185.SZ': 90,
 '002191.SZ': 91,
 '002192.SZ': 92,
 '002195.SZ': 93,
 '002203.SZ': 94,
 '002212.SZ': 95,
 '002221.SZ': 96,
 '002223.SZ': 97,
 '002242.SZ': 98,
 '002244.SZ': 99,
 '002249.SZ': 100,
 '002250.SZ': 101,
 '002266.SZ': 102,
 '002268.SZ': 103,
 '002273.SZ': 104,
 '002281.SZ': 105,
 '002294.SZ': 106,
 '002299.SZ': 107,
 '002326.SZ': 108,
 '002340.SZ': 109,
 '002353.SZ': 110,
 '002368.SZ': 111,
 '002372.SZ': 112,
 '002373.SZ': 113,
 '002384.SZ': 114,
 '002385.SZ': 115,
 '002396.SZ': 116,
 '002399.SZ': 117,
 '002407.SZ': 118,
 '002409.SZ': 119,
 '002416.SZ': 120,
 '002422.SZ': 121,
 '002423.SZ': 122,
 '002430.SZ': 123,
 '002434.SZ': 124,
 '002439.SZ': 125,
 '002444.SZ': 126,
 '002465.SZ': 127,
 '002468.SZ': 128,
 '002497.SZ': 129,
 '002500.SZ': 130,
 '002505.SZ': 131,
 '002506.SZ': 132,
 '002507.SZ': 133,
 '002508.SZ': 134,
 '002511.SZ': 135,
 '002531.SZ': 136,
 '002532.SZ': 137,
 '002557.SZ': 138,
 '002563.SZ': 139,
 '002568.SZ': 140,
 '002572.SZ': 141,
 '002595.SZ': 142,
 '002603.SZ': 143,
 '002624.SZ': 144,
 '002625.SZ': 145,
 '002653.SZ': 146,
 '002670.SZ': 147,
 '002673.SZ': 148,
 '002683.SZ': 149,
 '002690.SZ': 150,
 '002701.SZ': 151,
 '002705.SZ': 152,
 '002738.SZ': 153,
 '002739.SZ': 154,
 '002745.SZ': 155,
 '002797.SZ': 156,
 '002831.SZ': 157,
 '002850.SZ': 158,
 '002867.SZ': 159,
 '002901.SZ': 160,
 '002925.SZ': 161,
 '002926.SZ': 162,
 '002936.SZ': 163,
 '002939.SZ': 164,
 '002945.SZ': 165,
 '002958.SZ': 166,
 '002966.SZ': 167,
 '002985.SZ': 168,
 '003035.SZ': 169,
 '300001.SZ': 170,
 '300003.SZ': 171,
 '300009.SZ': 172,
 '300012.SZ': 173,
 '300017.SZ': 174,
 '300024.SZ': 175,
 '300026.SZ': 176,
 '300037.SZ': 177,
 '300058.SZ': 178,
 '300072.SZ': 179,
 '300073.SZ': 180,
 '300088.SZ': 181,
 '300115.SZ': 182,
 '300136.SZ': 183,
 '300144.SZ': 184,
 '300146.SZ': 185,
 '300168.SZ': 186,
 '300182.SZ': 187,
 '300212.SZ': 188,
 '300244.SZ': 189,
 '300251.SZ': 190,
 '300253.SZ': 191,
 '300257.SZ': 192,
 '300285.SZ': 193,
 '300296.SZ': 194,
 '300308.SZ': 195,
 '300357.SZ': 196,
 '300363.SZ': 197,
 '300373.SZ': 198,
 '300376.SZ': 199,
 '300383.SZ': 200,
 '300390.SZ': 201,
 '300418.SZ': 202,
 '300463.SZ': 203,
 '300474.SZ': 204,
 '300482.SZ': 205,
 '300558.SZ': 206,
 '300568.SZ': 207,
 '300618.SZ': 208,
 '300630.SZ': 209,
 '300676.SZ': 210,
 '300677.SZ': 211,
 '300682.SZ': 212,
 '300699.SZ': 213,
 '300724.SZ': 214,
 '300741.SZ': 215,
 '300776.SZ': 216,
 '300832.SZ': 217,
 '300861.SZ': 218,
 '300866.SZ': 219,
 '300869.SZ': 220,
 '300888.SZ': 221,
 '301029.SZ': 222,
 '600008.SH': 223,
 '600010.SH': 224,
 '600021.SH': 225,
 '600022.SH': 226,
 '600026.SH': 227,
 '600027.SH': 228,
 '600030.SH': 229,
 '600032.SH': 230,
 '600056.SH': 231,
 '600060.SH': 232,
 '600062.SH': 233,
 '600064.SH': 234,
 '600066.SH': 235,
 '600079.SH': 236,
 '600095.SH': 237,
 '600096.SH': 238,
 '600104.SH': 239,
 '600109.SH': 240,
 '600118.SH': 241,
 '600126.SH': 242,
 '600131.SH': 243,
 '600141.SH': 244,
 '600143.SH': 245,
 '600153.SH': 246,
 '600155.SH': 247,
 '600157.SH': 248,
 '600160.SH': 249,
 '600161.SH': 250,
 '600166.SH': 251,
 '600167.SH': 252,
 '600170.SH': 253,
 '600171.SH': 254,
 '600177.SH': 255,
 '600195.SH': 256,
 '600201.SH': 257,
 '600208.SH': 258,
 '600216.SH': 259,
 '600258.SH': 260,
 '600259.SH': 261,
 '600271.SH': 262,
 '600297.SH': 263,
 '600298.SH': 264,
 '600299.SH': 265,
 '600307.SH': 266,
 '600315.SH': 267,
 '600316.SH': 268,
 '600325.SH': 269,
 '600329.SH': 270,
 '600339.SH': 271,
 '600348.SH': 272,
 '600350.SH': 273,
 '600352.SH': 274,
 '600369.SH': 275,
 '600373.SH': 276,
 '600376.SH': 277,
 '600377.SH': 278,
 '600378.SH': 279,
 '600380.SH': 280,
 '600390.SH': 281,
 '600392.SH': 282,
 '600398.SH': 283,
 '600399.SH': 284,
 '600406.SH': 285,
 '600409.SH': 286,
 '600415.SH': 287,
 '600418.SH': 288,
 '600435.SH': 289,
 '600438.SH': 290,
 '600482.SH': 291,
 '600486.SH': 292,
 '600487.SH': 293,
 '600489.SH': 294,
 '600497.SH': 295,
 '600498.SH': 296,
 '600499.SH': 297,
 '600500.SH': 298,
 '600507.SH': 299,
 '600511.SH': 300,
 '600516.SH': 301,
 '600517.SH': 302,
 '600521.SH': 303,
 '600528.SH': 304,
 '600529.SH': 305,
 '600535.SH': 306,
 '600536.SH': 307,
 '600546.SH': 308,
 '600549.SH': 309,
 '600556.SH': 310,
 '600563.SH': 311,
 '600566.SH': 312,
 '600567.SH': 313,
 '600580.SH': 314,
 '600582.SH': 315,
 '600585.SH': 316,
 '600597.SH': 317,
 '600598.SH': 318,
 '600623.SH': 319,
 '600637.SH': 320,
 '600642.SH': 321,
 '600648.SH': 322,
 '600655.SH': 323,
 '600667.SH': 324,
 '600673.SH': 325,
 '600699.SH': 326,
 '600704.SH': 327,
 '600705.SH': 328,
 '600707.SH': 329,
 '600718.SH': 330,
 '600728.SH': 331,
 '600737.SH': 332,
 '600739.SH': 333,
 '600755.SH': 334,
 '600764.SH': 335,
 '600765.SH': 336,
 '600782.SH': 337,
 '600787.SH': 338,
 '600801.SH': 339,
 '600808.SH': 340,
 '600820.SH': 341,
 '600827.SH': 342,
 '600839.SH': 343,
 '600848.SH': 344,
 '600859.SH': 345,
 '600862.SH': 346,
 '600863.SH': 347,
 '600867.SH': 348,
 '600871.SH': 349,
 '600873.SH': 350,
 '600879.SH': 351,
 '600885.SH': 352,
 '600893.SH': 353,
 '600895.SH': 354,
 '600901.SH': 355,
 '600906.SH': 356,
 '600909.SH': 357,
 '600917.SH': 358,
 '600928.SH': 359,
 '600956.SH': 360,
 '600959.SH': 361,
 '600967.SH': 362,
 '600968.SH': 363,
 '600985.SH': 364,
 '600988.SH': 365,
 '600998.SH': 366,
 '601000.SH': 367,
 '601005.SH': 368,
 '601012.SH': 369,
 '601058.SH': 370,
 '601066.SH': 371,
 '601077.SH': 372,
 '601098.SH': 373,
 '601106.SH': 374,
 '601108.SH': 375,
 '601118.SH': 376,
 '601128.SH': 377,
 '601139.SH': 378,
 '601156.SH': 379,
 '601158.SH': 380,
 '601162.SH': 381,
 '601168.SH': 382,
 '601179.SH': 383,
 '601187.SH': 384,
 '601198.SH': 385,
 '601225.SH': 386,
 '601228.SH': 387,
 '601231.SH': 388,
 '601233.SH': 389,
 '601288.SH': 390,
 '601298.SH': 391,
 '601333.SH': 392,
 '601456.SH': 393,
 '601555.SH': 394,
 '601568.SH': 395,
 '601577.SH': 396,
 '601598.SH': 397,
 '601607.SH': 398,
 '601608.SH': 399,
 '601611.SH': 400,
 '601628.SH': 401,
 '601633.SH': 402,
 '601636.SH': 403,
 '601665.SH': 404,
 '601666.SH': 405,
 '601668.SH': 406,
 '601669.SH': 407,
 '601688.SH': 408,
 '601696.SH': 409,
 '601699.SH': 410,
 '601717.SH': 411,
 '601718.SH': 412,
 '601778.SH': 413,
 '601828.SH': 414,
 '601857.SH': 415,
 '601866.SH': 416,
 '601869.SH': 417,
 '601872.SH': 418,
 '601880.SH': 419,
 '601888.SH': 420,
 '601899.SH': 421,
 '601919.SH': 422,
 '601928.SH': 423,
 '601958.SH': 424,
 '601969.SH': 425,
 '601990.SH': 426,
 '601991.SH': 427,
 '601992.SH': 428,
 '601995.SH': 429,
 '601997.SH': 430,
 '603000.SH': 431,
 '603026.SH': 432,
 '603077.SH': 433,
 '603127.SH': 434,
 '603156.SH': 435,
 '603160.SH': 436,
 '603218.SH': 437,
 '603225.SH': 438,
 '603228.SH': 439,
 '603233.SH': 440,
 '603267.SH': 441,
 '603317.SH': 442,
 '603345.SH': 443,
 '603355.SH': 444,
 '603379.SH': 445,
 '603444.SH': 446,
 '603456.SH': 447,
 '603517.SH': 448,
 '603568.SH': 449,
 '603589.SH': 450,
 '603596.SH': 451,
 '603605.SH': 452,
 '603606.SH': 453,
 '603613.SH': 454,
 '603638.SH': 455,
 '603650.SH': 456,
 '603658.SH': 457,
 '603688.SH': 458,
 '603707.SH': 459,
 '603712.SH': 460,
 '603719.SH': 461,
 '603737.SH': 462,
 '603786.SH': 463,
 '603816.SH': 464,
 '603858.SH': 465,
 '603866.SH': 466,
 '603868.SH': 467,
 '603883.SH': 468,
 '603885.SH': 469,
 '603893.SH': 470,
 '603927.SH': 471,
 '603939.SH': 472,
 '605358.SH': 473,
 '688002.SH': 474,
 '688006.SH': 475,
 '688009.SH': 476,
 '688029.SH': 477,
 '688063.SH': 478,
 '688088.SH': 479,
 '688099.SH': 480,
 '688116.SH': 481,
 '688122.SH': 482,
 '688185.SH': 483,
 '688188.SH': 484,
 '688208.SH': 485,
 '688256.SH': 486,
 '688289.SH': 487,
 '688301.SH': 488,
 '688385.SH': 489,
 '688390.SH': 490,
 '688521.SH': 491,
 '688538.SH': 492,
 '688567.SH': 493,
 '688690.SH': 494,
 '688772.SH': 495,
 '688777.SH': 496,
 '688779.SH': 497,
 '688819.SH': 498,
 '688981.SH': 499}

input_path = sys.argv[1]
output_path = sys.argv[2]
symbol_file = './SampleStocks.csv'

tick_data = open(input_path, 'r')
order_time = open(output_path, 'w')
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()
idx_dict = dict(zip(symbol, list(range(len(symbol)))))
# ---------- Initialization ----------

target_vol = 100
basic_vol = 2
cum_vol_buy = [0] * len(symbol)  # accumulate buying volume
cum_vol_sell = [0] * len(symbol)  # accumulate selling volume
last_od_ms = [0] * len(symbol)  # last order time
hist_ms_prc = [[] for i in range(len(symbol))]  # historic time and price
state_info = [[] for i in range(len(symbol))]  # state information for each stock


def get_ms(tm):
    '''
    Function to return milliseconds equivalent of the input ticktime.
    '''
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms

# --------------- Loop ---------------
# recursively read all tick lines from tickdata file,
# do decision with your strategy and write order to the ordertime file

tick_data.readline()  # header
order_time.writelines('symbol,BSflag,dataIdx,volume\n')
order_time.flush()
env = SingleStockEnv
num_shares_prev = [0]*len(symbol)
buy_trades = [0]*len(symbol)
sell_trades = [0]*len(symbol)

while True:
    tick_line = tick_data.readline()  # read one tick line
    if tick_line.strip() == 'stop' or len(tick_line) == 0:
        break
    row = tick_line.split(',')
    nTick = row[0] # index
    sym = row[1] # sotck code
    tm = int(row[2]) # tick time
    tm_ms = get_ms(tm)
    op = float(row[3]) # opening price
    hp = float(row[4]) # high price
    lp = float(row[5]) # low price
    prc = float(row[6]) # closed price
    vol = float(row[47]) # accumulated txn volume
    txnAmt = float(row[48]) # txn amount

    #if the key is not present sample stocks, take the same action as no action taken
    if sym not in idx_dict:
        order_time.writelines(f'{sym},N,{nTick},0\n')
        order_time.flush()
        continue

    idx = idx_dict[sym]
    if (tm_ms - last_od_ms[idx] <= 60000):  # execute the order every 1 minute
        #print(f' inside if condition of <60000\n')
        order_time.writelines(f'{sym},N,{nTick},0\n')
        order_time.flush()
        continue

    buypower = ((float(row[7]) * float(row[17]))
                             + (float(row[8]) * float(row[18]))
                             + (float(row[9]) * float(row[19]))
                             + (float(row[10]) * float(row[20]))
                             + (float(row[11]) * float(row[21]))
                             + (float(row[12]) * float(row[22]))
                             + (float(row[13]) * float(row[23]))
                             + (float(row[14]) * float(row[24]))
                             + (float(row[15]) * float(row[25]))
                             + (float(row[16]) * float(row[26])))/10
    sellpower = ((float(row[27]) * float(row[37]))
                             + (float(row[28]) * float(row[38]))
                             + (float(row[29]) * float(row[39]))
                             + (float(row[30]) * float(row[40]))
                             + (float(row[31]) * float(row[41]))
                             + (float(row[32]) * float(row[42]))
                             + (float(row[33]) * float(row[43]))
                             + (float(row[34]) * float(row[44]))
                             + (float(row[35]) * float(row[45]))
                             + (float(row[36]) * float(row[46])))/10
    buy_sell_power = buypower-sellpower
 
    if sym not in symbol:
        order_time.writelines(f'{sym},N,{nTick},0\n')
        order_time.flush()
        continue

    # -------- Your Strategy Code Begin --------

    #print(f'predicting for stock {sym}')
    hist_ms_prc[idx] = [[nTick,stock_dict[sym],tm,op,hp,lp,prc,vol,buy_sell_power,txnAmt]]*2
    order = ('N', 0)

    df = pd.DataFrame(hist_ms_prc[idx])
    df.rename(columns = {0:'index',1:'symbol',2:'datadate',3:'open',4:'high',5:'low',6:'adjcp',7:'volume',8:'buy_sell_power',9:'txnAmount'},inplace=True)
    env_instance = env(df)
    model_ppo = PPO.load(filename)
    model_ppo.set_env(env_instance)
    if state_info[idx]:
        obs = state_info[idx]
        obs[1] = df['adjcp'].iloc[0]
        obs[6] = df['datadate'].iloc[0]
        obs[7] = df['open'].iloc[0]
        obs[8] = df['high'].iloc[0]
        obs[9] = df['low'].iloc[0]
        obs[10] = df['volume'].iloc[0]
        obs[11] = df['buy_sell_power'].iloc[0]
        obs[12] = df['txnAmount'].iloc[0]
        env_instance.state=obs
        env_instance.buy_trades = buy_trades[idx]
        env_instance.sell_trades = sell_trades[idx]
    else:
        obs = env_instance.reset()
    action, _states = model_ppo.predict(obs)

    if (int(action[0]*100) <= 0 or int(action[0]*100)==100 or int(action[0]*100)==99) and tm>=140000000 and cum_vol_buy[idx]!=100:
        action = np.array([0.98])
        obs, rewards, dones, info = env_instance.step(action)
    elif (int(action[0]*100)>=0 or int(action[0]*100)== -100 or int(action[0]*100)== -99) and tm>=140000000 and cum_vol_sell[idx]!=100:
        action = np.array([-0.98])
        obs, rewards, dones, info = env_instance.step(action)
    else:
        obs, rewards, dones, info = env_instance.step(action)
    num_shares = obs[2]

    state_info[idx] = obs
    cum_vol_buy[idx] = obs[3]
    cum_vol_sell[idx] = obs[4]

    if num_shares_prev[idx] < num_shares:
        last_od_ms[idx] = tm_ms
        order = 'B'
        nTick = df['index'].iloc[0]
        vol = num_shares - num_shares_prev[idx]
        order_time.writelines(f'{sym},{order},{nTick},{vol}\n')
        order_time.flush()
        num_shares_prev[idx] = num_shares
        buy_trades[idx]+=1

    elif num_shares_prev[idx] > num_shares:
        last_od_ms[idx] = tm_ms
        order = 'S'
        nTick = df['index'].iloc[0]
        vol = num_shares_prev[idx] - num_shares
        order_time.writelines(f'{sym},{order},{nTick},{vol}\n')
        order_time.flush()
        num_shares_prev[idx] = num_shares
        sell_trades[idx]+=1

    else:
        order = 'N'
        nTick = df['index'].iloc[0]
        vol = 0
        order_time.writelines(f'{sym},{order},{nTick},{vol}\n')
        order_time.flush()

tick_data.close()
order_time.close()
