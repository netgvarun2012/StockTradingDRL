import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global variables
HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE=100000000
STOCK_DIM = 1

# transaction fee: 
TRANSACTION_FEE_PERCENT = 0.000


class SingleStockEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        # date increment
        self.day = day
        self.df = df
        # action_space normalization and the shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 4: [Current Balance]+[prices]+[owned shares] +[macd] 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (13,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.tick =   self.df.datadate
        # termination
        self.terminal = False  
        # save the total number of trades
        self.trades = 0
        self.buying_prices =[]
        self.selling_prices =[]
        self.buy_trades = 0
        self.sell_trades = 0
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      [self.data.adjcp] + \
                      [0]*STOCK_DIM + \
                      [0]*STOCK_DIM + \
                      [0]*STOCK_DIM + \
                      [self.data.symbol] + \
                      [self.data.datadate] + \
                      [self.data.open] + \
                      [self.data.high] + \
                      [self.data.low] + \
                      [self.data.volume] + \
                      [self.data.buy_sell_power] + \
                      [self.data.txnAmount]
                      
                      
        # initialize reward and cost
        self.reward = 0
        self.cost = 0
        
        # memorize the total value, total rewards
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.cum_rewards_memory = []

    def _sell_stock(self, index, action):
        action = int(action)
        # perform sell action based on the sign of the action
        if self.state[index+STOCK_DIM+1] > 0:
            # update balance
            self.state[0] += \
            self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             (1- TRANSACTION_FEE_PERCENT)
            #print(f'curr_num_shares in sell state is {self.state[index+STOCK_DIM+1]}')
            self.state[index+STOCK_DIM+3] += min(abs(action), self.state[index+STOCK_DIM+1])
            # update held shares
            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
            #print(f'sell stocks number is {action}')
            # update transaction costs
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
            
            self.sell_trades+=1
            self.selling_prices.append(self.data.adjcp)
            if self.state[index+STOCK_DIM+3] >=100 and self.sell_trades==2:
                self.state[index+STOCK_DIM+3] = 99
                self.state[index+STOCK_DIM+1] = self.state[index+STOCK_DIM+2]-99



            #print(f'number of trades is {self.trades}')
            #print(f'Number of shares sold {list(self.state[(STOCK_DIM+3):(STOCK_DIM*2+3)])}')
        else:
            pass
            
    def _buy_stock(self, index, action):
        action = int(action)
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        #update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1+ TRANSACTION_FEE_PERCENT)
        # update held shares
        curr_num_shares = self.state[index+STOCK_DIM+1]
        #print(f'curr_num_shares in buy state is {curr_num_shares}')
        self.state[index+STOCK_DIM+1] += min(available_amount, action)
        #print(f'buy stocks number is {action}')
        # update transaction costs
        self.cost+=self.state[index+1]*min(available_amount, action)* \
                          TRANSACTION_FEE_PERCENT
        self.trades+=1
        self.buy_trades+=1
        self.buying_prices.append(self.data.adjcp)
        #print(f'number of trades is {self.trades}')
        curr_num_shares_bought =  self.state[index+STOCK_DIM+2] 
        self.state[index+STOCK_DIM+2] += min(available_amount, action)
        if self.state[index+STOCK_DIM+2] >= 100 and self.buy_trades>=2:
            self.state[index+STOCK_DIM+1] = curr_num_shares + 100-max(3-self.buy_trades,0)-curr_num_shares_bought
            self.state[index+STOCK_DIM+2] = 100-max((3-self.buy_trades),0)
            
            
            #print(f'Inside if {self.state[index+STOCK_DIM+1]}')

    def _calculate_reward(self, buy_prices, sell_prices):
        
        if len(buy_prices) == 0 or len(sell_prices) == 0:
            # no trades were made, so the reward is 0
            reward = 0
            return reward
        
        avg_buy_price = sum(buy_prices) / len(buy_prices)
        avg_sell_price = sum(sell_prices) / len(sell_prices)
    
        profit_pct = (avg_sell_price - avg_buy_price) / avg_buy_price
        reward  =  profit_pct
        return reward
        
    
    def step(self, actions):
        #print(f'State infromation inside step is {self.state}')
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            cumulative_rewards = np.cumsum(self.rewards_memory)
            self.cum_rewards_memory.append(cumulative_rewards[-1])
            #print("self.cum_rewards_memory ",self.cum_rewards_memory)
            df_rewards.to_csv('account_rewards.csv')
            return self.state, self.reward, self.terminal,{}

        else:
            
            # actions are the shares we need to buy, hold, or sell
            #print(f'actions is {actions}')
            actions = actions * HMAX_NORMALIZE           
           
            # perform buy or sell action
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            #print(f'sell_index is {sell_index}')
            #print(f'buy_index is {buy_index}')
            
            for index in sell_index:
                if (self.state[STOCK_DIM+3] < 100):
                    if not(int(actions[index])==-99):
                        if not(int(actions[index])==-100):
                            #print('take sell action'.format(actions[index]))
                            self._sell_stock(index, actions[index])
                else:
                    self.day = len(self.df.index.unique())-2
           
            for index in buy_index:
                if self.state[STOCK_DIM+2] < 100:
                    if not(int(actions[index])==99):
                        if not(int(actions[index])==100): # can't allow to buy 99 or 100 shares due to no_of_trades_buy>=3 condition
                            #print('take buy action: {}'.format(actions[index]))
                            self._buy_stock(index, actions[index])
                else:
                    self.day = len(self.df.index.unique())-2
            
            # update data, walk a step s'
            self.day += 1
            self.data = self.df.loc[self.day,:]         
            #load next state
            self.state =  [self.state[0]] + \
                          [self.data.adjcp] + \
                          list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) +\
                          [self.state[index+STOCK_DIM+2]] +\
                          [self.state[index+STOCK_DIM+3]] + \
                          [self.data.symbol]+ \
                          [self.data.datadate] + \
                          [self.data.open] + \
                          [self.data.high] + \
                          [self.data.low] + \
                          [self.data.volume] + \
                          [self.data.buy_sell_power] + \
                          [self.data.txnAmount]
            
            self.reward = self._calculate_reward(self.buying_prices, self.selling_prices)

            self.rewards_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}
    
    
    
    
    def _plot_rewards(self,symbol):
        # plot rewards earned in each episode
        fig = plt.figure(figsize=(15, 5))
        plt.plot(self.cum_rewards_memory,label=symbol)
        plt.xlabel('Time steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('Rewards_'+symbol+'.png')
    
    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      [self.data.adjcp] + \
                      [0]*STOCK_DIM + \
                      [0]*STOCK_DIM + \
                      [0]*STOCK_DIM + \
                      [self.data.symbol]+ \
                      [self.data.datadate] + \
                      [self.data.open] + \
                      [self.data.high] + \
                      [self.data.low] + \
                      [self.data.volume] + \
                      [self.data.buy_sell_power] + \
                      [self.data.txnAmount]
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
