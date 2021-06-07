#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("..")


# In[ ]:


import pandas as pd
print(pd.__version__)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

from pprint import pprint


# In[ ]:



import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# In[ ]:



df = YahooDownloader(start_date = '2019-05-30',
                     end_date = '2021-06-01',
                     ticker_list = config.DOW_30_TICKER).fetch_data()


# In[ ]:



fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)


# In[ ]:


processed['log_volume'] = np.log(processed.volume*processed.close)
processed['change'] = (processed.close-processed.open)/processed.close
processed['daily_variance'] = (processed.high-processed.low)/processed.close


# In[ ]:


train = data_split(processed, '2019-05-30','2020-05-30')
trade = data_split(processed, '2020-05-30','2021-06-01')
print(len(train))
print(len(trade))


# In[ ]:


information_cols = ['daily_variance', 'change', 'log_volume', 'close','day', 
                    'macd', 'rsi_30', 'cci_30', 'dx_30']

e_train_gym = StockTradingEnvCashpenalty(df = train,initial_amount = 1e6,hmax = 5000, 
                                turbulence_threshold = None, 
                                currency='$',
                                buy_cost_pct=0,
                                sell_cost_pct=0,
                                cash_penalty_proportion=0.2,
                                cache_indicator_data=True,
                                daily_information_cols = information_cols, 
                                print_verbosity = 500, 
                                random_start = True)

e_trade_gym = StockTradingEnvCashpenalty(df = trade,initial_amount = 1e6,hmax = 5000, 
                                turbulence_threshold = None, 
                                currency='$',
                                buy_cost_pct=0,
                                sell_cost_pct=0,
                                cash_penalty_proportion=0.2,
                                cache_indicator_data=True,
                                daily_information_cols = information_cols, 
                                print_verbosity = 500, 
                                random_start = False)


# In[ ]:



#this is our training env. It allows multiprocessing
env_train, _ = e_train_gym.get_sb_env()

#this is our observation environment. It allows full diagnostics
env_trade, _ = e_trade_gym.get_sb_env()


# In[ ]:


agent = DRLAgent(env = env_train)
# from stable_baselines3 import DDPG

ddpg_params ={'learning_rate': 0.000005, 
             'batch_size': 600, 
            'gamma': 0.99}

policy_kwargs = {
#     "activation_fn": ReLU,
    "net_arch": [600 for _ in range(10)], 
#     "squash_output": True
}

model = agent.get_model("ddpg",  
                        model_kwargs = ddpg_params, 
                        policy_kwargs = policy_kwargs, verbose = 0)


# In[ ]:


model.learn(total_timesteps = 5000000, 
            eval_env = env_trade, 
            eval_freq = 500,
            log_interval = 1, 
            tb_log_name = 'env_cashpenalty_highlr',
            n_eval_episodes = 1)


# In[ ]:


model.save("DDPG_dow30cons.model")


# In[ ]:




