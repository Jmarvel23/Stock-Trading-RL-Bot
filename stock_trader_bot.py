################# Import Packages
import sys

sys.path.append("..")

# import packages
import pandas as pd
import numpy as np
import datetime
from pytz import timezone
tz = timezone('EST')

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.model.models import DRLAgent

from pprint import pprint

from collections import Counter

import alpaca_trade_api as tradeapi
import time   # do I need this?
import tweepy

import re

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

################# API Connections - Finished

twitter_api_key = "o96nhfeenYeXyhAndVw0bBMcO"
twitter_api_secret_key = "kDGqWx5c4K0WKvKGp6EXZFmijz7jCc3JCTrcaHitm7XQwh7wAC"
twitter_bearer_token = "AAAAAAAAAAAAAAAAAAAAADHYOQEAAAAA94Y4Shc188NTs68AGIkhWSdWiW8%3DJFTVm5JDh82jxXidjkw8ZqUR7vaFKqfrMg9izvtTHcBqro7gWw"

alpaca_key = "PK6VSIINV96OQEVTT4VI"
alpaca_sec = "bJWDonVu3LzIQe4wrJ6ooE9JPChpIuAbkXUbr58s"
alpaca_url = "https://paper-api.alpaca.markets"

############ Alpaca
#api_version v2 refers to the version that we'll use
#very important for the documentation
alpaca_api = tradeapi.REST(alpaca_key, alpaca_sec, alpaca_url, api_version='v2')
#Init our account var
account = alpaca_api.get_account()

############ Twitter
# Connect to Twitter API using the secrets
auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret_key)
twitter_api = tweepy.API(auth)

################# Every time he tweets about a stock, add it to the list. Keep list distinct - Finished

def Extract_Tickers(tweet):
  Tickers = re.findall(r'[$][A-Za-z][\S]*', tweet)
  Tickers = [re.sub('[$]', '', ticker) for ticker in Tickers]
  return Tickers

def Pull_Tweets():
    # Pull tweets
    # currently limiting on 200 tweets, the limit that we can see. Don't want to exclude anything if he has tweeted a ton
    new_tweets = twitter_api.user_timeline(screen_name = "realwillmeade",count=200)

    # only look at tweets that have a ticker symbol in them
    reg = re.compile("[$][A-Za-z][\S]*")
    stock_tweets = [tweet.text for tweet in new_tweets if bool(re.search(reg, tweet.text))]
    return stock_tweets

def maintain_Tweets(Tickers):
    # maintain a distinct list of all the tickers
    stock_tweets = Pull_Tweets()
    for tweet in stock_tweets:
        tics = Extract_Tickers(tweet)
        for tic in tics:
            if tic.isalpha():
                Tickers.append(tic.upper())
    Tickers = list(set(Tickers))
    return Tickers

################# Continuously pull stock data - Finished

def Pull_Stock_Data(stocks):

    today = datetime.date.today().strftime('%Y-%m-%d')
    today_45 = (datetime.date.today()-datetime.timedelta(days = 45)).strftime('%Y-%m-%d')

    df = YahooDownloader(start_date = today_45,
                         end_date = today,
                         ticker_list = stocks).fetch_data()


    # Perform Feature Engineering:
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                        use_turbulence=False,
                        user_defined_feature = False)

    processed = fe.preprocess_data(df)

    processed['log_volume'] = np.log(processed.volume*processed.close)
    processed['change'] = (processed.close-processed.open)/processed.close
    processed['daily_variance'] = (processed.high-processed.low)/processed.close
    return processed

################# Set the current date as the trade environment - Finished

def create_trade_gym(stocks):

    processed = Pull_Stock_Data(stocks)

    today = datetime.date.today().strftime('%Y-%m-%d')
    last_monday = (datetime.datetime.now(tz) + datetime.timedelta(days=- (7+datetime.datetime.now(tz).weekday()))).strftime('%Y-%m-%d')
    trade = data_split(processed, last_monday,today)

    information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'day',
                        'macd', 'rsi_30', 'cci_30', 'dx_30']

    e_trade_gym = StockTradingEnvCashpenalty(df = trade,initial_amount = 1e6,hmax = 5000,
                                    turbulence_threshold = None,
                                    currency='$',
                                    buy_cost_pct=0,
                                    sell_cost_pct=0,
                                    cash_penalty_proportion=0.2,
                                    cache_indicator_data=False,
                                    daily_information_cols = information_cols,
                                    print_verbosity = 500,
                                    random_start = False)

    return e_trade_gym

def create_trade_env(e_trade_gym):
    env_trade, _ = e_trade_gym.get_sb_env()

    return env_trade

################# Alpaca - Order Placement (Ensemble Strategy) - Finished

def calc_act(i_list):
    counter = Counter(i_list)
    if counter[1] >= 2:
        return 1
    elif counter[-1] >= 2:
        return -1
    else :
        return 0

def map_act(num):
    n_list = [-1, 0, 1]
    t_list = ["sell", "hold", "buy"]
    index = n_list.index(num)
    return t_list[index]

def act(act_list, transact_list, Ticker):
    action = calc_act(act_list)

    if action == 0:
        print('do nothing')
    #     do nothing
    else:
        index = [i for i, e in enumerate(act_list) if e == action]
        quantity = round(np.average([transact_list[i] for i in index]))

        action_text = map_act(action)

        try:
            alpaca_api.submit_order(symbol=Ticker,
                                    qty=quantity,
                                    side=action_text,
                                    type="market",
                                    time_in_force="day")
            print(action_text, quantity, Ticker, 'executed')
        except Exception as ex:
            print(ex)


################# Final Functions

def time_to_open(current_time):
    if current_time.weekday() <= 4:
        d = (current_time + datetime.timedelta(days=1)).date()
    else:
        days_to_mon = 0 - current_time.weekday() + 7
        d = (current_time + datetime.timedelta(days=days_to_mon)).date()
    next_day = datetime.datetime.combine(d, datetime.time(9, 30, tzinfo=tz))
    seconds = (next_day - current_time).total_seconds()
    return seconds - (60 * 60)

def DRL_prediction_cust(model, environment):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs)
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)
#             if i == (len(environment.df.index.unique()) - 2):
#               account_memory = test_env.env_method(method_name="save_asset_memory")
#               actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

def run_bot():
    print('bot initiated')
    # c = 0
    while True:
        # Check if Monday-Friday
        if datetime.datetime.now(tz).weekday() >= 0 and datetime.datetime.now(tz).weekday() <= 4:
            # Checks market is open
            print('Trading day')
            if datetime.datetime.now(tz).time() > datetime.time(8, 30) and datetime.datetime.now(tz).time() <= datetime.time(14, 30):

                # if c == 0:
                #     Tickers = []
                # Tickers = maintain_Tweets(Tickers)
                # c += 1
                Tickers = config.DOW_30_TICKER

                e_trade_gym = create_trade_gym(Tickers)
                e_trade_gym.hmax = 5000

                e_trade_env = create_trade_env(e_trade_gym)

                agent = DRLAgent(env = e_trade_env)

                policy_kwargs = {"net_arch": [1 for _ in range(1)],}
                td3_params ={'learning_rate': 0.000005,'batch_size': 1,'gamma': 0.99}
                td3 = agent.get_model("td3",
                      model_kwargs = td3_params,
                      policy_kwargs = policy_kwargs, verbose = 0)
                TD3 = td3.load("TD3_dow30cons.model")
                TD3_account_value, TD3_actions = DRL_prediction_cust(model=TD3,environment = e_trade_gym)

                policy_kwargs = {"net_arch": [1 for _ in range(1)],}
                sac_params ={'ent_coef': 0.0,'learning_rate': 0.000005,'batch_size': 1,'gamma': 0.99}
                sac = agent.get_model("sac",
                        model_kwargs = sac_params,
                        policy_kwargs = policy_kwargs, verbose = 0)
                SAC = sac.load("SAC_dow30cons.model")
                SAC_account_value, SAC_actions = DRL_prediction_cust(model=SAC,environment = e_trade_gym)

                policy_kwargs = {"net_arch": [1 for _ in range(1)],}
                ddpg_params ={'learning_rate': 0.000005,'batch_size': 1,'gamma': 0.99}
                ddpg = agent.get_model("ddpg",
                        model_kwargs = ddpg_params,
                        policy_kwargs = policy_kwargs, verbose = 0)
                DDPG = ddpg.load("DDPG_dow30cons.model")
                DDPG_account_value, DDPG_actions = DRL_prediction_cust(model=DDPG,environment = e_trade_gym)

                for i in range(0, len(Tickers)):
                    act_list = [round(TD3_actions.actions[0][i]), round(SAC_actions.actions[0][i]), round(DDPG_actions.actions[0][i])]
                    transact_list = [round(TD3_actions.transactions[0][i]), round(SAC_actions.transactions[0][i]), round(DDPG_actions.transactions[0][i])]
                    Ticker = Tickers[i]
                    act(act_list, transact_list, Ticker)

                time.sleep(time_to_open(datetime.datetime.now(tz)))
            else:
                # Get time amount until open, sleep that amount
                print('Market closed ({})'.format(datetime.datetime.now(tz)))
                print('Sleeping', round(time_to_open(datetime.datetime.now(tz))/60/60, 2), 'hours')
                time.sleep(time_to_open(datetime.datetime.now(tz)))
        else:
            # If not trading day, find out how much until open, sleep that amount
            print('Market closed ({})'.format(datetime.datetime.now(tz)))
            print('Sleeping', round(time_to_open(datetime.datetime.now(tz))/60/60, 2), 'hours')
            time.sleep(time_to_open(datetime.datetime.now(tz)))

run_bot()
