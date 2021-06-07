# Stock-Trading-RL-Bot

Repository for Northwestern University MSDS Capstone Individual Project - Ensemble Reinforcement Learning Agent Stock Trader

This project highlights how to leverage the FinRL Python package to build 3 types of Reinforcment Learning Agents: DDPG, TD3, SAC. Each of these models are actor-critic reinforcement learning agents and differ more in their calculations/value estimations rather than architecture. To leverage each model, at each time step (each trading day), we evaluate each stock ticker from our watch list and have each model make a decision. Once we have a decision (buy, do nothing, sell) from each model for a given stock ticker, we use majority voting to determine which action to take. Once we have done this, we take the mean quantity of all models that voted this decision as our trade quantity. 

# Navigating the Repository & running the code

The "stock_trader_bot_Git.py" file runs the stock trader bot. The three RL models leveraged in the .py file are built & trained by the three .py files in the "Model Trainer Files" folder. Once these are trained and saved to a location that the .py file can access, the .py file can be run. The API keys and twitter account both need to be entered in the necessary locations in the "stock_trader_bot_Git.py" file as well in order to run.

# Why leverage RL for Stock Trading?

The stock market is one of the best tools for both wealth growth and preservation. Leveraging RL to do this both takes human emotion out of the equation, but also ensures that our portfolio is constantly being re-evaluated based on the current market conditions following a time-proven, consistent policy.

# Code Highlights Below:
## Ensemble/Majority Voting Strategy

These functions take the decision and quantity from each RL model for a given stock ticker and, based on majority voting, executes a trade.

``` Python
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

```

## Create a Watch List Based on Tweets

The below functions pull tweets from a particular twitter user, scrapes for stock ticker symbols in those tweets, and creates a watch list comprised of them.

``` Python
################# Every time he tweets about a stock, add it to the list. Keep list distinct - Finished

def Extract_Tickers(tweet):
  Tickers = re.findall(r'[$][A-Za-z][\S]*', tweet)
  Tickers = [re.sub('[$]', '', ticker) for ticker in Tickers]
  return Tickers

def Pull_Tweets():
    # Pull tweets
    # select twitter user below. a function could be created to call this function for multiple twitter users and append the ticker lists together
    new_tweets = twitter_api.user_timeline(screen_name = ,count=200)

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
```

# Future Developments

1) Build out these same models from scratch, rather than leveraging FinRL
  a) Build the environment to read current holdings and past trades from the Alpaca API rather than          storing history
2) Change to using 5-min stock market data rather than daily, to ensure that we make timely transitions;    which is critical for volatile markets.
