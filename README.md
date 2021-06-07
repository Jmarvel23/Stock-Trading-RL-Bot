# Stock-Trading-RL-Bot

Repository for Northwestern University MSDS Capstone Individual Project - Ensemble Reinforcement Learning Agent Stock Trader

This project highlights how to leverage the FinRL Python package to build 3 types of Reinforcment Learning Agents: DDPG, TD3, SAC. Each of these models are actor-critic reinforcement learning agents and differ more in their calculations/value estimations rather than architecture. To leverage each model, at each time step (each trading day), we evaluate each stock ticker from our watch list and have each model make a decision. Once we have a decision (buy, do nothing, sell) from each model for a given stock ticker, we use majority voting to determine which action to take. Once we have done this, we take the mean quantity of all models that voted this decision as our trade quantity. 


# Why leverage RL for Stock Trading?

The stock market is one of the best tools for both wealth growth and preservation. Leveraging RL to do this both takes human emotion out of the equation, but also ensures that our portfolio is constantly being re-evaluated based on the current market conditions following a time-proven, consistent policy.


# Ensemble/Majority Voting Strategy

``` Python

```


## Future Developments

1) Build out these same models from scratch, rather than leveraging FinRL
  a) Build the environment to read current holdings and past trades from the Alpaca API rather than          storing history
2) Change to using 5-min stock market data rather than daily, to ensure that we make timely transitions;    which is critical for volatile markets.
