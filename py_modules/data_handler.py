# SCRIPT TO HANDLE RAW DATA 
# Importing dependencies
import numpy as np

# Function to compute shock array and volatility
def compute_elements(prices, returns_array):
    
    # Computing log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Computing expected return (logarithmic - mu) and volatility before 180d
    mu = np.mean(log_returns)
    vol = log_returns.std()

    # Converting returns array into shocks
    shock_array = list(np.log(returns_array / returns_array.shift(1)).dropna() - mu)

    return vol, shock_array

# Function to compute mean
def compute_mean(returns_array):

    # Computing log returns using NumPy
    log = np.log(returns_array / returns_array.shift(1)).dropna()

    # Returning mean
    return log.mean()
