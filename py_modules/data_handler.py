# SCRIPT TO HANDLE RAW DATA 
# Importing dependencies
import numpy as np
import math

# Function to compute EGARCH elements 
def compute_EGARCH_elements(prices, returns_array):
    
    # Computing log returns
    volatility_array = np.log(prices / prices.shift(1)).dropna()

    # Computing expected return (logarithmic - mu) and volatility before 180d
    # Volatility is plugged into EGARCH to find forecasted volatility
    mu = np.mean(volatility_array)
    vol = volatility_array.std()

    # Converting returns array into shocks
    # Shock array help refine EGARCH volatility
    log_returns_array = np.log(returns_array / returns_array.shift(1)).dropna()
    shock_array = list(log_returns_array - mu)
    
    # Computing mean and annualizing
    mean = log_returns_array.mean() * 252

    return vol, shock_array, mean

# Function to compute Jump elements for MJD simulation (Historical)
def compute_jump_elements(jump_returns, time):

    # Computing threshold for jumps
    # Values over this threshold is taken as a "jump"
    threshold = 2 * jump_returns.std()

    # Obtaining jumps in fractions
    jumps = jump_returns[abs(jump_returns) > threshold].tolist()
    
    # JUMP COMPONENT 1
    # Computing Average jump frequency over given time
    # Average Jump Frequency = LAMBDA
    lambda_ = len(jumps)/ (1/time)

    # JUMP COMPONENT 2
    # Obtaining average jump (Logarithmic)
    # Average Jump = KAPPA
    ksum = 0
    for j in jumps:
        ksum += math.log(1+j)
    k = 0 if len(jumps) == 0 else ksum/len(jumps)

    # JUMP COMPONENT 3
    # Computing jump volatility
    # Jump volatility  = SIGMA_j
    jump_vol_array = [math.log(1+j) for j in jumps]
    sig_j = 0 if len(jump_vol_array) < 2 else np.std(jump_vol_array, ddof=1)

    return k, lambda_, sig_j