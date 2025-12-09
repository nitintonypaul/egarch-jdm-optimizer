# Importing dependencies
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import sys
import pandas as pd

# MVO Objective
def objective_function(w, cov, lam, mu):
    return (lam * 0.5 * np.dot(np.dot(w.T,cov), w)) - np.dot(w, mu.T)

# Optimizer
def optimize(data, risk_aversion, vol_arr):

    # Computing total investment
    total_investment = sum([x[1] for x in data])

    # Removing investments with losses
    vol_arr = [v for i,v in enumerate(vol_arr) if data[i][5] > 0]
    data = [x for x in data if x[5] > 0]

    # Checking if data length is 0
    # I.E no investments are present which returns a profit
    if len(data) == 0:
        return None
    
    # Changing EGARCH volatility array into diagonal matrix
    diag_vol = np.diag(np.array(vol_arr))

    # Downloading necessary data from yfinance
    ticker_array = [ticker[0] for ticker in data]
    realized_returns = yf.download(ticker_array, period="60d", auto_adjust=True)["Close"]
    
    # Collapsing to frame if there are multi-index columns for a single ticker
    if isinstance(realized_returns, pd.Series):
        realized_returns = realized_returns.to_frame()

    # Checking ticker to returns column match
    if len(vol_arr) != len(realized_returns.columns):
        print("----- COLUMNS DON'T MATCH -----")
        exit(0)
    
    # Using historical correlation and EGARCH volatility for covariance matrix
    log_returns = np.log(realized_returns / realized_returns.shift(1)).dropna()
    corr_matrix = log_returns.corr()
    cov_matrix = np.dot(np.dot(diag_vol, corr_matrix), diag_vol)

    # Computing expected_returns
    expected_returns = [item[5]/100 for item in data]

    # Solution
    n = len(expected_returns)
    x0 = np.ones(n) / n 

    # Setting bounds and constraints
    bounds = [(0, 1)] * n 
    constraints = ({'type': 'eq','fun': lambda w: np.sum(w) - 1})

    # Optimization using scipy
    res = minimize(objective_function, x0, args=(cov_matrix, risk_aversion, np.array(expected_returns)), method='SLSQP', bounds=bounds, constraints=constraints)
    if not(res.success):
        print("----- OPTIMIZATION FAILED -----")
        sys.exit(0)
    
    # Recieving and dropping insignificant weights
    weights = res.x
    weights[weights < 1e-7] = 0
    weights /= np.sum(weights)

    # Replacing investment and expected returns with optimized values
    for i in range(len(data)):
        data[i][4] /= data[i][1]
        data[i][1] = weights[i] * total_investment
        data[i][4] *= data[i][1]
    
    # Returning data
    return [x for x in data if x[1]!=0]