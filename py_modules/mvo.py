# Importing dependencies
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import time

# Minimizing function
def risk_return(w, cov, lam, mu):
    return (0.5 * np.dot(np.dot(w.T,cov), w)) - (lam * np.dot(w, mu.T))

# Optimization (MVO)
def optimize(data, risk_aversion):

    # Risk Aversion by default is 0.35 since the period of duration is 1 day
    # Risk aversion can be changed from default by argument when running main.py
    
    # Computing total investment
    total_investment = sum([x[1] for x in data])

    # Removing investment with losses
    data = [x for x in data if x[5] > 0]

    # Checking if data length is 0
    # I.E no investments are present which returns a profit
    if len(data) == 0:
        return None

    # Downloading necessary data from yfinance
    ticker_array = [ticker[0] for ticker in data]
    full_returns = yf.download(ticker_array, period="30d", auto_adjust=True)["Close"]

    # Computing correlation matrix
    corr = []
    for ticker in full_returns.columns:
        prices = full_returns[ticker]
        log_returns = np.array(np.log(prices / prices.shift(1)).dropna())
        corr.append(log_returns)

    # Computing expected_returns
    expected_returns = [item[5]/100 for item in data]

    # Computing covariance matrix
    corr = np.vstack(corr)
    cov = np.cov(corr)

    # Solution
    n = len(expected_returns)
    x0 = np.ones(n) / n  # Start with equal weights

    # Setting bounds and constraints
    bounds = [(0, 1)] * n 
    constraints = ({'type': 'eq','fun': lambda w: np.sum(w) - 1})

    # Optimization using scipy
    res = minimize(risk_return, x0, args=(cov, risk_aversion, np.array(expected_returns)), method='SLSQP', bounds=bounds, constraints=constraints)

    # Recieving and dropping insignificant weights
    weights = res.x
    weights[weights < 1e-7] = 0

    # Normalizing to fit complete investment capital
    weights /= np.sum(weights)

    # Replacing investment and expected returns with optimized values
    for i in range(len(data)):
        data[i][4] /= data[i][1]
        data[i][1] = weights[i]*total_investment
        data[i][4] *= data[i][1]
    
    # Returning data
    return [x for x in data if x[1]!=0]