# Importing dependencies
import yfinance as yf
import numpy as np
from scipy.optimize import minimize

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

    # Computing Correlation Matrix
    corr = []
    for datum in data:
        stock = yf.Ticker(datum[0])
        returns = stock.history(period="61d")["Close"]
        log_returns = np.array(np.log(returns / returns.shift(1)).dropna())
        corr.append(log_returns)

    # Computing expected_returns and declaring  risk aversion factor
    expected_returns = [(item[3]-item[2])/item[2] for item in data]

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