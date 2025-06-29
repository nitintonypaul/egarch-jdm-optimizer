import numpy as np

# Function to compute VAR and CVAR for given level of confidence
def find_VAR(totalInvestment, portfolio, confidence):

    # Computing losses array
    losses = totalInvestment - portfolio[:, -1]

    # Computing VAR and then CVAR from it
    VAR = np.percentile(losses, confidence)
    CVAR = np.mean([x for x in losses if x >= VAR])

    # Displaying both values at risk models
    print("\n---- VALUE AT RISK ----")
    
    # Displaying VAR and CVAR messages
    # Different VAR messages depending upon the value
    if VAR > 0:
        print(f"The {100-confidence}% Value at Risk (VaR) is {VAR:.2f}. There is a {100-confidence}% chance you will lose at least {VAR:.2f}")
    elif VAR <= 0:
        print(f"The {100-confidence}% Value at Risk (VaR) is {VAR:.2f}. There is a {100-confidence}% chance you will gain at least {-VAR:.2f}")
    print(f"The {100-confidence}% Conditional Value at Risk (CVaR) is {CVAR:.2f}. If you fall into that worst {100-confidence}%, the average loss is {CVAR:.2f}")