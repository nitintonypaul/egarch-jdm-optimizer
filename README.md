# EGARCH-JDM Based Portfolio Optimizer (MVO)

---

## Table of Contents

1. [Project Outline](#1-project-outline)  
2. [Disclaimer](#2-disclaimer)   
3. [EGARCH](#3-egarch)  
4. [Merton Jump Diffusion Model](#4-merton-jump-diffusion-model)  
5. [Mean Variance Optimization](#5-mean-variance-optimization)  
6. [Build Instructions](#6-build-instructions)  
7. [License](#7-license)

---

## 1. Project Outline

This project demonstrates a portfolio optimizer that integrates advanced volatility and jump-diffusion models with classical mean-variance optimization. The workflow is:

- The user provides a selection of stocks, an investment amount, and an optional risk aversion factor for MVO.
- For each stock, the model computes volatility using **EGARCH**, implemented from scratch in C++ using **Conjugate Gradient** method of optimization, capturing time-varying and asymmetric volatility effects.
- These volatilities feed into a **Monte Carlo simulation under the Merton Jump Diffusion (MJD) model** (also in C++) with 200,000 simulated price paths to estimate expected future prices.
- The current portfolio valuation based on these simulations is displayed.
- Finally, asset allocation is optimized via **Mean-Variance Optimization (MVO)** in Python, reallocating capital toward assets with superior risk-adjusted returns and pruning negligible positions.

This project serves as a practical example of combining volatility modelling, jump-diffusion, and classical portfolio theory into a single quantitative framework.

---

## 2. Disclaimer

This project is strictly intended for **demonstration and educational purposes only**. It serves to showcase the author’s ability to implement quantitative finance models, including EGARCH volatility modelling, Merton Jump Diffusion simulations, and Mean-Variance Optimization.

Users are explicitly advised **not to use this project for any real-world financial, investment, or trading decisions**. The models and optimizations presented herein may involve significant risk and simplifications that do not reflect live market conditions.

The author disclaims all liability for any financial losses or damages that may arise directly or indirectly from the use or reliance on this project or its outputs. Users assume full responsibility for any actions taken based on the information or results provided.

---
## 3. EGARCH

### 1. What is EGARCH?

The **Exponential Generalized Autoregressive Conditional Heteroskedasticity (EGARCH)** model is a statistical model primarily used in financial econometrics to analyze and forecast the volatility of financial time series, such as asset returns. It extends the traditional [GARCH](https://github.com/nitintonypaul/garch) model by addressing certain limitations, notably its ability to capture the **asymmetric response of volatility to positive and negative shocks**.

Unlike symmetric GARCH models, where only the magnitude of past innovations affects current volatility, EGARCH explicitly accounts for the **leverage effect**. This means that negative shocks (e.g., a stock price drop) can have a greater impact on future volatility than positive shocks of the same magnitude (e.g., a stock price increase). This asymmetry is a critical feature in financial markets, as bad news often leads to a more significant increase in volatility than good news.

A key advantage of the EGARCH model is that it models the **logarithm of the conditional variance**. This ensures that the forecasted variance is always positive, even if some of the model parameters are negative, removing the need for non-negativity constraints typically required in standard GARCH models.

### 2. Mathematics of EGARCH

The EGARCH(p, q) model for the logarithm of the conditional variance, $\ln(\sigma_t^2)$, is commonly expressed as:

$$\ln(\sigma_t^2) = \omega + \sum_{i=1}^p \alpha_i \left( |Z_{t-i}| - E[|Z_{t-i}|] \right) + \sum_{i=1}^p \gamma_i Z_{t-i} + \sum_{j=1}^q \beta_j \ln(\sigma_{t-j}^2)$$

Where:
* $\sigma_t^2$: The conditional variance at time $t$.
* $\omega$: A constant term, representing the baseline log-volatility.
* $Z_t = \frac{\epsilon_t}{\sigma_t}$: The standardized residual, with $\epsilon_t$ being the innovation (shock) at time $t$. $Z_t$ is typically assumed to follow a standard normal distribution ($$Z_t \sim N(0,1)$$).
* $\alpha_i$: Captures the symmetric effect of past standardized residuals on current volatility.
* $\gamma_i$: Represents the **asymmetric "leverage effect."** If $\gamma_i < 0$, it signifies that negative shocks increase volatility more than positive shocks of the same magnitude.
* $\beta_j$: Measures the **persistence of volatility**, indicating how slowly volatility reverts to its long-term mean after a shock.

Our implementation focuses on an **EGARCH(1,1)** model, which simplifies the equation to:

$$\ln(\sigma_t^2) = \omega + \beta_1 \ln(\sigma_{t-1}^2) + \alpha_1 |Z_{t-1}| + \gamma_1 Z_{t-1}$$

In the provided C++ code, the `compute_volatility` function directly implements this recurrence relation for $\ln(\sigma_t^2)$:

```cpp
double logsigsq = parameters[2] + (parameters[1] * std::log(std::pow(sig, 2))) + (parameters[0] * std::abs(shockarray[i-1] / sig)) + (parameters[3] * (shockarray[i-1] / sig));
// The 'parameters' array maps as follows:
// parameters[0] = alpha
// parameters[1] = beta
// parameters[2] = omega
// parameters[3] = gamma
sig = std::sqrt(std::exp(logsigsq)); // Converts log-variance back to standard deviation
```

### 3. Maximum Likelihood Estimation (MLE)

To estimate the parameters ($\omega, \alpha, \gamma, \beta$) of the EGARCH model, **Maximum Likelihood Estimation (MLE)** is the standard approach. Assuming that the standardized residuals $Z_t$ are independently and identically distributed (i.i.d.) and follow a standard normal distribution ($$Z_t \sim N(0,1)$$), the log-likelihood function for a series of $T$ observations is given by:

$$L(\theta) = \sum_{t=1}^T \left[ -\frac{1}{2} \ln(2\pi) - \frac{1}{2} \ln(\sigma_t^2) - \frac{1}{2} \frac{\epsilon_t^2}{\sigma_t^2} \right]$$

Where $\theta = (\alpha, \beta, \omega, \gamma)$ is the vector of parameters to be estimated, and $\epsilon_t$ are the observed shocks (returns).

In numerical optimization, it's common to minimize the **Negative Log-Likelihood (NLL)** function, which is simply $-L(\theta)$:

$$NLL(\theta) = \sum_{t=1}^T \left[ \frac{1}{2} \ln(2\pi) + \frac{1}{2} \ln(\sigma_t^2) + \frac{1}{2} \left(\frac{\epsilon_t}{\sigma_t}\right)^2 \right]$$

The term $\left(\frac{\epsilon_t}{\sigma_t}\right)^2$ is equivalent to $Z_t^2$.
The `armijo_condition` function in the C++ code calculates this Negative Log-Likelihood, where terms like `0.5 * (std::log(2 * M_PI) + logsigsq + Z_LHS)` directly correspond to the individual components of the NLL sum for each time step. The objective of MLE is to find the parameter values that minimize this NLL function, thereby maximizing the probability of observing the given data.

### 4. Conjugate Gradient Optimization

The estimation of EGARCH parameters through MLE requires a numerical optimization algorithm. This implementation utilizes the **Conjugate Gradient (CG) method**, specifically employing the **Fletcher-Reeves** formula, complemented by a **Backtracking Line Search** and the **Armijo Condition** to efficiently determine optimal step sizes.

#### Conjugate Gradient Method

The Conjugate Gradient method is an iterative optimization algorithm used to minimize functions of multiple variables. It is particularly well-suited for problems where direct calculation of the inverse Hessian matrix (as required by Newton's method) is computationally prohibitive.

The core principle of CG is to generate a sequence of search directions that are "conjugate" with respect to the Hessian matrix of the objective function. This ensures that each new search direction effectively makes progress towards the minimum without undoing the progress made in previous steps, leading to faster convergence than simple gradient descent.

The parameter update rule at each iteration $k$ is:
$$\theta_{k+1} = \theta_k + \alpha_k d_k$$
where $\theta_k$ is the parameter vector, $\alpha_k$ is the optimal step size, and $d_k$ is the search direction.

#### Fletcher-Reeves Formula

The search direction $d_k$ in Conjugate Gradient methods is typically a combination of the negative gradient at the current point and the previous search direction. The **Fletcher-Reeves** formula is a widely used method for calculating the $\beta_k$ coefficient, which determines this combination:

$$d_{k+1} = -\nabla f(\theta_{k+1}) + \beta_k d_k$$
$$\beta_k^{FR} = \frac{||\nabla f(\theta_{k+1})||^2}{||\nabla f(\theta_k)||^2} = \frac{\nabla f(\theta_{k+1})^T \nabla f(\theta_{k+1})}{\nabla f(\theta_k)^T \nabla f(\theta_k)}$$

Our `optimize` function implements this calculation:

```cpp
// Calculation of the denominator (||gradient_k||^2)
double beta_denominator = 0;
for (int j = 0; j < 4; j++) {
    beta_denominator += gradient[j] * gradient[j];
}

// ... Call to compute_gradient() to get gradient_{k+1} ...

// Calculation of the numerator (||gradient_{k+1}||^2)
double beta_numerator = 0;
for (int j = 0; j < 4; j++) {
    beta_numerator += gradient[j] * gradient[j];
}

beta = beta_numerator / beta_denominator; // Computes beta_k

// Update direction using Fletcher-Reeves formula
for(int j = 0; j < 4; j++) {
    direction[j] = -gradient[j] + (beta * direction[j]);
}
```

A **"DOT PRODUCT DIAGONOSTIC"** is also included in the `optimize` function. This is a practical heuristic: if the dot product of the current gradient and the computed direction becomes too large (`dot >= -1e-12`), it can indicate numerical instability or that the search direction is no longer a sufficient descent direction. In such cases, the search direction is reset to the simpler steepest descent direction (`-gradient[j]`) to ensure robust optimization.

#### Backtracking Line Search

Once a search direction $d_k$ is determined, finding an appropriate step size $\alpha_k$ (referred to as `lambda` in the code) along this direction is critical. The **Backtracking Line Search** algorithm is used for this purpose. It starts with an initial step size (e.g., $\alpha=1$) and repeatedly shrinks it by a fixed factor (e.g., `beta = 0.5`) until the Armijo condition is satisfied.

The `backtrack_line_search` function continuously reduces `alpha` until the `armijo_condition` returns `true`:

```cpp
double backtrack_line_search(double dir[], const std::vector<double> &shockarray) {
    double beta = 0.5; // Step multiplier (shrinks alpha by half)
    double alpha = 1;  // Initial step value

    while (!armijo_condition(alpha, dir, shockarray)) {
        alpha *= beta; // Decrease alpha
    }
    return alpha;
}
```

#### Armijo Condition

The **Armijo condition** is a fundamental criterion in line search algorithms that ensures the chosen step size $\alpha$ leads to a "sufficient decrease" in the objective function. This prevents oscillations or very slow convergence.

The condition states:
$$f(\theta + \alpha d) \le f(\theta) + c \alpha \nabla f(\theta)^T d$$

Where:
* $f(\theta)$: The objective function (Negative Log-Likelihood).
* $d$: The search direction.
* $\alpha$: The step size.
* $c$: A small constant between 0 and 1 (typically $10^{-4}$), defined as `ARMIJO_C` in the code.

The `armijo_condition` function directly implements this inequality:

```cpp
// Calculate LHS: f(x + ALPHA.d)
double LHS = /* computed NLL with modified parameters */;

// Calculate RHS: f(x) + c.ALPHA.T(grad(f(x))).d
double total_likelihood = /* computed NLL with current parameters */;
double RHS = total_likelihood;
double product = ARMIJO_C * alpha;
double tempsum = 0;
for (int i = 0; i < 4; i++) {
    tempsum += gradient[i]*dir[i]; // Dot product: gradient(f(x))^T * d
}
product *= tempsum;
RHS += product;

return (LHS <= RHS); // Returns true if Armijo condition is met
```

---

## 4. Merton Jump Diffusion Model

The **Merton Jump Diffusion (MJD) model** is a powerful extension of the standard Black-Scholes model, designed to capture sudden, discontinuous price movements (jumps) often observed in financial markets, alongside continuous, small fluctuations. It's particularly useful for modeling asset prices that exhibit fat tails and skewness in their return distributions, which standard models often miss.

### Mathematical Formulation

The price dynamics under Merton’s Jump Diffusion are governed by the following stochastic differential equation for the asset price $S_t$:

$$S_t = S_0 \exp\left(\left(\mu - \frac{1}{2}\sigma^2 - k\lambda\right)t + \sigma W_t + \sum_{i=1}^{N(t)} \ln(1+J_i)\right)$$

Let's break down the components of this equation:

* $S_t$: The asset price at time $t$.
* $S_0$: The initial (current) asset price.
* $\mu$: The **continuous drift term**, representing the expected return rate from the continuous part of the process.
* $\sigma$: The **diffusion volatility**, which is the standard deviation of the continuous returns (Brownian motion).
* $W_t$: A **Wiener process** ($$W_t \sim \mathcal{N}(0,t)$$), modeling the continuous, random fluctuations (Brownian motion).
* $t$: The time horizon, typically in trading years (e.g., $1/252$ for one trading day).
* $\lambda$: The **jump intensity**, representing the average number of jumps expected per unit of time.
* $N(t)$: A **Poisson process** ($$N(t)\sim \mathrm{Poisson}(\lambda t)$$), which counts the number of jumps that occur up to time $t$.
* $k$: The **mean jump size**, defined as $k = E[\ln(1+J)]$, where $J$ represents the proportional jump magnitude.
* $\ln(1+J_i)$: The **log-jump magnitudes**, which are random variables typically assumed to be normally distributed:
    $$\ln(1+J_i) \sim \mathcal{N}\left(\ln(1+k) - \frac{1}{2}\sigma_j^2, \sigma_j^2\right)$$
    Here, $\sigma_j$ is the **jump volatility**, representing the standard deviation of the log-jump magnitudes.

The equation effectively combines three forces driving asset prices:
1.  **Adjusted Continuous Drift:** $(\mu - 0.5\sigma^2 - k\lambda)t$ accounts for the average continuous growth, with adjustments for Itô's lemma ($0.5\sigma^2$) and the mean effect of jumps ($k\lambda$) to ensure the process remains unbiased.
2.  **Diffusion Term:** $\sigma W_t$ captures the everyday, small, continuous, and normally distributed price movements.
3.  **Jump Component:** $\sum_{i=1}^{N(t)} \ln(1+J_i)$ adds discrete, sudden shocks to the price whenever a jump event occurs. Each $J_i$ represents a proportional change, reflecting rare, impactful market moves.

### Monte Carlo Simulation for Expected Price

To estimate the expected terminal asset price $S_t$ under the Merton JDM, we employ a **Monte Carlo simulation**. This involves generating a large number of independent sample paths for the asset price and then averaging their terminal values. Our implementation performs $M=200,000$ such simulations, efficiently accumulating the results in a single `double` variable (no arrays needed for storing individual paths).

Here's the step-by-step process for each simulation:

1.  **Simulate Jump Events:**
    * First, we determine the number of jumps $N$ that occur over the time horizon $t$ by drawing from a Poisson distribution: $N \sim \mathrm{Poisson}(\lambda t)$.
    * If $N > 0$, for each jump $j = 1, \dots, N$, we draw a log-jump magnitude $\ln(1+J_j)$ from its specified normal distribution:
        $$\ln(1+J_j) \sim \mathcal{N}\left(\ln(1+k) - \frac{1}{2}\sigma_j^2, \sigma_j^2\right)$$
    * The total impact of all jumps for this path is then summed: $J_{sum} = \sum_{j=1}^{N} \ln(1+J_j)$.

2.  **Simulate Brownian Motion:**
    * A random variate $Z_i$ is drawn from a standard normal distribution ($$Z_i \sim \mathcal{N}(0,1)$$).
    * The Brownian increment is then calculated as $W_t = \sqrt{t} Z_i$.

3.  **Compute Terminal Price:**
    * Using the simulated jump sum and Brownian increment, the terminal price for this specific path is calculated directly from the MJD equation:
        $$S_t = S_0 \exp\left(\left(\mu - \frac{1}{2}\sigma^2 - k\lambda\right)t + \sigma W_t + J_{sum}\right)$$

4.  **Accumulate Results:**
    * The calculated $S_t$ for the current simulation is added to a running sum (`price_sum`). This ensures memory efficiency by not storing all $M$ individual paths.

After all $M$ simulations are completed, the **expected terminal price** is estimated by simply dividing the accumulated `price_sum` by the total number of simulations $M$:

```python
expected_price = price_sum / M
```

This average terminal price is then used for portfolio valuation, providing a more robust risk assessment by explicitly incorporating both continuous volatility and sudden market jumps.

---

## 5. Mean Variance Optimization

The core of this optimization lies in **Modern Portfolio Theory (MPT)**, pioneered by Harry Markowitz. MPT posits that investors are rational and seek to maximize return for a given risk level, or minimize risk for a given return level. This trade-off is central to MVO.

The problem is framed as minimizing a **Mean-Variance Utility Function**, which quantifies an investor's preference for return versus their aversion to risk. The objective function to be minimized is:

$$f(w) = \frac{1}{2} w^T \Sigma w - \lambda w^T \mu$$

This function represents the investor's utility, where:
* $w$: A column vector of **portfolio weights**, where each element $w_i$ denotes the proportion of total investment allocated to asset $i$.
* $\Sigma$: The **covariance matrix** of asset returns. This symmetric matrix quantifies the pairwise relationships between the returns of different assets, where diagonal elements are variances and off-diagonal elements are covariances. It is a critical measure of **portfolio risk**.
* $\mu$: A column vector of **expected returns** for each asset. These are the anticipated average returns of the individual assets in the portfolio.
* $\lambda$: The **risk aversion factor** (a scalar, denoted as `risk_aversion` in the code). This parameter governs the investor's preference for return over risk. A higher positive $\lambda$ indicates greater aversion to risk, leading the optimizer to favor portfolios with lower variance.

The first term, $\frac{1}{2} w^T \Sigma w$, represents the **portfolio variance**, which is the mathematical measure of portfolio risk. The second term, $\lambda w^T \mu$, represents the **risk-adjusted expected portfolio return**. By minimizing this combined objective, the algorithm finds a set of weights that optimally balances these two components based on the specified risk aversion.

The `risk_return` function in the code directly translates this mathematical objective:

```python
def risk_return(w, cov, lam, mu):
    return (0.5 * np.dot(np.dot(w.T,cov), w)) - (lam * np.dot(w, mu.T))
```

### 1. Optimization Process and Constraints

The `optimize` function employs a numerical optimization solver to find the optimal portfolio weights. The process involves several mathematical and practical steps:

1.  **Data Pre-processing and Filtering:** Initial investment data is filtered to include only assets that show a positive gain/loss percentage. This is a practical heuristic to focus optimization on potentially profitable ventures.
2.  **Estimation of Inputs ($\Sigma$, $\mu$):**
    * **Historical Log Returns:** Daily logarithmic returns are calculated from historical closing prices for each asset using `yfinance`. Log returns are preferred in financial modeling for their additive properties and more consistent statistical behavior.
    * **Covariance Matrix ($\Sigma$):** The covariance matrix is then computed from these historical log returns. This symmetric matrix quantifies the pairwise relationships between the returns of different assets, where diagonal elements are variances and off-diagonal elements are covariances. It is a critical measure of **portfolio risk**.
    * **Expected Returns ($\mu$):** Expected returns for individual assets are calculated as the percentage change from `current price` to `expected price`. This serves as the $\mu$ vector for the optimization problem.
3.  **Numerical Minimization:** The `scipy.optimize.minimize` function is used to solve the constrained minimization problem.
    * **Solver:** The 'SLSQP' (Sequential Least SQuares Programming) method is chosen. SLSQP is an iterative algorithm designed for non-linear optimization problems with both equality and inequality constraints. It approximates the Hessian of the Lagrangian function using a quasi-Newton method and solves a sequence of quadratic programming subproblems.
    * **Bounds:** Individual asset weights $w_i$ are constrained to be between $0$ and $1$ ($0 \le w_i \le 1$). This imposes a "long-only" portfolio constraint (no short-selling) and prevents leverage, ensuring that all capital is allocated to existing assets and that no asset holds a negative proportion of the portfolio.
    * **Equality Constraint:** A fundamental equality constraint is applied: $\sum_{i=1}^N w_i = 1$. This ensures that the sum of all portfolio weights equals exactly one, meaning 100% of the available capital is allocated across the selected assets.
4.  **Weight Post-processing:** After optimization, very small positive weights (below `1e-7`) are effectively rounded down to zero. This is a numerical hygiene step to eliminate negligible allocations that might arise from floating-point inaccuracies, making the resulting portfolio more practical. The remaining non-zero weights are then re-normalized to ensure their sum still equals one, proportionally redistributing the capital from the dropped assets.

### 2. Input & Output Data

#### Input Data (`data` parameter)

The `optimize` function expects a `list` of `list`s, where each inner list represents a stock with specific attributes in a defined order:

* `[0]`: **Stock Ticker Symbol** (e.g., "AAPL", "MSFT") - `str`
* `[1]`: **Initial Investment Amount** in this stock (used to derive `total_investment`) - `float`
* `[2]`: **Current Price** of the stock - `float`
* `[3]`: **Expected Price** of the stock - `float`
* `[4]`: **Expected Return** (initial, before optimization) - `float` (Note: the `expected_returns` vector for the objective function is calculated from `[2]` and `[3]` internally)
* `[5]`: **Gain/Loss Percentage** (used for initial filtering of assets) - `float`

#### Output Data

The function returns a filtered `list` of `list`s which is displayed using the `tabulate` module. This output list is structured identically to the input `data`, but with one crucial modification: the `Initial Investment Amount` (`[1]`) for each stock is updated to reflect the **optimized allocation** derived from the MVO. Stocks that receive zero allocation after optimization or were initially filtered out due to losses are excluded from the returned list. If, after filtering, no profitable investments remain for optimization, the function returns `None`.

---

## 6. Build Instructions

Follow these steps to set up and run the project on Windows, macOS, or Linux.

### 1. Clone or Download

1. **Clone the repository**  
   ```bash
   git clone https://github.com/nitintonypaul/egarch-jdm-optimizer.git
   ```
2. **Or download ZIP**

- Click “Code” → “Download ZIP” on GitHub.
- Extract the ZIP archive to a folder of your choice.

3. **Change to the project directory**

```bash
cd egarch-jdm-optimizer
```

### 2. Install Dependencies
The project requires Python packages listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Make sure you’re using a compatible Python version (e.g., `Python 3.8+`).

### 3. Windows: Run Directly
On Windows, you can run the main script without building extensions manually.


**Run `main.py`**

```bash
python main.py --stock AAPL --investment 10000 # Example
```

Add any other supported arguments as needed.

### 4. macOS/Linux: Build & Run
On macOS or Linux, you need to compile the C++ extension before running.


**Build the extension**

```bash
python setup.py build_ext --build-lib build_modules
```

This command compiles the C++ modules (egarch and merton) into the `build_modules` folder.


**Run `main.py`**

```bash
python main.py --stock AAPL --investment 10000
```

### 5. Troubleshooting

If you encounter build errors on macOS/Linux, ensure you have a **C++ compiler** installed (e.g., **gcc** or **clang**).

On Windows, make sure you have the appropriate **Visual C++ build tools** if you ever need to rebuild extensions.

### 6. Sample input

You can utiliize this sample input to see how the program works
```bash
python main.py --stock AAPL --investment 1000 --stock NFLX --investment 2000 --stock NVDA --investment 1250 --risk 0.4
```

- `--stock` contains the ticker symbol of the stock
- `--investment` contains the investment amount for the stock
- `--risk`, an optional argument, contains the risk aversion you wish to feed into the optimization (defaults to 0.35)

**Note**: Risk aversion factors above 0.4 **may** lead to corner solutions for certain investment amounts. Try different values to find your optimal outcome.

---

## 7. License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute the code, provided that you include proper attribution and retain the original license.

> See [LICENSE](LICENSE) for full details.

---
