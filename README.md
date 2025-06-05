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

- The user inputs a selection of stocks and the amount they want to invest.
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

## 4. Merton Jump Diffusion Model

### Model Equation

The price dynamics under Merton’s Jump Diffusion are given by:

$$
S_t = S_0 \exp\Bigl((\mu - 0.5\sigma^2 - k\lambda)t + \sigma W_t + \sum_{i=1}^{N(t)} J_i\Bigr)
$$

- $$S_t$$: Asset price at time $$t$$.
- $$S_0$$: Initial (current) asset price.
- $$\mu$$: Continuous drift term (expected return rate).
- $$\sigma$$: Diffusion volatility (standard deviation of continuous returns).
- $$W_t$$: Wiener process ($$W_t \sim \mathcal{N}(0,t)$$) capturing Brownian motion.
- $$t$$: Time horizon in trading years (e.g., $$t=1/252$$ for one trading day).
- $$\lambda$$: Jump intensity (expected number of jumps per unit time).
- $$k$$: Mean jump size, defined as $$k = mean(ln(1 + J))$$ where $$J$$ are proportional jump magnitudes.
  
- $$N(t)$$: Poisson process counting the number of jumps by time $$t$$, with $$N(t)\sim \mathrm{Poisson}(\lambda t)$$.
- $$J_i$$: Log-jump magnitudes, typically drawn from

$$
ln(J_i) \sim \mathcal{N}\bigl(\ln(1 + k) - 0.5\sigma_j^2,\sigma_j^2\bigr)
$$
  
  where $$\sigma_j$$ is jump volatility (std of $$\ln(1+J_i)$$).

### Variable Interpretations

1. **Drift Adjustment $$(\mu - 0.5\sigma^2 - k\lambda)$$**  
   - $$\mu$$ is reduced by $$0.5\sigma^2$$ (Itô correction) and by $$k\lambda$$ to maintain unbiased growth when jumps occur.

2. **Diffusion Term $$\sigma W_t$$**  
   - Captures continuous, normally distributed fluctuations around the adjusted drift.

3. **Jump Component $$\sum_{i=1}^{N(t)} J_i$$**  
   - Adds discrete, log-normally distributed shocks whenever a jump event occurs.  
   - Each $$J_i$$ shifts returns by $$\exp(J_i)$$, simulating rare, impactful moves.

### Monte Carlo Simulation

To approximate the expected terminal price $$S_t$$ under the Merton JDM, we perform $$M=200,000$$ independent simulations and accumulate results in a single double accumulator (no arrays).

#### 1. Per-Simulation Step
For each simulation $$i=1,...,M$$:

##### 1.1 Jumps
- Sample $$N \sim Poisson(\lambda t)$$
- If $$N \gt 0$$, draw $$ln(J_j)$$:

$$
ln(J_j) \sim N(ln(1+k) - 0.5\sigma_j², \sigma_j²)
$$

for $$j=1,...,N$$
- Compute total jump sum:

$$
J_{sum} = \sum_{j=1}^{N} J_j
$$

##### 1.2 Brownian Increment
Draw $$Z_i \sim N(0,1)$$ and set $$W_t = \sqrt{t Z_i}$$

##### 1.3 Compute Terminal Price

$$
S_t = S_0 exp((\mu - 0.5\sigma² - k\lambda)t + \sigma W_t + J_{sum})
$$

<p align="center">
  OR
</p>

$$
S_t = S_0 exp((\bar{r} - \sigma² - k\lambda)t + \sigma W_t + J_{sum})
$$

##### 1.4 Accumulate
- Add $$S_t$$ to a running sum:

  ```python
  price_sum += S_t
  ```

#### 2. Estimate Expected Price
After all $$M$$ simulations, compute `expected_price`:

```python
expected_price = price_sum / M
```

This single double variable yields the average terminal price without storing individual paths.

This streamlined approach directly implements the closed-form update for each path and then divides the accumulated sum by 200,000 to obtain the expected price. No intermediate arrays or multi-step loops are used—each simulation computes one $$S_t$$ and immediately contributes to the average.

These simulated terminal prices feed into portfolio valuation, allowing for more accurate risk assessment by capturing both continuous volatility and discrete jumps. The sample average estimates the expected terminal price under jump diffusion. These simulated prices then feed into portfolio valuation and subsequent optimization.

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

---

## 7. License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute the code, provided that you include proper attribution and retain the original license.

> See [LICENSE](LICENSE) for full details.

---
