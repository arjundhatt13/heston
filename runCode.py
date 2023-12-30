from heston import *
from GBM import *
from plotting import *
from historicDataScraper import *
from riskMetrics import *
from scipy import *
from seaborn import *
from deltaHedging import *

# User Inputs
ticker = "AAPL"                     # Stock Ticker
T = 252                             # Forecast Horizon (Trading Days)
M = 10000                           # Number of Simulations

a = 0.95                            # Confidence Level

parameterStart='2022-12-01'         # Historic Data Scrape Start Date

model = "Heston"                       # GBM, Heston

# Delta Hedge Mode Writing Puts (Y/N)
deltaHedge = "Y"  
optSize = 10000                     # Number of Options (Number of Contracts * Contract Size)
r = 0.06                            # Risk-Free Rate
K = 260                             # Strike Price


# Scrapes Historic Data
parameterData, forecastData = getData(ticker, parameterStart, T, model)
# Simulation
if model == "GBM":
    # Estimate Model Parameters (Initial Guess)
    initialParameterGuess = initialGBMGuess(parameterData)

    # Calculate Model Parameters (Estimation)
    gbmParameters = (optimize.minimize(calculateGBMParameters, initialParameterGuess, args = parameterData, method = 'Nelder-Mead')).x
    
    # Run Simulation
    n, T, M, S0, St = gbmSimulation(gbmParameters, M, forecastData, T)

    # Calculate Risk Metrics
    aIndex, VaR, ES, sortedLoss = riskMetrics(a, T, M, S0, St)

    # Prints Model Output
    modelOutput(model, ticker, T, M, St, sortedLoss, a, VaR, ES, gbmParameters[0], gbmParameters[1], None)    # Model Output (Table)

    # Delta Hedging
    if deltaHedge == "Y":
        outflow = gbmDeltaHedge(St, T, optSize, r, gbmParameters[1], K)

elif model == "Heston":
    # Returns the mu parameter calculated using MLE
    initialParameterGuess = initialGBMGuess(parameterData)
    gbmParameters = (optimize.minimize(calculateGBMParameters, initialParameterGuess, args = parameterData, method = 'Nelder-Mead')).x

    # Estimate Model Parameters (Initial Guess)
    initialParameterGuess = initialHestonGuess(parameterData, gbmParameters)

    # Calculate Model Parameters (Estimation)
    hestonParameters = (optimize.minimize(calculateHestonParameters, initialParameterGuess[:5], args = (parameterData, initialParameterGuess[5]), method = 'Nelder-Mead')).x
    hestonParameters = np.append(hestonParameters, initialParameterGuess[5])

    # Run Simulation
    n, T, M, S0, St, v = hestonSimulation(T, M, hestonParameters, parameterData)

    # Calculate Risk Metrics
    aIndex, VaR, ES, sortedLoss = riskMetrics(a, T, M, S0, St)

    # Prints Model Output
    modelOutput(model, ticker, T, M, St, sortedLoss, a, VaR, ES, None, None, hestonParameters)

    # Delta Hedging
    if deltaHedge == "Y":
        outflow = hestonDeltaHedge(St, T, optSize, r, hestonParameters[0], K)

# Plots Data
mcPathsPlot(n, T, St, aIndex, a, VaR, ES, ticker, model, M)             # Monte Carlo Simulation Paths
lossDistPlot(a, M, VaR, ES, sortedLoss, ticker)                         # Loss Distribution
if deltaHedge == "Y":
    deltaHedgeDistPlot(outflow, a, ticker)                                  # Simulated Hedging Cost Distribution

plt.show()