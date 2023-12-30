import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime as dt1
import pandas_market_calendars as mcal
import scipy as scipy
from scipy import optimize
import astropy as ast
from astropy.modeling import models, fitting

# Uses the historical data to calculate an initial guess used in the estimation
def initialHestonGuess(parameterData, gbmParameters):
    
    # mu = daily logReturns 
    logReturns = (np.log(parameterData.pct_change()+1))
    mu=gbmParameters[0]        # Drift
    
    # Residuals & Realized Variance Calculation
    residuals = logReturns - mu
    realizedVariance = residuals ** 2 

    # Heston Volatility Parameters
    theta = residuals.var().iloc[0]         # Long Variance
    kappa = 1                               # Mean Reversion
    sigma = realizedVariance.std().iloc[0]  # Volatility of Volatility
    rho = 0                                 # Correlation between Wiener Processes
    v0 = theta                              # Starting Variance
    
    return [theta, kappa, sigma, rho, v0, mu]

# Estimates the parameters using maximum likelihood estimation (MLE)
def calculateHestonParameters(parameterList, parameterData, mu):
    logReturns = (np.log(parameterData.pct_change()+1))
    residuals = logReturns - mu
    realizedVariance = residuals ** 2 

    expectedVariance = np.array([parameterList[4]])
    realized_variances = realizedVariance.iloc[1:-1, 0].values
    expectedVariance = np.append(expectedVariance, realized_variances + parameterList[1] * (parameterList[0] - realized_variances))

    logLikelihood=[]
    for i in range(1,len(parameterData)):        
        
        # Provides "safe" values to avoid errors
        safe_sqrt_expected_variance = np.sqrt(expectedVariance[i-1]) if expectedVariance[i-1] >= 0 else -1/1000
        safe_rho = 0.9999 if parameterList[3] == 1 else parameterList[3]

        # Joint PDF of asset prices and volatility (Log-Likelihood)
        jointPdf = ((1 / ( 2 * np.pi * safe_sqrt_expected_variance * parameterList[2] * np.sqrt(1-safe_rho**2))) * np.e ** (-(((residuals.iloc[i][0] / safe_sqrt_expected_variance) ** 2 - 2 * parameterList[3] * (residuals.iloc[i][0] / safe_sqrt_expected_variance) * ((realizedVariance.iloc[i][0] - expectedVariance[i-1]) / parameterList[2]) + ((realizedVariance.iloc[i][0] - expectedVariance[i-1]) / parameterList[2]) ** 2 ) / (2 * (1- safe_rho**2)))))
        
        # Penalize for parameter sets that give errors (ln(negative))
        if jointPdf <= 0:
            logLikelihood.append(-1000)

        else:
            logLikelihood.append(np.log(jointPdf))
    return -sum(logLikelihood) # Maximizing logLikelihood is the same as minimizing -logLikelihood

# Conducts the Monte-Carlo Simulation 
def hestonSimulation(forecastDays, M, hestonParameters, parameterData):
    
    # Parameter Definitions
    theta = hestonParameters[0]       # Long Variance
    kappa = hestonParameters[1]       # Mean Reversion
    sigma = hestonParameters[2]       # Volatility of Volatility
    rho = hestonParameters[3]         # Correlation between Wiener Processes
    v0 = hestonParameters[4]          # Starting Variance
    mu = hestonParameters[5]          # Drift

    # Sets initial stock adjusted price 
    S0 = (parameterData.iloc[-1,0]).astype(float)
    
    # Number of Steps
    n = forecastDays

    # Daily discretized time step
    T = n    

    # Calculates each time step
    dt = T/n

    # Arrays for storing prices and variances
    St = np.full(shape=(n+1,M), fill_value=S0)
    v = np.full(shape=(n+1,M), fill_value=v0)

    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(np.array([0,0]), np.array([[1,rho],[rho,1]]), (n,M))

    for i in range(1,n+1):
        St[i] = St[i-1] * np.exp((mu) * dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0])
        v[i] = np.maximum(v[i-1] + kappa * (theta - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * Z[i-1,:,1], 0)
        
    return n, T, M, S0, St, v



