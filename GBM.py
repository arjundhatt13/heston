# Import modules
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime as dt1
import pandas_market_calendars as mcal
import astropy as ast
from astropy.modeling import models, fitting

# Uses parameter data to return historical drift and volatility coefficients for GBM
def initialGBMGuess(parameterData):
    
    # Stores the daily log returns
    logReturns = (np.log(parameterData.pct_change()+1))

    sigma=(logReturns.std()).iloc[0] 
    mu=(logReturns.mean()).iloc[0]

    return [mu, sigma]

# Estimates the parameters using maximum likelihood estimation (MLE)
def calculateGBMParameters(parameterList, parameterData):
    logLikelihood=[]
    logReturns = (np.log(parameterData.pct_change()+1))
    residuals = logReturns - parameterList[0]

    for i in range(1,len(parameterData)):        
        # Joint PDF of asset prices and volatility (Log-Likelihood)
        jointPdf = 1/(np.sqrt(2*np.pi)*parameterList[1])*np.exp(-1/2*((residuals.iloc[i][0]/parameterList[1])**2))

        # Penalize for parameter sets that give errors (ln(negative))
        if jointPdf <= 0:
            logLikelihood.append(-1000)

        else:
            logLikelihood.append(np.log(jointPdf))

    return -sum(logLikelihood) # Maximizing logLikelihood is the same as minimizing -logLikelihood

# Conducts the Monte-Carlo Simulation 
def gbmSimulation(parameterList, M, forecastData, forecastDays):
    
    mu = parameterList[0]
    sigma = parameterList[1]

    # Sets initial stock adjusted price 
    S0=forecastData['Adj Close'].iloc[0]

    # Number of Steps
    n = forecastDays

    # Daily discretized time step
    T = n    

    # Calculates each time step
    dt = T/n

    # GBM model with normally distributed Wiener Process. Simulation is done using Arrays, which is more efficient than using a loop. 
    St = np.exp((mu* dt) + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
    St = np.vstack([np.ones(M), St])


    # This will calculate the cumulative return at each time step, for each simulation. Multiplying through by S0 will change the St matrix from returns to stock prices.
    St = St.cumprod(axis=0)
    St = S0*St

    a=St[-1,:]-St[-2,:]
    
    return n, T, M, S0, St

