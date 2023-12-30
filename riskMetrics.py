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

# VaR and ES Calculations
def riskMetrics(a, T, M, S0, St):   
    aIndex = int(np.ceil(M*a))-1
    
    sortedPrice = sorted([item for sublist in St[T:] for item in sublist], reverse=True)
    sortedLoss = -np.vectorize(lambda i: i-S0)(sortedPrice)

    VaR = sortedLoss[aIndex]
    ES = np.mean(sortedLoss[aIndex:M])

    return aIndex, VaR, ES, sortedLoss
