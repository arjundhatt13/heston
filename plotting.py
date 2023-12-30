# Import modules
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
from datetime import datetime as dt
import datetime as dt1
import pandas_market_calendars as mcal
import astropy as ast
from astropy.modeling import models, fitting
import seaborn as sns
import matplotlib.patches as mpatches

# Plot the simulation paths
def mcPathsPlot(n, T, St, aIndex, a, VaR, ES, ticker, model, M):

    tempList=St.transpose().tolist()
    tempList.sort(key=lambda ele: (ele[n]), reverse=True)

    pathsWithinLevel = np.asarray(tempList[:aIndex]).T
    pathsBreachedLevel = np.asarray(tempList[aIndex:]).T

    # Confidence Interval
    upperCI = np.mean(np.asarray(tempList).T, axis=1) + np.std(np.asarray(tempList).T, axis=1) * norm.ppf(a)
    lowerCI = np.mean(np.asarray(tempList).T, axis=1) - np.std(np.asarray(tempList).T, axis=1) * norm.ppf(a)

    # Used for the x-axis of the plot
    time = np.linspace(0,T,n+1)

    # Plots
    plt.plot(time, pathsWithinLevel, color='green', label = "Non-Breaching Paths")
    plt.plot(time, pathsBreachedLevel, color='red', label = "VaR Breaching Paths")
    plt.plot(time, upperCI, color='black', label = "Non-Breaching Paths")
    plt.plot(time, lowerCI, color='black', label = "Non-Breaching Paths")

    # Axis Labels
    plt.xlabel("Trading Days $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title(r"$\bf{" + ticker + "\ Stock\ Price\ Paths\ for\ " + model + "\ Simulation" + "}$" + "\n Median Price = " + '\${:,.2f}'.format(np.median(St[-1, :])) + ",  $\mathregular{VaR_{" + str(a) + "}}$ =" + "\${:,.2f}".format(VaR) + ",  $\mathregular{ES_{" + str(a) + "}}$ =" + "\${:,.2f}".format(ES))
    
    # Legend
    green_patch = mpatches.Patch(color = 'green', label = "Non-Breaching Paths")
    red_patch = mpatches.Patch(color = 'red', label = "VaR Breaching Paths")
    black_patch = mpatches.Patch(color = 'black', label = str(a*100) + "% Confidence Interval")

    plt.legend(handles = [green_patch, red_patch, black_patch])

    plt.show()

def lossDistPlot(a, M, VaR, ES, sortedLoss, ticker):
    # Histogram Bins
    ax = sns.distplot(sortedLoss)
    
    data_x, data_y = ax.lines[0].get_data()

    # Plotting
    plt.fill_between(sortedLoss, np.interp(sortedLoss, data_x, data_y), where = sortedLoss >= VaR, color='red', alpha=0.8)
    
    #Labeling
    plt.xlabel("Loss ($)")
    plt.ylabel("Relative Frequency (%)")
    plt.title(r"$\bf{" + ticker + "\ Loss\ Distribution" + "}$" + "\n Median Loss = " + '\${:,.2f}'.format(np.median(-sortedLoss)) + ",  $\mathregular{VaR_{" + str(a) + "}}$ =" + "\${:,.2f}".format(VaR) + ",  $\mathregular{ES_{" + str(a) + "}}$ =" + "\${:,.2f}".format(ES))
    plt.show()

# Model output in tabular form
def modelOutput(model, ticker, T, M, St, sortedLoss, a, VaR, ES, mu, sigma, hestonParameters):
    fig = plt.figure(figsize = (6,4)) 
    ax = fig.add_subplot(111)

    if model == "GBM":
        df = pd.DataFrame(["",
                        model, 
                        ticker, 
                        T,
                        M,
                        "",
                        "", 
                        '\${:,.2f}'.format(np.median(St[-1, :])), 
                        '\${:,.2f}'.format(np.median(-sortedLoss)),
                        "",
                        "",  
                        "\${:,.2f}".format(VaR), 
                        "\${:,.2f}".format(ES), 
                        "",
                        "",
                        "{:.2f}%".format(mu*252*100), 
                        "{:.2F}%".format(sigma*np.sqrt(252)*100)
                        ],

        index=["Model Information (User Inputted)", 
            "       Model Used", 
            "       Stock",
            "       Time Horizon (T) in Trading Days",
            "       Number of Simulations (M)",
            "", 
            "Projected Price Information",  
            "       Projected Price", 
            "       Projected P&L",
            "",
            "Risk Metrics", 
            "       $\mathregular{VaR_{" + str(a) + "}}$", 
            "       $\mathregular{ES_{" + str(a) + "}}$",
            "",
            "Model Parameters", 
            "       μ (Annualized)",
            "       σ (Annualized)" 
                    ])
    
    elif model == "Heston":
        df = pd.DataFrame(["",
                        model, 
                        ticker, 
                        T,
                        M,
                        "",
                        "", 
                        '\${:,.2f}'.format(np.median(St[-1, :])), 
                        '\${:,.2f}'.format(np.median(-sortedLoss)),
                        "",
                        "",  
                        "\${:,.2f}".format(VaR), 
                        "\${:,.2f}".format(ES), 
                        "",
                        "",
                        "{:.2f}%".format(hestonParameters[5]*252*100), 
                        "{:.2f}%".format(hestonParameters[0]*np.sqrt(252)*10000),
                        "{:.2f}%".format(hestonParameters[1]*100),
                        "{:.2e}%".format(hestonParameters[2]),
                        "{:.2f}%".format(hestonParameters[3]*100),
                        "{:.2f}%".format(hestonParameters[4]*np.sqrt(252)*10000),
                        ],

        index=["Model Information (User Inputted)", 
            "       Model Used", 
            "       Stock",
            "       Time Horizon (T) in Trading Days",
            "       Number of Simulations (M)", 
            "", 
            "Projected Price Information",  
            "       Projected Price", 
            "       Projected P&L",
            "",
            "Risk Metrics", 
            "       $\mathregular{VaR_{" + str(a) + "}}$", 
            "       $\mathregular{ES_{" + str(a) + "}}$",
            "",
            "Model Parameters", 
            "       μ (Annualized)",
            "       θ (Annualized Long-Run Volatility)",
            "       κ (Rate of Reversion to θ)",
            "       ξ (Daily Volatility of Volatility)",
            "       ρ (Correlation Between Wiener Processes)",
            "       $\mathregular{v_{" + str(0) + "}}$ Starting Variance",

                    ])


    table = ax.table(cellText=df.values, rowLabels=df.index, loc = "upper center", colWidths=[0.3]*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    
    if model == "Heston":
        ax.set_title("Heston Diffusion Model Output", loc='left')
    else:
        ax.set_title("GBM Model Output", loc='left')
       
    ax.axis("off")
    fig.tight_layout()
    plt.show()

def deltaHedgeDistPlot(outflow, a, ticker):
    # Histogram Bins
    fig, ax = plt.subplots()
    sns.distplot(outflow)
        
    # Confidence Interval
    upperCI = np.mean(outflow) + np.std(outflow) * norm.ppf(a)
    lowerCI = np.mean(outflow) - np.std(outflow) * norm.ppf(a)

    #Labeling
    plt.xlabel("Loss ($)")
    plt.ylabel("Relative Frequency (%)")
    plt.title(r"$\bf{" + ticker + "\ Discounted\ Simulated\ Hedging\ Cost" + "}$" + "\n Average Hedging Cost = " + '\${:,.2f},  '.format(np.mean(outflow)) + str(a*100) + "% CI = $" + '({:,.2f}, {:,.2f})'.format(lowerCI, upperCI))
    plt.show()
