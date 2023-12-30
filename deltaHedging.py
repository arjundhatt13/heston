import numpy as np
from scipy.stats import norm

def gbmDeltaHedge(St, T, optSize, r, sigma, K):
    i_values = np.arange(0, St.shape[0])
    
    # Annualize Volatility & Time to Maturity
    sigma*=np.sqrt(252)
    T/=252

    # Calculate d1 & normdist
    d1 = (np.log(St[:-1] / K) + (r + sigma**2/2) * (T - i_values[:-1, np.newaxis] / 252)) / (sigma * np.sqrt(T - i_values[:-1, np.newaxis] / 252))
    
    d1 = np.vstack([d1, np.zeros_like(d1[0, :])])    
    normdist = norm.cdf(d1, loc=0, scale=1) - 1
    normdist[-1, :] = np.where(St[-1, :] > K, 0, -1)

    # Calculate cost of shares purchased
    cost_of_shares_purchased = np.diff(normdist, axis=0, prepend=0) * optSize * St

    # Initialize arrays for cumulative and interest
    cumulative = np.zeros_like(St)
    interest = np.zeros_like(St)

    # Calculate cumulative and interest
    cumulative[0, :] = cost_of_shares_purchased[0, :]
    interest[0, :] = cost_of_shares_purchased[0, :] * (1/252) * r

    for i in range(1, cost_of_shares_purchased.shape[0]):
        cumulative[i, :] = cost_of_shares_purchased[i, :] + cumulative[i-1, :] + interest[i-1, :]
        interest[i, :] = cumulative[i, :] * (1/252) * r

    # Calculate Outflow After Execution Point
    outflow = np.zeros_like(St[-1, :])

    # Find the indices where St[-1, :] is less than K
    indices_below_K = St[-1, :] < K

    # Calculate outflow based on the condition
    outflow[indices_below_K] = cumulative[-1, indices_below_K] + optSize * K
    outflow[~indices_below_K] = cumulative[-1, ~indices_below_K]

    return outflow

def hestonDeltaHedge(St, T, optSize, r, theta, K):
    i_values = np.arange(0, St.shape[0])
    
    # Annualize Volatility & Time to Maturity
    theta*=np.sqrt(252)*100
    T/=252

    # Calculate d1 & normdist
    d1 = (np.log(St[:-1] / K) + (r + theta**2/2) * (T - i_values[:-1, np.newaxis] / 252)) / (theta * np.sqrt(T - i_values[:-1, np.newaxis] / 252))
    
    d1 = np.vstack([d1, np.zeros_like(d1[0, :])])

    normdist = norm.cdf(d1, loc=0, scale=1) - 1
    normdist[-1, :] = np.where(St[-1, :] > K, 0, -1)


    # Calculate cost of shares purchased
    cost_of_shares_purchased = np.diff(normdist, axis=0, prepend=0) * optSize * St

    # Initialize arrays for cumulative and interest
    cumulative = np.zeros_like(St)
    interest = np.zeros_like(St)

    # Calculate cumulative and interest
    cumulative[0, :] = cost_of_shares_purchased[0, :]
    interest[0, :] = cost_of_shares_purchased[0, :] * (1/252) * r

    for i in range(1, cost_of_shares_purchased.shape[0]):
        cumulative[i, :] = cost_of_shares_purchased[i, :] + cumulative[i-1, :] + interest[i-1, :]
        interest[i, :] = cumulative[i, :] * (1/252) * r

    # Calculate Outflow After Execution Point
    outflow = np.zeros_like(St[-1, :])

    # Find the indices where St[-1, :] is less than K
    indices_below_K = St[-1, :] < K

    # Calculate outflow based on the condition
    outflow[indices_below_K] = cumulative[-1, indices_below_K] + optSize * K
    outflow[~indices_below_K] = cumulative[-1, ~indices_below_K]

    return outflow

