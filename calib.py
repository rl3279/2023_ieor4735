import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.stats import norm
from tabulate import tabulate
import requests
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import scipy.optimize as optimize
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def vasicek_regression(rates, changes):
    b = rates.mean()
    X = -1 * (rates[:-1] - b).values.reshape(-1, 1)
    y = changes.values

    model = LinearRegression()
    model.fit(X, y)
    a = -model.coef_[0]

    # Estimate sigma
    residuals = y - model.predict(X)
    sigma = np.std(residuals)
        
    return a, sigma, b

def vasicek_regression_rol(df, dt=1/252, W=100):
    rate_colname = 'Rate (%)'
    X = df.loc[:, [rate_colname]]
    y = df.loc[:, "changes"]
    X = sm.add_constant(X)
    rols = RollingOLS(y, X, window=W)
    rres = rols.fit()
    params = rres.params.copy()
    params.loc[:,"b"] = params["const"] / dt
    params.loc[:,"a"] = -params[rate_colname] / dt
    params.loc[:,"sigma"] = np.sqrt(rres.mse_resid / dt)
    return params.loc[:,["a","b","sigma"]].dropna()

def read_sofr():
    filename = 'SOFR.xlsx'
    rate_colname = 'Rate (%)'
    sofr_data = pd.read_excel(filename)
    sofr_rates = sofr_data[[rate_colname]]/100
    sofr_rates.loc[:,"changes"] = sofr_rates.loc[:,rate_colname].diff()
    return sofr_rates.dropna()

if __name__ == "__main__":
    sofr = read_sofr()
    filename = 'SOFR.xlsx'
    rate_colname = 'Rate (%)'
    sofr_ma = sofr.rolling(90).mean().dropna()
    sofr_ma.changes = sofr_ma.loc[:,rate_colname].diff()
    sofr_ma.dropna(inplace=True )
    result = vasicek_regression_rol(sofr_ma, W=100)
    print(result)