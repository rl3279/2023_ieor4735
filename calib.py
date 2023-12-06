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

def read_sofr(ma=False, ma_w = 90):
    print("Reading SOFR...")
    sofr_data = pd.read_excel('SOFR.xlsx')
    sofr_rates = sofr_data[['Rate (%)']]/100
    sofr_rates.loc[:,"dr"] = sofr_rates.loc[:,'Rate (%)'].diff()
    sofr_rates.rename(columns={'Rate (%)':"r"}, inplace=True)
    if ma:
        sofr_rates = sofr_rates.rolling(ma_w).mean().dropna()
        sofr_rates.loc[:,"dr"] = sofr_rates.loc[:,"r"].diff()
    return sofr_rates.dropna()

def read_stock():
    print("Downloading euro stock and exchange rate...")
    stock_data = yf.download('^STOXX50E','2018-11-01','2023-11-01')
    exchange_rate = yf.download('EURUSD=X','2018-11-01','2023-11-01')
    return stock_data, exchange_rate

def calibrate_quanto():
    stock_data, exchange_rate = read_stock()
    delta_f = stock_data["Close"].apply(np.log).diff().std() * np.sqrt(252)
    delta_x = exchange_rate["Close"].apply(np.log).diff().std() * np.sqrt(252)
    aligned_data = pd.merge(
        stock_data['Close'], exchange_rate['Close'], 
        how="inner", left_index=True, right_index=True)
    logret = aligned_data.apply(np.log).diff()

    rho_fx = logret.corr().values[0,1]
    sigma_s = (delta_f**2 + delta_x**2 + 2 * delta_f * delta_x * rho_fx)**0.5
    return sigma_s

def vasicek_regression_rol(df, dt=1/252, W=100):
    rate_colname = 'r'
    change_colname = "dr"
    X = df.loc[:, [rate_colname]]
    y = df.loc[:, change_colname]
    X = sm.add_constant(X)
    rols = RollingOLS(y, X, window=W)
    rres = rols.fit()
    params = rres.params.copy()
    params.loc[:,"b"] = params["const"] / dt
    params.loc[:,"a"] = -params[rate_colname] / dt
    params.loc[:,"sigma"] = np.sqrt(rres.mse_resid / dt)
    return params.loc[:,["a","b","sigma"]].dropna()

def cir_regression_rol(df, dt=1/252, W=100):
    rate_colname = 'r'
    change_colname = "dr"
    b1 = "sqrt(r)"
    b2 = "1/sqrt(r)"
    X = df.copy()
    X.loc[:,b1] = df.loc[:,rate_colname]**(1/2)
    X.loc[:,b2] = 1 / X.loc[:,b1]
    y = X.loc[:, change_colname] / X.loc[:,b2]
    X = X.loc[:,[b1, b2]]
    rols = RollingOLS(y, X, window=W)
    rres = rols.fit()
    params = rres.params.copy()
    params.loc[:,"a"] = -params[b1] / dt
    params.loc[:,"b"] = params[b2] /params.loc[:,"a"]/ dt
    params.loc[:,"sigma"] = np.sqrt(rres.mse_resid / dt)
    return params.loc[:,["a","b","sigma"]].dropna()

def calibrate(df, dt=1/252, W=200, model="vasicek"):
    if model == "vasicek":
        return vasicek_regression_rol(df, dt, W)
    elif model == "cir":
        return cir_regression_rol(df, dt, W)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    sigma_s = calibrate_quanto()
    print(sigma_s)