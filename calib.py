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

PARAM_DICT = {
    "vasicek": ("a", "b", "sigma"),
    "cir": ("a", "b", "sigma"),
    "dothan": ("a", "sigma"),
    "holee1": ("k", "sigma"),
    "holee2": ("k", "sigma"),
    "hw_v_ma": ("a", "sigma")
}

def read_sofr(ma=False, ma_w = 90):
    print("Reading SOFR...")
    sofr_data = pd.read_excel('SOFR.xlsx').iloc[::-1].reset_index()
    sofr_data.rename(columns = {"Effective Date":"DATE"}, inplace=True)
    # sofr_data.set_index("DATE", inplace=True)
    sofr_rates = sofr_data[['DATE', 'Rate (%)']].copy()
    # sofr_rates = sofr_data[['Rate (%)']]/100
    sofr_rates.loc[:,'Rate (%)'] = sofr_rates.loc[:,'Rate (%)']/100
    sofr_rates.loc[:,"dr"] = sofr_rates.loc[:,'Rate (%)'].diff().shift(-1)
    sofr_rates.rename(columns={'Rate (%)':"r"}, inplace=True)
    if ma:
        sofr_rates.loc[:,"r"] =sofr_rates.loc[:,"r"].rolling(ma_w).mean().dropna()
        sofr_rates.loc[:,"dr"] = sofr_rates.loc[:,"r"].diff().shift(-1)
    return sofr_rates.dropna()

def read_sofr_ffr(ma=False, W_ma=90):
    eps = 1e-6
    print("Reading SOFR and FFR...")
    sofr_data = pd.read_excel('SOFR.xlsx').iloc[::-1].reset_index()
    sofr_data.rename(columns = {"Effective Date":"DATE"}, inplace=True)
    # sofr_data.set_index("DATE", inplace=True)
    sofr_rates = sofr_data[['DATE', 'Rate (%)']].copy()
    # sofr_rates = sofr_data[['Rate (%)']]/100
    sofr_rates.loc[:,'Rate (%)'] = sofr_rates.loc[:,'Rate (%)']/100
    sofr_rates.rename(columns={'Rate (%)':"r"}, inplace=True)
    ffr = pd.read_csv("FEDFUNDS.csv")
    df = sofr_rates.merge(ffr, how = "outer", left_on="DATE", right_on="DATE")
    df.DATE = df.DATE.astype("datetime64")
    df = df.sort_values(by = "DATE")
    df = df.ffill(axis=0).dropna()
    df.loc[:,"FEDFUNDS"] = df.loc[:,"FEDFUNDS"]/ 100
    ffr = df.loc[:,"FEDFUNDS"]
    df.loc[:,"r-ffr"] = df.loc[:,"FEDFUNDS"] - df.loc[:,"r"] 
    df = df.loc[:,["r-ffr"]]
    df.rename(columns = {"r-ffr":"r"}, inplace=True)
    if ma:
        df.loc[:,"r"] = df.loc[:,"r"].rolling(W_ma).mean().dropna()
    df.loc[:,"dr"] = df.loc[:,"r"].diff().shift(-1)
    # calib.vasicek_regression_rol(df1, W=len(df1))
    df.loc[df.r == 0, "r"] = eps
    return df.dropna().reset_index(drop=True), ffr.values


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

def dothan_regression_rol(df, dt=1/252, W=100):
    rate_colname = 'r'
    change_colname = 'dr'
    logret = df.loc[:,change_colname] / df.loc[:,rate_colname]
    a = logret.rolling(W).mean() / dt
    sigma = logret.rolling(W).std() / np.sqrt(dt)
    params = pd.DataFrame()
    params['a'] = a
    params['sigma'] = sigma
    return params.dropna()

def holee_1_regression_rol(df, dt=1/252, W=100):
    rate_colname = 'r'
    change_colname = "dr"
    X = df.loc[:, [rate_colname]]
    y = df.loc[:, change_colname]   
    rols = RollingOLS(y, X, window=W)
    rres = rols.fit()
    params = rres.params.copy()
    params.loc[:,"k"] = -params[rate_colname] / dt
    params.loc[:,"sigma"] = np.sqrt(rres.mse_resid / dt)
    return params.loc[:,["k","sigma"]].dropna()

def holee_2_regression_rol(df, dt=1/252, W=100, W_ma = 10):
    rate_colname = 'r'
    change_colname = "dr"
    rate_ma_colname = "rma"
    df[rate_ma_colname] = df.loc[:,rate_colname].rolling(W_ma).mean()
    df = df.dropna().reset_index(drop=True)
    X = df.loc[:, [rate_ma_colname]]
    y = df.loc[:, change_colname]   
    rols = RollingOLS(y, X, window=W)
    rres = rols.fit()
    params = rres.params.copy()
    params.loc[:,"k"] = -params[rate_ma_colname] / dt
    params.loc[:,"sigma"] = np.sqrt(rres.mse_resid / dt)
    return params.loc[:,["k","sigma"]].dropna()

def hw_vasicek_ma_regression_rol(df, dt=1/252, W=100, W_ma=30):
    rate_colname = 'r'
    change_colname = "dr"
    rate_ma_colname = "rma"
    df[rate_ma_colname] = df.loc[:,rate_colname].rolling(W_ma).mean()
    df = df.dropna().reset_index(drop=True)
    print(df)
    X = df.loc[:,[rate_colname, rate_ma_colname]]
    y = df.loc[:, change_colname]   
    rols = RollingOLS(y, X, window=W)
    rres = rols.fit()
    params = rres.params.copy()
    params.loc[:,"a"] = -params[rate_colname] / dt
    params.loc[:,"sigma"] = np.sqrt(rres.mse_resid / dt)
    return params.loc[:,["a","sigma"]].dropna()







def calibrate(df, dt=1/252, W=200, model="vasicek"):
    if model == "vasicek":
        return vasicek_regression_rol(df, dt, W)
    elif model == "cir":
        return cir_regression_rol(df, dt, W)
    elif model == "dothan":
        return dothan_regression_rol(df, dt, W)
    elif model == "holee1":
        return holee_1_regression_rol(df, dt, W)
    elif model == "holee2":
        return holee_2_regression_rol(df, dt, W)
    elif model == "hw_v_ma":
        return hw_vasicek_ma_regression_rol(df, dt, W)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    sigma_s = calibrate_quanto()
    print(sigma_s)