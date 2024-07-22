# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:31:41 2024

@author: Daniel Roizman

"""

# ======================================================
# PART 1 - FINANCIAL DATA AND DAILY EV/EBITDA SERIES
# ------------------------------------------------------


import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

def TTM_ev_ebitda (tickers) :
    
    # Get TTM EBITDA data
    data = yf.download(tickers, period="3mo")['Adj Close']
    
    EV_EBITDA_series = {}
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    for tick in tickers:
        try: # tick = "AMZN"
            stock = yf.Ticker(tick)
    
            # Get first and second-last release dates
            second, first = pd.DataFrame(stock.earnings_dates).dropna(how='any').iloc[:2].index
            second, first = first.strftime("%Y-%m-%d"), second.strftime("%Y-%m-%d")
    
            # Get first and second-last LTM EBITDAs (conferir se 'bate' o primeiro e o segundo LTM ebitdas)
            ebitdas = stock.quarterly_income_stmt.loc['EBITDA']
            ebitda_series = pd.DataFrame(data=[sum(ebitdas.iloc[1:5]), sum(ebitdas.iloc[:4]), sum(ebitdas.iloc[:4])], 
                                         index = pd.to_datetime([first, second, today]))
    
            # Get mkt cap and net debt
            balance = stock.quarterly_balance_sheet.iloc[:,:2] # last two quarters
    
                # (short term debt)
            if 'Current Debt And Capital Lease Obligation' in balance.index:
                st_term_debt = balance.loc['Current Debt And Capital Lease Obligation']
            elif 'Current Debt' in balance.index:
                st_term_debt = balance.loc['Current Debt']
            elif 'Capital Lease Obligations' in balance.index: 
                st_term_debt = balance.loc['Capital Lease Obligations']
            
                # (long term debt)
            if 'Long Term Debt And Capital Lease Obligation' in balance.index:
                lg_term_debt = balance.loc['Long Term Debt And Capital Lease Obligation']
            elif 'Long Term Debt' in balance.index:
                lg_term_debt = balance.loc['Long Term Debt']
    
                # (cash and cash equivalents)
            if 'Cash Cash Equivalents And Short Term Investments' in balance.index:
                cash = balance.loc['Cash Cash Equivalents And Short Term Investments']
            elif 'Cash And Cash Equivalents' in balance.index:
                cash = balance.loc['Cash And Cash Equivalents']
            
            net_debt = lg_term_debt + st_term_debt - cash 
            net_debt_series = pd.DataFrame([net_debt[0], net_debt[1], net_debt[1]], index = pd.to_datetime([first, second, today]))
    
            market_cap = (data[tick] * stock.info['sharesOutstanding'])
     
            # Forward fill  EBITDA and net debt values to match the specified daily frequency
            ebitda_series = ebitda_series.resample('D').ffill() #.reindex(data.index, method = 'ffill')
            net_debt_series = net_debt_series.resample('D').ffill() #.reindex(data.index, method = 'ffill')
                    
            # Calculate, EV/EBITDA series only for last 3 months
            EV_EBITDA_series[tick] = ((market_cap + net_debt_series[0]) / ebitda_series[0])[
                data.index[0].strftime("%Y-%m-%d"):
                    data.index[len(data.index)-1].strftime("%Y-%m-%d")].dropna()

        except KeyError:
            tickers.remove(tick)
            continue
    
    return pd.DataFrame(EV_EBITDA_series)

# Example
tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO",
         "TSLA", "JPM", "WMT", "V", "XOM", "UNH", "MA", "ORCL", "PG", "COST",
         "JNJ", "HD", "BAC", "MRK", "ABBV", "AMD", "CVX", "NFLX", "KO", "ADBE",
         "CRM", "PEP", "QCOM", "WFC", "TMUS", "LIN", "TMO", "AMAT", "CSCO", "ACN",
         "MCD", "DHR", "TXN", "ABT", "GE", "DIS", "INTU", "AMGN", "VZ", "AXP"]

ev_ebitda_data = TTM_ev_ebitda(tickers)

    
# plot
plt.Figure(figsize=(10,6), dpi = 600)
plt.plot(ev_ebitda_data)

# Calculate the correlation matrix
ev_ebitda_corr = ev_ebitda_data.corr(numeric_only=False)

# Plot the heatmap of the correlations
plt.figure(figsize=(8, 6), dpi = 600)
sns.heatmap(ev_ebitda_corr, annot=False, vmin=-1, vmax=1)

# ======================================================
# PART 2 - COINTEGRATION TEST
# ------------------------------------------------------

from statsmodels.tsa.stattools import coint

# function to check possible candidates for cointegration test (choosing 50%+ correlated stocks only)
def get_correlated_stocks(data, correlation_threshold):

    correlation_matrix = data.corr(numeric_only=False)
    correlated_stocks = {}
    tickers = list(correlation_matrix.keys())
    while tickers: 
        x = tickers.pop()
        x_correlations = correlation_matrix[x]
        for y in tickers:
            if y != x and x_correlations[y] > correlation_threshold:
                if x in correlated_stocks.keys():
                    correlated_stocks[x].append(y)
                else:
                    correlated_stocks[x] = [y]
    return correlated_stocks

correlated_stocks = get_correlated_stocks(ev_ebitda_data, correlation_threshold = 0.9)
#correlated_stocks

# function to perform Johanssen cointegration test for every pair of highly pseudo-correlated stocks
def get_cointegrated_stocks(data): # how does the test work ?
    
    cointegrated_stocks = {}    
    correlated_stocks = get_correlated_stocks(data, 0.5)
    tickers = list(correlated_stocks.keys())
    while tickers:
        x = tickers.pop()
        x_data = data[x]
        for y in correlated_stocks[x]:
            y_data = data[y]
            
            # Perform Cointegration test and discard t_statistics more than 5%
            t_statistic, p_val, critical_p_val = coint(x_data,y_data)
            if t_statistic < critical_p_val[1]:
                if x in cointegrated_stocks.keys():
                    cointegrated_stocks[x].append(y)
                else:
                    cointegrated_stocks[x] = [y]
    
    pairs = []
    keys = list(cointegrated_stocks.keys())
    while keys:
        key = keys.pop()
        pairs += [(key, value) for value in cointegrated_stocks[key] if value not in keys]
    
    return pairs

cointegrated_stocks = get_cointegrated_stocks(ev_ebitda_data)
cointegrated_stocks

# ======================================================
# PART 3 - BUY/SELL SIGNALS
# ------------------------------------------------------

import numpy as np

# Calculate EV/EBITDA ratios
for x, y in cointegrated_stocks:
    ratio = ev_ebitda_data[x]/ev_ebitda_data[y]
    avg = np.average(ratio)
    stdev = np.std(ratio)
    
    # plot ev/ebitda ratios
    plt.plot(ratio)
    plt.axhline(avg, color = 'black')
    plt.axhline(avg + 2*stdev, color = 'grey')
    plt.axhline(avg - 2*stdev, color = 'grey')
    plt.title(f' {x}/{y} EV/EBITDA ratio')
    plt.show()
    
    if ratio[len(ratio) - 1] > avg + 2 * stdev:
        print(f'Short {x}, Buy {y}')
    elif ratio[len(ratio) - 1] < avg - 2 * stdev:
        print(f'Short {y}, Buy {x}')










