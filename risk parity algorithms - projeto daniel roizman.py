# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 19:28:08 2023

@author: Daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyfeng as pf
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import time

# ________________________________________________________________
# reading CSV data 
# ________________________________________________________________

'''
# run the the following code if you don't have the csv file in your compute... 
# ________________________________________________________________
# function to retreive sp500 stock data and store into csv file
# ________________________________________________________________

# in case you can't simply import the packages...
# pip install requests beautifulsoup4 pandas yfinance

# Step 1: Get a list of S&P 500 companies
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
html_content = requests.get(url).text
soup = BeautifulSoup(html_content, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
table_rows = table.find_all('tr')[1:]

sp500_symbols = []
for row in table_rows:
    symbol = row.find_all('td')[0].text.strip()
    sp500_symbols.append(symbol)

# Step 2: Retrieve historical stock data for each company
data = pd.DataFrame()
start_date = '2018-06-08'
end_date = '2023-06-08'

for symbol in sp500_symbols:
    try:
        stock = yf.download(symbol, start=start_date, end=end_date)
        if not stock.empty:
            stock['Symbol'] = symbol  # Add a 'Symbol' column to identify the company
            data = pd.concat([data, stock])
            print(f'Retrieved data for {symbol}')
    except:
        print(f'Error retrieving data for {symbol}')

# Step 3: Save the data to a single CSV file
data.to_csv('sp500_data.csv')

'''

data = pd.read_csv(r'C:\Users\Daniel\OneDrive\Área de Trabalho\DRV\IMPA\algebra linear e aplicacoes - 2023.1\sp500_data.csv') # o 'r' antes do caminho do arquivo possibiltou a leitura

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# separating columns and computing returns
data_pivot = data.pivot(index='Date', columns='Symbol', values='Adj Close')
sp_returns = data_pivot.pct_change()
sp_returns.drop(sp_returns.index[0], inplace=True)
sp_returns_clean = sp_returns.dropna(axis=1) # remove columns with 'NaN's
sp_returns_clean = sp_returns_clean.drop('GOOGL',axis=1) # removing because there is GOOG and GOOGL
sp_names = [i for i in sp_returns_clean.columns]
len(sp_names) # 490 remaining...

# ________________________________________________________________
# COVARIANCE MATRICES
# ________________________________________________________________

# obtaining (sample) covariance matrices and its eigenvalues
covar_490 = np.array(sp_returns_clean.cov())
eigval_490, eigvec_490 = np.linalg.eig(covar_490)
names_eigvals = np.column_stack((sp_names,eigval_490))

# sorted 2xN matrix with names and corresponding eigenvalues of each asset
sorted_names_eigvals = names_eigvals[names_eigvals[:, 1].argsort()]
sort_names, sort_eigs = sorted_names_eigvals[:,0], sorted_names_eigvals[:,1]

# plotando os autovalores
plt.bar(range(len(eigval_490)), np.sort(eigval_490), width=0.45)
plt.xlabel('Stocks')
plt.ylabel('Eigenvalues')
plt.yticks([])  # Hide y-axis ticks
plt.ylim(0,.012)
plt.show()

# testing positive definiteness
np.all(eigval_490> 0) # True

# condition numbers
n_cond_490 = np.linalg.cond(covar_490)

# ________________________________________________________________
# NAIVE ALGORITHM TO SELECT SUBMATRICES AND GET COVARIANCE (SUB)MATRICES
# ________________________________________________________________

def posto_efetivo(M, tol):
    sigma = np.linalg.svd(M)[1]
    cont = 0
    n = np.shape(M)[1]
    for i in sigma:
        if i < tol: 
            cont += 1
    posto_efetivo = n - cont
    return posto_efetivo

def keep_p_eigval(covar, p, index_names):
    eigvals, eigvecs = np.linalg.eig(covar)
    names_eigvals = np.column_stack((index_names, eigvals))
    sorted_names_eigvals = names_eigvals[names_eigvals[:, 1].argsort()[::-1]]
    sort_names = sorted_names_eigvals[:, 0]
    sort_eigs = sorted_names_eigvals[:, 1]
    return sort_names[:p], sort_eigs[:p]


# calculando os postos efetivos da matriz principal, com diferentes margens de erro
p1 = posto_efetivo(covar_490, np.exp(-10))
p2 = posto_efetivo(covar_490, np.exp(-9))
p3 = posto_efetivo(covar_490, np.exp(-8))

# obtendo as listas com os ativos associados aos p maiores autovalores
names_1 = keep_p_eigval(covar_490, p = p1, index_names = sp_names)[0]
names_2 = keep_p_eigval(covar_490, p = p2, index_names = sp_names)[0]
names_3 = keep_p_eigval(covar_490, p = p3, index_names = sp_names)[0]

# matrizes de covariancia obtidas a partir dos ativos selecionados
rets_1 = [sp_returns_clean[i] for i in names_1]
cov_1 = np.array(np.cov(rets_1))
rets_2 = [sp_returns_clean[i] for i in names_2]
cov_2 = np.array(np.cov(rets_2))
rets_3 = [sp_returns_clean[i] for i in names_3]
cov_3 = np.array(np.cov(rets_3))

# calculando numeros de condicao
np.linalg.cond(cov_1)
np.linalg.cond(cov_2)
np.linalg.cond(cov_3)

# ________________________________________________________________
# ALGO 1 : NEWTON-RAPHSON (NR) METHOD
# ________________________________________________________________

'''
references: Thierry Roncalli
URL: https://deliverypdf.ssrn.com/delivery.php?ID=824100098007121005104030074028006090038051063013063017078066105109101086112121094065039020017097126100053082024067018071073077045032091044031102087028113114029119021002039026022123084067084000114125064127066097113099027089084089002108125088022093110&EXT=pdf&INDEX=TRUE

we are trying to find y* = argmin R, 
u.c. sum {ln(y_i) | i = 1,...,n}
where R is the portfolio risk (volatility), R(y) = y_T * Covar * y


Computing the optimal value of lambda_k may be time consuming. In this case, we may also
prefer the half method which consists in dividing the test value by one half each time the
function fails to decrease – λ then takes the respective values 1, 1/2, 1/4, 1/8, etc. – and to
stop when the criteria f (xk + λkdk) < f (xk) is satisfied

'''

def ERC_NewtonRaphson(covar):
    
    variances = np.diag(covar)
    stdevs = np.sqrt(variances)
    
    max_iter = 20 # numero maximo de iterações
    i = 0 # contador de iterações
    n = np.shape(covar)[1] # dimensão da matriz de covariancia (= numero de ativos)
    x = np.array(np.reciprocal(stdevs))/(sum(np.reciprocal(stdevs))) # palpite inicial para o vetor de pesos, assumindo correlações uniformes
    vol = np.sqrt(x @ covar @ x) # volatilidade do portfolio em funcao dos pesos iniciais
    lambda_c = (vol ** 2)/n # optimal step length. We could use lambda = argmin f(x_k + lambda*d_k). We could also use Nesterov (2004) results to improve lambda_c at each step
    tol = np.exp(-15) # = 0.000,000,000,001
    
    x_inv1 = np.reciprocal(x)
    x_inv2 = np.diag(x_inv1 ** 2)
    gradient = covar @ x - lambda_c * x_inv1
    max_G = max(np.abs(gradient))
    hessian = covar + lambda_c * x_inv2 # approximation of the Hessian (avoids the problem of singularity)
    delta = np.linalg.inv(hessian) @ gradient
    
    while max_G > tol and i < max_iter:
        
        x_inv1 = np.reciprocal(x)
        x_inv2 = np.diag(x_inv1 ** 2)
        
        gradient = covar @ x - lambda_c * x_inv1
        max_G = max(np.abs(gradient))
        hessian = covar + lambda_c * x_inv2       
        
        delta = np.linalg.inv(hessian) @ gradient
        x = x - delta
        i += 1
    
    solution = x/sum(x) # normalizando o vetor dos pesos dos ativos
    return [solution,i] #resultado.format(solution,i)

# PROVA REAL - CALCULANDO CONTRIBUICOES DE RISCO
def check_risks(w,covar):
    portfolio_risk = float(np.sqrt(w @ covar @ w))
    risk_contributions = []
    a = covar @ w
    for i in range(len(w)):
        risk = w[i] * a[i] / portfolio_risk
        risk_contributions.append(risk)
    r = np.array(risk_contributions)
    return max(r) - min(r) < 0.0001

# analisando resultados pelo metodo NR
times_NR = []
weights_NR = []
iterations_NR = []
checks_NR = []

# resultados algoritmo NR
for cov in [covar_490, cov_1, cov_2, cov_3]:
    start = time.time()
    NR = ERC_NewtonRaphson(cov)
    end = time.time()
    
    times_NR.append(end - start)
    weights_NR.append(NR[0])
    iterations_NR.append(NR[1])
    checks_NR.append(check_risks(NR[0],cov))


# ________________________________________________________________
# ALGO 2 : Cyclical Coordinate Descent (CCD) Method
# ________________________________________________________________

# usando codigo já existente (escrito por Kaehyuk Choi)
# analisando os resultados pelo metodo CCD
times_CCD = []

#for i in [covar_490, cov_1, cov_2, cov_3]:
start = time.time()
m = pf.RiskParity(cov=cov_3)
erc = m.weight_ccd_original(tol=np.exp(-15))
print(m._result)
end = time.time()
total=end-start
    times_CCD.append(end - start)
print(f"4 decimal places: {total:.6f}")    

