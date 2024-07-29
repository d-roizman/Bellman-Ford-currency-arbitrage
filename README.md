# Bellman Ford currency arbitrage detection algorithm

The Bellman Ford algorithm got famous for quickly solving the 'shortest path problem' on directed graphs. It's application to currency trading is due to the fact that if there are still "relaxable edges" on a currency rates' matrix, then there is a negative cycle, which represents an arbitrage opportunity. This repository contains a Python script to identify arbitrage opportunities in currency trading using the Bellman-Ford algorithm. The script fetches current exchange rates and processes them to find profitable arbitrage cycles.

## Features

- Fetches real-time currency exchange rates.
- Uses the Bellman-Ford algorithm to detect currency arbitrage opportunities.
- Outputs potential profitable currency trading paths.

## Usage

1. **Clone the repository:**

   ```bash
   https://github.com/your-username/currency-arbitrage.git](https://github.com/d-roizman/Bellman-Ford-currency-arbitrage/blob/Quant_Finance/currency_arbitrage_bellman_ford.py
```


2. **Install the required packages and set API key:**

  ```bash
  from math import log
  import pandas as pd
  import requests
```

The API key is set on the [ExchangeRate-API website](https://www.exchangerate-api.com/), which is used to provide free real-time currency data.


3. **Get currency rates**

The function 'get_currency_rates' receives an API key (string), a list of currencies and a True if you want the negative logarithm of the currencies.
```bash
def get_currency_rates(api_key, currencies, neg_log=False):
    data = pd.DataFrame(index = currencies, columns = currencies)
    for currency in currencies:
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{currency}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
    
        rates = response.json()
        for currency_2 in currencies:
            if currency_2 != currency:
                data[currency][currency_2] = rates['conversion_rates'][currency_2]
            else:
                data[currency][currency_2] = 1

    if neg_log == False:
        data = data.astype(float)
        return data
    else:
        for currency_1 in currencies: # currency_1 = currency
            for currency_2 in currencies:
                data[currency_1][currency_2] = -log(data[currency_1][currency_2]) 
        return data
```
The idea of the negative log is to be able to run the Bellman Ford algorithm. That is because of how the edge-relaxation works.


4. **Bellman Ford algorithm**

See the references for a detailed explanation of how the algorithm works.
´´´bash

def Bellman_Ford_Arbitrage(rates_matrix, log_margin = 0.001):

    currencies = rates_matrix.index    
    source = 0 # (obs #3)
    n = len(rates_matrix)
    min_dist = [float('inf')] * n

    # List of 'father-nodes'
    pre = [-1] * n
    min_dist[source] = 0

    # Relax all edges (V-1) times
    for _ in range(n-1):
        for source_curr in range(n): # source_curr = 0
            for dest_curr in range(n): # dest_curr = 1
                if min_dist[dest_curr] > min_dist[source_curr] + rates_matrix.iloc[source_curr][dest_curr]:
                    min_dist[dest_curr] = min_dist[source_curr] + rates_matrix.iloc[source_curr][dest_curr]
                    pre[dest_curr] = source_curr

    # Test whether there are still 'relaxable' edges (which imply a negative cycle)
    opportunities = []
    for source_curr in range(n): # source_curr = 0
        for dest_curr in range(n): # dest_curr = 2
            if min_dist[dest_curr] > min_dist[source_curr] + rates_matrix.iloc[source_curr][dest_curr] + log_margin:
                # negative cycle exists, and use the predecessor chain to print the cycle
                cycle = [dest_curr]
                # Start from the source and go backwards until you see the source vertex again
                while True:
                    source_curr = pre[source_curr]
                    if source_curr in cycle:
                        break
                    cycle.append(source_curr)
                cycle.append(dest_curr)
                if len(cycle) > 3:
                    path = [currencies[p] for p in cycle[::-1]]
                    if path not in opportunities:
                        opportunities.append(path)
    
    return list(opportunities)

´´´


5. **Check if the algorithm found any relevant arbitrage opportunities**

´´´bash
api_key = 'your_api_key'
top10_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CNY', 'AUD', 'CAD', 'CHF', 'HKD', 'SGD']
rates = get_currency_rates(api_key, top10_currencies)
neg_log_rates = get_currency_rates(api_key, top10_currencies, neg_log = True)
print(rates)
arbitrage_opportunities = Bellman_Ford_Arbitrage(neg_log_rates)
[print(a) for a in arbitrage_opportunities]

# Testing

for path in arbitrage_opportunities:
    arbitrage_1 = path.copy()    
    
    initial_balance = 100 # in the respective source-currency listed first on 'arbitrage_1'
    final_balance = initial_balance
    
    source_currency = arbitrage_1.pop()
    while arbitrage_1:
        dest_currency = arbitrage_1.pop()
        final_balance *= rates[source_currency][dest_currency]        
        source_currency = dest_currency
    
    if final_balance - initial_balance > 0.5:
        d = final_balance - initial_balance
        print(f'ARBITRAGE OPPORTUNITY ({d}% gain): {path} ')

´´´

Thats it!

   
6. **References**
   [4.4 Shortest Paths - ALGORITHMS (Sedgewick, R., WAYNE, K.)](https://algs4.cs.princeton.edu/44sp/)
