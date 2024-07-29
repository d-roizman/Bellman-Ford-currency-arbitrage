# Bellman Ford currency arbitrage detection algorithm

The Bellman Ford algorithm got famous for solving "quickly" the shortest path problem on directed graphs. It's application to currency trading is due to the fact that if there are still "relaxable vertices" on the ajcacency matrix of currency rates, then there is what is called a negative cycle, which represents an arbitrage opportunity. This repository contains a Python script to identify arbitrage opportunities in currency trading using the Bellman-Ford algorithm. The script fetches current exchange rates and processes them to find profitable arbitrage cycles.

## Features

- Fetches real-time currency exchange rates.
- Uses the Bellman-Ford algorithm to detect currency arbitrage opportunities.
- Outputs potential profitable currency trading paths.

## Usage

1. **Clone the repository:**

   ```bash
   https://github.com/your-username/currency-arbitrage.git](https://github.com/d-roizman/Bellman-Ford-currency-arbitrage/blob/Quant_Finance/currency_arbitrage_bellman_ford.py
   exit

2. **Install the required packages and set API key:**

  ```bash
  from math import log
  import pandas as pd
  import requests
```
The API key is set on the [ExchangeRate-API website](https://www.exchangerate-api.com/), which is used to provide free real-time currency data.

3. **Run the script**
