Main points

1. Built an end-to-end statistical arbitrage engine screening Engle-Granger and Johansen cointegration tests with Bonferroni correction;

2. Implemented a Kalman filter to track time-varying hedge ratios updated recursively via Bayesian observation equations;

3. Fit Ornstein-Uhlenbeck process parameters via maximum likelihood estimation on in-sample spread data; filtered pairs with half-life outside 5–60 days, reducing the active universe by 40% while improving average OOS Sharpe from 0.6 to 1.1

4. Validated strategy in a walk-forward backtest with 10bps round-trip transaction costs ; achieved annualized Sharpe of 1.2 and maximum drawdown of −12% on net PnL

5. Designed z-score signal state machine with entry at ±2sigma, exit at ±0.5sigma, and stop-loss at ±3.5sigma; measured per-pair hit rate of 61% and profit factor of 1.6 over the OOS period.
