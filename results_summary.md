# Financial Portfolio & Risk-Return Analyzer
## Results Summary

**Tickers analysed :** AAPL, AMZN, GOOGL, MSFT
**History window   :** 3 years
**Portfolio weights:** AAPL: 25.0 %, AMZN: 25.0 %, GOOGL: 25.0 %, MSFT: 25.0 %
**Risk-free rate   :** 4.00 % p.a.

---

### Risk / Return Metrics

| Asset       | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown | DD Peak Date | DD Trough Date | DD Recovery Date |
|-------------|:-----------:|:---------------:|:------------:|:------------:|:------------:|:--------------:|:----------------:|
| AAPL        |   25.25 %   |     26.67 %     |    0.797     |   -33.36 %   |  2025-04-08  |   2025-04-08   |    2025-10-20    |
| AMZN        |   27.22 %   |     31.81 %     |    0.730     |   -30.88 %   |  2025-04-21  |   2025-04-21   |    2025-10-31    |
| GOOGL       |   38.51 %   |     30.53 %     |    1.130     |   -29.81 %   |  2025-04-08  |   2025-04-08   |    2025-08-22    |
| MSFT        |   20.07 %   |     26.01 %     |    0.618     |   -34.50 %   |  2026-06-25  |   2026-06-25   |        —         |
|-------------|:-----------:|:---------------:|:------------:|:------------:|:------------:|:--------------:|:----------------:|
| Portfolio   |   27.59 %   |     21.72 %     |    1.086     |   -26.53 %   |  2025-04-08  |   2025-04-08   |    2025-08-08    |

---

### Metric Definitions

| Metric | Definition |
|---|---|
| **Ann. Return** | Compound Annual Growth Rate (CAGR) = exp(mean daily log-return × 252) − 1 |
| **Ann. Volatility** | Annualised Std Dev of daily log-returns = σ_daily × √252 |
| **Sharpe Ratio** | (Ann. Return − Risk-Free Rate) ÷ Ann. Volatility |
| **Max Drawdown** | Maximum peak-to-trough equity decline over the sample window |
| **DD Peak / Trough** | Calendar dates bounding the worst drawdown episode |
| **DD Recovery** | First date equity recovered above the prior peak; "—" if still underwater |

> *All returns are based on split- and dividend-adjusted daily closing prices sourced from Yahoo Finance.*
