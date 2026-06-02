import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")


class CointegrationScreener:
    def __init__(
        self, pvalue_thresh=0.05, corr_thresh=0.70, min_halflife=5, max_halflife=60
    ):
        self.pval = pvalue_thresh
        self.corr = corr_thresh
        self.min_halflife = min_halflife
        self.max_halflife = max_halflife

    def adf_test(self, series: np.ndarray) -> dict:
        result = adfuller(series, maxlag=1, regression="c", autolag="AIC")
        return {
            "stat": result[0],
            "pvalue": result[1],
            "is_stationary": result[1] < self.pval,
        }

    def engle_granger(self, y: np.ndarray, x: np.ndarray) -> dict:
        stat_yx, pval_yx, _ = coint(y, x)
        stat_xy, pval_xy, _ = coint(x, y)

        pval = min(pval_yx, pval_xy)

        if pval_yx <= pval_xy:
            beta = np.polyfit(x, y, 1)[0]
            spread = y - beta * x
        else:
            beta = np.polyfit(y, x, 1)[0]
            spread = y - beta * x

        half_life = self._half_life(spread)

        return {
            "pvalue": pval,
            "cointegrated": pval < self.pval,
            "hedge_ratio": beta,
            "half_life": half_life,
            "spread": spread,
        }

    def johansen_test(self, price_df: pd.DataFrame) -> dict:
        result = coint_johansen(price_df.values, det_order=0, k_ar_diff=1)
        trace_stats = result.lr1
        crit_vals = result.cvt[:, 1]
        rank = int(np.sum(trace_stats > crit_vals))

        if rank >= 1:
            coint_vec = result.evec[:, 0]
            spread = price_df.values @ coint_vec
        else:
            coint_vec = None
            spread = None

        return {
            "rank": rank,
            "cointegrated": rank >= 1,
            "coint_vector": coint_vec,
            "spread": spread,
        }

    def _half_life(self, spread: np.ndarray) -> float:
        spread_lag = spread[:-1]
        delta = spread[1:] - spread[:-1]
        phi = np.polyfit(spread_lag, delta, 1)[0]
        if phi >= 0:
            return np.inf
        return float(-np.log(2) / phi)

    def screen_universe(
        self, log_prices: pd.DataFrame, same_sector_only=True, sector_map: dict = None
    ) -> pd.DataFrame:
        tickers = log_prices.columns.tolist()
        returns = log_prices.pct_change().dropna()
        corr_matrix = returns.corr()
        n_pairs = len(list(combinations(tickers, 2)))
        bonferroni_alpha = self.pval / n_pairs

        records = []

        for A, B in combinations(tickers, 2):
            if same_sector_only and sector_map:
                if sector_map.get(A) != sector_map.get(B):
                    continue

            if abs(corr_matrix.loc[A, B]) < self.corr:
                continue

            y = log_prices[A].values
            x = log_prices[B].values

            eg = self.engle_granger(y, x)

            if not eg["cointegrated"]:
                continue

            hl = eg["half_life"]
            if hl < self.min_halflife or hl > self.max_halflife:
                continue

            records.append(
                {
                    "stock_A": A,
                    "stock_B": B,
                    "hedge_ratio": round(eg["hedge_ratio"], 4),
                    "pvalue_eg": round(eg["pvalue"], 4),
                    "half_life_days": round(hl, 1),
                    "correlation": round(corr_matrix.loc[A, B], 3),
                    "bonferroni_ok": eg["pvalue"] < bonferroni_alpha,
                }
            )

        df = pd.DataFrame(records).sort_values("pvalue_eg")
        return df.reset_index(drop=True)
