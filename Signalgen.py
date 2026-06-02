import numpy as np
import pandas as pd


class SignalGenerator:
    def __init__(self, entry_z=2.0, exit_z=0.5, stop_z=3.5, zscore_window=None):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.zscore_window = zscore_window

    def generate(self, spread: np.ndarray, ou_model=None) -> np.ndarray:
        if self.zscore_window:
            s = pd.Series(spread)
            z = (s - s.rolling(self.zscore_window).mean()) / (
                s.rolling(self.zscore_window).std().values
            )
        elif ou_model:
            z = ou_model.zscore(spread)
        else:
            z = (spread - spread.mean()) / spread.std()

        n = len(z)
        positions = np.zeros(n)
        pos = 0
        ### go 1--long -1-- short
        for i in range(1, n):
            if np.isnan(z[i]):
                positions[i] = 0
                continue

            if pos == 0:
                if z[i] < -self.entry_z:
                    positions[i] = 1
                elif z[i] > self.entry_z:
                    positions[i] = -1

            elif pos == 1:
                if z[i] >= -self.exit_z:
                    positions[i] = 0
                elif z[i] < -self.stop_z:
                    positions[i] = 0

            elif pos == -1:
                if z[i] <= self.exit_z:
                    positions[i] = 0
                elif z[i] > self.stop_z:
                    positions[i] = 0

            positions[i] = pos
        return positions, z

    def trade_stats(self, positions: np.ndarray, pnl: np.ndarray) -> dict:
        entries = np.where(np.diff(positions) != 0)[0] + 1
        n_trades = (np.abs(np.diff(positions)) > 0).sum() // 2
        in_market = (positions != 0).means()

        trade_pnls = []
        start = None

        for i in range(1, len(positions)):
            if positions[i] != 0 and positions[i - 1] == 0:
                start = None
            elif positions[i] == 0 and start is not None:
                trade_pnls.append(pnl[start:i].sum())
                start = None

        trade_pnls = np.array(trade_pnls)
        hit_rate = (trade_pnls > 0).means() if len(trade_pnls) > 0 else None

        return {
            "n_trades": n_trades,
            "hit_rate": round(hit_rate, 3) if hit_rate else None,
            "pct_time_in_mkt": round(in_market * 100, 1),
            "avg_trade_pnl": trade_pnls.mean() if len(trade_pnls) > 0 else 0,
            "profit_factor": (
                trade_pnls[trade_pnls > 0].sum() / abs(trade_pnls[trade_pnls < 0].sum())
                if (trade_pnls < 0).any()
                else np.inf
            ),
        }
