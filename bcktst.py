import numpy as np
import pandas as pd


class WalkForwardPairsBacktest:
    def __init__(
        self,
        log_prices: pd.DataFrame,
        prices: pd.DataFrame,
        sector_map: dict,
        train_days=252,
        oos_days=63,
        max_pairs=15,
        capital_per_pair=100_000,
    ):
        self.lp = log_prices
        self.px = prices
        self.sm = sector_map
        self.train = train_days
        self.oos = oos_days
        self.max_p = max_pairs
        self.cap = capital_per_pair

    def run(self) -> pd.DataFrame:
        T = len(self.lp)
        all_pnl = pd.Series(0.0, index=self.lp.index, dtype=float)
        all_positions = {}

        windows = range(self.train, T - self.oos, self.oos)
        print(f"Running {len(windows)} walk-forward windows...")

        for start in windows:
            train_idx = slice(start - self.train, start)
            oos_idx = slice(start, start + self.oos)

            lp_train = self.lp.iloc[train_idx]
            lp_oos = self.lp.iloc[oos_idx]
            px_train = self.px.iloc[train_idx]
            px_oos = self.px.iloc[oos_idx]

            screener = CointegrationScreener(
                pvalue_thresh=0.05, corr_thresh=0.70, min_halflife=5, max_halflife=60
            )
            pairs_df = screener.screen_universe(lp_train, sector_map=self.sm)
            selected = pairs_df.head(self.max_p)  # top N by p-value

            if len(selected) == 0:
                continue

            period_pnl = np.zeros(self.oos)

            for _, row in selected.iterrows():
                A, B = row["stock_A"], row["stock_B"]

                lp_full_A = (
                    self.lp[A].iloc[start - self.train : start + self.oos].values
                )
                lp_full_B = (
                    self.lp[B].iloc[start - self.train : start + self.oos].values
                )
                px_full_A = (
                    self.px[A].iloc[start - self.train : start + self.oos].values
                )
                px_full_B = (
                    self.px[B].iloc[start - self.train : start + self.oos].values
                )

                kf = KalmanHedgeRatio(delta=1e-4)
                kf_df = kf.run(lp_full_A, lp_full_B)

                oos_spread = kf_df["spread"].values[-self.oos :]
                oos_betas = kf_df["beta"].values[-self.oos :]

                train_spread = kf_df["spread"].values[: -self.oos]
                mu_s = train_spread.mean()
                std_s = train_spread.std()
                z_oos = (oos_spread - mu_s) / std_s

                sg = SignalGenerator(entry_z=2.0, exit_z=0.5, stop_z=3.5)
                pos, _ = sg.generate(
                    oos_spread, ou_model=None
                )  # uses rolling z internally

                ret_A = np.diff(lp_full_A[-self.oos :], prepend=lp_full_A[-self.oos])
                ret_B = np.diff(lp_full_B[-self.oos :], prepend=lp_full_B[-self.oos])
                spread_ret = ret_A - oos_betas * ret_B
                pair_pnl = pos * spread_ret * (self.cap / len(selected))

                trades = (np.abs(np.diff(pos, prepend=0)) > 0).astype(float)
                pair_pnl -= trades * (self.cap / len(selected)) * 0.002

                period_pnl += pair_pnl

            oos_dates = self.lp.index[oos_idx]
            all_pnl.iloc[oos_idx] += period_pnl

        return all_pnl

    def report(self, pnl: pd.Series) -> dict:
        r = pnl.values / (self.max_p * self.cap)  # normalize to % returns
        r = r[r != 0]  # exclude non-trading days
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        cum = (1 + r).cumprod()
        max_dd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()
        ann_ret = (cum[-1]) ** (252 / len(r)) - 1
        return {
            "Annual Return": f"{ann_ret:.1%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.1%}",
            "Total PnL $": f"${pnl.sum():,.0f}",
        }
