import numpy as np
import pandas as pd


class PairsStrategy:
    def __init__(
        self,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        kalman_delta=1e-4,
        max_pairs=20,
        capital_per_pair=100_000,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.kd = kalman_delta
        self.max_pairs = max_pairs
        self.capital = capital_per_pair

    def run_pair(
        self,
        log_A: np.ndarray,
        log_B: np.ndarray,
        price_A: np.ndarray,
        price_B: np.ndarray,
    ) -> dict:
        """Full pipeline for a single pair."""
        # 1. Cointegration test
        screener = CointegrationScreener()
        coint = screener.engle_granger(log_A, log_B)
        if not coint["cointegrated"]:
            return None

        # 2. Kalman filter for dynamic hedge ratio
        kf = KalmanHedgeRatio(delta=self.kd, obs_noise=1e-3)
        kf_df = kf.run(log_A, log_B)
        spread = kf_df["spread"].values
        betas = kf_df["beta"].values

        # 3. OU model on spread
        ou = OUSpreadModel().fit(spread)
        if ou.half_life > 60 or ou.half_life < 5:
            return None

        # 4. Signal generation
        sg = SignalGenerator(
            self.entry_z, self.exit_z, self.stop_z, zscore_window=int(2 * ou.half_life)
        )
        positions, z = sg.generate(spread)

        # 5. Compute daily P&L
        # Long spread: buy A, sell β units of B
        ret_A = np.diff(log_A, prepend=log_A[0])
        ret_B = np.diff(log_B, prepend=log_B[0])
        spread_ret = ret_A - betas * ret_B

        # Dollar PnL: position × spread_return × capital
        gross_pnl = positions * spread_ret * self.capital

        # Transaction costs: 10bps each leg, charged on entry/exit
        trades = np.abs(np.diff(positions, prepend=0)) > 0
        tc = trades.astype(float) * self.capital * (2 * 10e-4)
        net_pnl = gross_pnl - tc

        stats = sg.trade_stats(positions, net_pnl)

        return {
            "positions": positions,
            "zscore": z,
            "spread": spread,
            "betas": betas,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "ou_params": {
                "kappa": ou.kappa,
                "mu": ou.mu,
                "sigma": ou.sigma,
                "half_life": ou.half_life,
            },
            "coint_pval": coint["pvalue"],
            **stats,
        }
