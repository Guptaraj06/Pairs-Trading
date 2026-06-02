import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    def __init__(self, delta=1e-4, obs_noise=1e-3):
        self.delta = delta
        self.obs_noise = obs_noise
        self.R = obs_noise

        self.Q = np.eye(2) * delta
        self.P = np.zeros((2, 2))
        self.theta = np.zeros(2)

    def update(self, y: float, x: float) -> tuple:
        H = np.array([x, 1.0])
        P_pred = self.P + self.Q

        y_pred = H @ self.theta
        nu = y - y_pred
        S = H @ P_pred @ H.T + self.R
        K = (P_pred @ H.T) / S
        self.theta = self.theta + K * nu
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        beta = self.theta[0]
        alpha = self.theta[1]
        spread = y - beta * x - alpha
        return beta, alpha, spread, nu

    def run(self, y: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        n = len(y)
        betas = np.zeros(n)
        alphas = np.zeros(n)
        spreads = np.zeros(n)
        innovations = np.zeros(n)

        for t in range(n):
            betas[t], alphas[t], spreads[t], innovations[t] = self.update(y[t], x[t])

        return pd.DataFrame(
            {
                "beta": betas,
                "alpha": alphas,
                "spread": spreads,
                "innovation": innovations,
            }
        )

    def tune_delta(self, y: np.ndarray, x: np.ndarray, deltas=None) -> float:
        deltas = deltas or np.logspace(-6, -1, 30)
        best_ll = -np.inf
        best_delta = deltas[0]

        for d in deltas:
            self.__init__(delta=d, obs_noise=self.obs_noise)
            df = self.run(y, x)
            nu = df["innovation"].values[10:]
            ll = -0.5 * np.log(2 * np.pi * nu.var()) - (0.5 / nu.var()) * (nu**2).mean()

            if ll > best_ll:
                best_ll = ll
                best_delta = d

        self.__init__(delta=best_delta, obs_noise=self.obs_noise)
        return best_delta
