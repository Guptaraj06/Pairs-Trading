import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class OUSpreadModel:
    def __init__(self):
        self.kappa = None
        self.mu = None
        self.sigma = None
        self.sigma_eq = None
        self.half_life = None

    def _neg_log_likelihood(self, params, spread, dt=0.1):
        kappa, mu, sigma = params

        if kappa <= 0 or sigma <= 0:
            return 1e10

        e_kdt = np.exp(-kappa * dt)
        sigma_d = np.sqrt(sigma**2 * (1 - e_kdt**2) / (2 * kappa))
        S_t = spread[1:]
        S_tm1 = spread[:-1]
        mean_cond = S_tm1 * e_kdt + mu * (1 - e_kdt)
        ll = norm.logpdf(S_t, loc=mean_cond, scale=sigma_d).sum()
        return -ll

    def fit(self, spread: np.ndarray, dt=1.0) -> "OUSpreadModel":
        phi = np.polyfit(spread[:-1], spread[1:], 1)[0]
        phi = np.clip(phi, 1e-4, 1 - 1e-4)
        kappa0 = -np.log(phi) / dt
        mu0 = spread.mean()
        sigma0 = spread.std() * np.sqrt(2 * kappa0)

        result = minimize(
            self._neg_log_likelihood,
            x0=[kappa0, mu0, sigma0],
            args=(spread, dt),
            method="L-BFGS-B",
            bounds=[(1e-4, None), (None, None), (1e-6, None)],
            options={"maxiter": 1000},
        )

        self.kappa, self.mu, self.sigma = result.x
        self.sigma_eq = self.sigma / np.sqrt(2 * self.kappa)
        self.half_life = np.log(2) / self.kappa
        return self

    def zscore(self, spread: np.ndarray, rolling_window: int = None) -> np.ndarray:
        if rolling_window:
            s = pd.Series(spread)
            mu_r = s.rolling(rolling_window).mean()
            sd_r = s.rolling(rolling_window).std()
            return ((s - mu_r) / sd_r).values
        else:
            return (spread - self.mu) / self.sigma_eq

    def expected_return_time(self, current_z: float) -> float:
        return -np.log(0.5) / self.kappa

    def simulate(self, T=252, n_paths=1000, s0=None) -> np.ndarray:
        s0 = s0 or self.mu
        dt = 1.0
        paths = np.zeros((T, n_paths))
        paths[0] = s0
        e_kdt = np.exp(-self.kappa * dt)
        sigma_d = np.sqrt(self.sigma**2 * (1 - e_kdt**2) / (2 * self.kappa))
        for t in range(1, T):
            mean_c = paths[t - 1] * e_kdt + self.mu * (1 - e_kdt)
            paths[t] = mean_c + sigma_d * np.randdom.randn(n_paths)

        return paths
