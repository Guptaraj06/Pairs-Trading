import yfinance as yf
import pandas as pd
import numpy as np


class UniverseBuilder:
    SECTOR_TICKERS = {
        "tech": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "NVDA",
            "AMD",
            "INTC",
            "QCOM",
            "AVGO",
            "TXN",
            "MU",
            "AMAT",
        ],
        "finance": [
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            "BK",
            "STT",
            "AXP",
            "COF",
            "USB",
            "PNC",
        ],
        "energy": [
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "OXY",
            "PXD",
            "EOG",
            "VLO",
            "MPC",
            "PSX",
            "HAL",
            "BKR",
        ],
        "consumer": [
            "KO",
            "PEP",
            "MCD",
            "YUM",
            "SBUX",
            "CMG",
            "WMT",
            "TGT",
            "COST",
            "DG",
            "DLTR",
            "KR",
        ],
        "pharma": [
            "JNJ",
            "PFE",
            "MRK",
            "ABBV",
            "BMY",
            "LLY",
            "AMGN",
            "GILD",
            "BIIB",
            "VRTX",
            "REGN",
            "MRNA",
        ],
    }

    def download(self, start="2015-01-01", end="2024-12-31"):
        all_tickers = [t for s in self.SECTOR_TICKERS.values() for t in s]
        prices = yf.download(all_tickers, start=start, end=end, auto_adjust=True)[
            "Close"
        ]
        prices = prices.dropna(axis=1, thresh=int(0.95 * len(prices)))
        prices = prices.ffill().dropna()
        self.log_prices = np.log(prices)
        self.prices = prices

        # Build sector membership map
        self.sector_map = {}
        for sector, tickers in self.SECTOR_TICKERS.items():
            for t in tickers:
                if t in prices.columns:
                    self.sector_map[t] = sector
        return self
