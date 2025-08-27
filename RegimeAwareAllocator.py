import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from fredapi import Fred

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture   
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

class RegimeAwareAllocator:
    def __init__(self, start_date, verbose, out_of_sample_years, end_date=None):
        self.start_date = start_date
        self.end_date = pd.Timestamp.today() - pd.DateOffset(years=out_of_sample_years)

        if end_date:
            end_date = pd.to_datetime(end_date)
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=out_of_sample_years)
            if end_date > cutoff:
                print(f"end_date must be at least {out_of_sample_years} years before today (<= {cutoff.date()})")
                print(f"Adjusting end_date accordingly to start at {cutoff}")
            else:
                self.end_date = end_date

        self.start_date_oos = self.end_date + pd.DateOffset(days=1)
        self.end_date_oos = pd.Timestamp.today()

        self.assets_series = {
            "SPY": "SPY", 
            "BONDS": "IEF", 
            "GOLD": "GLD", 
            "USD": "UUP", 
            "VIX": "^VIX"
        }
        self.fred_series = {
            "CPI": "CPIAUCSL",
            "UNRATE": "UNRATE",
            "T10Y2Y": "T10Y2Y",
            "GDP": "GDPC1"  
        }
        self.macro = None
        self.assets = None
        self.features = None
        self.regimes = None
        self.verbose = verbose
        self.is_returns = None
        self.oos_returns = None

    def fetch_asset_data(self):
        print("Fetching asset prices from yfinance...")
        px = yf.download(list(self.assets_series.values()), start=self.start_date, end=self.end_date_oos)['Close']
        px = px.rename(columns={v: k for k, v in self.assets_series.items()})
        self.assets = px.dropna(how='all')
        print(f"Prices fetched: {self.assets.shape}")
        if self.verbose:
            print(self.assets.tail(10))
        print("="*50)

        print("Fetching macro data from FRED...")
        load_dotenv()
        fred_key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=fred_key)

        macro = {}
        for name, code in self.fred_series.items():
            series = fred.get_series(code)
            macro[name] = series.to_frame(name)
        
        # Preprocess cpi and gdp 
        cpi = macro['CPI']
        gdp = macro['GDP']
        cpi_yoy = np.log(cpi['CPI'] / cpi['CPI'].shift(12)).to_frame('CPI_YoY')
        gdp_yoy = np.log(gdp['GDP'] / gdp['GDP'].shift(4)).to_frame('GDP_YoY')
        macro['CPI'] = cpi
        macro['GDP'] = gdp
        macro['CPI_YoY'] = cpi_yoy
        macro['GDP_YoY'] = gdp_yoy

        macro_df = pd.concat(macro.values(), axis=1)
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_df = macro_df.sort_index()
        self.macro = macro_df
        print(f"Macro data fetched: {self.macro.shape}")
        if self.verbose:
            print(self.macro[self.macro.notnull().all(axis=1)])
        print("="*50)

    def preprocess_features(self, vol_window=21):
        print("Preprocessing features...")

        # Asset features
        prices = self.assets.copy()
        log_returns = np.log(prices.drop(columns="VIX") / prices.drop(columns="VIX").shift(1))
        log_returns.columns = [f"ret_{col}" for col in log_returns.columns]

        rolling_vol = log_returns.rolling(window=vol_window).std()
        rolling_vol.columns = [f"vol_{col[4:]}" for col in log_returns.columns]

        vix_series = prices["VIX"].to_frame("VIX")
        asset_feats = pd.concat([log_returns, rolling_vol, vix_series], axis=1)

        # Macro features
        macro = self.macro.copy()
        macro_feats = macro[['CPI_YoY', 'GDP_YoY', 'UNRATE', 'T10Y2Y']]

        # Combine features
        combined = asset_feats.join(macro_feats, how='left')
        combined[['CPI_YoY', 'GDP_YoY', 'UNRATE', 'T10Y2Y']] = combined[['CPI_YoY', 'GDP_YoY', 'UNRATE', 'T10Y2Y']].ffill() 
        self.features = combined
        self.features.dropna(inplace=True)
        self.features.to_csv("outputs/features.csv")

        print(f"Features ready: {self.features.shape[0]} rows × {self.features.shape[1]} columns")
        if self.verbose:
            print(self.features.tail(10))
        print("="*50)

    def optimize_n_components_by_bic(self, window=750, step=21, k_min=2, k_max=4, random_state=42):
        print("Optimizing number of GMM components using average BIC...")

        X = self.features.loc[:self.end_date].copy()
        k_range = range(k_min, k_max + 1)
        bic_scores = {k: [] for k in k_range}

        for start in range(0, len(X) - window, step):
            train = X.iloc[start:start + window]

            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)

            for k in k_range:
                gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
                gmm.fit(train_scaled)
                bic = gmm.bic(train_scaled)
                bic_scores[k].append(bic)

        avg_bics = {k: np.mean(bic_scores[k]) for k in k_range}
        best_k = min(avg_bics, key=avg_bics.get)

        print("Average BIC per k:")
        for k, score in avg_bics.items():
            print(f"  k = {k}: avg BIC = {score:.2f}")

        print(f"Best number of components based on average BIC: {best_k}")
        print("=" * 50)

        return best_k

    def identify_regimes(self, window=750, step=21, n_components=3, random_state=42):
        print("Identifying regimes...")
        X = self.features.copy()
        regimes = pd.Series(index=X.index, dtype=int)

        for start in range(0, len(X) - window, step):
            train = X.iloc[start:start + window]
            predict = X.iloc[start + window:start + window + step]

            if len(predict) == 0:
                break  

            scaler = StandardScaler()
            train_scaled = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)
            predict_scaled = pd.DataFrame(scaler.transform(predict), index=predict.index, columns=predict.columns)

            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
            gmm.fit(train_scaled)

            pred_labels = gmm.predict(predict_scaled)
            regimes[predict.index] = pred_labels

            if self.verbose:
                print(f"GMM window: {predict.index[0].date()} to {predict.index[-1].date()} — labels: {np.unique(pred_labels)}")

        self.regimes = regimes.to_frame("regime")

        print(f"Regime identification complete. {self.regimes['regime'].nunique()} regimes detected.")
        print("="*50)

    def combine_regimes_to_features(self):
        print("Combining regimes to feature df")

        if self.regimes is None or self.features is None:
            raise ValueError("Run identify_regimes and preprocess_features first.")

        df = self.features.copy()
        df["regime"] = self.regimes["regime"]

        self.features = df
        self.features.dropna(inplace=True)
        self.regimes.dropna(inplace=True)

        self.regimes.to_csv("outputs/regimes.csv")
        self.features.to_csv("outputs/features_with_regimes.csv")   

        if self.verbose:
            print(self.regimes.tail())
        print("="*50)

    def convert_log_to_simple_returns(self):
        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        print("Converting log returns to simple returns for assets...")
        self.features.loc[:, asset_cols] = np.expm1(self.features.loc[:, asset_cols])
        print("="*50)

    def plot_regime_distribution(self):
        print("Plotting regime distribution...")

        raw_avg = self.features.loc[:self.end_date].groupby("regime").mean().T.round(6)
        raw_avg.to_csv("outputs/raw_avg_features_by_regime.csv")
        print(raw_avg)

        counts = self.regimes.loc[:self.end_date, "regime"].value_counts(normalize=True).sort_index()
        counts_percentage = (counts * 100).round(2)
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=counts_percentage.index, y=counts_percentage.values, hue=counts_percentage.index, palette="Set2", legend=False)
        for i, val in enumerate(counts_percentage.values):
            ax.text(i, val + 1, f"{val:.0f}%", ha='center', va='bottom', fontsize=11, weight='bold')

        plt.title("Regime Distribution (% of Time)")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Regime")
        plt.ylim(0, 100) 
        plt.tight_layout()
        plt.show()

        print("Regime distribution (%):")
        print(counts_percentage.to_frame(name="Percentage"))
        print("=" * 50)

    def plot_oos_regime_distribution(self):
        print("Plotting oos regime distribution...")

        raw_avg = self.features.loc[self.end_date:].groupby("regime").mean().T.round(6)
        raw_avg.to_csv("outputs/raw_avg_features_by_oos_regime.csv")
        print(raw_avg)

        counts = self.regimes.loc[self.end_date:, "regime"].value_counts(normalize=True).sort_index()
        counts_percentage = (counts * 100).round(2)
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=counts_percentage.index, y=counts_percentage.values, hue=counts_percentage.index, palette="Set2", legend=False)
        for i, val in enumerate(counts_percentage.values):
            ax.text(i, val + 1, f"{val:.0f}%", ha='center', va='bottom', fontsize=11, weight='bold')

        plt.title("OOS Regime Distribution (% of Time)")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Regime")
        plt.ylim(0, 100) 
        plt.tight_layout()
        plt.show()

        print("Regime distribution (%):")
        print(counts_percentage.to_frame(name="Percentage"))
        print("=" * 50)

    def plot_transition_likelihood(self):
        prev = self.regimes.loc[:self.end_date, "regime"].shift(1)
        curr = self.regimes.loc[:self.end_date, "regime"]

        mask = prev.notna() & curr.notna()
        transitions = prev[mask].astype(int).astype(str) + " → " + curr[mask].astype(int).astype(str)

        transition_counts = transitions.value_counts(normalize=True).sort_index()
        transition_percentage = (transition_counts * 100).round(2)

        print("Transition likelihood (%):")
        print(transition_percentage.to_frame(name="Percentage"))
        print("=" * 50)

    def shift_returns(self): # Prevent lookahead bias
        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        print("Shifting returns by -1")
        self.features.loc[:, asset_cols] = self.features.loc[:, asset_cols].shift(-1)
        self.features.dropna(inplace=True)
        print("="*50)

    def compute_all_regime_weights(self, target_return_ann=0.10, risk_free_rate=0.0):
        if self.regimes is None or self.features is None:
            raise ValueError("Run preprocess_features() and identify_regimes() first.")

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]

        X = self.features.loc[:self.end_date, asset_cols].dropna()
        reg = self.regimes.loc[:self.end_date, "regime"]
        reg = reg.dropna().astype(int)

        rf_d = risk_free_rate / 252.0

        cols = []
        for rid in sorted(reg.unique()):
            R = X.loc[reg[reg == rid].index]

            mu = R.mean().values
            Sigma = R.cov().values
            n = len(mu)

            def neg_sharpe(w):
                port_mu = w @ mu - rf_d
                port_sd = np.sqrt(w @ Sigma @ w + 1e-12)
                return -port_mu / port_sd

            constraints = [
                {'type': 'eq',  'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'ineq','fun': lambda w: ((w @ mu) - rf_d) * 252.0 - target_return_ann},
            ]
            bnds = [(-1.0, 1.0)] * n
            x0 = np.ones(n) / n

            res = minimize(neg_sharpe, x0, bounds=bnds, constraints=constraints)
            w = res.x if res.success else x0

            s = pd.Series(w, index=[c.replace("ret_", "") for c in asset_cols], name=str(rid))
            cols.append(s)
        
        print("Computed weights for asset allocation in different regimes:")
        weight_df = pd.concat(cols, axis=1)
        print(weight_df)
        return weight_df

    def calculate_regime_aware_max_sharpe_is(self, weights_df=None):
        if self.features is None or self.regimes is None:
            raise ValueError("Run preprocess_features() and identify_regimes() first.")

        print("Calculating in-sample regime-aware max Sharpe returns...")
        if weights_df is None:
            weights_df = self.compute_all_regime_weights()

        weights_df = weights_df.copy()
        weights_df.columns = weights_df.columns.astype(str)

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R = self.features.loc[:self.end_date, asset_cols].dropna()

        reg = self.regimes.reindex(R.index)["regime"].astype("Int64").astype(str)
        mask = reg.notna()
        R = R.loc[mask]
        reg = reg.loc[mask]

        asset_index = ["SPY", "BONDS", "GOLD", "USD"]
        W = np.vstack([weights_df[reg_i].reindex(asset_index).values for reg_i in reg])

        port = pd.Series((R.values * W).sum(axis=1), index=R.index, name="regime_max_sharpe")

        if getattr(self, "is_returns", None) is None:
            self.is_returns = pd.DataFrame(port)
        else:
            self.is_returns = self.is_returns.join(port, how="inner")

        print("=" * 50)

    def calculate_regime_aware_max_sharpe_oos(self, weights_df=None):
        if self.features is None or self.regimes is None:
            raise ValueError("Run preprocess_features() and identify_regimes() first.")

        print("Calculating out-of-sample regime-aware max Sharpe returns...")
        if weights_df is None:
            weights_df = self.compute_all_regime_weights()

        weights_df = weights_df.copy()
        weights_df.columns = weights_df.columns.astype(str)

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R = self.features.loc[self.end_date:, asset_cols].dropna()

        reg = self.regimes.reindex(R.index)["regime"].astype("Int64").astype(str)
        mask = reg.notna()
        R = R.loc[mask]
        reg = reg.loc[mask]

        asset_index = ["SPY", "BONDS", "GOLD", "USD"]
        W = np.vstack([weights_df[reg_i].reindex(asset_index).values for reg_i in reg])

        port = pd.Series((R.values * W).sum(axis=1), index=R.index, name="regime_max_sharpe")

        if getattr(self, "oos_returns", None) is None:
            self.oos_returns = pd.DataFrame(port)
        else:
            self.oos_returns = self.oos_returns.join(port, how="inner")

        print("=" * 50)

    def calculate_rolling_rpp_is(self, window=252, step=21):
        print("Calculating in-sample rolling RPP returns...")

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R = self.features.loc[:self.end_date, asset_cols].dropna()

        port_returns = []

        for i in range(window, len(R) - 1, step):
            train = R.iloc[i - window:i]
            Sigma = train.cov().values
            n = Sigma.shape[0]

            def obj(w):
                cw = Sigma @ w
                tot = max(w @ cw, 1e-12)
                rc = (w * cw) / tot
                return np.sum((rc - 1.0/n) ** 2)

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bnds = [(0.0, 1.0)] * n
            x0 = np.ones(n) / n

            res = minimize(obj, x0, bounds=bnds, constraints=constraints)
            w = res.x if res.success else x0

            test_window = R.iloc[i + 1 : min(i + 1 + step, len(R))]
            for date, row in test_window.iterrows():
                ret = row.values @ w
                port_returns.append((date, ret))

        out = pd.Series(dict(port_returns)).sort_index()
        out.name = "rolling_rpp"

        if getattr(self, "is_returns", None) is None:
            self.is_returns = pd.DataFrame(out)
        else:
            self.is_returns = self.is_returns.join(out, how="inner")

    def calculate_rolling_rpp_oos(self, window=252, step=21):
        print("Calculating out-of-sample rolling RPP returns...")

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R = self.features.loc[self.end_date:, asset_cols].dropna()

        port_returns = []

        for i in range(window, len(R) - 1, step):
            train = R.iloc[i - window:i]
            Sigma = train.cov().values
            n = Sigma.shape[0]

            def obj(w):
                cw = Sigma @ w
                tot = max(w @ cw, 1e-12)
                rc = (w * cw) / tot
                return np.sum((rc - 1.0/n) ** 2)

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bnds = [(0.0, 1.0)] * n
            x0 = np.ones(n) / n

            res = minimize(obj, x0, bounds=bnds, constraints=constraints)
            w = res.x if res.success else x0

            test_window = R.iloc[i + 1 : min(i + 1 + step, len(R))]
            for date, row in test_window.iterrows():
                ret = row.values @ w
                port_returns.append((date, ret))

        out = pd.Series(dict(port_returns)).sort_index()
        out.name = "rolling_rpp"

        if getattr(self, "oos_returns", None) is None:
            self.oos_returns = pd.DataFrame(out)
        else:
            self.oos_returns = self.oos_returns.join(out, how="inner")

    def calculate_rolling_max_sharpe_is(self, window=252, step=21, risk_free_rate=0.0):
        print("Calculating in-sample rolling max Sharpe returns...")

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R = self.features.loc[:self.end_date, asset_cols].dropna()
        rf_d = risk_free_rate / 252.0

        port_returns = []

        for i in range(window, len(R) - 1, step):
            train = R.iloc[i - window:i]
            mu = train.mean().values
            Sigma = train.cov().values
            n = len(mu)

            def neg_sharpe(w):
                port_mu = w @ mu - rf_d
                port_sd = np.sqrt(w @ Sigma @ w + 1e-12)
                return -port_mu / port_sd

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bnds = [(-1.0, 1.0)] * n
            x0 = np.ones(n) / n

            res = minimize(neg_sharpe, x0, bounds=bnds, constraints=constraints)
            w = res.x if res.success else x0

            test_window = R.iloc[i + 1 : min(i + 1 + step, len(R))]
            for date, row in test_window.iterrows():
                ret = row.values @ w
                port_returns.append((date, ret))

        out = pd.Series(dict(port_returns)).sort_index()
        out.name = "rolling_max_sharpe"

        if getattr(self, "is_returns", None) is None:
            self.is_returns = pd.DataFrame(out)
        else:
            self.is_returns = self.is_returns.join(out, how="inner")

    def calculate_rolling_max_sharpe_oos(self, window=252, step=21, risk_free_rate=0.0):
        print("Calculating out-of-sample rolling max Sharpe returns...")

        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R = self.features.loc[self.end_date:, asset_cols].dropna()
        rf_d = risk_free_rate / 252.0

        port_returns = []

        for i in range(window, len(R) - 1, step):
            train = R.iloc[i - window:i]
            mu = train.mean().values
            Sigma = train.cov().values
            n = len(mu)

            def neg_sharpe(w):
                port_mu = w @ mu - rf_d
                port_sd = np.sqrt(w @ Sigma @ w + 1e-12)
                return -port_mu / port_sd

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bnds = [(-1.0, 1.0)] * n
            x0 = np.ones(n) / n

            res = minimize(neg_sharpe, x0, bounds=bnds, constraints=constraints)
            w = res.x if res.success else x0

            test_window = R.iloc[i + 1 : min(i + 1 + step, len(R))]
            for date, row in test_window.iterrows():
                ret = row.values @ w
                port_returns.append((date, ret))

        out = pd.Series(dict(port_returns)).sort_index()
        out.name = "rolling_max_sharpe"

        if getattr(self, "oos_returns", None) is None:
            self.oos_returns = pd.DataFrame(out)
        else:
            self.oos_returns = self.oos_returns.join(out, how="inner")

    def calculate_6040_is(self):
        if self.features is None:
            raise ValueError("Run preprocess_features() first.")

        print("Calculating in-sample 60/40 portfolio returns...")
        assets = ["ret_SPY", "ret_BONDS"]
        R = self.features.loc[:self.end_date, assets].dropna()
        if R.empty:
            raise ValueError("No in-sample returns found for 60/40 portfolio.")

        w = [0.6, 0.4]
        p_ret = pd.Series(R.values @ w, index=R.index, name="60_40")

        if getattr(self, "is_returns", None) is None:
            self.is_returns = pd.DataFrame(p_ret)
        else:
            self.is_returns = self.is_returns.join(p_ret, how="inner")

        print("=" * 50)

    def calculate_6040_oos(self):
        if self.features is None:
            raise ValueError("Run preprocess_features() first.")

        print("Calculating out-of-sample 60/40 portfolio returns...")
        assets = ["ret_SPY", "ret_BONDS"]
        R = self.features.loc[self.end_date:, assets].dropna()
        if R.empty:
            raise ValueError("No out-of-sample returns found for 60/40 portfolio.")

        w = [0.6, 0.4]
        p_ret = pd.Series(R.values @ w, index=R.index, name="60_40")

        if getattr(self, "oos_returns", None) is None:
            self.oos_returns = pd.DataFrame(p_ret)
        else:
            self.oos_returns = self.oos_returns.join(p_ret, how="inner")

        print("=" * 50)

    def plot_cumulative_wealth(self, returns_df):
        print("Plotting cumulative wealth ...")
        wealth = (1.0 + returns_df.fillna(0)).cumprod()

        plt.figure(figsize=(12, 6))
        for col in wealth.columns:
            plt.plot(wealth.index, wealth[col], label=col, linewidth=2)

        plt.title("Cumulative Wealth", fontsize=14, weight="bold")
        plt.xlabel("Date")
        plt.ylabel("Wealth (Initial = 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _ann_factor(self, freq="D"):
        return {"D": 252, "W": 52, "M": 12}.get(freq, 252)

    def _series_metrics(self, r, rf_annual=0.0, mar_annual=0.0):
        r = r.dropna()
        if r.empty:
            return {
                "n_days": 0,
                "ann_return": np.nan,
                "ann_vol": np.nan,
                "sharpe": np.nan,
                "sortino": np.nan,
                "max_drawdown": np.nan,
                "var_5": np.nan,
                "cvar_5": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
            }

        AF = 252.0
        rf_d = rf_annual / AF           # daily risk-free
        mar_d = mar_annual / AF         # daily minimum acceptable return (MAR) for Sortino

        # Cumulative wealth from simple returns
        wealth = (1.0 + r).cumprod()

        # Daily stats
        mu_d  = r.mean()
        vol_d = r.std(ddof=1)

        # CAGR (handles non-integer years robustly)
        ann_return = wealth.iloc[-1] ** (AF / len(r)) - 1.0

        # Annualized volatility
        ann_vol = vol_d * np.sqrt(AF)

        # Sharpe (annualized)
        sharpe = ((mu_d - rf_d) / (vol_d + 1e-12)) * np.sqrt(AF)

        # Sortino (annualized): downside deviation relative to MAR
        downside = np.minimum(r - mar_d, 0.0)
        dd_dev_d = np.sqrt((downside ** 2).mean())
        sortino = ((mu_d - mar_d) * AF) / (dd_dev_d * np.sqrt(AF) + 1e-12)

        # Max Drawdown from wealth curve
        roll_max = wealth.cummax()
        drawdown = (wealth / roll_max) - 1.0
        max_dd = drawdown.min()

        # Additional risk metrics
        var_5 = np.percentile(r, 5) # VAR
        cvar_5 = r[r <= var_5].mean() # CVAR
        skewness = skew(r) # Skewness
        kurt = kurtosis(r, fisher=False) # Kurtosis

        return {
            "n_days": int(len(r)),
            "ann_return": float(ann_return),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "var_5": float(var_5),
            "cvar_5": float(cvar_5),
            "skewness": float(skewness),
            "kurtosis": float(kurt),
        }

    def compute_metrics(self, returns_df=None, rf_annual=0.0, mar_annual=0.0):
        if returns_df is None:
            returns_df = self.is_returns

        if returns_df is None or returns_df.empty:
            raise ValueError("No returns to evaluate. Provide returns_df or populate self.is_returns first.")

        out = {}
        for col in returns_df.columns:
            out[col] = self._series_metrics(returns_df[col], rf_annual=rf_annual, mar_annual=mar_annual)

        df = pd.DataFrame(out)
        order = ["n_days", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "var_5", "cvar_5", "skewness", "kurtosis"]
        return df.loc[order]

    def print_metrics(self, returns_df, rf_annual=0.0, mar_annual=0.0):
        print("Computing Performance Metrics ...")
        df = self.compute_metrics(returns_df=returns_df, rf_annual=rf_annual, mar_annual=mar_annual)

        fmt = df.copy()
        for c in fmt.columns:
            fmt.loc["ann_return", c]   = f"{100 * df.loc['ann_return', c]:.2f}%"
            fmt.loc["ann_vol", c]      = f"{100 * df.loc['ann_vol', c]:.2f}%"
            fmt.loc["sharpe", c]       = f"{df.loc['sharpe', c]:.2f}"
            fmt.loc["sortino", c]      = f"{df.loc['sortino', c]:.2f}"
            fmt.loc["max_drawdown", c] = f"{100 * df.loc['max_drawdown', c]:.2f}%"
            fmt.loc["var_5", c]        = f"{100 * df.loc['var_5', c]:.2f}%"
            fmt.loc["cvar_5", c]       = f"{100 * df.loc['cvar_5', c]:.2f}%"
            fmt.loc["skewness", c]     = f"{df.loc['skewness', c]:.2f}"
            fmt.loc["kurtosis", c]     = f"{df.loc['kurtosis', c]:.2f}"
            fmt.loc["n_days", c]       = f"{int(df.loc['n_days', c])}"

        print("Performance Metrics")
        print(fmt)
        print("=" * 50)
    
    def simulate_block_bootstrap_mc(
        self,
        n_sims=2000,
        block_len=None,
        seed=42,
        include_6040=True,
        include_regime_strategy=True,
        weights_df=None
    ):
        print("Simulating Block Bootstrap Monte Carlo...")
        if self.features is None:
            raise ValueError("Run preprocess_features() first.")

        rng = np.random.default_rng(seed)
        asset_cols = ["ret_SPY", "ret_BONDS", "ret_GOLD", "ret_USD"]
        R_is = self.features.loc[:self.end_date, asset_cols].dropna()
        if R_is.empty:
            raise ValueError("No IS returns available for bootstrap.")

        R_oos = self.features.loc[self.end_date:, asset_cols].dropna()
        horizon_days = len(R_oos)
        oos_index = R_oos.index

        T = len(R_is)
        if block_len is None:
            block_len = max(5, int(round(1.5 * (T ** (1/3)))))

        starts = np.arange(0, T - block_len + 1)
        sims = {}

        if include_6040:
            w6040 = np.array([0.6, 0.4, 0.0, 0.0])
            sim_mat = np.zeros((horizon_days, n_sims))
            for s in range(n_sims):
                chunks = []
                while len(chunks) * block_len < horizon_days:
                    st = int(rng.choice(starts))
                    chunks.append(R_is.iloc[st:st + block_len].values)
                path = np.vstack(chunks)[:horizon_days]
                sim_mat[:, s] = path @ w6040
            sims["60_40"] = pd.DataFrame(sim_mat, index=oos_index,
                                          columns=[f"sim_{i}" for i in range(n_sims)])

        if include_regime_strategy:
            if weights_df is None:
                weights_df = self.compute_all_regime_weights()
            weights_df = weights_df.copy()
            weights_df.columns = weights_df.columns.astype(str)

            reg_is = self.regimes.loc[:self.end_date, "regime"].dropna().astype(int)
            asset_order = ["SPY", "BONDS", "GOLD", "USD"]

            sim_mat = np.zeros((horizon_days, n_sims))
            for s in range(n_sims):
                chunks = []
                regime_weights = []
                while len(chunks) * block_len < horizon_days:
                    st = int(rng.choice(starts))
                    chunk = R_is.iloc[st:st + block_len].values
                    regime_chunk = reg_is.iloc[st:st + block_len].values
                    chunks.append(chunk)
                    regime_weights.extend(regime_chunk)
                path = np.vstack(chunks)[:horizon_days]
                regime_weights = regime_weights[:horizon_days]

                port_returns = []
                for t, row in enumerate(path):
                    regime = str(regime_weights[t])
                    if regime in weights_df.columns:
                        w = weights_df[regime].reindex(asset_order).values
                        port_returns.append(np.dot(row, w))
                    else:
                        port_returns.append(np.nan)

                sim_mat[:, s] = port_returns
            sims["regime_max_sharpe"] = pd.DataFrame(sim_mat, index=oos_index,
                                                      columns=[f"sim_{i}" for i in range(n_sims)])
        print("Simulation complete")
        print("="*50)
        return sims

    def summarize_mc(self, sim_dict, rf_annual=0.0, mar_annual=0.0):
        def _metrics_for_each(df):
            met = {}
            for c in df.columns:
                met[c] = self._series_metrics(df[c], rf_annual=rf_annual, mar_annual=mar_annual)
            return pd.DataFrame(met)

        def _summary(M):
            return pd.DataFrame({
                "mean": M.mean(axis=1),
                "std":  M.std(axis=1),
                "p05":  M.quantile(0.05, axis=1),
                "p25":  M.quantile(0.25, axis=1),
                "p50":  M.quantile(0.50, axis=1),
                "p75":  M.quantile(0.75, axis=1),
                "p95":  M.quantile(0.95, axis=1),
            })

        metrics_raw = {}
        metrics_summary = {}
        for name, sims in sim_dict.items():
            M = _metrics_for_each(sims)
            metrics_raw[name] = M
            metrics_summary[name] = _summary(M)
        
        full_summary_table = pd.concat(metrics_summary, axis=1)
        print(full_summary_table.round(4))
        print("="*50)
        full_summary_table.to_csv("outputs/mc_summary_table.csv")
        return {"metrics_raw": metrics_raw, "metrics_summary": metrics_summary}

    def compare_mc_vs_actual_oos(self, sim_dict, rf_annual=0.0, mar_annual=0.0):
        oos_returns_df = self.oos_returns.copy()

        fields = [
            "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "var_5", "cvar_5", "skewness", "kurtosis"
        ]

        report = {}
        for name, sims in sim_dict.items():
            if name not in oos_returns_df.columns:
                continue

            combined = sims.join(oos_returns_df[[name]], how="inner").dropna()
            sims_aligned = combined.drop(columns=name)
            r_actual = combined[name]

            a = self._series_metrics(r_actual, rf_annual=rf_annual, mar_annual=mar_annual)
            S = {c: self._series_metrics(sims_aligned[c], rf_annual=rf_annual, mar_annual=mar_annual)
                for c in sims_aligned.columns}
            M = pd.DataFrame(S)

            pct = {f: float((M.loc[f].values < a[f]).mean()) for f in fields}
            report[name] = {"actual": a, "mc_percentile": pct}

            print("="*50)
            print(f"Monte Carlo vs Actual OOS Performance: **{name.upper()}**")
            for k in fields:
                val = a[k]
                pv = 100 * pct[k]
                if isinstance(val, float):
                    print(f" {k:<15}: Actual = {val:>8.4f} | MC Percentile Rank = {pv:>6.2f}%")
                else:
                    print(f" {k:<15}: Actual = {val}       | MC Percentile Rank = {pv:>6.2f}%")
            print("="*50)
        return report

    def plot_mc_wealth_fan(self, sim_df, strategy_name=None):
        sim_df = sim_df.copy()[strategy_name]
        oos_series = self.oos_returns[strategy_name]

        wealth = (1.0 + sim_df.fillna(0)).cumprod()
        p05 = wealth.quantile(0.05, axis=1)
        p50 = wealth.quantile(0.50, axis=1)
        p95 = wealth.quantile(0.95, axis=1)

        plt.figure(figsize=(12, 6))
        plt.fill_between(wealth.index, p05.values, p95.values, alpha=0.25, label="MC 5–95% band")
        plt.plot(wealth.index, p50.values, linewidth=2, label="MC median")

        if oos_series is not None:
            oos_w = (1.0 + oos_series.reindex(wealth.index).fillna(0)).cumprod()
            plt.plot(oos_w.index, oos_w.values, linewidth=2, linestyle="--", label="Realized OOS")

        plt.title(strategy_name.upper() + " - Block Bootstrap MC vs Realized OOS")
        plt.ylabel("Wealth (Initial=1.0)")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_mc_sharpe_hist(self, sim_df, strategy_name=None, rf_annual=0.0, mar_annual=0.0, bins=40):
        sim_df = sim_df.copy()[strategy_name]
        oos_series = self.oos_returns[strategy_name]

        stats = {c: self._series_metrics(sim_df[c], rf_annual=rf_annual, mar_annual=mar_annual)
                for c in sim_df.columns}
        M = pd.DataFrame(stats)
        sharpe_vals = M.loc["sharpe"].astype(float).values

        plt.figure(figsize=(8, 5))
        plt.hist(sharpe_vals, bins=bins, alpha=0.85)
        plt.title(strategy_name.upper() + " - MC Sharpe Distribution")
        plt.xlabel("Sharpe")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        if oos_series is not None:
            a = self._series_metrics(oos_series.reindex(sim_df.index).dropna(),
                                    rf_annual=rf_annual, mar_annual=mar_annual)
            plt.axvline(a["sharpe"], linewidth=2, linestyle="--", label=f"OOS Sharpe = {a['sharpe']:.2f}")
            plt.legend()

        plt.tight_layout()
        plt.show()
