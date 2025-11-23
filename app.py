import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.lines import Line2D
from pathlib import Path
from urllib.request import urlretrieve
from typing import Dict, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 100

LEGEND_BAD = "|拟合-期货| > 70%"

st.set_page_config(page_title="多资产隐含降息次数与资产价格对照", layout="wide")

START_DATE = "2015-10-01"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FONTS_DIR = BASE_DIR / "fonts"

ASSET_CONFIG = {
    "GOLD": {"label": "黄金(现货)", "type": "yfinance", "ticker": "GC=F"},
    "SP500": {"label": "标普500指数", "type": "yfinance", "ticker": "^GSPC"},
    "UST3M": {"label": "美国3M国债收益率", "type": "yfinance", "ticker": "^IRX"},
    "UST10Y": {"label": "美国10Y国债收益率", "type": "yfinance", "ticker": "^TNX"},
    "DXY": {"label": "美元指数", "type": "yfinance", "ticker": "DX-Y.NYB"},
    "USDCNY": {"label": "美元兑人民币", "type": "yfinance", "ticker": "USDCNY=X"},
}

CANDIDATES = {
    "Linear": make_pipeline(PolynomialFeatures(degree=1, include_bias=False), LinearRegression()),
    "Quadratic": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
    "Cubic": make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression()),
}


def select_single_file(pattern: str) -> Path:
    matches = sorted(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"未找到匹配 {pattern} 的文件")
    return matches[0]


def ensure_cn_font():
    """
    确保 Matplotlib 有可用的中文字体，并同步到 rcParams。
    优先使用本地 fonts 目录下的 NotoSansSC-Regular.otf，不存在时自动下载一次。
    """
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    target = FONTS_DIR / "NotoSansSC-Regular.otf"
    if not target.exists():
        url = (
            "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/"
            "SimplifiedChinese/NotoSansSC-Regular.otf"
        )
        urlretrieve(url, target)
    fm.fontManager.addfont(str(target))
    plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Arial Unicode MS",
        "sans-serif",
    ]


@st.cache_data(show_spinner=False)
def load_reference_data():
    decisions_path = select_single_file("FOMC_decisions*.csv")
    meetings_path = select_single_file("FOMC_Meeting*.xlsx")

    zq = yf.download("ZQ=F", start=START_DATE, progress=False, auto_adjust=False)
    if zq.empty:
        raise ValueError("无法获取 ZQ=F 期货数据")
    zq = zq[["Close"]].dropna()
    zq.index = pd.to_datetime(zq.index).tz_localize(None)
    zq["implied_rate"] = 100.0 - zq["Close"]
    zq = zq.loc[zq.index >= START_DATE]

    meetings_raw = pd.read_excel(meetings_path)

    def parse_meeting_range(value):
        if isinstance(value, pd.Timestamp):
            ts = value.normalize()
            return ts, ts
        text = str(value)
        if "to" in text:
            start_str, end_str = text.split("to", 1)
        else:
            start_str = end_str = text
        start_ts = pd.to_datetime(start_str.strip()).normalize()
        end_ts = pd.to_datetime(end_str.strip()).normalize()
        return start_ts, end_ts

    meeting_ranges = meetings_raw["meeting_date"].apply(parse_meeting_range)
    meetings = (
        pd.DataFrame(meeting_ranges.tolist(), columns=["meeting_start", "meeting_end"])
        .sort_values("meeting_end")
        .reset_index(drop=True)
    )

    decisions = (
        pd.read_csv(decisions_path, parse_dates=["date"])
        .rename(columns={"date": "decision_date"})
        .sort_values("decision_date")
    )
    for col in ["change_bps", "level_mid", "level_low", "level_high"]:
        if col in decisions.columns:
            decisions[col] = pd.to_numeric(decisions[col], errors="coerce")
    decisions["actual_change"] = decisions["change_bps"] / 25.0
    decisions = decisions.loc[decisions["actual_change"] != 0].reset_index(drop=True)

    aligned = pd.merge_asof(
        decisions,
        meetings[["meeting_start", "meeting_end"]],
        left_on="decision_date",
        right_on="meeting_end",
        direction="nearest",
        tolerance=pd.Timedelta(days=7),
    )
    aligned["meeting_date"] = aligned["meeting_end"].fillna(aligned["decision_date"])
    aligned["meeting_month"] = aligned["meeting_date"].dt.to_period("M")
    aligned = aligned.loc[aligned["meeting_month"] >= pd.Period(START_DATE, "M")]

    monthly_events = (
        aligned.sort_values("meeting_date")
        .groupby("meeting_month")
        .agg(
            meeting_date=("meeting_date", "max"),
            actual_change_total=("actual_change", "sum"),
            actual_rate_post=("level_mid", "last"),
        )
        .reset_index(drop=True)
        .sort_values("meeting_date")
    )
    monthly_events["actual_change_total"] = monthly_events["actual_change_total"].astype(float)
    monthly_events["actual_rate_post"] = monthly_events["actual_rate_post"].astype(float)
    fallback_pre = monthly_events["actual_rate_post"] - monthly_events["actual_change_total"] * 0.25
    monthly_events["actual_rate_pre"] = monthly_events["actual_rate_post"].shift(1)
    monthly_events["actual_rate_pre"] = monthly_events["actual_rate_pre"].fillna(fallback_pre)
    monthly_events["actual_rate_pre"] = monthly_events["actual_rate_pre"].astype(float)
    meeting_base = monthly_events.copy()

    def implied_rate_at(ts: pd.Timestamp, mode: str = "pre") -> float:
        ts = pd.Timestamp(ts)
        if mode == "pre":
            values = zq["implied_rate"].loc[zq.index < ts]
            return float(values.iloc[-1]) if not values.empty else np.nan
        values = zq["implied_rate"].loc[zq.index >= ts]
        return float(values.iloc[0]) if not values.empty else np.nan

    meeting_base["implied_rate_pre"] = meeting_base["meeting_date"].apply(
        lambda d: implied_rate_at(d, "pre")
    )
    meeting_base["implied_rate_post"] = meeting_base["meeting_date"].apply(
        lambda d: implied_rate_at(d, "post")
    )
    meeting_base["expected_change_pre_quarters"] = (
        meeting_base["implied_rate_pre"] - meeting_base["actual_rate_pre"]
    ) / 0.25
    meeting_base["expected_change_post_quarters"] = (
        meeting_base["implied_rate_post"] - meeting_base["actual_rate_post"]
    ) / 0.25
    meeting_base["implied_change_quarters"] = meeting_base["expected_change_pre_quarters"]

    asset_prices: Dict[str, pd.Series] = {}
    for key, cfg in ASSET_CONFIG.items():
        if cfg.get("type") != "yfinance":
            continue
        data = yf.download(cfg["ticker"], start=START_DATE, progress=False, auto_adjust=False)
        if data.empty:
            continue
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        series = data[col].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.loc[series.index >= START_DATE]
        series = series.asfreq("D").ffill().dropna()
        asset_prices[key] = series

    if not asset_prices:
        raise RuntimeError("未能加载任何资产行情")

    return meeting_base, asset_prices, zq


def price_snapshots(series: pd.Series, ts: pd.Timestamp) -> Tuple[float, float, float]:
    ts = pd.Timestamp(ts)
    before = series.loc[series.index < ts]
    after = series.loc[series.index >= ts]
    price_pre = float(before.iloc[-1]) if not before.empty else np.nan
    price_post = float(after.iloc[0]) if not after.empty else np.nan
    avg_price = np.nanmean([price_pre, price_post])
    return price_pre, price_post, avg_price


def build_asset_dataset(series: pd.Series, meeting_base: pd.DataFrame) -> pd.DataFrame:
    df = meeting_base.copy()
    snapshots = df["meeting_date"].apply(lambda d: price_snapshots(series, d))
    df["price_pre"] = [item[0] for item in snapshots]
    df["price_post"] = [item[1] for item in snapshots]
    df["avg_price_around"] = [item[2] for item in snapshots]
    df["price_change"] = df["price_post"] - df["price_pre"]
    df = df.dropna(subset=["avg_price_around", "implied_change_quarters"]).reset_index(drop=True)
    df = df.loc[df["avg_price_around"] > 0].copy()
    return df


def fit_candidate_models(df: pd.DataFrame):
    X = df[["implied_change_quarters"]].values
    y = np.log(df["avg_price_around"].values)
    summaries = []
    for name, estimator in CANDIDATES.items():
        estimator.fit(X, y)
        preds = estimator.predict(X)
        r2 = r2_score(y, preds)
        summaries.append({"name": name, "estimator": estimator, "predictions": preds, "r_squared": r2})
    best = max(summaries, key=lambda item: item["r_squared"])
    return best


def analyze_assets(meeting_base: pd.DataFrame, asset_prices: Dict[str, pd.Series]):
    asset_results = {}
    for key, cfg in ASSET_CONFIG.items():
        series = asset_prices.get(key)
        if series is None:
            continue
        df = build_asset_dataset(series, meeting_base)
        if df.empty:
            continue
        best_model = fit_candidate_models(df)
        df["predicted_log_price"] = best_model["predictions"]
        df["predicted_price"] = np.exp(best_model["predictions"])
        df["best_model_name"] = best_model["name"]
        asset_results[key] = {
            "data": df,
            "best_model": best_model,
            "label": cfg["label"],
            "series": series,
        }
    if not asset_results:
        raise RuntimeError("所有资产都缺少有效结果")
    return asset_results


def compute_cubic_summary(asset_results, meeting_base: pd.DataFrame, zq: pd.DataFrame):
    latest_implied_rate = float(zq["implied_rate"].iloc[-1])
    current_actual_rate = float(meeting_base["actual_rate_post"].dropna().iloc[-1])
    futures_implied_change = (latest_implied_rate - current_actual_rate) / 0.25
    futures_implied_change = float(np.clip(futures_implied_change, -8, 8))

    cubic_summary_rows = []
    for key, result in asset_results.items():
        if key == "DXY":
            continue
        df_asset = result["data"]
        if df_asset.empty:
            continue
        series = result["series"].dropna()
        if series.empty:
            continue
        X_vals = df_asset[["implied_change_quarters"]].values
        y_vals = np.log(df_asset["avg_price_around"].values)
        cubic_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
        cubic_model.fit(X_vals, y_vals)
        lr_step = cubic_model.named_steps["linearregression"]
        coef = lr_step.coef_
        intercept = float(lr_step.intercept_)
        function_str = (
            f"log(price) = {intercept:.4f} + {coef[0]:.4f}·x + {coef[1]:.4f}·x² + {coef[2]:.4f}·x³"
        )
        latest_price = float(series.iloc[-1])
        latest_date = pd.Timestamp(series.index[-1]).date()
        log_target = np.log(latest_price)
        poly_coeffs = np.array([coef[2], coef[1], coef[0], intercept - log_target], dtype=float)
        coeffs_trimmed = np.trim_zeros(poly_coeffs, trim="f")
        implied_today = np.nan
        if coeffs_trimmed.size >= 2:
            roots = np.roots(coeffs_trimmed)
            real_roots = [root.real for root in roots if abs(root.imag) < 1e-6]
            ref_value = df_asset["implied_change_quarters"].iloc[-1]
            if real_roots:
                implied_today = min(real_roots, key=lambda r: abs(r - ref_value))
            else:
                implied_today = min(roots, key=lambda r: abs(r.imag)).real
            implied_today = float(np.clip(implied_today, -8, 8))
        cubic_summary_rows.append(
            {
                "资产": result["label"],
                "Cubic函数": function_str,
                "当日资产价格": latest_price,
                "拟合隐含降息次数": implied_today,
                "期货隐含降息次数": futures_implied_change,
                "当日日期": latest_date,
            }
        )
    cubic_summary_df = pd.DataFrame(cubic_summary_rows)
    return cubic_summary_df


def scale_cubic_summary(cubic_summary_df: pd.DataFrame) -> pd.DataFrame:
    if cubic_summary_df.empty:
        return cubic_summary_df
    values = cubic_summary_df["拟合隐含降息次数"].astype(float)
    if values.nunique() <= 1 or np.isclose(values.max() - values.min(), 0.0):
        normalized = pd.Series(0.0, index=cubic_summary_df.index)
    else:
        normalized = 2 * (values - values.min()) / (values.max() - values.min()) - 1
    mu_original = float(normalized.mean()) if not normalized.empty else 0.0
    target_center = float(cubic_summary_df["期货隐含降息次数"].mean())
    delta = target_center - mu_original
    scaled = normalized + delta
    scaled_df = cubic_summary_df.copy()
    scaled_df["拟合隐含降息次数（负=降息，正=加息）"] = scaled
    scaled_df.drop(columns=["拟合隐含降息次数"], inplace=True)
    return scaled_df


def compute_daily_fit_summary(asset_prices: Dict[str, pd.Series], zq: pd.DataFrame) -> pd.DataFrame:
    rate_series = zq["implied_rate"].dropna()
    if rate_series.empty:
        return pd.DataFrame()
    latest_rate = float(rate_series.iloc[-1])
    latest_rate_date = pd.Timestamp(rate_series.index[-1]).date()
    rows = []
    for key, cfg in ASSET_CONFIG.items():
        if key == "DXY":
            continue
        price_series = asset_prices.get(key)
        if price_series is None:
            continue
        series_obj = price_series.iloc[:, 0] if isinstance(price_series, pd.DataFrame) else price_series
        series_obj = series_obj.dropna()
        if series_obj.empty:
            continue
        df_daily = pd.concat(
            [rate_series.rename("implied_rate"), series_obj.rename("price")], axis=1, join="inner"
        ).dropna()
        df_daily = df_daily.loc[df_daily["price"] > 0]
        if df_daily.empty:
            continue
        X = df_daily[["implied_rate"]].values
        y = np.log(df_daily["price"].values)
        cubic_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
        cubic_model.fit(X, y)
        lr_step = cubic_model.named_steps["linearregression"]
        coef = lr_step.coef_
        intercept = float(lr_step.intercept_)
        function_str = (
            f"log(price) = {intercept:.4f} + {coef[0]:.4f}·r + {coef[1]:.4f}·r² + {coef[2]:.4f}·r³"
        )
        latest_price = float(series_obj.iloc[-1])
        latest_date = pd.Timestamp(series_obj.index[-1]).date()
        log_target = np.log(latest_price)
        poly_coeffs = np.array([coef[2], coef[1], coef[0], intercept - log_target], dtype=float)
        coeffs_trimmed = np.trim_zeros(poly_coeffs, trim="f")
        implied_rate_fit = np.nan
        if coeffs_trimmed.size >= 2:
            roots = np.roots(coeffs_trimmed)
            real_roots = [root.real for root in roots if abs(root.imag) < 1e-6]
            ref_rate = float(df_daily["implied_rate"].iloc[-1])
            if real_roots:
                implied_rate_fit = min(real_roots, key=lambda r: abs(r - ref_rate))
            else:
                implied_rate_fit = min(roots, key=lambda r: abs(r.imag)).real
        rows.append(
            {
                "资产": cfg["label"],
                "Cubic函数": function_str,
                "当日资产价格": latest_price,
                "拟合隐含利率": implied_rate_fit,
                "期货隐含利率": latest_rate,
                "当日资产对应日期": latest_date,
                "期货隐含利率对应日期": latest_rate_date,
            }
        )
    return pd.DataFrame(rows)


def plot_sample_windows(samples, asset_daily):
    n = len(samples)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")

    for i, row in enumerate(samples):
        ax = axes[i]
        info = asset_daily.get(row["asset_key"])
        if info is None:
            ax.axis("off")
            continue
        series_obj = info["series"]
        dt = pd.Timestamp(row["日期"])
        window = series_obj.loc[dt - pd.Timedelta(days=10) : dt + pd.Timedelta(days=10)]
        if window.empty:
            ax.axis("off")
            continue
        color = "green" if row["偏差是否在70%内"] == "是" else "red"
        ax.plot(window.index, window.values, color=color, alpha=0.75)
        if dt in window.index:
            ax.scatter([dt], [window.loc[dt]], color="black", marker="o", zorder=5, s=30)
            ax.axvline(dt, color="gray", linestyle="--", alpha=0.4)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        if color == "green":
            legend_label = "|拟合-期货| <= 70%"
        else:
            fit_val = f"{row['拟合隐含利率']:.3f}"
            fut_val = f"{row['期货隐含利率']:.3f}"
            legend_label = f"{LEGEND_BAD} ({fit_val}-{fut_val})"
        handles = [
            Line2D([0], [0], color=color, lw=2, label=legend_label),
            Line2D([0], [0], color="black", marker="o", linestyle="", label="样本点"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
        ax.set_title(f"{row['资产']} {dt.date()}", fontsize=9)
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.tight_layout()
    return fig


def compute_daily_samples(asset_prices: Dict[str, pd.Series], zq: pd.DataFrame):
    if not asset_prices or zq.empty or "implied_rate" not in zq:
        return pd.DataFrame(), None
    ensure_cn_font()

    rate_series = zq["implied_rate"].dropna().rename("implied_rate")
    if rate_series.empty:
        return pd.DataFrame(), None

    rng = np.random.default_rng(42)
    asset_daily = {}
    for key, cfg in ASSET_CONFIG.items():
        if key == "DXY":
            continue
        price_series = asset_prices.get(key)
        if price_series is None:
            continue
        series_obj = price_series.iloc[:, 0] if isinstance(price_series, pd.DataFrame) else price_series
        series_obj = series_obj.dropna()
        if series_obj.empty:
            continue
        df_daily = pd.concat(
            [rate_series, series_obj.rename("price")],
            axis=1,
            join="inner",
        ).dropna()
        df_daily = df_daily.loc[df_daily["price"] > 0]
        if df_daily.empty:
            continue
        X = df_daily[["implied_rate"]].values
        y = np.log(df_daily["price"].values)
        model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
        model.fit(X, y)
        lr_step = model.named_steps["linearregression"]
        asset_daily[key] = {
            "label": cfg["label"],
            "df": df_daily,
            "coef": lr_step.coef_,
            "intercept": float(lr_step.intercept_),
            "series": series_obj,
        }

    if not asset_daily:
        return pd.DataFrame(), None

    years = sorted(rate_series.index.year.unique())
    chosen_dates = []
    for year in years:
        year_dates = rate_series.index[rate_series.index.year == year]
        if year_dates.empty:
            continue
        dt = pd.Timestamp(rng.choice(year_dates))
        chosen_dates.append(dt)
        if len(chosen_dates) >= 10:
            break

    samples = []
    for dt in chosen_dates:
        for key, info in asset_daily.items():
            df_daily = info["df"]
            if dt not in df_daily.index:
                continue
            futures_rate = float(df_daily.loc[dt, "implied_rate"])
            price_today = float(df_daily.loc[dt, "price"])
            log_target = np.log(price_today)
            coef = info["coef"]
            intercept = info["intercept"]
            poly_coeffs = np.array([coef[2], coef[1], coef[0], intercept - log_target], dtype=float)
            coeffs_trimmed = np.trim_zeros(poly_coeffs, trim="f")
            implied_rate_fit = np.nan
            if coeffs_trimmed.size >= 2:
                roots = np.roots(coeffs_trimmed)
                real_roots = [root.real for root in roots if abs(root.imag) < 1e-6]
                ref_rate = float(df_daily["implied_rate"].iloc[-1])
                if real_roots:
                    implied_rate_fit = min(real_roots, key=lambda r: abs(r - ref_rate))
                else:
                    implied_rate_fit = min(roots, key=lambda r: abs(r.imag)).real
            if np.isnan(implied_rate_fit):
                continue
            threshold = 0.7 * abs(futures_rate)
            within_band = abs(implied_rate_fit - futures_rate) <= threshold
            samples.append(
                {
                    "asset_key": key,
                    "资产": info["label"],
                    "日期": dt,
                    "价格": price_today,
                    "拟合隐含利率": float(implied_rate_fit),
                    "期货隐含利率": futures_rate,
                    "偏差是否在70%内": "是" if within_band else "否",
                }
            )

    if not samples:
        return pd.DataFrame(), None

    if len(samples) > 50:
        samples = samples[:50]

    sampled_df = pd.DataFrame(samples)
    fig = plot_sample_windows(samples, asset_daily)
    return sampled_df, fig


def main():
    st.title("多资产隐含降息次数与日度拟合结果")
    st.caption("数据来源：Yahoo Finance / FOMC 决议文件，支持 `streamlit run app.py --server.address 0.0.0.0` 远程访问。")
    ensure_cn_font()
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Noto Sans SC","Microsoft YaHei","PingFang SC","SimHei",sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("加载数据并计算模型…"):
        meeting_base, asset_prices, zq = load_reference_data()
        asset_results = analyze_assets(meeting_base, asset_prices)
        cubic_summary_df = compute_cubic_summary(asset_results, meeting_base, zq)
        scaled_cubic_df = scale_cubic_summary(cubic_summary_df)
        daily_fit_summary = compute_daily_fit_summary(asset_prices, zq)

    st.subheader("缩放后隐含降息次数（负=降息，正=加息）")
    if scaled_cubic_df.empty:
        st.info("暂无可显示的 cubic 汇总结果。")
    else:
        st.dataframe(scaled_cubic_df, use_container_width=True)

    st.subheader("日度隐含利率拟合概览（Cubic 最优）")
    if daily_fit_summary.empty:
        st.info("暂无可显示的日度拟合结果。")
    else:
        st.dataframe(daily_fit_summary, use_container_width=True)

    st.subheader("日度抽样走势与拟合偏差")
    sampled_df, sampled_fig = compute_daily_samples(asset_prices, zq)
    if sampled_df.empty or sampled_fig is None:
        st.info("缺少资产价格数据或抽样结果，无法绘制。")
    else:
        st.dataframe(sampled_df.drop(columns=["asset_key"]), use_container_width=True)
        st.pyplot(sampled_fig, clear_figure=True)


if __name__ == "__main__":
    main()
