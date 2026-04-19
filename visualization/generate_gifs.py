"""
Generate README GIFs showcasing institutional features.
Each function creates a polished animated GIF for the README.
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.engine import BacktestEngine, Strategy
from screener.strategies import (
    momentum_crossover_factory, bollinger_factory, momentum_crossover,
    mean_reversion_bollinger, regime_adaptive,
)
from risk.tail_risk import TailRiskScreener
from risk.drift_detector import DriftDetector
from signals.alpha_engine import AlphaSignalEngine

logging.basicConfig(level=logging.WARNING)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Consistent styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "savefig.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#30363d",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})


def _make_synthetic_prices(n=504, seed=42, regime_change=True):
    """Generate realistic synthetic OHLCV with regime change."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)

    returns = np.zeros(n)
    if regime_change:
        returns[:250] = np.random.normal(0.0008, 0.012, 250)    # bull
        returns[250:380] = np.random.normal(-0.0005, 0.025, 130)  # volatile
        returns[380:] = np.random.normal(0.0010, 0.014, n - 380)  # recovery
    else:
        returns = np.random.normal(0.0005, 0.015, n)

    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "close": close,
        "high": close * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "open": close * (1 + np.random.normal(0, 0.003, n)),
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


# ═══════════════════════════════════════════════════════════════════════
# GIF 1: Time-Evolving 3D PnL Surface
# ═══════════════════════════════════════════════════════════════════════

def gif_pnl_surface_animated(output_path=None, fps=2):
    """Animated 3D PnL surface evolving through time with rotation."""
    output_path = output_path or os.path.join(ASSETS_DIR, "pnl_surface_3d.gif")
    prices = _make_synthetic_prices()
    engine = BacktestEngine()

    # Generate surfaces for 6 time windows
    n_periods = 6
    window = len(prices) // 2
    step = (len(prices) - window) // (n_periods - 1)
    fast_grid = [5, 8, 12, 16, 20, 25]
    slow_grid = [25, 35, 45, 55, 65]

    time_surfaces = []
    for i in range(n_periods):
        start = i * step
        end = start + window
        if end > len(prices):
            break
        window_prices = prices.iloc[start:end]
        label = str(window_prices.index[-1].date())
        try:
            result = engine.parameter_sweep(
                window_prices, signal_fn_factory=momentum_crossover_factory,
                param_grid={"fast": fast_grid, "slow": slow_grid},
            )
            pivot = result.pnl_by_param.pivot_table(
                values="sharpe", index="fast", columns="slow"
            )
            time_surfaces.append((pivot, label))
        except ValueError:
            continue

    if not time_surfaces:
        print("No surfaces generated")
        return None

    # Global z-limits for consistent scale
    z_min = min(np.nanmin(p.values) for p, _ in time_surfaces)
    z_max = max(np.nanmax(p.values) for p, _ in time_surfaces)

    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Total frames = periods x rotation steps
    rotation_steps = 8
    total_frames = len(time_surfaces) * rotation_steps

    def update(frame_idx):
        ax.clear()
        period_idx = frame_idx // rotation_steps
        rot_idx = frame_idx % rotation_steps
        period_idx = min(period_idx, len(time_surfaces) - 1)

        pivot, label = time_surfaces[period_idx]
        X, Y = np.meshgrid(pivot.columns.values.astype(float),
                           pivot.index.values.astype(float))
        Z = pivot.values.astype(float)

        # Plot surface
        surf = ax.plot_surface(
            X, Y, Z, cmap="RdYlGn", alpha=0.9,
            vmin=z_min, vmax=z_max,
            edgecolor="#30363d", linewidth=0.3, antialiased=True,
        )

        # Highlight best point
        best_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.scatter([X[best_idx]], [Y[best_idx]], [Z[best_idx]],
                   color="gold", s=150, marker="*",
                   edgecolor="white", linewidths=1.5, zorder=10)

        # Rotating view
        azim = -60 + rot_idx * 10
        ax.view_init(elev=25, azim=azim)

        ax.set_xlabel("Slow MA", color="#c9d1d9", labelpad=8)
        ax.set_ylabel("Fast MA", color="#c9d1d9", labelpad=8)
        ax.set_zlabel("Sharpe Ratio", color="#c9d1d9", labelpad=8)
        ax.set_zlim(z_min, z_max)
        ax.set_title(
            f"Time-Evolving PnL Surface · Period: {label}\n"
            f"Best Sharpe: {Z[best_idx]:.2f}  ·  Optimal Params: "
            f"fast={int(Y[best_idx])}, slow={int(X[best_idx])}",
            color="#f0f6fc", fontsize=12, pad=18, fontweight="bold",
        )
        ax.tick_params(colors="#8b949e")
        ax.xaxis.set_pane_color((0.05, 0.07, 0.09, 1.0))
        ax.yaxis.set_pane_color((0.05, 0.07, 0.09, 1.0))
        ax.zaxis.set_pane_color((0.05, 0.07, 0.09, 1.0))
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 // fps, blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 2: Strategy Equity Curves (animated progression)
# ═══════════════════════════════════════════════════════════════════════

def gif_equity_curves(output_path=None, fps=20):
    """Animated equity curves progressing through time."""
    output_path = output_path or os.path.join(ASSETS_DIR, "equity_curves.gif")
    prices = _make_synthetic_prices()
    engine = BacktestEngine()

    strategies = {
        "Momentum 10/30": Strategy("Momentum", momentum_crossover(10, 30)),
        "Bollinger 20/2": Strategy("Bollinger", mean_reversion_bollinger(20, 2.0)),
        "Regime Adaptive": Strategy("Regime", regime_adaptive()),
    }
    colors = {"Momentum 10/30": "#58a6ff", "Bollinger 20/2": "#f85149",
              "Regime Adaptive": "#3fb950"}

    equity_curves = {}
    for name, strat in strategies.items():
        result = engine.run(prices, strat)
        equity_curves[name] = result.equity_curve

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    n_frames = 80
    dates = prices.index

    def update(frame_idx):
        ax.clear()
        progress = int(len(dates) * (frame_idx + 1) / n_frames)
        progress = max(progress, 5)

        # Buy-and-hold benchmark
        bh = 100_000 * (prices["close"] / prices["close"].iloc[0])
        ax.plot(bh.index[:progress], bh.values[:progress],
                color="#8b949e", linestyle="--", linewidth=1.5,
                label="Buy & Hold", alpha=0.7)

        max_val = 100_000
        for name, eq in equity_curves.items():
            ax.plot(eq.index[:progress], eq.values[:progress],
                    color=colors[name], linewidth=2.2, label=name)
            # Add endpoint marker
            if progress > 0:
                ax.scatter([eq.index[progress-1]], [eq.values[progress-1]],
                           color=colors[name], s=60, zorder=5,
                           edgecolor="white", linewidths=1)
            max_val = max(max_val, eq.values[:progress].max())

        ax.axhline(100_000, color="#30363d", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Date", color="#c9d1d9")
        ax.set_ylabel("Portfolio Value ($)", color="#c9d1d9")
        ax.set_title("Strategy Equity Curves Evolution",
                     color="#f0f6fc", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(70_000, max(max_val * 1.05, 110_000))
        ax.grid(True, alpha=0.3, color="#30363d")
        ax.legend(loc="upper left", facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Format y-axis as dollars
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 3: Risk Dashboard (multi-panel animated)
# ═══════════════════════════════════════════════════════════════════════

def gif_risk_dashboard(output_path=None, fps=12):
    """Animated 4-panel risk dashboard evolving over time."""
    output_path = output_path or os.path.join(ASSETS_DIR, "risk_dashboard.gif")
    prices = _make_synthetic_prices()
    tail = TailRiskScreener()

    returns = prices["close"].pct_change().dropna()
    equity = 100_000 * (1 + returns).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor("#0d1117")

    n_frames = 60
    min_obs = 60  # minimum observations before animating

    def update(frame_idx):
        progress = min_obs + int((len(returns) - min_obs) * (frame_idx + 1) / n_frames)
        r = returns.iloc[:progress]
        eq = equity.iloc[:progress]

        # ── Panel 1: Equity + Drawdown overlay ──────────────────────
        ax1 = axes[0, 0]
        ax1.clear()
        ax1.set_facecolor("#161b22")
        ax1.plot(eq.index, eq.values, color="#58a6ff", linewidth=2)
        ax1.fill_between(eq.index, eq.values, 100_000,
                         where=(eq.values >= 100_000),
                         color="#3fb950", alpha=0.15)
        ax1.fill_between(eq.index, eq.values, 100_000,
                         where=(eq.values < 100_000),
                         color="#f85149", alpha=0.15)
        ax1.axhline(100_000, color="#30363d", linestyle=":", alpha=0.7)
        ax1.set_title("Equity Curve", color="#f0f6fc",
                      fontweight="bold", fontsize=11)
        ax1.grid(True, alpha=0.2, color="#30363d")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax1.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363d")

        # ── Panel 2: Drawdown ────────────────────────────────────────
        ax2 = axes[0, 1]
        ax2.clear()
        ax2.set_facecolor("#161b22")
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd.values, 0, color="#f85149", alpha=0.6)
        ax2.plot(dd.index, dd.values, color="#f85149", linewidth=1.5)
        ax2.axhline(-10, color="#d29922", linestyle="--", alpha=0.7,
                    label="Alert: -10%")
        current_dd = dd.iloc[-1] if len(dd) > 0 else 0
        ax2.set_title(f"Drawdown  ·  Current: {current_dd:.1f}%",
                      color="#f0f6fc", fontweight="bold", fontsize=11)
        ax2.grid(True, alpha=0.2, color="#30363d")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.legend(loc="lower left", fontsize=8, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="#c9d1d9")
        ax2.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")

        # ── Panel 3: Rolling Vol (21d + EWMA) ───────────────────────
        ax3 = axes[1, 0]
        ax3.clear()
        ax3.set_facecolor("#161b22")
        vol_21 = r.rolling(21).std() * np.sqrt(252) * 100
        ewma = tail.ewma_volatility(r, annualize=True) * 100
        ax3.plot(vol_21.index, vol_21.values, color="#ff7b72",
                 linewidth=1.8, label="21d Realized")
        ax3.plot(ewma.index, ewma.values, color="#f0883e",
                 linewidth=1.8, label="EWMA (λ=0.94)", alpha=0.9)
        ax3.set_title("Volatility", color="#f0f6fc",
                      fontweight="bold", fontsize=11)
        ax3.grid(True, alpha=0.2, color="#30363d")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax3.legend(loc="upper left", fontsize=8, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="#c9d1d9")
        ax3.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax3.spines.values():
            spine.set_edgecolor("#30363d")

        # ── Panel 4: VaR comparison (4 methods) ──────────────────────
        ax4 = axes[1, 1]
        ax4.clear()
        ax4.set_facecolor("#161b22")
        if len(r) >= 30:
            var_hist = tail.compute_var(r, 0.99, "historical") * 100
            var_param = tail.compute_var(r, 0.99, "parametric") * 100
            var_cf = tail.compute_var(r, 0.99, "cornish_fisher") * 100
            var_evt = tail.compute_var(r, 0.99, "evt") * 100
            cvar = tail.compute_cvar(r, 0.975) * 100

            methods = ["Hist", "Param", "CF", "EVT", "CVaR"]
            values = [var_hist, var_param, var_cf, var_evt, cvar]
            colors_bar = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#f85149"]
            bars = ax4.bar(methods, values, color=colors_bar,
                           edgecolor="#30363d", linewidth=1)
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         val - 0.15, f"{val:.1f}%",
                         ha="center", va="top", color="#c9d1d9", fontsize=9,
                         fontweight="bold")
            ax4.axhline(0, color="#30363d", linewidth=1)
            ax4.set_title("VaR 99% · CVaR 97.5%", color="#f0f6fc",
                          fontweight="bold", fontsize=11)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax4.grid(True, alpha=0.2, color="#30363d", axis="y")
            ax4.tick_params(colors="#8b949e", labelsize=9)
            for spine in ax4.spines.values():
                spine.set_edgecolor("#30363d")

        fig.suptitle(
            f"Institutional Risk Dashboard  ·  Observations: {progress}",
            color="#f0f6fc", fontsize=13, fontweight="bold", y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 4: Regime Detection (price + regime overlay + Hurst exponent)
# ═══════════════════════════════════════════════════════════════════════

def gif_regime_detection(output_path=None, fps=15):
    """Animated price chart with live regime detection overlay."""
    output_path = output_path or os.path.join(ASSETS_DIR, "regime_detection.gif")
    prices = _make_synthetic_prices()
    returns = prices["close"].pct_change().dropna()
    close = prices["close"]

    # Pre-compute regime indicators
    vol_21 = returns.rolling(21).std() * np.sqrt(252)
    vol_63 = returns.rolling(63).std() * np.sqrt(252)
    vol_ratio = (vol_21 / vol_63).replace([np.inf, -np.inf], np.nan)

    # Pre-compute rolling Hurst exponent
    engine_alpha = AlphaSignalEngine()
    hurst_series = pd.Series(np.nan, index=close.index)
    window = 100
    for i in range(window, len(close), 10):
        segment = close.iloc[i - window:i]
        h = engine_alpha.compute_hurst_exponent(segment, max_lag=20)
        hurst_series.iloc[i] = h
    hurst_series = hurst_series.ffill()

    fig, (ax_price, ax_vol, ax_hurst) = plt.subplots(
        3, 1, figsize=(11, 7), dpi=100,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )
    fig.patch.set_facecolor("#0d1117")

    n_frames = 80
    min_obs = 100

    def update(frame_idx):
        progress = min_obs + int((len(close) - min_obs) * (frame_idx + 1) / n_frames)

        # ── Panel 1: Price with regime shading ───────────────────────
        ax_price.clear()
        ax_price.set_facecolor("#161b22")

        c = close.iloc[:progress]
        # Shade regions by vol regime
        vr = vol_ratio.iloc[:progress]
        for i in range(0, len(c) - 1, 5):
            if np.isnan(vr.iloc[i]):
                continue
            color = ("#f85149" if vr.iloc[i] > 1.3
                     else "#d29922" if vr.iloc[i] > 1.0
                     else "#3fb950")
            ax_price.axvspan(c.index[i], c.index[min(i + 5, len(c) - 1)],
                             alpha=0.08, color=color, zorder=0)

        ax_price.plot(c.index, c.values, color="#58a6ff", linewidth=2, zorder=2)
        ax_price.scatter([c.index[-1]], [c.iloc[-1]], color="gold",
                         s=100, zorder=5, edgecolor="white", linewidths=1.5)

        # Current regime annotation
        current_vr = vr.iloc[-1] if len(vr) > 0 and not np.isnan(vr.iloc[-1]) else 1.0
        regime_label = (
            "HIGH VOL" if current_vr > 1.3
            else "ELEVATED" if current_vr > 1.0
            else "LOW VOL"
        )
        regime_color = ("#f85149" if current_vr > 1.3
                        else "#d29922" if current_vr > 1.0
                        else "#3fb950")
        ax_price.text(0.02, 0.95, f"● {regime_label}",
                      transform=ax_price.transAxes, color=regime_color,
                      fontsize=12, fontweight="bold", verticalalignment="top")
        ax_price.text(0.02, 0.88, f"vol ratio: {current_vr:.2f}x",
                      transform=ax_price.transAxes, color="#8b949e",
                      fontsize=9, verticalalignment="top")

        ax_price.set_title("Live Regime Detection",
                           color="#f0f6fc", fontweight="bold", fontsize=13)
        ax_price.set_ylabel("Price", color="#c9d1d9")
        ax_price.grid(True, alpha=0.2, color="#30363d")
        ax_price.set_xlim(close.index[0], close.index[-1])
        ax_price.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_price.spines.values():
            spine.set_edgecolor("#30363d")

        # ── Panel 2: Vol ratio ───────────────────────────────────────
        ax_vol.clear()
        ax_vol.set_facecolor("#161b22")
        vr = vol_ratio.iloc[:progress]
        ax_vol.plot(vr.index, vr.values, color="#f0883e", linewidth=1.8)
        ax_vol.axhline(1.0, color="#30363d", linestyle="-", alpha=0.7)
        ax_vol.axhline(1.3, color="#f85149", linestyle="--", alpha=0.7,
                       label="High vol threshold")
        ax_vol.fill_between(vr.index, vr.values, 1.0,
                            where=(vr.values > 1.0),
                            color="#f85149", alpha=0.2)
        ax_vol.fill_between(vr.index, vr.values, 1.0,
                            where=(vr.values <= 1.0),
                            color="#3fb950", alpha=0.2)
        ax_vol.set_ylabel("Vol Ratio (21d/63d)", color="#c9d1d9", fontsize=9)
        ax_vol.set_xlim(close.index[0], close.index[-1])
        ax_vol.legend(loc="upper right", fontsize=8, facecolor="#161b22",
                      edgecolor="#30363d", labelcolor="#c9d1d9")
        ax_vol.grid(True, alpha=0.2, color="#30363d")
        ax_vol.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_vol.spines.values():
            spine.set_edgecolor("#30363d")

        # ── Panel 3: Hurst exponent ──────────────────────────────────
        ax_hurst.clear()
        ax_hurst.set_facecolor("#161b22")
        h = hurst_series.iloc[:progress].dropna()
        if len(h) > 0:
            ax_hurst.plot(h.index, h.values, color="#bc8cff", linewidth=1.8)
            ax_hurst.axhline(0.5, color="#30363d", linestyle="-", alpha=0.7,
                             label="Random walk")
            ax_hurst.fill_between(h.index, h.values, 0.5,
                                  where=(h.values > 0.5),
                                  color="#3fb950", alpha=0.2,
                                  label="Trending (H > 0.5)")
            ax_hurst.fill_between(h.index, h.values, 0.5,
                                  where=(h.values < 0.5),
                                  color="#58a6ff", alpha=0.2,
                                  label="Mean-reverting (H < 0.5)")
            current_h = h.iloc[-1]
            ax_hurst.scatter([h.index[-1]], [current_h], color="gold",
                             s=60, zorder=5, edgecolor="white", linewidths=1)
        ax_hurst.set_ylabel("Hurst Exponent H", color="#c9d1d9", fontsize=9)
        ax_hurst.set_xlabel("Date", color="#c9d1d9", fontsize=9)
        ax_hurst.set_xlim(close.index[0], close.index[-1])
        ax_hurst.set_ylim(0.2, 0.8)
        ax_hurst.legend(loc="upper right", fontsize=7, facecolor="#161b22",
                        edgecolor="#30363d", labelcolor="#c9d1d9", ncol=3)
        ax_hurst.grid(True, alpha=0.2, color="#30363d")
        ax_hurst.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_hurst.spines.values():
            spine.set_edgecolor("#30363d")

        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 5: Parameter sweep heatmap (parameter landscape)
# ═══════════════════════════════════════════════════════════════════════

def gif_parameter_heatmap(output_path=None, fps=3):
    """Animated parameter sweep heatmap evolving through time."""
    output_path = output_path or os.path.join(ASSETS_DIR, "parameter_sweep.gif")
    prices = _make_synthetic_prices()
    engine = BacktestEngine()

    n_periods = 6
    window = len(prices) // 2
    step = (len(prices) - window) // (n_periods - 1)
    fast_grid = [3, 5, 8, 10, 12, 15, 18, 20, 25]
    slow_grid = [20, 25, 30, 35, 40, 45, 50, 55, 60]

    surfaces = []
    for i in range(n_periods):
        start = i * step
        end = start + window
        if end > len(prices):
            break
        wp = prices.iloc[start:end]
        label = str(wp.index[-1].date())
        try:
            result = engine.parameter_sweep(
                wp, signal_fn_factory=momentum_crossover_factory,
                param_grid={"fast": fast_grid, "slow": slow_grid},
            )
            pivot = result.pnl_by_param.pivot_table(
                values="sharpe", index="fast", columns="slow"
            )
            surfaces.append((pivot, label, result.metrics))
        except ValueError:
            continue

    z_min = min(np.nanmin(p.values) for p, _, _ in surfaces)
    z_max = max(np.nanmax(p.values) for p, _, _ in surfaces)

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=100)
    fig.patch.set_facecolor("#0d1117")

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor("#161b22")
        pivot, label, metrics = surfaces[frame_idx]

        im = ax.imshow(
            pivot.values, cmap="RdYlGn", aspect="auto",
            vmin=z_min, vmax=z_max, origin="lower",
            extent=[0, len(pivot.columns), 0, len(pivot.index)],
        )

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black",
                        fontsize=8, fontweight="bold")

        # Mark best
        best = np.unravel_index(np.nanargmax(pivot.values), pivot.shape)
        ax.add_patch(plt.Rectangle(
            (best[1], best[0]), 1, 1, fill=False,
            edgecolor="gold", linewidth=3,
        ))

        ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
        ax.set_xticklabels(pivot.columns, color="#c9d1d9", fontsize=9)
        ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
        ax.set_yticklabels(pivot.index, color="#c9d1d9", fontsize=9)
        ax.set_xlabel("Slow MA", color="#c9d1d9", fontsize=11)
        ax.set_ylabel("Fast MA", color="#c9d1d9", fontsize=11)

        best_sharpe = pivot.values[best]
        best_fast = pivot.index[best[0]]
        best_slow = pivot.columns[best[1]]
        dsr = metrics.get("deflated_sharpe", {})
        dsr_p = dsr.get("p_value", 1.0) if dsr else 1.0
        sig_label = "✓ significant" if dsr_p < 0.05 else "✗ likely overfit"
        sig_color = "#3fb950" if dsr_p < 0.05 else "#f85149"

        ax.set_title(
            f"Parameter Sweep · {label}\n"
            f"Best: fast={best_fast}, slow={best_slow}, "
            f"Sharpe={best_sharpe:.2f}",
            color="#f0f6fc", fontweight="bold", fontsize=12, pad=10,
        )
        # Deflated Sharpe annotation
        ax.text(0.99, -0.18, f"Deflated Sharpe p={dsr_p:.3f}  {sig_label}",
                transform=ax.transAxes, ha="right", va="top",
                color=sig_color, fontsize=10, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")

        fig.tight_layout()
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(surfaces), interval=1000 // fps,
        blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 6: ARIMA Walk-Forward Backtest (multi-panel)
# ═══════════════════════════════════════════════════════════════════════

def gif_arima_backtest(output_path=None, fps=12):
    """
    Animated ARIMA walk-forward: OOS equity curves with t-stat and
    forecast vs actual overlay. Loaded from pre-computed results.
    """
    import pickle
    output_path = output_path or os.path.join(ASSETS_DIR, "arima_walk_forward.gif")
    results_path = os.path.join(ASSETS_DIR, "arima_results.pkl")

    if not os.path.exists(results_path):
        print(f"Missing {results_path}. Run: python scripts/arima_optimizer.py")
        return None

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    prices = data["prices"]
    returns = data["returns"]
    forecasts = data["forecasts"]
    results = data["results"]

    # Sort by t-stat descending
    results = sorted(results,
                     key=lambda r: r["metrics"].get("hac_t_stat", -999),
                     reverse=True)

    colors = ["#3fb950", "#58a6ff", "#bc8cff", "#f0883e", "#f85149"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=85,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#0d1117")
    (ax_eq, ax_fc), (ax_tstat, ax_drawdown) = axes

    # Equity curves and masks
    equity_curves = {r["name"]: r["equity"] for r in results}
    oos_masks = {r["name"]: r.get("oos_mask", pd.Series(False, index=r["equity"].index))
                 for r in results}

    n_frames = 48
    dates = prices.index
    min_progress_idx = int(len(dates) * 0.25)

    def update(frame_idx):
        progress_frac = (frame_idx + 1) / n_frames
        progress = int(min_progress_idx + (len(dates) - min_progress_idx) * progress_frac)
        progress = min(progress, len(dates) - 1)
        current_date = dates[progress]

        # ── Panel 1: OOS Equity Curves ──────────────────────────────
        ax_eq.clear()
        ax_eq.set_facecolor("#161b22")

        bh_equity = 100_000 * (prices["close"] / prices["close"].iloc[0])
        ax_eq.plot(bh_equity.index[:progress], bh_equity.values[:progress],
                   color="#8b949e", linewidth=1.3, linestyle="--",
                   label="Buy & Hold", alpha=0.6)

        for i, (name, eq) in enumerate(equity_curves.items()):
            m = results[i]["metrics"]
            t = m.get("hac_t_stat", 0)
            marker = "★" if abs(t) > 5 else "✓" if abs(t) > 2.576 else "·"
            label = f"{marker} {name}  t={t:.2f}"
            ax_eq.plot(eq.index[:progress], eq.values[:progress],
                       color=colors[i], linewidth=2.2, label=label)

        ax_eq.axhline(100_000, color="#30363d", linestyle=":", alpha=0.5)
        ax_eq.set_title(
            f"ARIMA Walk-Forward OOS Equity  ·  {current_date.date()}",
            color="#f0f6fc", fontweight="bold", fontsize=12,
        )
        ax_eq.set_ylabel("Portfolio ($)", color="#c9d1d9", fontsize=9)
        ax_eq.grid(True, alpha=0.2, color="#30363d")
        ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax_eq.legend(loc="upper left", fontsize=8.5, facecolor="#161b22",
                     edgecolor="#30363d", labelcolor="#c9d1d9")
        ax_eq.tick_params(colors="#8b949e", labelsize=8)
        ax_eq.set_xlim(dates[0], dates[-1])
        for s in ax_eq.spines.values():
            s.set_edgecolor("#30363d")

        # ── Panel 2: ARIMA Forecast vs Actual ──────────────────────
        ax_fc.clear()
        ax_fc.set_facecolor("#161b22")

        # Recent window of forecasts vs actuals
        window = 80
        end = min(progress, len(forecasts))
        start_w = max(0, end - window)
        if start_w < end:
            fc_window = forecasts.iloc[start_w:end]
            actual = fc_window["actual"] * 100
            mean = fc_window["forecast_mean"] * 100
            lower = fc_window["forecast_lower"] * 100
            upper = fc_window["forecast_upper"] * 100

            ax_fc.fill_between(fc_window.index, lower, upper,
                               alpha=0.25, color="#58a6ff",
                               label="95% CI")
            ax_fc.plot(fc_window.index, mean, color="#58a6ff",
                       linewidth=1.8, label="Forecast")
            ax_fc.plot(fc_window.index, actual, color="#f85149",
                       linewidth=1.2, alpha=0.8, label="Actual")
            ax_fc.axhline(0, color="#30363d", linestyle="-", alpha=0.5)

        ax_fc.set_title(
            f"ARIMA(2,0,2) One-Step Forecast vs Actual (last {window} days)",
            color="#f0f6fc", fontweight="bold", fontsize=11,
        )
        ax_fc.set_ylabel("Return (%)", color="#c9d1d9", fontsize=9)
        ax_fc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax_fc.legend(loc="upper right", fontsize=8, facecolor="#161b22",
                     edgecolor="#30363d", labelcolor="#c9d1d9")
        ax_fc.grid(True, alpha=0.2, color="#30363d")
        ax_fc.tick_params(colors="#8b949e", labelsize=8)
        for s in ax_fc.spines.values():
            s.set_edgecolor("#30363d")

        # ── Panel 3: t-statistic bar race ──────────────────────────
        ax_tstat.clear()
        ax_tstat.set_facecolor("#161b22")

        names = [r["name"] for r in results]
        t_stats_current = []
        for i, r in enumerate(results):
            eq = equity_curves[r["name"]]
            oos_ret = eq.iloc[:progress].pct_change().dropna()
            oos_ret = oos_ret[oos_ret != 0]
            if len(oos_ret) < 20:
                t_stats_current.append(0)
            else:
                mean = oos_ret.mean()
                std = oos_ret.std()
                t = (mean / std) * np.sqrt(len(oos_ret)) if std > 0 else 0
                t_stats_current.append(t)

        bars = ax_tstat.barh(names, t_stats_current,
                              color=[colors[i] for i in range(len(names))],
                              edgecolor="#30363d", linewidth=1)
        ax_tstat.axvline(0, color="#30363d", linewidth=1)
        ax_tstat.axvline(2.576, color="#d29922", linestyle="--",
                          linewidth=1.2, alpha=0.8, label="1% sig")
        ax_tstat.axvline(5.0, color="#3fb950", linestyle="--",
                          linewidth=1.5, alpha=0.9, label="5σ target")

        for bar, t in zip(bars, t_stats_current):
            ax_tstat.text(t + (0.1 if t >= 0 else -0.1),
                           bar.get_y() + bar.get_height() / 2,
                           f"{t:.2f}",
                           ha="left" if t >= 0 else "right",
                           va="center", color="#c9d1d9",
                           fontsize=9, fontweight="bold")

        ax_tstat.set_title("Live t-statistic (rolling)",
                            color="#f0f6fc", fontweight="bold", fontsize=11)
        ax_tstat.set_xlabel("t-statistic", color="#c9d1d9", fontsize=9)
        max_t = max(max(t_stats_current, default=5), 6)
        min_t = min(min(t_stats_current, default=-1), -1)
        ax_tstat.set_xlim(min_t - 0.5, max_t + 0.5)
        ax_tstat.legend(loc="lower right", fontsize=8, facecolor="#161b22",
                         edgecolor="#30363d", labelcolor="#c9d1d9")
        ax_tstat.grid(True, alpha=0.2, color="#30363d", axis="x")
        ax_tstat.tick_params(colors="#8b949e", labelsize=8)
        for s in ax_tstat.spines.values():
            s.set_edgecolor("#30363d")

        # ── Panel 4: Drawdown ───────────────────────────────────────
        ax_drawdown.clear()
        ax_drawdown.set_facecolor("#161b22")

        for i, (name, eq) in enumerate(equity_curves.items()):
            eq_slice = eq.iloc[:progress]
            if len(eq_slice) > 1:
                dd = (eq_slice - eq_slice.cummax()) / eq_slice.cummax() * 100
                ax_drawdown.fill_between(dd.index, dd.values, 0,
                                          alpha=0.3, color=colors[i])
                ax_drawdown.plot(dd.index, dd.values,
                                  color=colors[i], linewidth=1.2)

        ax_drawdown.axhline(0, color="#30363d", linewidth=1)
        ax_drawdown.set_title("Drawdown",
                               color="#f0f6fc", fontweight="bold", fontsize=11)
        ax_drawdown.set_ylabel("Drawdown (%)", color="#c9d1d9", fontsize=9)
        ax_drawdown.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_drawdown.set_xlim(dates[0], dates[-1])
        ax_drawdown.grid(True, alpha=0.2, color="#30363d")
        ax_drawdown.tick_params(colors="#8b949e", labelsize=8)
        for s in ax_drawdown.spines.values():
            s.set_edgecolor("#30363d")

        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 7: Walk-Forward Schedule (train/test gantt chart)
# ═══════════════════════════════════════════════════════════════════════

def gif_walk_forward_schedule(output_path=None, fps=4):
    """Animated Gantt chart showing train/purge/test windows rolling forward."""
    import pickle
    output_path = output_path or os.path.join(ASSETS_DIR, "walk_forward.gif")
    results_path = os.path.join(ASSETS_DIR, "arima_results.pkl")

    if not os.path.exists(results_path):
        return None

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    prices = data["prices"]
    results = data["results"]
    # Use best result's param history
    best = max(results, key=lambda r: r["metrics"].get("hac_t_stat", 0))
    param_history = best.get("param_history", [])

    if not param_history:
        return None

    fig, ax = plt.subplots(figsize=(13, 5), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    n_windows = len(param_history)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor("#161b22")
        window_idx = min(frame_idx, n_windows - 1)

        # Draw price in background
        close = prices["close"]
        price_norm = (close - close.min()) / (close.max() - close.min()) * 0.6
        ax.plot(close.index, price_norm - 2.5, color="#8b949e",
                linewidth=0.9, alpha=0.5, label="Price (normalized)")

        # Draw all windows up to current
        for i, p in enumerate(param_history[:window_idx + 1]):
            alpha = 0.9 if i == window_idx else 0.35
            # Train
            ax.barh(0, (p["train_end"] - p["train_start"]).days,
                    left=p["train_start"], height=0.55,
                    color="#58a6ff", alpha=alpha, edgecolor="#30363d")
            # Purge gap
            purge_width = (p["test_start"] - p["train_end"]).days
            if purge_width > 0:
                ax.barh(0, purge_width, left=p["train_end"],
                        height=0.55, color="#f85149", alpha=alpha * 0.7,
                        edgecolor="#30363d")
            # Test
            ax.barh(0, (p["test_end"] - p["test_start"]).days,
                    left=p["test_start"], height=0.55,
                    color="#3fb950", alpha=alpha, edgecolor="#30363d")

        # Current window highlighted with arrow + annotation
        p = param_history[window_idx]
        train_mid = p["train_start"] + (p["train_end"] - p["train_start"]) / 2
        test_mid = p["test_start"] + (p["test_end"] - p["test_start"]) / 2

        ax.annotate("TRAIN", xy=(train_mid, 0.45), ha="center",
                    color="#f0f6fc", fontsize=10, fontweight="bold")
        ax.annotate("OOS", xy=(test_mid, 0.45), ha="center",
                    color="#f0f6fc", fontsize=10, fontweight="bold")

        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#58a6ff", label="In-sample (train)"),
            Patch(facecolor="#f85149", alpha=0.7, label="Purge gap"),
            Patch(facecolor="#3fb950", label="Out-of-sample (test)"),
        ]

        params_str = ", ".join(f"{k}={v}" for k, v in p["params"].items())
        ax.set_title(
            f"Walk-Forward Window {window_idx+1}/{n_windows}  ·  "
            f"IS score: {p['is_score']:.2f}\n"
            f"Optimal params: {params_str}",
            color="#f0f6fc", fontweight="bold", fontsize=12, pad=10,
        )
        ax.set_ylim(-3, 1.2)
        ax.set_yticks([])
        ax.set_xlim(prices.index[0], prices.index[-1])
        ax.legend(handles=legend_elements, loc="lower left",
                  fontsize=9, facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.grid(True, alpha=0.15, color="#30363d", axis="x")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")

        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_windows, interval=1000 // fps,
        blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"✓ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# GIF 8: Multi-Asset Cross-Category Backtest
# ═══════════════════════════════════════════════════════════════════════

def gif_multi_asset_backtest(
    output_path: str = os.path.join(ASSETS_DIR, "multi_asset_backtest.gif"),
    fps: int = 2,
):
    """
    Animated 4-panel GIF showing cross-asset ARIMA backtest results.
    Panel 1: Equity curves (best strategy per asset, revealed one by one)
    Panel 2: t-statistic bar chart (animated reveal)
    Panel 3: Category summary radar
    Panel 4: Drawdown comparison
    """
    import pickle

    pkl_path = os.path.join(ASSETS_DIR, "multi_asset_results.pkl")
    if not os.path.exists(pkl_path):
        print(f"  Skipping multi-asset GIF (no {pkl_path})")
        return None

    with open(pkl_path, "rb") as f:
        all_results = pickle.load(f)

    assets = list(all_results.keys())
    n_assets = len(assets)

    category_colors = {
        "Stocks": "#58a6ff",
        "Crypto": "#f0883e",
        "Indices": "#3fb950",
        "Commodities": "#d2a8ff",
        "Bonds": "#f85149",
        "Forex": "#79c0ff",
    }
    asset_colors = {a: category_colors.get(all_results[a]["category"], "#8b949e")
                    for a in assets}

    # Pre-compute equity curves for best strategy per asset
    equities = {}
    for asset in assets:
        r = all_results[asset]
        best_name = r["best_strategy"]
        best_result = r["results"][best_name]
        eq = best_result.get("equity", None)
        if eq is not None:
            equities[asset] = eq / eq.iloc[0] * 100  # normalize to 100

    # Build frames: reveal assets one at a time
    n_frames = n_assets + 6  # reveal + hold summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=85)
    fig.patch.set_facecolor("#0d1117")

    def update(frame_idx):
        for ax in axes.flat:
            ax.clear()
            ax.set_facecolor("#161b22")

        reveal = min(frame_idx + 1, n_assets)
        shown_assets = assets[:reveal]

        # ── Panel 1: Equity curves ──────────────────────────────────
        ax1 = axes[0, 0]
        for asset in shown_assets:
            if asset in equities:
                eq = equities[asset]
                ax1.plot(eq.index, eq.values, color=asset_colors[asset],
                         linewidth=1.3, alpha=0.85, label=asset)
        ax1.axhline(100, color="#8b949e", linestyle="--", alpha=0.4, linewidth=0.8)
        ax1.set_title("Equity Curves (Best Strategy per Asset)",
                       color="#f0f6fc", fontweight="bold", fontsize=10)
        ax1.set_ylabel("Equity (indexed=100)", color="#c9d1d9", fontsize=9)
        if shown_assets:
            ax1.legend(loc="upper right", fontsize=7, ncol=2,
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#c9d1d9")
        ax1.grid(True, alpha=0.15)

        # ── Panel 2: t-stat bar chart ───────────────────────────────
        ax2 = axes[0, 1]
        t_stats = []
        bar_colors = []
        for asset in shown_assets:
            m = all_results[asset]["best_metrics"]
            t = m.get("hac_t_stat", 0) if m else 0
            t_stats.append(t)
            bar_colors.append(asset_colors[asset])

        y_pos = range(len(shown_assets))
        bars = ax2.barh(y_pos, t_stats, color=bar_colors, alpha=0.85,
                        edgecolor="#30363d", height=0.6)

        # Significance lines
        ax2.axvline(-1.96, color="#d29922", linestyle="--", alpha=0.5, linewidth=0.8)
        ax2.axvline(-2.576, color="#f85149", linestyle="--", alpha=0.5, linewidth=0.8)
        ax2.axvline(0, color="#8b949e", linestyle="-", alpha=0.3, linewidth=0.8)
        if any(abs(t) > 4 for t in t_stats):
            ax2.axvline(-5.0, color="#f85149", linestyle=":", alpha=0.5, linewidth=0.8)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(shown_assets, fontsize=8)
        ax2.set_title("HAC t-statistic (Best Strategy)",
                       color="#f0f6fc", fontweight="bold", fontsize=10)
        ax2.set_xlabel("t-stat", color="#c9d1d9", fontsize=9)

        # Annotate significance
        for i, t in enumerate(t_stats):
            sig = "5sig" if abs(t) > 5 else "1%" if abs(t) > 2.576 else \
                  "5%" if abs(t) > 1.96 else "n.s."
            color = "#f85149" if abs(t) > 2.576 else "#d29922" if abs(t) > 1.96 else "#8b949e"
            ax2.text(t - 0.15 if t < 0 else t + 0.1, i, f" {sig}",
                     va="center", fontsize=7, color=color, fontweight="bold")
        ax2.grid(True, alpha=0.15, axis="x")

        # ── Panel 3: Category summary ──────────────────────────────
        ax3 = axes[1, 0]
        shown_cats = {}
        for asset in shown_assets:
            cat = all_results[asset]["category"]
            m = all_results[asset]["best_metrics"]
            if not m:
                continue
            if cat not in shown_cats:
                shown_cats[cat] = {"t_stats": [], "sharpes": [], "returns": []}
            shown_cats[cat]["t_stats"].append(m.get("hac_t_stat", 0))
            shown_cats[cat]["sharpes"].append(m.get("oos_sharpe", 0))
            shown_cats[cat]["returns"].append(m.get("total_return", 0))

        cats = list(shown_cats.keys())
        if cats:
            x = range(len(cats))
            avg_t = [np.mean(shown_cats[c]["t_stats"]) for c in cats]
            avg_s = [np.mean(shown_cats[c]["sharpes"]) for c in cats]
            cat_cols = [category_colors.get(c, "#8b949e") for c in cats]

            w = 0.35
            ax3.bar([i - w/2 for i in x], avg_t, w, label="Avg t-stat",
                    color=cat_cols, alpha=0.7, edgecolor="#30363d")
            ax3.bar([i + w/2 for i in x], avg_s, w, label="Avg Sharpe",
                    color=cat_cols, alpha=0.4, edgecolor="#30363d",
                    hatch="//")
            ax3.set_xticks(list(x))
            ax3.set_xticklabels(cats, fontsize=8, rotation=15)
            ax3.axhline(0, color="#8b949e", linestyle="-", alpha=0.3, linewidth=0.8)
            ax3.legend(loc="lower left", fontsize=7,
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#c9d1d9")
        ax3.set_title("Category Averages",
                       color="#f0f6fc", fontweight="bold", fontsize=10)
        ax3.grid(True, alpha=0.15, axis="y")

        # ── Panel 4: Drawdown comparison ────────────────────────────
        ax4 = axes[1, 1]
        for asset in shown_assets:
            if asset in equities:
                eq = equities[asset]
                running_max = eq.cummax()
                dd = (eq - running_max) / running_max * 100
                ax4.plot(dd.index, dd.values, color=asset_colors[asset],
                         linewidth=1.0, alpha=0.75, label=asset)

        ax4.axhline(-10, color="#d29922", linestyle="--", alpha=0.4, linewidth=0.8)
        ax4.axhline(-50, color="#f85149", linestyle="--", alpha=0.4, linewidth=0.8)
        ax4.set_title("Drawdowns (%)",
                       color="#f0f6fc", fontweight="bold", fontsize=10)
        ax4.set_ylabel("Drawdown %", color="#c9d1d9", fontsize=9)
        if shown_assets:
            ax4.legend(loc="lower left", fontsize=7, ncol=2,
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#c9d1d9")
        ax4.grid(True, alpha=0.15)

        # Style all panels
        for ax in axes.flat:
            for s in ax.spines.values():
                s.set_edgecolor("#30363d")
            ax.tick_params(colors="#8b949e", labelsize=7)

        # Super title
        if reveal < n_assets:
            title = f"Multi-Asset ARIMA Backtest · Revealing {reveal}/{n_assets} assets..."
        else:
            title = f"Multi-Asset ARIMA Backtest · All {n_assets} assets (6 categories)"
        fig.suptitle(title, color="#f0f6fc", fontweight="bold", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps,
        blit=False, repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  {output_path}")
    return output_path


if __name__ == "__main__":
    gif_pnl_surface_animated()
    gif_equity_curves()
    gif_risk_dashboard()
    gif_regime_detection()
    gif_parameter_heatmap()
    gif_arima_backtest()
    gif_walk_forward_schedule()
    gif_multi_asset_backtest()
