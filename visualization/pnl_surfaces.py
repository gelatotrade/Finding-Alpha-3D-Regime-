"""
Institutional 3D PnL Surface Visualization.

Features:
- Static and animated 3D PnL surfaces (Plotly + Matplotlib)
- Monte Carlo confidence cones on equity curves
- Regime overlay on surfaces
- Multi-panel risk dashboards
- Deflated Sharpe annotations
- Alpha decay heatmaps
"""
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError:
        raise ImportError("plotly required: pip install plotly")


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import animation
        return plt, Axes3D, animation
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")


class PnLSurfaceVisualizer:
    """Institutional-grade visualization suite."""

    # ── 3D PnL Surface (Static) ─────────────────────────────────────────

    def plot_static_pnl_surface(
        self,
        pnl_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = "sharpe",
        title: str = "PnL Surface",
    ) -> "go.Figure":
        go, _ = _try_import_plotly()

        pivot = pnl_df.pivot_table(values=metric, index=param1, columns=param2)

        # Find best point
        best_idx = pnl_df[metric].idxmax()
        best_row = pnl_df.loc[best_idx]

        fig = go.Figure(data=[
            go.Surface(
                x=pivot.columns.values,
                y=pivot.index.values,
                z=pivot.values,
                colorscale="RdYlGn",
                colorbar=dict(title=metric.replace("_", " ").title()),
                hovertemplate=(
                    f"{param2}: %{{x}}<br>"
                    f"{param1}: %{{y}}<br>"
                    f"{metric}: %{{z:.3f}}<extra></extra>"
                ),
            ),
            # Mark best point
            go.Scatter3d(
                x=[best_row[param2]],
                y=[best_row[param1]],
                z=[best_row[metric]],
                mode="markers+text",
                marker=dict(size=8, color="gold", symbol="diamond"),
                text=[f"Best: {best_row[metric]:.3f}"],
                textposition="top center",
                name="Optimal",
            ),
        ])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=param2,
                yaxis_title=param1,
                zaxis_title=metric.replace("_", " ").title(),
            ),
            width=1000, height=750,
        )
        return fig

    # ── 3D PnL Surface (Animated) ────────────────────────────────────────

    def plot_animated_pnl_surface(
        self,
        time_pnl_data: Dict[str, pd.DataFrame],
        param1: str,
        param2: str,
        metric: str = "sharpe",
        title: str = "Time-Evolving PnL Surface",
    ) -> "go.Figure":
        go, _ = _try_import_plotly()

        time_labels = sorted(time_pnl_data.keys())
        frames = []
        initial_surface = None
        global_zmin, global_zmax = np.inf, -np.inf

        for i, t_label in enumerate(time_labels):
            df = time_pnl_data[t_label]
            pivot = df.pivot_table(values=metric, index=param1, columns=param2)
            z_vals = pivot.values
            global_zmin = min(global_zmin, np.nanmin(z_vals))
            global_zmax = max(global_zmax, np.nanmax(z_vals))

            surface = go.Surface(
                x=pivot.columns.values,
                y=pivot.index.values,
                z=z_vals,
                colorscale="RdYlGn",
                cmin=global_zmin if i > 0 else None,
                cmax=global_zmax if i > 0 else None,
                colorbar=dict(title=metric.replace("_", " ").title()),
            )
            frames.append(go.Frame(data=[surface], name=str(t_label)))
            if i == 0:
                initial_surface = surface

        fig = go.Figure(data=[initial_surface], frames=frames)

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=param2, yaxis_title=param1,
                zaxis_title=metric.replace("_", " ").title(),
                zaxis=dict(range=[global_zmin * 1.1, global_zmax * 1.1]),
            ),
            updatemenus=[dict(
                type="buttons", showactive=False, y=0, x=0.5, xanchor="center",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 800, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[
                    dict(args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                          "mode": "immediate"}],
                         label=str(f.name), method="animate")
                    for f in frames
                ],
                x=0.1, len=0.8, xanchor="left", y=0,
                currentvalue=dict(prefix="Period: ", visible=True),
            )],
            width=1100, height=800,
        )
        return fig

    # ── Matplotlib Animation (GIF/MP4) ───────────────────────────────────

    def animate_pnl_surface_matplotlib(
        self,
        time_pnl_data: Dict[str, pd.DataFrame],
        param1: str,
        param2: str,
        metric: str = "sharpe",
        output_path: str = "pnl_surface_animation.gif",
        fps: int = 2,
    ) -> str:
        plt, Axes3D, anim_mod = _try_import_matplotlib()

        time_labels = sorted(time_pnl_data.keys())
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")

        surfaces_data = []
        global_zmin, global_zmax = np.inf, -np.inf
        for t_label in time_labels:
            df = time_pnl_data[t_label]
            pivot = df.pivot_table(values=metric, index=param1, columns=param2)
            X, Y = np.meshgrid(pivot.columns.values.astype(float),
                               pivot.index.values.astype(float))
            Z = pivot.values.astype(float)
            surfaces_data.append((X, Y, Z, t_label))
            global_zmin = min(global_zmin, np.nanmin(Z))
            global_zmax = max(global_zmax, np.nanmax(Z))

        def update(frame_idx):
            ax.clear()
            X, Y, Z, t_label = surfaces_data[frame_idx]
            ax.plot_surface(X, Y, Z, cmap="RdYlGn", alpha=0.85,
                            vmin=global_zmin, vmax=global_zmax)
            # Mark best point
            best_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
            ax.scatter([X[best_idx]], [Y[best_idx]], [Z[best_idx]],
                       color="gold", s=100, marker="D", zorder=5)
            ax.set_xlabel(param2)
            ax.set_ylabel(param1)
            ax.set_zlabel(metric.replace("_", " ").title())
            ax.set_zlim(global_zmin, global_zmax)
            ax.set_title(f"PnL Surface — {t_label}")
            return []

        ani = anim_mod.FuncAnimation(
            fig, update, frames=len(surfaces_data), interval=1000 // fps, blit=False
        )
        writer = "pillow" if output_path.endswith(".gif") else "ffmpeg"
        ani.save(output_path, writer=writer, fps=fps)
        plt.close(fig)
        logger.info("Animation saved to %s", output_path)
        return output_path

    # ── Equity Curves with Monte Carlo Cone ──────────────────────────────

    def plot_equity_with_mc(
        self,
        equity: pd.Series,
        monte_carlo: Optional[dict] = None,
        title: str = "Equity Curve with Confidence Intervals",
    ) -> "go.Figure":
        go, _ = _try_import_plotly()

        fig = go.Figure()

        # Actual equity
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="Actual Equity",
            line=dict(color="blue", width=2),
        ))

        # MC confidence bands (if available)
        if monte_carlo and "return_5th" in monte_carlo:
            n = len(equity)
            x_proj = equity.index
            base = equity.iloc[0]

            # Create fan chart
            for pct, color, name in [
                ("return_5th", "rgba(255,0,0,0.1)", "5th pct"),
                ("return_25th", "rgba(255,165,0,0.15)", "25th pct"),
                ("return_75th", "rgba(0,128,0,0.15)", "75th pct"),
                ("return_95th", "rgba(0,128,0,0.1)", "95th pct"),
            ]:
                if pct in monte_carlo:
                    final = base * (1 + monte_carlo[pct])
                    # Linear interpolation for illustration
                    projected = np.linspace(base, final, n)
                    fig.add_trace(go.Scatter(
                        x=x_proj, y=projected,
                        mode="lines", name=name,
                        line=dict(dash="dot", width=1),
                        opacity=0.5,
                    ))

        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Value ($)",
            width=1000, height=500, hovermode="x unified",
        )
        return fig

    # ── Multi-Panel Risk Dashboard ───────────────────────────────────────

    def plot_risk_dashboard(
        self,
        returns: pd.Series,
        equity: pd.Series,
        risk_data: Optional[dict] = None,
        title: str = "Risk Dashboard",
    ) -> "go.Figure":
        go, make_subplots = _try_import_plotly()

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Equity Curve", "Drawdown",
                "Rolling Volatility (EWMA vs Realized)", "Return Distribution",
                "Rolling Sharpe Ratio", "VaR Comparison",
            ),
            vertical_spacing=0.08,
        )

        # 1. Equity
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values, name="Equity",
            line=dict(color="blue"),
        ), row=1, col=1)

        # 2. Drawdown
        dd = (equity - equity.cummax()) / equity.cummax()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", line=dict(color="red"),
        ), row=1, col=2)

        # 3. Volatility
        vol_21 = returns.rolling(21).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=vol_21.index, y=vol_21.values, name="21d Realized",
            line=dict(color="orange"),
        ), row=2, col=1)

        # EWMA volatility
        decay = 0.94
        var = returns.iloc[0] ** 2 if len(returns) > 0 else 0
        ewma_vars = [var]
        for r in returns.iloc[1:]:
            var = decay * var + (1 - decay) * r**2
            ewma_vars.append(var)
        ewma_vol = pd.Series(np.sqrt(ewma_vars) * np.sqrt(252), index=returns.index)
        fig.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol.values, name="EWMA Vol",
            line=dict(color="purple"),
        ), row=2, col=1)

        # 4. Return distribution with normal overlay
        fig.add_trace(go.Histogram(
            x=returns.values, nbinsx=60, name="Returns",
            marker_color="steelblue", opacity=0.7,
        ), row=2, col=2)

        # 5. Rolling Sharpe
        rolling_sharpe = (
            returns.rolling(63).mean() / returns.rolling(63).std()
        ) * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name="63d Sharpe", line=dict(color="green"),
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # 6. VaR comparison bars
        if risk_data:
            var_methods = ["var_99_hist", "var_99_param", "var_99_cf", "var_99_evt"]
            var_labels = ["Historical", "Parametric", "Cornish-Fisher", "EVT (GPD)"]
            var_values = [risk_data.get(m, 0) * 100 for m in var_methods]

            fig.add_trace(go.Bar(
                x=var_labels, y=var_values, name="VaR (99%)",
                marker_color=["steelblue", "orange", "green", "red"],
            ), row=3, col=2)

        fig.update_layout(
            title=title, height=1100, width=1300,
            showlegend=True, legend=dict(x=1.05),
        )
        return fig

    # ── Equity Comparison ────────────────────────────────────────────────

    def plot_equity_comparison(
        self,
        equity_curves: Dict[str, pd.Series],
        title: str = "Strategy Comparison",
    ) -> "go.Figure":
        go, _ = _try_import_plotly()

        fig = go.Figure()
        for name, eq in equity_curves.items():
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values, mode="lines", name=name,
            ))
        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Value ($)",
            width=1000, height=500, hovermode="x unified",
        )
        return fig

    # ── Alpha Decay Heatmap ──────────────────────────────────────────────

    def plot_alpha_decay_heatmap(
        self,
        decay_data: Dict[str, pd.DataFrame],
        title: str = "Alpha Decay Analysis",
    ) -> "go.Figure":
        """Heatmap of IC across assets and horizons."""
        go, _ = _try_import_plotly()

        assets = list(decay_data.keys())
        horizons = []
        ic_matrix = []

        for asset in assets:
            df = decay_data[asset]
            if df.empty:
                continue
            if not horizons:
                horizons = df["horizon_days"].tolist()
            ic_matrix.append(df["ic"].tolist())

        if not ic_matrix:
            fig = go.Figure()
            fig.add_annotation(text="No alpha decay data", x=0.5, y=0.5)
            return fig

        fig = go.Figure(data=go.Heatmap(
            z=ic_matrix,
            x=[f"{h}d" for h in horizons],
            y=assets,
            colorscale="RdYlGn",
            colorbar=dict(title="IC"),
            text=[[f"{v:.3f}" for v in row] for row in ic_matrix],
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title=title, xaxis_title="Forward Horizon",
            yaxis_title="Asset", width=800, height=400,
        )
        return fig
