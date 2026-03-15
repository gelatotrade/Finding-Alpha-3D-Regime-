"""
Time-Evolving Animated 3D PnL Surfaces.
Visualizes how trading strategy performance evolves across
parameter space and time.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError:
        logger.error("plotly is required for 3D surfaces: pip install plotly")
        raise


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import animation
        return plt, Axes3D, animation
    except ImportError:
        logger.error("matplotlib required: pip install matplotlib")
        raise


class PnLSurfaceVisualizer:
    """
    Generate animated 3D PnL surfaces showing strategy behavior
    across parameter space and through time.
    """

    # ── Plotly Interactive 3D ────────────────────────────────────────────

    def plot_static_pnl_surface(
        self,
        pnl_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = "sharpe",
        title: str = "PnL Surface",
    ) -> "go.Figure":
        """
        Static 3D surface of a metric over 2 parameter dimensions.

        Parameters
        ----------
        pnl_df : DataFrame from BacktestEngine.parameter_sweep with columns
                  [param1, param2, metric]
        """
        go, _ = _try_import_plotly()

        pivot = pnl_df.pivot_table(values=metric, index=param1, columns=param2)

        fig = go.Figure(data=[
            go.Surface(
                x=pivot.columns.values,
                y=pivot.index.values,
                z=pivot.values,
                colorscale="RdYlGn",
                colorbar=dict(title=metric.replace("_", " ").title()),
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=param2,
                yaxis_title=param1,
                zaxis_title=metric.replace("_", " ").title(),
            ),
            width=900, height=700,
        )
        return fig

    def plot_animated_pnl_surface(
        self,
        time_pnl_data: Dict[str, pd.DataFrame],
        param1: str,
        param2: str,
        metric: str = "sharpe",
        title: str = "Time-Evolving PnL Surface",
    ) -> "go.Figure":
        """
        Animated 3D surface that evolves through time.

        Parameters
        ----------
        time_pnl_data : {time_label: pnl_df} where each pnl_df has
                        [param1, param2, metric] columns from parameter sweeps
                        at different time windows.
        """
        go, _ = _try_import_plotly()

        time_labels = sorted(time_pnl_data.keys())
        frames = []
        initial_surface = None

        for i, t_label in enumerate(time_labels):
            df = time_pnl_data[t_label]
            pivot = df.pivot_table(values=metric, index=param1, columns=param2)

            surface = go.Surface(
                x=pivot.columns.values,
                y=pivot.index.values,
                z=pivot.values,
                colorscale="RdYlGn",
                colorbar=dict(title=metric.replace("_", " ").title()),
            )
            frames.append(go.Frame(data=[surface], name=str(t_label)))
            if i == 0:
                initial_surface = surface

        fig = go.Figure(data=[initial_surface], frames=frames)

        # Animation controls
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=param2,
                yaxis_title=param1,
                zaxis_title=metric.replace("_", " ").title(),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 800, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
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
                x=0.1, len=0.8,
                xanchor="left",
                y=0, yanchor="top",
                currentvalue=dict(prefix="Period: ", visible=True),
                transition=dict(duration=300),
            )],
            width=1000, height=750,
        )
        return fig

    # ── Matplotlib Animated GIF/MP4 ──────────────────────────────────────

    def animate_pnl_surface_matplotlib(
        self,
        time_pnl_data: Dict[str, pd.DataFrame],
        param1: str,
        param2: str,
        metric: str = "sharpe",
        output_path: str = "pnl_surface_animation.gif",
        fps: int = 2,
    ) -> str:
        """
        Create an animated GIF/MP4 of the PnL surface evolving over time.
        """
        plt, Axes3D, anim_mod = _try_import_matplotlib()

        time_labels = sorted(time_pnl_data.keys())
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Precompute all surfaces
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
            ax.set_xlabel(param2)
            ax.set_ylabel(param1)
            ax.set_zlabel(metric.replace("_", " ").title())
            ax.set_zlim(global_zmin, global_zmax)
            ax.set_title(f"PnL Surface — Period: {t_label}")
            return []

        ani = anim_mod.FuncAnimation(
            fig, update, frames=len(surfaces_data), interval=1000 // fps, blit=False
        )
        ani.save(output_path, writer="pillow" if output_path.endswith(".gif") else "ffmpeg",
                 fps=fps)
        plt.close(fig)
        logger.info("Animation saved to %s", output_path)
        return output_path

    # ── Equity Curve Comparison ──────────────────────────────────────────

    def plot_equity_comparison(
        self,
        equity_curves: Dict[str, pd.Series],
        title: str = "Strategy Equity Curves",
    ) -> "go.Figure":
        """Plot multiple equity curves for comparison."""
        go, _ = _try_import_plotly()

        fig = go.Figure()
        for name, eq in equity_curves.items():
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode="lines", name=name,
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            width=1000, height=500,
            hovermode="x unified",
        )
        return fig

    def plot_risk_dashboard(
        self,
        returns: pd.Series,
        equity: pd.Series,
        title: str = "Risk Dashboard",
    ) -> "go.Figure":
        """Multi-panel risk visualization."""
        go, make_subplots = _try_import_plotly()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Equity Curve", "Drawdown",
                            "Rolling Volatility", "Return Distribution"),
        )

        # Equity
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="blue")),
            row=1, col=1,
        )

        # Drawdown
        dd = (equity - equity.cummax()) / equity.cummax()
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                       fill="tozeroy", line=dict(color="red")),
            row=1, col=2,
        )

        # Rolling vol
        vol_21 = returns.rolling(21).std() * np.sqrt(252)
        vol_63 = returns.rolling(63).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=vol_21.index, y=vol_21.values, name="21d Vol", line=dict(color="orange")),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=vol_63.index, y=vol_63.values, name="63d Vol", line=dict(color="purple")),
            row=2, col=1,
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns.values, nbinsx=50, name="Returns", marker_color="steelblue"),
            row=2, col=2,
        )

        fig.update_layout(title=title, height=800, width=1200, showlegend=True)
        return fig
