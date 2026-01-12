"""Visualization tools for IS/OOS validation results.

Provides plotting functions for analyzing cross-validation
and walk-forward results.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..validation.results import CrossValidationResult, SplitResult
    from .parameter_stability import ParameterStability


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


class ValidationVisualizer:
    """Visualizer for validation results.

    Provides various plots for analyzing cross-validation
    and walk-forward results.

    Example:
        >>> viz = ValidationVisualizer(figsize=(12, 8))
        >>> viz.plot_equity_curves(cv_result)
        >>> viz.plot_parameter_stability(stability)
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """Initialize ValidationVisualizer.

        Args:
            figsize: Default figure size.
            style: Matplotlib style to use.
        """
        _check_matplotlib()
        self.figsize = figsize
        self.style = style

    def plot_equity_curves(
        self,
        result: "CrossValidationResult",
        show_splits: bool = True,
        title: str = "Walk-Forward Equity Curves",
    ) -> None:
        """Plot IS and OOS equity curves.

        Args:
            result: CrossValidationResult with splits.
            show_splits: Show vertical lines at split boundaries.
            title: Plot title.
        """
        with plt.style.context(self.style):
            fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

            # IS scores
            split_indices = [s.split_idx for s in result.splits]
            train_scores = [s.train_score for s in result.splits]
            test_scores = [s.test_score for s in result.splits]

            ax1 = axes[0]
            ax1.bar(split_indices, train_scores, color='blue', alpha=0.7, label='Train (IS)')
            ax1.set_ylabel('Score')
            ax1.set_title('In-Sample Performance')
            ax1.legend()
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # OOS scores
            ax2 = axes[1]
            colors = ['green' if s > 0 else 'red' for s in test_scores]
            ax2.bar(split_indices, test_scores, color=colors, alpha=0.7, label='Test (OOS)')
            ax2.set_xlabel('Split')
            ax2.set_ylabel('Score')
            ax2.set_title('Out-of-Sample Performance')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Add summary
            fig.suptitle(
                f"{title}\n"
                f"Robustness: {result.robustness_score:.2f} | "
                f"Mean OOS: {result.mean_test_score:.4f} | "
                f"Positive OOS: {result.positive_oos_ratio:.1%}",
                fontsize=12,
            )

            plt.tight_layout()
            plt.show()

    def plot_degradation_heatmap(
        self,
        result: "CrossValidationResult",
        title: str = "IS→OOS Degradation",
    ) -> None:
        """Plot heatmap of IS→OOS degradation by split.

        Args:
            result: CrossValidationResult with splits.
            title: Plot title.
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create degradation matrix
            n_splits = len(result.splits)
            degradations = [s.degradation_pct for s in result.splits]
            train_scores = [s.train_score for s in result.splits]
            test_scores = [s.test_score for s in result.splits]

            # Create DataFrame for heatmap
            data = pd.DataFrame({
                'Train': train_scores,
                'Test': test_scores,
                'Degradation %': degradations,
            }, index=[f'Split {i}' for i in range(n_splits)])

            # Normalize for color mapping
            norm_deg = np.array(degradations)
            vmin, vmax = min(norm_deg.min(), -50), max(norm_deg.max(), 50)

            # Plot as bar chart with color gradient
            colors = plt.cm.RdYlGn_r((norm_deg - vmin) / (vmax - vmin))

            bars = ax.barh(range(n_splits), degradations, color=colors)
            ax.set_yticks(range(n_splits))
            ax.set_yticklabels([f'Split {i}' for i in range(n_splits)])
            ax.set_xlabel('Degradation (%)')
            ax.set_title(title)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            # Add value labels
            for i, (bar, deg) in enumerate(zip(bars, degradations)):
                ax.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f'{deg:.1f}%',
                    va='center',
                )

            plt.tight_layout()
            plt.show()

    def plot_parameter_stability(
        self,
        stability: Dict[str, "ParameterStability"],
        title: str = "Parameter Stability Across Splits",
    ) -> None:
        """Plot parameter stability box plots.

        Args:
            stability: Dictionary of ParameterStability objects.
            title: Plot title.
        """
        with plt.style.context(self.style):
            # Filter to numeric parameters only
            numeric_params = {}
            for name, param in stability.items():
                try:
                    values = [float(v) for v in param.values_per_split if v is not None]
                    if values:
                        numeric_params[name] = values
                except (TypeError, ValueError):
                    continue

            if not numeric_params:
                print("No numeric parameters to plot")
                return

            fig, ax = plt.subplots(figsize=self.figsize)

            # Create box plot
            positions = list(range(len(numeric_params)))
            bp = ax.boxplot(
                list(numeric_params.values()),
                positions=positions,
                patch_artist=True,
            )

            # Color stable vs unstable
            for i, (name, values) in enumerate(numeric_params.items()):
                is_stable = stability[name].is_stable
                color = 'lightgreen' if is_stable else 'lightcoral'
                bp['boxes'][i].set_facecolor(color)

            ax.set_xticks(positions)
            ax.set_xticklabels(list(numeric_params.keys()), rotation=45, ha='right')
            ax.set_ylabel('Parameter Value')
            ax.set_title(title)

            # Legend
            legend_elements = [
                Patch(facecolor='lightgreen', label='Stable'),
                Patch(facecolor='lightcoral', label='Unstable'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            plt.show()

    def plot_split_timeline(
        self,
        data: pd.DataFrame,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        title: str = "Train/Test Split Timeline",
    ) -> None:
        """Plot timeline of train/test periods.

        Args:
            data: DataFrame with datetime index.
            splits: List of (train_idx, test_idx) tuples.
            title: Plot title.
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            n_splits = len(splits)
            has_datetime = isinstance(data.index, pd.DatetimeIndex)

            for i, (train_idx, test_idx) in enumerate(splits):
                y_pos = n_splits - i - 1

                if has_datetime:
                    train_start = data.index[train_idx[0]]
                    train_end = data.index[train_idx[-1]]
                    test_start = data.index[test_idx[0]]
                    test_end = data.index[test_idx[-1]]
                else:
                    train_start = train_idx[0]
                    train_end = train_idx[-1]
                    test_start = test_idx[0]
                    test_end = test_idx[-1]

                # Plot train period
                ax.barh(
                    y_pos, train_end - train_start, left=train_start,
                    color='blue', alpha=0.6, height=0.4, label='Train' if i == 0 else ''
                )

                # Plot test period
                ax.barh(
                    y_pos, test_end - test_start, left=test_start,
                    color='orange', alpha=0.6, height=0.4, label='Test' if i == 0 else ''
                )

            ax.set_yticks(range(n_splits))
            ax.set_yticklabels([f'Split {i}' for i in range(n_splits - 1, -1, -1)])

            if has_datetime:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)

            ax.set_xlabel('Date' if has_datetime else 'Index')
            ax.set_title(title)
            ax.legend()

            plt.tight_layout()
            plt.show()

    def plot_robustness_breakdown(
        self,
        result: "CrossValidationResult",
        title: str = "Robustness Score Breakdown",
    ) -> None:
        """Plot breakdown of robustness score components.

        Args:
            result: CrossValidationResult with robustness info.
            title: Plot title.
        """
        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)

            # Pie chart of components
            ax1 = axes[0]
            positive_oos = result.positive_oos_ratio
            consistency = result.consistency_score
            low_deg = max(0, 1 - result.mean_degradation / 100)

            labels = ['Positive OOS', 'Consistency', 'Low Degradation']
            sizes = [positive_oos * 0.4, consistency * 0.3, low_deg * 0.3]
            colors = ['#2ecc71', '#3498db', '#9b59b6']

            ax1.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90
            )
            ax1.set_title('Robustness Components')

            # Bar chart of metrics
            ax2 = axes[1]
            metrics = {
                'Positive OOS Ratio': positive_oos,
                'Consistency': consistency,
                'Low Degradation': low_deg,
                'Robustness Score': result.robustness_score,
            }

            bars = ax2.bar(
                metrics.keys(),
                metrics.values(),
                color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
            )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Score')
            ax2.set_title('Metric Values')

            for bar, val in zip(bars, metrics.values()):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}',
                    ha='center',
                )

            plt.xticks(rotation=45, ha='right')
            fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            plt.show()

    def plot_parameter_evolution(
        self,
        stability: Dict[str, "ParameterStability"],
        param_name: str,
        title: Optional[str] = None,
    ) -> None:
        """Plot evolution of a parameter across splits.

        Args:
            stability: Dictionary of ParameterStability.
            param_name: Name of parameter to plot.
            title: Plot title.
        """
        if param_name not in stability:
            raise ValueError(f"Parameter {param_name} not found")

        param = stability[param_name]

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(10, 6))

            values = param.values_per_split
            splits = range(len(values))

            # Try numeric plot
            try:
                numeric_values = [float(v) for v in values]
                ax.plot(splits, numeric_values, 'o-', linewidth=2, markersize=8)
                ax.axhline(
                    y=param.mean, color='r', linestyle='--',
                    label=f'Mean: {param.mean:.4f}'
                )
                ax.fill_between(
                    splits,
                    [param.mean - param.std] * len(splits),
                    [param.mean + param.std] * len(splits),
                    alpha=0.2, color='red', label=f'±1 Std: {param.std:.4f}'
                )
            except (TypeError, ValueError):
                # Categorical
                ax.scatter(splits, values, s=100)

            ax.set_xlabel('Split')
            ax.set_ylabel(param_name)
            ax.set_title(title or f'Parameter Evolution: {param_name}')
            ax.set_xticks(list(splits))

            # Add stability info
            status = "Stable" if param.is_stable else "Unstable"
            ax.text(
                0.02, 0.98,
                f'CV: {param.coefficient_of_variation:.3f} ({status})',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            )

            ax.legend()
            plt.tight_layout()
            plt.show()


def plot_equity_curves(
    result: "CrossValidationResult",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot IS and OOS equity curves.

    Convenience function.

    Args:
        result: CrossValidationResult.
        figsize: Figure size.
    """
    viz = ValidationVisualizer(figsize=figsize)
    viz.plot_equity_curves(result)


def plot_parameter_evolution(
    stability: Dict[str, "ParameterStability"],
    param_name: str,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot parameter evolution across splits.

    Convenience function.

    Args:
        stability: Parameter stability results.
        param_name: Parameter to plot.
        figsize: Figure size.
    """
    viz = ValidationVisualizer(figsize=figsize)
    viz.plot_parameter_evolution(stability, param_name)


def plot_split_timeline(
    data: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot timeline of train/test splits.

    Convenience function.

    Args:
        data: DataFrame with time series data.
        splits: List of (train_idx, test_idx) tuples.
        figsize: Figure size.
    """
    viz = ValidationVisualizer(figsize=figsize)
    viz.plot_split_timeline(data, splits)
