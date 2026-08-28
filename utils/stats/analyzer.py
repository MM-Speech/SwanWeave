# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis


Number = Union[int, float, np.number]
ArrayLike = Union[Sequence[Number], np.ndarray, pd.Series]
DataLike = Union[ArrayLike, pd.DataFrame]


# ---------------------------------------------------------------------------
# Helper dataclasses for configuration
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Configuration for plotting behavior."""
    figsize: Tuple[float, float] = (8.0, 5.0)
    style: str = "default"          # matplotlib style name, e.g. 'ggplot', 'seaborn-v0_8'
    dpi: int = 100
    grid: bool = True
    tight_layout: bool = True

    def apply(self):
        """Apply global plotting style (per plot)."""
        plt.style.use(self.style)


@dataclass
class SaveConfig:
    """Configuration for saving figure to disk."""
    save_path: Optional[str] = None    # If None, no file will be saved.
    dpi: Optional[int] = None          # If None, use PlotConfig.dpi
    bbox_inches: str = "tight"
    transparent: bool = False

    def effective_dpi(self, default_dpi: int) -> int:
        return self.dpi if self.dpi is not None else default_dpi


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseStatsAnalyzer(ABC):
    """
    Abstract base class for statistical data analyzers.

    Parameters
    ----------
    data : DataLike
        Input data: 1D/2D array-like, Series or DataFrame.
    columns : Optional[Sequence[str]]
        Columns to focus on (for DataFrame). If None, all numeric columns.
    copy : bool
        Whether to copy input data.
    dropna : bool
        Whether to drop NA values for statistics by default.
    """

    def __init__(
        self,
        data: DataLike,
        columns: Optional[Sequence[str]] = None,
        copy: bool = True,
        dropna: bool = True,
    ) -> None:
        self.dropna = dropna
        self._original_data = data

        self.data = self._to_dataframe(data, copy=copy)
        self.columns = self._resolve_columns(columns)

    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _to_dataframe(data: DataLike, copy: bool = True) -> pd.DataFrame:
        """
        Convert input data to a pandas DataFrame for unified processing.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy() if copy else data

        if isinstance(data, pd.Series):
            df = data.to_frame(name=data.name or "value")
            return df.copy() if copy else df

        if isinstance(data, (list, tuple, np.ndarray)):
            arr = np.asarray(data)
            if arr.ndim == 1:
                return pd.DataFrame({"value": arr.copy() if copy else arr})
            else:
                # Create generic column names
                cols = [f"col_{i}" for i in range(arr.shape[1])]
                return pd.DataFrame(arr.copy() if copy else arr, columns=cols)

        raise TypeError(f"Unsupported data type: {type(data)}")

    def _resolve_columns(self, columns: Optional[Sequence[str]]) -> List[str]:
        """
        Determine which columns to analyze: numeric columns by default.
        """
        df = self.data
        if columns is not None:
            # Only keep columns that exist & are numeric
            cols = [c for c in columns if c in df.columns]
            numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in provided 'columns'.")
            return numeric_cols

        # Default: all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in data.")
        return numeric_cols

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _select_series(self, column: Optional[str] = None) -> pd.Series:
        """
        Select a 1D numeric Series for analysis.
        """
        if column is None:
            if len(self.columns) != 1:
                raise ValueError(
                    f"Multiple numeric columns available {self.columns}, "
                    "please specify 'column'."
                )
            column = self.columns[0]

        if column not in self.columns:
            raise ValueError(f"Column '{column}' is not in numeric columns {self.columns}.")

        s = self.data[column]
        if self.dropna:
            return s.dropna()
        return s

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------
    def describe(
        self,
        percentiles: Sequence[float] = (0.25, 0.5, 0.75),
        include_skew_kurt: bool = True,
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics for numeric columns.

        Similar to pandas.DataFrame.describe, but adds skewness & kurtosis.

        Returns
        -------
        stats : DataFrame
            Index: statistic names.
            Columns: numeric columns.
        """
        df = self.data[self.columns]
        if self.dropna:
            df = df.dropna(axis=0, how="any")

        desc = df.describe(percentiles=list(percentiles))  # count, mean, std, min, percentiles, max

        if include_skew_kurt:
            # Skew & kurtosis for each column
            skew_vals = df.apply(lambda x: skew(x, nan_policy="omit"))
            kurt_vals = df.apply(lambda x: kurtosis(x, nan_policy="omit"))

            desc.loc["skew"] = skew_vals
            desc.loc["kurtosis"] = kurt_vals

        return desc

    def missing_report(self) -> pd.Series:
        """
        Report missing value ratio per column (0~1).
        """
        total = len(self.data)
        if total == 0:
            return pd.Series(dtype=float)

        miss_ratio = self.data[self.columns].isna().sum() / total
        miss_ratio.name = "missing_ratio"
        return miss_ratio

    def basic_stats(
        self,
        column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute basic stats for a single column: min, max, mean, median, std, var,
        q1, q3, iqr, skew, kurtosis.
        """
        s = self._select_series(column)
        if len(s) == 0:
            return {}

        arr = s.to_numpy()
        result = {
            "count": int(arr.size),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "std": float(np.nanstd(arr, ddof=1)),
            "var": float(np.nanvar(arr, ddof=1)),
        }

        q1, q3 = np.nanpercentile(arr, [25, 75])
        result["q1"] = float(q1)
        result["q3"] = float(q3)
        result["iqr"] = float(q3 - q1)
        result["skew"] = float(skew(arr, nan_policy="omit"))
        result["kurtosis"] = float(kurtosis(arr, nan_policy="omit"))

        return result

    # ------------------------------------------------------------------
    # Plotting helper
    # ------------------------------------------------------------------
    def _finalize_figure(
        self,
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Apply layout, save figure if needed, and show/close.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        if plot_config.tight_layout:
            plt.tight_layout()

        if save_config and save_config.save_path:
            dpi = save_config.effective_dpi(plot_config.dpi)
            plt.savefig(
                save_config.save_path,
                dpi=dpi,
                bbox_inches=save_config.bbox_inches,
                transparent=save_config.transparent,
            )

        if show:
            plt.show()
        else:
            # To release memory when many plots are generated in batch
            plt.close()

    # ------------------------------------------------------------------
    # Abstract methods for children to implement key plots
    # ------------------------------------------------------------------
    @abstractmethod
    def summary_report(self) -> Dict[str, Any]:
        """
        A high-level summary report, combining several statistics.

        Must be implemented by subclass, depending on the nature of data
        (distribution vs time series).
        """
        ...


# ---------------------------------------------------------------------------
# Distribution analyzer (histogram, pie chart, boxplot, etc.)
# ---------------------------------------------------------------------------

class DistributionStatsAnalyzer(BaseStatsAnalyzer):
    """
    Analyzer for general numeric distributions (i.e. unordered samples).

    Suitable for:
    - Histogram, density plot (KDE)
    - Boxplot, violin (if extended)
    - Bar chart, pie chart (for categorical counts)
    """

    # ----------------------------------------------
    # Implementation of summary report
    # ----------------------------------------------
    def summary_report(self) -> Dict[str, Any]:
        """
        High-level summary: describe + missing + per-column basic stats.
        """
        report = {
            "describe": self.describe(),
            "missing": self.missing_report(),
            "basic_stats": {
                col: self.basic_stats(col) for col in self.columns
            },
        }
        return report

    # ----------------------------------------------
    # Histogram
    # ----------------------------------------------
    def plot_histogram(
        self,
        column: Optional[str] = None,
        bins: Union[int, Sequence[float]] = 30,
        density: bool = False,
        alpha: float = 0.75,
        color: str = "C0",
        edgecolor: str = "black",
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Plot a histogram for the given numeric column.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        s = self._select_series(column)
        plt.hist(
            s.values,
            bins=bins,
            density=density,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
        )

        col = s.name
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Density" if density else "Frequency")
        if plot_config.grid:
            plt.grid(True, linestyle="--", alpha=0.5)

        self._finalize_figure(plot_config, save_config, show)

    # ----------------------------------------------
    # Boxplot
    # ----------------------------------------------
    def plot_boxplot(
        self,
        columns: Optional[Sequence[str]] = None,
        vert: bool = True,
        showfliers: bool = True,
        patch_artist: bool = True,
        box_colors: Optional[Sequence[str]] = None,
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Plot boxplots for one or multiple numeric columns.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        cols = list(columns) if columns is not None else self.columns
        data = [self._select_series(c).values for c in cols]

        bp = plt.boxplot(
            data,
            vert=vert,
            showfliers=showfliers,
            patch_artist=patch_artist,
        )

        if patch_artist:
            if box_colors is None:
                box_colors = [f"C{i}" for i in range(len(cols))]
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        axis = plt.gca()
        if vert:
            axis.set_xticklabels(cols)
            axis.set_ylabel("Value")
        else:
            axis.set_yticklabels(cols)
            axis.set_xlabel("Value")

        plt.title("Boxplot")
        if plot_config.grid:
            plt.grid(True, axis="y" if vert else "x", linestyle="--", alpha=0.5)

        self._finalize_figure(plot_config, save_config, show)

    # ----------------------------------------------
    # Bar / Pie chart for categorical distribution
    # ----------------------------------------------
    def _value_counts(
        self,
        column: str,
        normalize: bool = False,
        top_n: Optional[int] = None,
        others_label: str = "Others",
    ) -> pd.Series:
        """
        Compute value counts for a single column, possibly grouped into top-N + others.
        """
        s = self.data[column]
        if self.dropna:
            s = s.dropna()

        counts = s.value_counts(normalize=normalize)

        if top_n is not None and len(counts) > top_n:
            top = counts.iloc[:top_n]
            others = counts.iloc[top_n:].sum()
            top[others_label] = others
            return top

        return counts

    def plot_bar(
        self,
        column: str,
        top_n: Optional[int] = None,
        normalize: bool = False,
        color: str = "C0",
        rotation: int = 45,
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Plot bar chart for categorical distribution (value counts).
        """
        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        counts = self._value_counts(column, normalize=normalize, top_n=top_n)

        counts.plot(kind="bar", color=color)

        ylabel = "Proportion" if normalize else "Count"
        plt.ylabel(ylabel)
        plt.xlabel(column)
        plt.title(f"Bar chart of {column}")
        plt.xticks(rotation=rotation)
        if plot_config.grid:
            plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        self._finalize_figure(plot_config, save_config, show)

    def plot_pie(
        self,
        column: str,
        top_n: Optional[int] = None,
        normalize: bool = True,
        autopct: str = "%1.1f%%",
        startangle: int = 90,
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Plot pie chart for categorical distribution.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        counts = self._value_counts(column, normalize=normalize, top_n=top_n)
        plt.pie(
            counts.values,
            labels=counts.index,
            autopct=autopct,
            startangle=startangle,
        )
        plt.title(f"Pie chart of {column}")
        plt.axis("equal")

        self._finalize_figure(plot_config, save_config, show)


# ---------------------------------------------------------------------------
# Time-series / ordered series analyzer (line plots, rolling stats, etc.)
# ---------------------------------------------------------------------------

class TimeSeriesStatsAnalyzer(BaseStatsAnalyzer):
    """
    Analyzer for time series or generally ordered numeric sequences.

    If index is datetime-like, additional time-specific methods can be extended.
    """

    def __init__(
        self,
        data: DataLike,
        columns: Optional[Sequence[str]] = None,
        copy: bool = True,
        dropna: bool = True,
        sort_index: bool = True,
    ) -> None:
        super().__init__(data, columns=columns, copy=copy, dropna=dropna)

        # For time series, it's usually safer to sort by index
        if sort_index:
            self.data = self.data.sort_index()

    # ----------------------------------------------
    # Summary report
    # ----------------------------------------------
    def summary_report(self) -> Dict[str, Any]:
        """
        High-level summary for time series:
        - describe
        - missing report
        - per column basic stats
        - index info (range, frequency estimate)
        """
        idx = self.data.index

        idx_info = {
            "index_type": type(idx).__name__,
            "start": idx[0] if len(idx) > 0 else None,
            "end": idx[-1] if len(idx) > 0 else None,
            "length": len(idx),
        }

        if hasattr(idx, "inferred_freq"):
            idx_info["inferred_freq"] = getattr(idx, "inferred_freq", None)

        report = {
            "describe": self.describe(),
            "missing": self.missing_report(),
            "basic_stats": {
                col: self.basic_stats(col) for col in self.columns
            },
            "index_info": idx_info,
        }
        return report

    # ----------------------------------------------
    # Line plot
    # ----------------------------------------------
    def plot_line(
        self,
        columns: Optional[Sequence[str]] = None,
        linewidth: float = 1.5,
        alpha: float = 0.9,
        markers: bool = False,
        marker: str = "o",
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
        legend: bool = True,
    ) -> None:
        """
        Plot line chart(s) for time series.

        Parameters
        ----------
        columns : Optional[Sequence[str]]
            Which numeric columns to plot. Default: self.columns.
        markers : bool
            Whether to show markers on each point.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        cols = list(columns) if columns is not None else self.columns

        for i, col in enumerate(cols):
            s = self._select_series(col)
            m = marker if markers else None
            plt.plot(
                s.index,
                s.values,
                label=col,
                linewidth=linewidth,
                alpha=alpha,
                marker=m,
            )

        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Time Series Line Plot")
        if legend:
            plt.legend()
        if plot_config.grid:
            plt.grid(True, linestyle="--", alpha=0.5)

        self._finalize_figure(plot_config, save_config, show)

    # ----------------------------------------------
    # Rolling statistics
    # ----------------------------------------------
    def plot_rolling_stats(
        self,
        column: str,
        window: int = 10,
        center: bool = False,
        plot_std: bool = True,
        linewidth: float = 1.5,
        alpha: float = 0.8,
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Plot original series and its rolling mean (and optionally rolling std).
        """
        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        s = self._select_series(column)

        roll_mean = s.rolling(window=window, center=center).mean()
        roll_std = s.rolling(window=window, center=center).std()

        plt.plot(s.index, s.values, label="Original", color="C0", alpha=0.5)
        plt.plot(
            roll_mean.index,
            roll_mean.values,
            label=f"Rolling Mean (window={window})",
            color="C1",
            linewidth=linewidth,
            alpha=alpha,
        )

        if plot_std:
            plt.plot(
                roll_std.index,
                roll_std.values,
                label=f"Rolling Std (window={window})",
                color="C2",
                linewidth=linewidth,
                alpha=alpha,
            )

        plt.xlabel("Index")
        plt.ylabel(column)
        plt.title(f"Rolling Statistics for {column}")
        plt.legend()
        if plot_config.grid:
            plt.grid(True, linestyle="--", alpha=0.5)

        self._finalize_figure(plot_config, save_config, show)

    # ----------------------------------------------
    # Lag plot (auto-correlation visually)
    # ----------------------------------------------
    def plot_lag(
        self,
        column: str,
        lag: int = 1,
        plot_config: Optional[PlotConfig] = None,
        save_config: Optional[SaveConfig] = None,
        show: bool = True,
    ) -> None:
        """
        Simple lag plot: x_t vs x_{t-lag}.
        """
        if lag <= 0:
            raise ValueError("lag must be > 0.")

        if plot_config is None:
            plot_config = PlotConfig()

        plot_config.apply()
        plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)

        s = self._select_series(column)
        x = s.iloc[:-lag].values
        y = s.iloc[lag:].values

        plt.scatter(x, y, alpha=0.6, s=20)
        plt.xlabel(f"{column}(t-lag)")
        plt.ylabel(f"{column}(t)")
        plt.title(f"Lag Plot (lag={lag}) for {column}")
        if plot_config.grid:
            plt.grid(True, linestyle="--", alpha=0.5)

        self._finalize_figure(plot_config, save_config, show)


if __name__ == '__main__':
    ################
    # Distribution #
    ################

    np.random.seed(42)
    df = pd.DataFrame({
        "height": np.random.normal(loc=170, scale=10, size=1000),
        "weight": np.random.normal(loc=65, scale=8, size=1000),
        "city": np.random.choice(["Hangzhou", "Beijing", "Shanghai"], size=1000)
    })

    dist_analyzer = DistributionStatsAnalyzer(df)

    report = dist_analyzer.summary_report()
    print(report["describe"])
    print(report["missing"])
    print(report["basic_stats"]["height"])

    dist_analyzer.plot_histogram(
        column="height",
        bins=40,
        density=True,
        color="steelblue",
        plot_config=PlotConfig(style="ggplot", figsize=(7, 4))
    )

    dist_analyzer.plot_boxplot(
        columns=["height", "weight"],
        plot_config=PlotConfig(style="ggplot", figsize=(6, 4))
    )

    dist_analyzer.plot_pie(
        column="city",
        top_n=3,
        normalize=True,
        autopct="%1.0f%%",
        plot_config=PlotConfig(style="ggplot"),
        save_config=SaveConfig(save_path="city_pie.png"),
        show=False,  # 不弹出图像窗口
    )



