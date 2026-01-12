"""Overfitting detection tools for strategy validation.

Implements statistical tests for detecting overfitting in backtests,
including Deflated Sharpe Ratio and Probability of Backtest Overfitting.

Based on research by Marcos López de Prado and Bailey & López de Prado.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class OverfittingTestResult:
    """Results from an overfitting detection test."""

    test_name: str
    statistic: float
    p_value: float
    is_overfitted: bool
    confidence_level: float
    details: Dict[str, Any]


class OverfittingDetector:
    """Detects overfitting using statistical methods.

    Implements multiple tests for detecting backtest overfitting:

    1. **Deflated Sharpe Ratio (DSR)**: Adjusts Sharpe ratio for multiple testing
    2. **Probability of Backtest Overfitting (PBO)**: Estimates probability that
       the best IS strategy is worse than median OOS
    3. **Performance Degradation Test**: Statistical test for IS→OOS degradation

    Example:
        >>> detector = OverfittingDetector()
        >>> pbo = detector.probability_of_backtest_overfitting(
        ...     is_sharpes=[0.8, 0.9, 1.0, 1.1],
        ...     oos_sharpes=[0.3, 0.4, 0.5, 0.6]
        ... )
        >>> print(f"PBO: {pbo:.1%}")
    """

    def __init__(
        self,
        significance_level: float = 0.05,
    ):
        """Initialize OverfittingDetector.

        Args:
            significance_level: Alpha level for statistical tests.
        """
        if not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")

        self.significance_level = significance_level

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        variance_sharpe: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        annualization_factor: float = 252.0,
    ) -> OverfittingTestResult:
        """Calculate Deflated Sharpe Ratio.

        The DSR adjusts the Sharpe ratio for multiple testing by accounting
        for the expected maximum Sharpe ratio under the null hypothesis.

        Based on Bailey & López de Prado (2014):
        "The Deflated Sharpe Ratio: Correcting for Selection Bias,
        Backtest Overfitting, and Non-Normality"

        Args:
            observed_sharpe: Observed Sharpe ratio from backtest.
            n_trials: Number of strategy configurations tested.
            variance_sharpe: Variance of Sharpe ratios across trials.
            skewness: Skewness of returns (0 for normal).
            kurtosis: Kurtosis of returns (3 for normal).
            annualization_factor: Factor for annualization (252 for daily).

        Returns:
            OverfittingTestResult with DSR and probability of overfitting.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if variance_sharpe < 0:
            raise ValueError("variance_sharpe must be non-negative")

        # Expected maximum Sharpe ratio under null (Euler-Mascheroni)
        euler_gamma = 0.5772156649
        expected_max_sr = np.sqrt(variance_sharpe) * (
            (1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) +
            euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

        # Deflated Sharpe Ratio
        if variance_sharpe == 0:
            dsr = float('inf') if observed_sharpe > 0 else 0
            p_value = 0.0
        else:
            # Standard error of Sharpe ratio (accounting for non-normality)
            sr_std = np.sqrt(
                (1 + 0.5 * observed_sharpe ** 2 -
                 skewness * observed_sharpe +
                 (kurtosis - 3) / 4 * observed_sharpe ** 2) / annualization_factor
            )

            # Z-score for deflated test
            z_score = (observed_sharpe - expected_max_sr) / sr_std
            dsr = float(z_score)

            # P-value (one-tailed)
            p_value = float(1 - stats.norm.cdf(z_score))

        is_overfitted = p_value > self.significance_level

        return OverfittingTestResult(
            test_name="Deflated Sharpe Ratio",
            statistic=dsr,
            p_value=p_value,
            is_overfitted=is_overfitted,
            confidence_level=1 - self.significance_level,
            details={
                "observed_sharpe": observed_sharpe,
                "expected_max_sharpe": expected_max_sr,
                "n_trials": n_trials,
                "variance_sharpe": variance_sharpe,
            },
        )

    def probability_of_backtest_overfitting(
        self,
        is_sharpes: List[float],
        oos_sharpes: List[float],
    ) -> float:
        """Calculate Probability of Backtest Overfitting (PBO).

        PBO estimates the probability that the best in-sample strategy
        will perform worse than the median out-of-sample.

        Based on Bailey et al. (2015):
        "Probability of Backtest Overfitting"

        Args:
            is_sharpes: List of in-sample Sharpe ratios.
            oos_sharpes: Corresponding list of out-of-sample Sharpe ratios.

        Returns:
            PBO as a probability (0-1).
        """
        if len(is_sharpes) != len(oos_sharpes):
            raise ValueError("is_sharpes and oos_sharpes must have same length")
        if len(is_sharpes) < 2:
            raise ValueError("Need at least 2 strategies for PBO calculation")

        is_arr = np.array(is_sharpes)
        oos_arr = np.array(oos_sharpes)

        # Find best IS strategy
        best_is_idx = int(np.argmax(is_arr))
        best_oos_sharpe = float(oos_arr[best_is_idx])

        # Calculate median OOS Sharpe
        median_oos = float(np.median(oos_arr))

        # PBO: probability that best IS is worse than median OOS
        # Use rank-based estimation
        rank_of_best = int(np.sum(oos_arr < best_oos_sharpe))

        # Simple PBO estimate
        pbo = rank_of_best / len(oos_arr)

        # Adjust if best IS has negative OOS rank
        if best_oos_sharpe < median_oos:
            pbo = max(pbo, 0.5)

        return float(pbo)

    def performance_degradation_test(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray,
        test_type: str = "welch",
    ) -> OverfittingTestResult:
        """Test for significant performance degradation from IS to OOS.

        Args:
            is_returns: In-sample returns array.
            oos_returns: Out-of-sample returns array.
            test_type: Type of test ('welch', 'mann_whitney', 'bootstrap').

        Returns:
            OverfittingTestResult with degradation assessment.
        """
        if len(is_returns) < 5 or len(oos_returns) < 5:
            raise ValueError("Need at least 5 samples in each set")

        is_returns = np.array(is_returns)
        oos_returns = np.array(oos_returns)

        # Calculate summary statistics
        is_mean = float(np.mean(is_returns))
        oos_mean = float(np.mean(oos_returns))
        degradation = is_mean - oos_mean
        degradation_pct = (degradation / abs(is_mean) * 100) if is_mean != 0 else 0

        statistic: float
        p_value: float

        if test_type == "welch":
            # Welch's t-test (unequal variances)
            result = stats.ttest_ind(
                is_returns, oos_returns,
                equal_var=False,
                alternative='greater'
            )
            statistic = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
        elif test_type == "mann_whitney":
            # Non-parametric Mann-Whitney U test
            result = stats.mannwhitneyu(
                is_returns, oos_returns,
                alternative='greater'
            )
            statistic = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
        elif test_type == "bootstrap":
            # Bootstrap test for mean difference
            statistic, p_value = self._bootstrap_test(is_returns, oos_returns)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        is_overfitted = bool(p_value < self.significance_level and degradation > 0)

        return OverfittingTestResult(
            test_name=f"Performance Degradation ({test_type})",
            statistic=statistic,
            p_value=p_value,
            is_overfitted=is_overfitted,
            confidence_level=1 - self.significance_level,
            details={
                "is_mean": is_mean,
                "oos_mean": oos_mean,
                "degradation": degradation,
                "degradation_pct": degradation_pct,
                "is_std": float(np.std(is_returns)),
                "oos_std": float(np.std(oos_returns)),
            },
        )

    def _bootstrap_test(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray,
        n_bootstrap: int = 10000,
    ) -> Tuple[float, float]:
        """Bootstrap test for mean difference.

        Args:
            is_returns: In-sample returns.
            oos_returns: Out-of-sample returns.
            n_bootstrap: Number of bootstrap samples.

        Returns:
            Tuple of (statistic, p_value).
        """
        observed_diff = np.mean(is_returns) - np.mean(oos_returns)

        # Pool all returns under null
        pooled = np.concatenate([is_returns, oos_returns])
        n_is = len(is_returns)

        # Bootstrap sampling
        diffs = []
        for _ in range(n_bootstrap):
            permuted = np.random.permutation(pooled)
            boot_is = permuted[:n_is]
            boot_oos = permuted[n_is:]
            diffs.append(np.mean(boot_is) - np.mean(boot_oos))

        diffs = np.array(diffs)

        # P-value: proportion of bootstrap diffs >= observed
        p_value = np.mean(diffs >= observed_diff)

        return float(observed_diff), float(p_value)

    def comprehensive_test(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray,
        n_trials: int = 1,
    ) -> Dict[str, OverfittingTestResult]:
        """Run all overfitting tests.

        Args:
            is_returns: In-sample returns array.
            oos_returns: Out-of-sample returns array.
            n_trials: Number of strategies tested.

        Returns:
            Dictionary of test results.
        """
        is_returns = np.array(is_returns)
        oos_returns = np.array(oos_returns)

        results = {}

        # Calculate Sharpe ratios
        is_sharpe = float(np.mean(is_returns) / np.std(is_returns)) if np.std(is_returns) > 0 else 0
        oos_sharpe = float(np.mean(oos_returns) / np.std(oos_returns)) if np.std(oos_returns) > 0 else 0

        # DSR test
        if n_trials > 1:
            variance = float(np.var(is_returns)) if len(is_returns) > 1 else 0.1
            results["dsr"] = self.deflated_sharpe_ratio(
                observed_sharpe=is_sharpe,
                n_trials=n_trials,
                variance_sharpe=variance,
            )

        # Degradation test
        results["degradation"] = self.performance_degradation_test(
            is_returns=is_returns,
            oos_returns=oos_returns,
        )

        return results

    def print_summary(
        self,
        results: Dict[str, OverfittingTestResult],
    ) -> None:
        """Print summary of overfitting test results.

        Args:
            results: Dictionary of test results.
        """
        print("=" * 60)
        print("OVERFITTING DETECTION RESULTS")
        print("=" * 60)

        for name, result in results.items():
            status = "OVERFITTED" if result.is_overfitted else "OK"
            print(f"\n{result.test_name}:")
            print(f"  Statistic: {result.statistic:.4f}")
            print(f"  P-value:   {result.p_value:.4f}")
            print(f"  Status:    {status}")

            for key, value in result.details.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 60)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    variance_sharpe: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> Tuple[float, float]:
    """Calculate Deflated Sharpe Ratio.

    Convenience function for quick DSR calculation.

    Args:
        observed_sharpe: Observed Sharpe ratio.
        n_trials: Number of trials/strategies tested.
        variance_sharpe: Variance of Sharpe ratios.
        skewness: Return skewness.
        kurtosis: Return kurtosis.

    Returns:
        Tuple of (deflated_sharpe, probability_of_overfitting).
    """
    detector = OverfittingDetector()
    result = detector.deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        variance_sharpe=variance_sharpe,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return result.statistic, result.p_value


def probability_of_backtest_overfitting(
    is_sharpes: List[float],
    oos_sharpes: List[float],
) -> float:
    """Calculate Probability of Backtest Overfitting (PBO).

    Convenience function for quick PBO calculation.

    Args:
        is_sharpes: In-sample Sharpe ratios.
        oos_sharpes: Out-of-sample Sharpe ratios.

    Returns:
        PBO probability (0-1).
    """
    detector = OverfittingDetector()
    return detector.probability_of_backtest_overfitting(is_sharpes, oos_sharpes)


def performance_degradation_test(
    is_returns: np.ndarray,
    oos_returns: np.ndarray,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Test for performance degradation.

    Convenience function for quick degradation test.

    Args:
        is_returns: In-sample returns.
        oos_returns: Out-of-sample returns.
        significance_level: Alpha level.

    Returns:
        Dictionary with test results.
    """
    detector = OverfittingDetector(significance_level=significance_level)
    result = detector.performance_degradation_test(is_returns, oos_returns)

    return {
        "statistic": result.statistic,
        "p_value": result.p_value,
        "is_overfitted": result.is_overfitted,
        **result.details,
    }


def estimate_haircut_sharpe(
    observed_sharpe: float,
    n_trials: int,
    returns_std: float = 0.01,
) -> float:
    """Estimate haircut to apply to Sharpe ratio.

    Returns a more realistic Sharpe estimate accounting for
    multiple testing and selection bias.

    Args:
        observed_sharpe: Observed (potentially inflated) Sharpe.
        n_trials: Number of strategies tested.
        returns_std: Standard deviation of returns.

    Returns:
        Haircut Sharpe ratio.
    """
    if n_trials <= 1:
        return observed_sharpe

    # Expected maximum Sharpe under null
    euler_gamma = 0.5772156649
    expected_max = np.sqrt(2 * np.log(n_trials)) - (
        (np.log(np.log(n_trials)) + np.log(4 * np.pi)) /
        (2 * np.sqrt(2 * np.log(n_trials)))
    )

    # Haircut: subtract expected maximum under null
    haircut_sharpe = max(0, observed_sharpe - expected_max * returns_std)

    return float(haircut_sharpe)
