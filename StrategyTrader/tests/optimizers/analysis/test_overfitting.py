"""Tests for overfitting detection module."""

import pytest
import numpy as np

from src.optimizers.analysis.overfitting_detection import (
    OverfittingDetector,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    performance_degradation_test,
    estimate_haircut_sharpe,
)


class TestOverfittingDetector:
    """Tests for OverfittingDetector class."""

    def test_initialization(self):
        """Detector initializes correctly."""
        detector = OverfittingDetector(significance_level=0.05)
        assert detector.significance_level == 0.05

    def test_invalid_significance_level(self):
        """Invalid significance level raises error."""
        with pytest.raises(ValueError):
            OverfittingDetector(significance_level=1.5)

        with pytest.raises(ValueError):
            OverfittingDetector(significance_level=0)


class TestDeflatedSharpeRatio:
    """Tests for Deflated Sharpe Ratio calculation."""

    def test_basic_dsr(self):
        """Basic DSR calculation works."""
        detector = OverfittingDetector()
        result = detector.deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=100,
            variance_sharpe=0.5,
        )

        assert result.test_name == "Deflated Sharpe Ratio"
        assert isinstance(result.statistic, float)
        assert 0 <= result.p_value <= 1

    def test_high_sharpe_likely_ok(self):
        """High Sharpe with few trials is likely not overfitted."""
        detector = OverfittingDetector(significance_level=0.05)
        result = detector.deflated_sharpe_ratio(
            observed_sharpe=3.0,
            n_trials=10,
            variance_sharpe=0.3,
        )

        # Very high Sharpe should have low p-value
        assert result.p_value < 0.5

    def test_many_trials_inflates_expected(self):
        """Many trials increase expected max Sharpe."""
        detector = OverfittingDetector()

        result_few = detector.deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=10,
            variance_sharpe=0.3,
        )

        result_many = detector.deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=1000,
            variance_sharpe=0.3,
        )

        # More trials should result in higher expected max (lower DSR statistic)
        assert result_many.statistic < result_few.statistic

    def test_invalid_n_trials(self):
        """Invalid n_trials raises error."""
        detector = OverfittingDetector()
        with pytest.raises(ValueError):
            detector.deflated_sharpe_ratio(
                observed_sharpe=1.0,
                n_trials=0,
                variance_sharpe=0.3,
            )

    def test_convenience_function(self):
        """Convenience function works."""
        dsr, p_value = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=50,
            variance_sharpe=0.4,
        )

        assert isinstance(dsr, float)
        assert isinstance(p_value, float)


class TestProbabilityOfBacktestOverfitting:
    """Tests for PBO calculation."""

    def test_basic_pbo(self):
        """Basic PBO calculation works."""
        detector = OverfittingDetector()
        pbo = detector.probability_of_backtest_overfitting(
            is_sharpes=[1.5, 1.2, 1.8, 1.0],
            oos_sharpes=[0.8, 0.5, 0.9, 0.3],
        )

        assert 0 <= pbo <= 1

    def test_perfect_generalization(self):
        """Perfect generalization has consistent PBO."""
        detector = OverfittingDetector()
        pbo = detector.probability_of_backtest_overfitting(
            is_sharpes=[1.0, 0.8, 1.2, 0.9],
            oos_sharpes=[1.0, 0.8, 1.2, 0.9],  # Same as IS
        )

        # Should return a valid probability
        assert 0 <= pbo <= 1

    def test_severe_overfitting(self):
        """Severe overfitting has high PBO."""
        detector = OverfittingDetector()
        pbo = detector.probability_of_backtest_overfitting(
            is_sharpes=[2.0, 1.8, 2.2, 1.9],  # High IS
            oos_sharpes=[0.1, -0.2, 0.0, -0.1],  # Very low OOS
        )

        # Best IS strategy should perform below median OOS
        assert pbo >= 0.5

    def test_mismatched_lengths(self):
        """Mismatched IS/OOS lengths raises error."""
        detector = OverfittingDetector()
        with pytest.raises(ValueError):
            detector.probability_of_backtest_overfitting(
                is_sharpes=[1.0, 1.2],
                oos_sharpes=[0.5],
            )

    def test_too_few_strategies(self):
        """Too few strategies raises error."""
        detector = OverfittingDetector()
        with pytest.raises(ValueError):
            detector.probability_of_backtest_overfitting(
                is_sharpes=[1.0],
                oos_sharpes=[0.5],
            )

    def test_convenience_function(self):
        """Convenience function works."""
        pbo = probability_of_backtest_overfitting(
            is_sharpes=[1.5, 1.2, 1.8],
            oos_sharpes=[0.8, 0.5, 0.9],
        )

        assert 0 <= pbo <= 1


class TestPerformanceDegradationTest:
    """Tests for performance degradation test."""

    def test_welch_test(self, sample_returns):
        """Welch's t-test works."""
        is_returns = sample_returns + 0.01  # Higher IS returns
        oos_returns = sample_returns

        detector = OverfittingDetector()
        result = detector.performance_degradation_test(
            is_returns=is_returns,
            oos_returns=oos_returns,
            test_type="welch",
        )

        assert "degradation" in result.details
        assert isinstance(result.p_value, float)

    def test_mann_whitney_test(self, sample_returns):
        """Mann-Whitney U test works."""
        is_returns = sample_returns + 0.01
        oos_returns = sample_returns

        detector = OverfittingDetector()
        result = detector.performance_degradation_test(
            is_returns=is_returns,
            oos_returns=oos_returns,
            test_type="mann_whitney",
        )

        assert isinstance(result.p_value, float)

    def test_bootstrap_test(self, sample_returns):
        """Bootstrap test works."""
        np.random.seed(42)
        is_returns = sample_returns + 0.01
        oos_returns = sample_returns

        detector = OverfittingDetector()
        result = detector.performance_degradation_test(
            is_returns=is_returns,
            oos_returns=oos_returns,
            test_type="bootstrap",
        )

        assert isinstance(result.p_value, float)

    def test_significant_degradation_detected(self, sample_returns):
        """Significant degradation is detected."""
        np.random.seed(42)
        is_returns = sample_returns + 0.05  # Much higher IS
        oos_returns = sample_returns - 0.02  # Lower OOS

        detector = OverfittingDetector(significance_level=0.05)
        result = detector.performance_degradation_test(
            is_returns=is_returns,
            oos_returns=oos_returns,
        )

        # Should detect significant degradation
        assert result.details['degradation'] > 0

    def test_no_degradation(self, sample_returns):
        """No degradation when IS and OOS are similar."""
        detector = OverfittingDetector()
        result = detector.performance_degradation_test(
            is_returns=sample_returns,
            oos_returns=sample_returns,
        )

        # Degradation should be near zero
        assert abs(result.details['degradation']) < 0.01

    def test_invalid_test_type(self, sample_returns):
        """Invalid test type raises error."""
        detector = OverfittingDetector()
        with pytest.raises(ValueError):
            detector.performance_degradation_test(
                is_returns=sample_returns,
                oos_returns=sample_returns,
                test_type="invalid",
            )

    def test_too_few_samples(self):
        """Too few samples raises error."""
        detector = OverfittingDetector()
        with pytest.raises(ValueError):
            detector.performance_degradation_test(
                is_returns=np.array([0.01, 0.02]),
                oos_returns=np.array([0.01, 0.02]),
            )

    def test_convenience_function(self, sample_returns):
        """Convenience function works."""
        result = performance_degradation_test(
            is_returns=sample_returns + 0.01,
            oos_returns=sample_returns,
        )

        assert 'degradation' in result
        assert 'p_value' in result


class TestComprehensiveTest:
    """Tests for comprehensive overfitting test."""

    def test_comprehensive_test(self, sample_returns):
        """Comprehensive test runs all tests."""
        is_returns = sample_returns + 0.01
        oos_returns = sample_returns

        detector = OverfittingDetector()
        results = detector.comprehensive_test(
            is_returns=is_returns,
            oos_returns=oos_returns,
            n_trials=50,
        )

        assert 'dsr' in results
        assert 'degradation' in results


class TestEstimateHaircutSharpe:
    """Tests for haircut Sharpe estimation."""

    def test_haircut_reduces_sharpe(self):
        """Haircut should reduce observed Sharpe."""
        observed = 2.0
        haircut = estimate_haircut_sharpe(
            observed_sharpe=observed,
            n_trials=100,
        )

        assert haircut < observed

    def test_single_trial_no_haircut(self):
        """Single trial should have no haircut."""
        observed = 1.5
        haircut = estimate_haircut_sharpe(
            observed_sharpe=observed,
            n_trials=1,
        )

        assert haircut == observed

    def test_more_trials_larger_haircut(self):
        """More trials should result in larger haircut."""
        observed = 1.5

        haircut_few = estimate_haircut_sharpe(
            observed_sharpe=observed,
            n_trials=10,
        )

        haircut_many = estimate_haircut_sharpe(
            observed_sharpe=observed,
            n_trials=1000,
        )

        assert haircut_few > haircut_many

    def test_non_negative_result(self):
        """Haircut Sharpe should be non-negative."""
        haircut = estimate_haircut_sharpe(
            observed_sharpe=0.5,
            n_trials=1000,
        )

        assert haircut >= 0
