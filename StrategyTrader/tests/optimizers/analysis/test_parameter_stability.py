"""Tests for parameter stability analysis."""

import pytest
import numpy as np

from src.optimizers.analysis.parameter_stability import (
    ParameterStabilityAnalyzer,
    analyze_parameter_drift,
    get_consensus_parameters,
)


class TestParameterStabilityAnalyzer:
    """Tests for ParameterStabilityAnalyzer class."""

    def test_initialization(self):
        """Analyzer initializes correctly."""
        analyzer = ParameterStabilityAnalyzer(stability_threshold=0.3)
        assert analyzer.stability_threshold == 0.3

    def test_invalid_threshold(self):
        """Invalid threshold raises error."""
        with pytest.raises(ValueError):
            ParameterStabilityAnalyzer(stability_threshold=1.5)

    def test_analyze_stable_params(self, sample_split_results):
        """Stable parameters are identified correctly."""
        analyzer = ParameterStabilityAnalyzer(stability_threshold=0.5)
        stability = analyzer.analyze(sample_split_results)

        assert len(stability) > 0
        for param_name, param_stability in stability.items():
            assert hasattr(param_stability, 'is_stable')

    def test_analyze_empty_list(self):
        """Empty list returns empty dict."""
        analyzer = ParameterStabilityAnalyzer()
        stability = analyzer.analyze([])
        assert stability == {}

    def test_get_unstable_parameters(self, sample_split_results):
        """Get unstable parameters works."""
        analyzer = ParameterStabilityAnalyzer(stability_threshold=0.1)  # Strict
        analyzer.analyze(sample_split_results)

        unstable = analyzer.get_unstable_parameters()
        assert isinstance(unstable, list)

    def test_get_stable_parameters(self, sample_split_results):
        """Get stable parameters works."""
        analyzer = ParameterStabilityAnalyzer(stability_threshold=0.5)  # Lenient
        analyzer.analyze(sample_split_results)

        stable = analyzer.get_stable_parameters()
        assert isinstance(stable, list)

    def test_analyze_drift(self, sample_split_results):
        """Drift analysis works."""
        analyzer = ParameterStabilityAnalyzer()
        drift = analyzer.analyze_drift(sample_split_results)

        assert isinstance(drift, dict)

    def test_get_drifting_parameters(self, sample_split_results):
        """Get drifting parameters works."""
        analyzer = ParameterStabilityAnalyzer()
        analyzer.analyze_drift(sample_split_results)

        drifting = analyzer.get_drifting_parameters()
        assert isinstance(drifting, list)

    def test_get_summary(self, sample_split_results):
        """Summary returns correct structure."""
        analyzer = ParameterStabilityAnalyzer()
        analyzer.analyze(sample_split_results)

        summary = analyzer.get_summary()

        assert 'total_parameters' in summary
        assert 'stable_parameters' in summary
        assert 'unstable_parameters' in summary

    def test_to_dataframe(self, sample_split_results):
        """Export to DataFrame works."""
        analyzer = ParameterStabilityAnalyzer()
        analyzer.analyze(sample_split_results)
        analyzer.analyze_drift(sample_split_results)

        df = analyzer.to_dataframe()

        assert len(df) > 0
        assert 'parameter' in df.columns
        assert 'is_stable' in df.columns


class TestAnalyzeParameterDrift:
    """Tests for analyze_parameter_drift function."""

    def test_basic_drift_analysis(self, sample_params_list):
        """Basic drift analysis works."""
        drift = analyze_parameter_drift(sample_params_list)

        assert len(drift) > 0

    def test_increasing_trend(self):
        """Increasing trend is detected."""
        params = [
            {'value': 10},
            {'value': 15},
            {'value': 20},
            {'value': 25},
            {'value': 30},
        ]

        drift = analyze_parameter_drift(params, trend_threshold=0.8)

        assert 'value' in drift
        # Should detect increasing trend
        assert drift['value'].correlation_with_split > 0.5

    def test_decreasing_trend(self):
        """Decreasing trend is detected."""
        params = [
            {'value': 30},
            {'value': 25},
            {'value': 20},
            {'value': 15},
            {'value': 10},
        ]

        drift = analyze_parameter_drift(params, trend_threshold=0.8)

        assert 'value' in drift
        # Should detect decreasing trend
        assert drift['value'].correlation_with_split < -0.5

    def test_stable_no_drift(self):
        """Stable parameter shows no significant drift."""
        params = [
            {'value': 20},
            {'value': 20},
            {'value': 20},
            {'value': 20},
            {'value': 20},
        ]

        drift = analyze_parameter_drift(params)

        assert 'value' in drift
        # When all values are same, drift magnitude is 0
        assert drift['value'].drift_magnitude == 0.0
        # And should not be significant
        assert not drift['value'].is_significant

    def test_too_few_params(self):
        """Too few params returns empty dict."""
        params = [{'value': 10}, {'value': 15}]
        drift = analyze_parameter_drift(params)
        assert drift == {}


class TestGetConsensusParameters:
    """Tests for get_consensus_parameters function."""

    def test_median_method(self, sample_params_list):
        """Median method works."""
        consensus = get_consensus_parameters(sample_params_list, method='median')

        assert len(consensus.parameters) > 0
        assert consensus.method == 'median'

    def test_mean_method(self, sample_params_list):
        """Mean method works."""
        consensus = get_consensus_parameters(sample_params_list, method='mean')

        assert len(consensus.parameters) > 0
        assert consensus.method == 'mean'

    def test_last_method(self, sample_params_list):
        """Last method returns last value."""
        consensus = get_consensus_parameters(sample_params_list, method='last')

        # Last value should match
        for param, value in consensus.parameters.items():
            assert value == sample_params_list[-1][param]

    def test_confidence_values(self, sample_params_list):
        """Confidence values are reasonable."""
        consensus = get_consensus_parameters(sample_params_list)

        for param, conf in consensus.confidence.items():
            assert 0 <= conf <= 1

    def test_stable_param_high_confidence(self):
        """Stable parameter has high confidence."""
        params = [
            {'value': 20},
            {'value': 20},
            {'value': 20},
            {'value': 20},
            {'value': 20},
        ]

        consensus = get_consensus_parameters(params)

        # Constant parameter should have confidence 1.0
        assert consensus.confidence['value'] == 1.0

    def test_variable_param_low_confidence(self):
        """Variable parameter has lower confidence."""
        params = [
            {'value': 10},
            {'value': 50},
            {'value': 20},
            {'value': 80},
            {'value': 30},
        ]

        consensus = get_consensus_parameters(params)

        # High variance should result in lower confidence
        assert consensus.confidence['value'] < 0.8

    def test_empty_params(self):
        """Empty params returns empty consensus."""
        consensus = get_consensus_parameters([])

        assert len(consensus.parameters) == 0
        assert consensus.n_splits_used == 0

    def test_n_splits_tracked(self, sample_params_list):
        """Number of splits is tracked."""
        consensus = get_consensus_parameters(sample_params_list)

        assert consensus.n_splits_used == len(sample_params_list)

    def test_non_numeric_params(self):
        """Non-numeric parameters handled correctly."""
        params = [
            {'method': 'sma', 'value': 10},
            {'method': 'ema', 'value': 15},
            {'method': 'sma', 'value': 20},
            {'method': 'sma', 'value': 25},
            {'method': 'sma', 'value': 30},
        ]

        consensus = get_consensus_parameters(params)

        # Should handle string parameter
        assert 'method' in consensus.parameters
        assert consensus.parameters['method'] == 'sma'  # Mode
