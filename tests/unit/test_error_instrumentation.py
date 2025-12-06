"""
Unit tests for error instrumentation module.

Tests log levels, latency metrics, and error tracking.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock


from arc_saga.error_instrumentation import LatencyMetrics, log_with_context


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_critical_level(self) -> None:
        """Test CRITICAL log level is handled correctly."""
        with patch(
            "arc_saga.error_instrumentation.logging.getLogger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            log_with_context("critical", "test_event", test_data="value")
            mock_logger.critical.assert_called_once()
            call_args = mock_logger.critical.call_args[0][0]
            assert "test_event" in call_args
            assert "test_data" in call_args

    def test_log_info_level(self) -> None:
        """Test INFO log level is handled correctly."""
        with patch(
            "arc_saga.error_instrumentation.logging.getLogger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            log_with_context("info", "test_event", test_data="value")
            mock_logger.info.assert_called_once()

    def test_log_warning_level(self) -> None:
        """Test WARNING log level is handled correctly."""
        with patch(
            "arc_saga.error_instrumentation.logging.getLogger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            log_with_context("warning", "test_event", test_data="value")
            mock_logger.warning.assert_called_once()

    def test_log_error_level(self) -> None:
        """Test ERROR log level is handled correctly."""
        with patch(
            "arc_saga.error_instrumentation.logging.getLogger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            log_with_context("error", "test_event", test_data="value")
            mock_logger.error.assert_called_once()

    def test_log_unknown_level_defaults_to_info(self) -> None:
        """Test unknown log level defaults to INFO."""
        with patch(
            "arc_saga.error_instrumentation.logging.getLogger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            log_with_context("unknown_level", "test_event", test_data="value")
            mock_logger.info.assert_called_once()


class TestLatencyMetrics:
    """Tests for LatencyMetrics class."""

    def test_p50_with_empty_latencies(self) -> None:
        """Test p50 returns 0 when latencies list is empty."""
        metrics = LatencyMetrics("test_operation")
        assert metrics.p50 == 0

    def test_p95_with_empty_latencies(self) -> None:
        """Test p95 returns 0 when latencies list is empty."""
        metrics = LatencyMetrics("test_operation")
        assert metrics.p95 == 0

    def test_p99_with_empty_latencies(self) -> None:
        """Test p99 returns 0 when latencies list is empty."""
        metrics = LatencyMetrics("test_operation")
        assert metrics.p99 == 0

    def test_p50_with_latencies(self) -> None:
        """Test p50 calculation with actual latencies."""
        metrics = LatencyMetrics("test_operation")
        metrics.add(100.0)
        metrics.add(200.0)
        metrics.add(300.0)
        metrics.add(400.0)
        metrics.add(500.0)
        # Median of [100, 200, 300, 400, 500] is 300
        assert metrics.p50 == 300.0

    def test_p95_with_latencies(self) -> None:
        """Test p95 calculation with actual latencies."""
        metrics = LatencyMetrics("test_operation")
        # Add 100 latencies
        for i in range(100):
            metrics.add(float(i))
        # p95 should be around index 94 (95th percentile)
        assert metrics.p95 >= 94.0

    def test_p99_with_latencies(self) -> None:
        """Test p99 calculation with actual latencies."""
        metrics = LatencyMetrics("test_operation")
        # Add 100 latencies
        for i in range(100):
            metrics.add(float(i))
        # p99 should be around index 98 (99th percentile)
        assert metrics.p99 >= 98.0

    def test_add_latency_above_threshold_logs_warning(self) -> None:
        """Test adding latency > 1000ms logs warning."""
        metrics = LatencyMetrics("test_operation")
        with patch("arc_saga.error_instrumentation.log_with_context") as mock_log:
            metrics.add(1500.0)
            # Verify warning was logged
            warning_calls = [
                call
                for call in mock_log.call_args_list
                if len(call[0]) > 1 and call[0][1].endswith("slow")
            ]
            assert len(warning_calls) > 0

    def test_add_latency_below_threshold_no_warning(self) -> None:
        """Test adding latency < 1000ms does not log warning."""
        metrics = LatencyMetrics("test_operation")
        with patch("arc_saga.error_instrumentation.log_with_context") as mock_log:
            metrics.add(500.0)
            # Verify no slow warning was logged
            warning_calls = [
                call
                for call in mock_log.call_args_list
                if len(call[0]) > 1 and call[0][1].endswith("slow")
            ]
            assert len(warning_calls) == 0

    def test_add_multiple_latencies(self) -> None:
        """Test adding multiple latencies updates metrics correctly."""
        metrics = LatencyMetrics("test_operation")
        metrics.add(100.0)
        metrics.add(200.0)
        metrics.add(300.0)
        assert len(metrics.latencies) == 3
        assert metrics.p50 == 200.0
