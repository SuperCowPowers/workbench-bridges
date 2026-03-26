"""Tests for logger"""

import logging

from workbench_bridges.utils.logger import (
    logging_setup,
    ColoredFormatter,
    exception_log_forward,
    IMPORTANT_LEVEL_NUM,
    MONITOR_LEVEL_NUM,
)


def test_logging_setup():
    """Test that logging_setup configures the logger"""
    # Reset the setup flag so we can test
    log = logging.getLogger("workbench-bridges")
    log._is_setup = False
    log.handlers.clear()

    logging_setup()
    assert log._is_setup is True
    assert len(log.handlers) > 0


def test_logging_setup_idempotent():
    """Test that calling logging_setup twice doesn't add duplicate handlers"""
    log = logging.getLogger("workbench-bridges")
    log._is_setup = False
    log.handlers.clear()

    logging_setup()
    handler_count = len(log.handlers)
    logging_setup()  # Second call
    assert len(log.handlers) == handler_count


def test_logging_setup_no_color():
    """Test setup without color logs"""
    log = logging.getLogger("workbench-bridges")
    log._is_setup = False
    log.handlers.clear()

    logging_setup(color_logs=False)
    assert log._is_setup is True
    # Handler should use plain Formatter, not ColoredFormatter
    assert not isinstance(log.handlers[0].formatter, ColoredFormatter)


def test_logging_setup_debug_mode(monkeypatch):
    """Test that WORKBENCH_DEBUG enables debug logging"""
    log = logging.getLogger("workbench-bridges")
    log._is_setup = False
    log.handlers.clear()

    monkeypatch.setenv("WORKBENCH_DEBUG", "True")
    logging_setup()
    assert log.level == logging.DEBUG

    # Clean up
    log._is_setup = False
    log.handlers.clear()
    monkeypatch.delenv("WORKBENCH_DEBUG")
    logging_setup()


def test_important_level():
    """Test that IMPORTANT level is between INFO and WARNING"""
    assert logging.INFO < IMPORTANT_LEVEL_NUM < logging.WARNING


def test_monitor_level():
    """Test that MONITOR level is between WARNING and ERROR"""
    assert logging.WARNING < MONITOR_LEVEL_NUM < logging.ERROR


def test_important_method():
    """Test that logger has an important() method"""
    log = logging.getLogger("workbench-bridges")
    assert hasattr(log, "important")
    # Should not raise
    log.important("Test important message")


def test_monitor_method():
    """Test that logger has a monitor() method"""
    log = logging.getLogger("workbench-bridges")
    assert hasattr(log, "monitor")
    log.monitor("Test monitor message")


def test_colored_formatter():
    """Test ColoredFormatter applies colors"""
    formatter = ColoredFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    result = formatter.format(record)
    # Should contain ANSI escape codes
    assert "\x1b[" in result
    assert "hello" in result


def test_colored_formatter_all_levels():
    """Test ColoredFormatter handles all custom levels"""
    formatter = ColoredFormatter("%(levelname)s %(message)s")
    for level_name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        level = getattr(logging, level_name)
        record = logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        result = formatter.format(record)
        assert "test" in result


def test_exception_log_forward_no_exception():
    """Test context manager with no exception"""
    with exception_log_forward():
        x = 1 + 1
    assert x == 2


def test_exception_log_forward_with_exception():
    """Test context manager re-raises exception"""
    try:
        with exception_log_forward():
            raise ValueError("test error")
        assert False, "Should have raised"
    except ValueError as e:
        assert "test error" in str(e)


def test_exception_log_forward_with_callback():
    """Test context manager calls callback on exception"""
    callback_called = []

    def my_callback(exc):
        callback_called.append(str(exc))

    with exception_log_forward(call_on_exception=my_callback):
        raise ValueError("callback test")

    assert len(callback_called) == 1
    assert "callback test" in callback_called[0]


if __name__ == "__main__":
    test_logging_setup()
    test_colored_formatter()
    print("All logger tests passed!")
