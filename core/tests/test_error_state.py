import logging

import pytest

from core.services.error_state import ErrorState


@pytest.fixture
def error_state() -> ErrorState:
    return ErrorState()


class TestInitialState:
    def test_starts_without_error(self, error_state: ErrorState):
        assert error_state.has_error() is False
        assert error_state.get_last_exception() is None


class TestMarkError:
    def test_records_the_exception(self, error_state: ErrorState):
        exc = RuntimeError("boom")

        error_state.mark_error(exc)

        assert error_state.has_error() is True
        assert error_state.get_last_exception() is exc

    def test_does_not_reraise(self, error_state: ErrorState):
        exc = RuntimeError("boom")

        error_state.mark_error(exc)  # 不應拋出

    def test_logs_the_exception(self, error_state: ErrorState, caplog: pytest.LogCaptureFixture):
        exc = RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            error_state.mark_error(exc)

        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_later_call_overwrites_previous_exception(self, error_state: ErrorState):
        first = RuntimeError("first")
        second = ValueError("second")

        error_state.mark_error(first)
        error_state.mark_error(second)

        assert error_state.get_last_exception() is second


class TestClear:
    def test_resets_has_error_and_last_exception(self, error_state: ErrorState):
        error_state.mark_error(RuntimeError("boom"))

        error_state.clear()

        assert error_state.has_error() is False
        assert error_state.get_last_exception() is None

    def test_clear_on_fresh_instance_is_noop(self, error_state: ErrorState):
        error_state.clear()

        assert error_state.has_error() is False
        assert error_state.get_last_exception() is None
