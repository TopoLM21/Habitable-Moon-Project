from __future__ import annotations

import pytest

from moon_gui.timing import RunTiming, format_duration


def finish(timing: RunTiming, target: float, start: float, end: float) -> None:
    timing.start_segment(target, start)
    timing.finish_segment(end)


def test_eta_waits_for_two_completed_segments_and_counts_active_time() -> None:
    timing = RunTiming(0.0, 100.0)
    assert timing.estimate(0.0).remaining_seconds is None
    finish(timing, 20.0, 0.0, 30.0)
    assert timing.estimate(30.0).remaining_seconds is None
    finish(timing, 40.0, 30.0, 60.0)
    timing.start_segment(60.0, 60.0)
    estimate = timing.estimate(75.0)
    assert estimate.elapsed_seconds == 75.0
    assert estimate.remaining_seconds == 75.0
    assert estimate.sample_count == 2


def test_resume_counts_only_new_simulated_duration_and_excludes_pauses() -> None:
    timing = RunTiming(400.0, 500.0)
    finish(timing, 420.0, 1000.0, 1030.0)
    finish(timing, 440.0, 1030.0, 1060.0)
    before = timing.estimate(1060.0)
    assert timing.estimate(9000.0) == before
    assert before.elapsed_seconds == 60.0
    assert before.remaining_seconds == 90.0
    timing.start_segment(460.0, 9000.0)
    assert timing.estimate(9015.0).elapsed_seconds == 75.0
    assert timing.estimate(9015.0).remaining_seconds == 75.0


def test_eta_uses_last_five_samples_and_accounts_for_short_final_segment() -> None:
    timing = RunTiming(0.0, 132.0)
    finish(timing, 20.0, 0.0, 1000.0)
    for index in range(5):
        finish(timing, 40.0 + 20.0 * index, 1000.0 + 20.0 * index, 1020.0 + 20.0 * index)
    estimate = timing.estimate(1100.0)
    assert estimate.sample_count == 5
    assert estimate.remaining_seconds == 12.0
    timing.start_segment(132.0, 1100.0)
    assert timing.estimate(1105.0).remaining_seconds == 7.0
    timing.finish_segment(1112.0)
    assert timing.estimate(9999.0).remaining_seconds == 0.0
    assert timing.estimate(9999.0).elapsed_seconds == 1112.0


def test_rate_is_weighted_by_simulated_time_not_number_of_segments() -> None:
    timing = RunTiming(0.0, 100.0)
    finish(timing, 20.0, 0.0, 20.0)
    finish(timing, 30.0, 20.0, 40.0)
    assert timing.estimate(40.0).remaining_seconds == pytest.approx(70.0 * 40.0 / 30.0)


def test_overdue_finalization_does_not_report_false_zero() -> None:
    timing = RunTiming(0.0, 60.0)
    finish(timing, 20.0, 0.0, 20.0)
    finish(timing, 40.0, 20.0, 40.0)
    timing.start_segment(60.0, 40.0)
    estimate = timing.estimate(90.0)
    assert estimate.remaining_seconds is None
    assert estimate.segment_overdue
    timing.finish_segment(100.0)
    assert timing.estimate(100.0).remaining_seconds == 0.0


def test_interrupted_segment_counts_time_but_not_completed_progress() -> None:
    timing = RunTiming(0.0, 100.0)
    finish(timing, 20.0, 0.0, 20.0)
    finish(timing, 40.0, 20.0, 40.0)
    timing.start_segment(60.0, 40.0)
    timing.stop_segment(50.0)
    assert timing.completed_myr == 40.0
    assert timing.estimate(9000.0).elapsed_seconds == 50.0
    assert timing.estimate(9000.0).sample_count == 2
    assert timing.estimate(9000.0).remaining_seconds == 60.0
    timing.stop_segment(9000.0)
    assert timing.estimate(9000.0).elapsed_seconds == 50.0


def test_invalid_segment_lifecycle_is_rejected() -> None:
    timing = RunTiming(20.0, 100.0)
    with pytest.raises(ValueError):
        timing.start_segment(20.0, 0.0)
    with pytest.raises(ValueError):
        timing.start_segment(120.0, 0.0)
    with pytest.raises(RuntimeError):
        timing.finish_segment(0.0)
    timing.start_segment(40.0, 0.0)
    with pytest.raises(RuntimeError):
        timing.start_segment(60.0, 5.0)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0 с"), (0.1, "1 с"), (75.0, "1 мин 15 с"), (4320.0, "1 ч 12 мин")],
)
def test_duration_format(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected
