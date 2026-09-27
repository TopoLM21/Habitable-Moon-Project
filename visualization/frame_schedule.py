"""Frame timing on an existing integration grid; never changes physical time."""
import math


def frame_due_at_step(previous_time_myr, time_myr, interval_myr, origin_myr=None):
    """Keep legacy absolute frames, or capture the first step crossing a phase.

    Explicit phases survive segment boundaries and time-step changes: a frame
    is due when a scheduled age lies inside (previous_time, time]. The saved
    image still displays the actual computed age, not the scheduled threshold.
    """
    if interval_myr <= 0 or not all(math.isfinite(value) for value in
                                   (previous_time_myr, time_myr, interval_myr)):
        raise ValueError("Frame times must be finite and interval positive")
    if origin_myr is None:
        ratio = time_myr / interval_myr
        return abs(ratio - round(ratio)) < 1e-9
    if not math.isfinite(origin_myr):
        raise ValueError("Frame origin must be finite")
    if time_myr <= previous_time_myr:
        return False
    before = math.floor((previous_time_myr - origin_myr) / interval_myr + 1e-9)
    after = math.floor((time_myr - origin_myr) / interval_myr + 1e-9)
    return after >= 0 and after > before
