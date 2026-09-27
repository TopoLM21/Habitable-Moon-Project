import pytest

from visualization.frame_schedule import frame_due_at_step


def test_legacy_frames_keep_absolute_phase():
    assert frame_due_at_step(4., 8., 4.)
    assert not frame_due_at_step(4.984, 8.984, 4.)
    assert not frame_due_at_step(3., 5., 4.)


def test_fractional_origin_produces_each_requested_frame():
    origin = .9906219482421875
    assert all(frame_due_at_step(origin+i-1, origin+i, 1., origin) for i in range(1, 5))
    assert not frame_due_at_step(origin, origin+.5, 1., origin)


def test_changed_step_on_resume_does_not_disable_or_duplicate_frames():
    origin = .9843719482421875
    # Resume at 1.5 elapsed, then change to one-Myr steps. The next scheduled
    # age is two Myr elapsed and is first available at the 2.5-Myr step.
    assert frame_due_at_step(origin+1.5, origin+2.5, 2., origin)
    assert not frame_due_at_step(origin+2.5, origin+3.5, 2., origin)
    assert frame_due_at_step(origin+3.5, origin+4.5, 2., origin)
    assert not frame_due_at_step(origin+4., origin+4., 2., origin)


def test_segmentation_preserves_requested_frame_ages():
    origin = .9906219482421875
    times = [origin+i*.5 for i in range(13)]
    capture = lambda values: [b for a,b in zip(values, values[1:]) if frame_due_at_step(a,b,2.,origin)]
    assert capture(times) == capture(times[:6])+capture(times[5:])


@pytest.mark.parametrize("args", [(0,1,0,None), (0,1,float('nan'),None), (0,1,1,float('inf'))])
def test_invalid_schedule_parameters_fail(args):
    with pytest.raises(ValueError):
        frame_due_at_step(*args)
