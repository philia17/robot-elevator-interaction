from __future__ import annotations

import numpy as np

from press.button_light_observer import (
    ButtonLightObserver,
    CLEAR_NO_LIGHT,
    LIGHT_ON,
    UNOBSERVABLE,
)


ROI = (10, 10, 30, 30)


def frame(level=40, roi_level=None):
    image = np.full((40, 40, 3), level, dtype=np.uint8)
    if roi_level is not None:
        image[10:30, 10:30] = roi_level
    return image


def baseline(observer, stamp=0):
    for index in range(5):
        observer.observe(
            frame(), stamp_ns=stamp + index + 1,
            target_identity="target-1", phase="BASELINE",
        )


def test_off_through_back_is_clear_no_light():
    observer = ButtonLightObserver(ROI, "target-1")
    baseline(observer)
    for stamp in range(6, 10):
        observer.observe(frame(), stamp_ns=stamp, target_identity="target-1", phase="ACTIVE")
    observer.mark_back_proven(True)
    for stamp in range(10, 13):
        observer.observe(frame(), stamp_ns=stamp, target_identity="target-1", phase="POST_BACK")
    assert observer.post_back_ready is True
    assert observer.classify() == CLEAR_NO_LIGHT


def test_baseline_uses_latest_required_frames():
    observer = ButtonLightObserver(ROI, "target-1")
    for stamp in range(1, 8):
        observer.observe(
            frame(level=30 + stamp),
            stamp_ns=stamp,
            target_identity="target-1",
            phase="BASELINE",
        )
    assert observer.baseline_ready is True
    assert len(observer._baseline_rois) == observer.config.baseline_frames


def test_sustained_roi_light_latches_only_from_post_back_frames():
    observer = ButtonLightObserver(ROI, "target-1")
    baseline(observer)
    for stamp in range(6, 11):
        observer.observe(frame(40, 110), stamp_ns=stamp, target_identity="target-1", phase="ACTIVE")
    assert observer.light_latched is False
    assert all(item["latch_eligible"] is False for item in observer.metrics)
    assert observer.classify() == UNOBSERVABLE
    observer.mark_back_proven(True)
    for stamp in range(11, 14):
        observer.observe(frame(40, 110), stamp_ns=stamp, target_identity="target-1", phase="POST_BACK")
    assert observer.light_latched is True
    assert observer.classify() == LIGHT_ON


def test_active_reflection_cannot_latch_light_when_post_back_is_off():
    observer = ButtonLightObserver(ROI, "target-1")
    baseline(observer)
    for stamp in range(6, 11):
        observer.observe(
            frame(40, 110),
            stamp_ns=stamp,
            target_identity="target-1",
            phase="ACTIVE",
        )
    assert len(observer.metrics) == 5
    assert all(item["on"] is True for item in observer.metrics)
    assert observer.light_latched is False
    observer.mark_back_proven(True)
    for stamp in range(11, 14):
        observer.observe(
            frame(),
            stamp_ns=stamp,
            target_identity="target-1",
            phase="POST_BACK",
        )
    assert observer.light_latched is False
    assert observer.classify() == CLEAR_NO_LIGHT


def test_stale_frame_is_unobservable():
    observer = ButtonLightObserver(ROI, "target-1")
    baseline(observer)
    observer.observe(frame(), stamp_ns=5, target_identity="target-1", phase="ACTIVE")
    observer.mark_back_proven(True)
    assert observer.classify() == UNOBSERVABLE
    assert observer.failure_reason == "FRAME_NOT_FRESH"


def test_occluded_roi_is_unobservable():
    observer = ButtonLightObserver(ROI, "target-1")
    baseline(observer)
    observer.observe(
        frame(), stamp_ns=6, target_identity="target-1", phase="ACTIVE", occluded=True,
    )
    observer.mark_back_proven(True)
    assert observer.classify() == UNOBSERVABLE


def test_global_exposure_shift_is_not_a_light():
    observer = ButtonLightObserver(ROI, "target-1")
    baseline(observer)
    observer.observe(frame(100), stamp_ns=6, target_identity="target-1", phase="ACTIVE")
    observer.mark_back_proven(True)
    assert observer.classify() == UNOBSERVABLE
    assert observer.failure_reason == "EXPOSURE_SHIFT"
