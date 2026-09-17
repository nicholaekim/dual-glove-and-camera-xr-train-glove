"""Validate a raw OSC payload before parsing.

Errors  -> packet is unusable, drop it.
Warnings -> packet is usable but something looks off (log it).

Per-packet checks done by `validate_raw_message`:
  - exact length == 187
  - no NaN / inf
  - per joint: not all-zero, quaternion magnitude ~ 1

Cross-packet (stream-level) checks done by `StreamMonitor`:
  - persistent counter freeze (default: 20+ identical values in a row)
  - large counter gap (default: 10+ missed ticks)
  - counter going backwards
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math

from .joints import (
    EXPECTED_VALUE_COUNT,
    HEADER_LEN,
    JOINT_COUNT,
    JOINT_NAMES,
    VALUES_PER_JOINT,
)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


QUAT_MAG_TOLERANCE = 0.1
ZERO_EPSILON = 1e-6


def validate_raw_message(
    values: Sequence[float], prev_counter: Optional[int] = None
) -> ValidationResult:
    """Per-packet structural validation. Use StreamMonitor for counter checks.

    prev_counter is accepted for backward compatibility but ignored. Counter
    behaviour is now tracked by StreamMonitor since real-device tick counters
    legitimately repeat and skip within batched emissions.
    """
    del prev_counter  # intentionally unused
    result = ValidationResult(is_valid=True)

    if len(values) != EXPECTED_VALUE_COUNT:
        result.errors.append(
            f"length mismatch: got {len(values)}, expected {EXPECTED_VALUE_COUNT}"
        )
        result.is_valid = False
        return result

    nan_idx = [i for i, v in enumerate(values) if not math.isfinite(v)]
    if nan_idx:
        preview = nan_idx[:10]
        more = "..." if len(nan_idx) > 10 else ""
        result.errors.append(f"non-finite values at indices {preview}{more}")
        result.is_valid = False

    for j in range(JOINT_COUNT):
        base = HEADER_LEN + j * VALUES_PER_JOINT
        chunk = values[base : base + VALUES_PER_JOINT]
        name = JOINT_NAMES[j]

        if all(math.isfinite(v) and abs(v) < ZERO_EPSILON for v in chunk):
            result.warnings.append(f"joint {name} is all-zero")
            continue

        qw, qx, qy, qz = chunk[3:7]
        if all(math.isfinite(v) for v in (qw, qx, qy, qz)):
            mag = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if abs(mag - 1.0) > QUAT_MAG_TOLERANCE:
                result.warnings.append(
                    f"joint {name} quaternion magnitude {mag:.3f} (expected ~1.0)"
                )

    return result


class StreamMonitor:
    """Tracks counter behaviour across packets for ONE hand.

    Real-device tick counters batch: the same value can repeat in consecutive
    packets (when one tick spans multiple OSC messages) and small skips happen
    (some ticks emit no kinematic frame). Only persistent freezes and large
    gaps are real problems.

    Defaults assume ~100 Hz packet rate. Adjust thresholds if your stream
    rate is very different.
    """

    def __init__(
        self,
        name: str = "",
        freeze_threshold: int = 20,
        drop_threshold: int = 10,
    ):
        self.name = name
        self.freeze_threshold = freeze_threshold
        self.drop_threshold = drop_threshold
        self.prev_counter: Optional[int] = None
        self.repeat_streak = 0
        self.in_freeze = False
        self.packets_seen = 0

    def update(self, counter: int) -> List[str]:
        warnings: List[str] = []
        self.packets_seen += 1

        if self.prev_counter is None:
            self.prev_counter = counter
            return warnings

        if counter == self.prev_counter:
            self.repeat_streak += 1
            if self.repeat_streak == self.freeze_threshold and not self.in_freeze:
                warnings.append(
                    f"counter frozen at {counter} for {self.repeat_streak}+ packets"
                )
                self.in_freeze = True
        elif counter < self.prev_counter:
            warnings.append(
                f"counter went backwards: {self.prev_counter} -> {counter}"
            )
            self.repeat_streak = 0
            self.in_freeze = False
        else:
            if self.in_freeze:
                warnings.append(
                    f"counter resumed: {self.prev_counter} -> {counter}"
                )
                self.in_freeze = False
            skipped = counter - self.prev_counter - 1
            if skipped >= self.drop_threshold:
                warnings.append(
                    f"counter jumped: {self.prev_counter} -> {counter} (dropped {skipped})"
                )
            self.repeat_streak = 0

        self.prev_counter = counter
        return warnings
