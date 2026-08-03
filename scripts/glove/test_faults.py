"""Headless fault-injection verification.

For each fault toggle, runs the mock generator through the validator (and
StreamMonitor where relevant) for N frames and reports whether the expected
error/warning fired.
"""
from xr_hand.mock import MockHandGenerator
from xr_hand.validator import StreamMonitor, validate_raw_message

FRAMES = 60


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {detail}")
    return ok


def check_clean_stream() -> bool:
    gen = MockHandGenerator(hand="right")
    mon = StreamMonitor("right", freeze_threshold=2, drop_threshold=2)
    for _ in range(FRAMES):
        raw = gen.next_frame()
        r = validate_raw_message(raw)
        if not r.is_valid or r.warnings:
            return report("clean stream", False,
                          f"unexpected: errors={r.errors} warnings={r.warnings}")
        if mon.update(int(raw[0])):
            return report("clean stream", False, "monitor flagged a clean stream")
    return report("clean stream", True, f"no errors/warnings over {FRAMES} frames")


def check_stuck_fingertip_stays_valid() -> bool:
    gen = MockHandGenerator(hand="right")
    gen.stuck_fingertip = "INDEX_TIP"
    for _ in range(FRAMES):
        raw = gen.next_frame()
        if not validate_raw_message(raw).is_valid:
            return report("stuck_fingertip", False, "stream went invalid")
    return report("stuck_fingertip", True, "stream stays valid (visual-only fault)")


def check_freeze_counter() -> bool:
    gen = MockHandGenerator(hand="right")
    gen.freeze_counter = True
    mon = StreamMonitor("right", freeze_threshold=3, drop_threshold=10)
    warnings = []
    for _ in range(FRAMES):
        raw = gen.next_frame()
        warnings.extend(mon.update(int(raw[0])))
    hit = next((w for w in warnings if "frozen" in w), None)
    return report("freeze_counter", hit is not None, f"got: {hit!r}")


def check_zero_joint() -> bool:
    gen = MockHandGenerator(hand="right")
    gen.zero_joint = "MIDDLE_PROXIMAL"
    raw = gen.next_frame()
    r = validate_raw_message(raw)
    hit = next((w for w in r.warnings if "MIDDLE_PROXIMAL" in w), None)
    return report("zero_joint", hit is not None, f"got: {hit!r}")


def check_wrong_length() -> bool:
    gen = MockHandGenerator(hand="right")
    gen.wrong_length = True
    raw = gen.next_frame()
    r = validate_raw_message(raw)
    hit = next((e for e in r.errors if "length mismatch" in e), None)
    return report("wrong_length", hit is not None, f"got: {hit!r}")


def check_random_missing() -> bool:
    gen = MockHandGenerator(hand="right")
    gen.random_missing_prob = 0.01
    hit = None
    for _ in range(FRAMES):
        raw = gen.next_frame()
        r = validate_raw_message(raw)
        for e in r.errors:
            if "non-finite" in e:
                hit = e
                break
        if hit:
            break
    return report("random_missing", hit is not None, f"got: {hit!r}")


if __name__ == "__main__":
    results = [
        check_clean_stream(),
        check_stuck_fingertip_stays_valid(),
        check_freeze_counter(),
        check_zero_joint(),
        check_wrong_length(),
        check_random_missing(),
    ]
    print("=" * 50)
    print(f"  {sum(results)}/{len(results)} checks passed")
