"""Run the full mock pipeline: generate -> validate -> parse -> 3D viz.

Edit the FAULT toggles near the top to inject specific failures and verify
the validator/logger catch them.
"""
import logging

from xr_hand.mock import MockHandGenerator
from xr_hand.parser import parse_hand_message
from xr_hand.validator import StreamMonitor, validate_raw_message
from xr_hand.viz3d import HandViewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mock")

left_gen = MockHandGenerator(hand="left")
right_gen = MockHandGenerator(hand="right")

# --- FAULT INJECTION (uncomment to try) ----------------------------------
# right_gen.stuck_fingertip = "INDEX_TIP"
# right_gen.freeze_counter  = True
# right_gen.zero_joint      = "MIDDLE_PROXIMAL"
# right_gen.wrong_length    = True
# right_gen.random_missing_prob = 0.005
# -------------------------------------------------------------------------

monitors = {
    "left":  StreamMonitor("left",  freeze_threshold=2, drop_threshold=2),
    "right": StreamMonitor("right", freeze_threshold=2, drop_threshold=2),
}
viewer = HandViewer(title="XR Hand (mock)")


def tick() -> None:
    for gen in (left_gen, right_gen):
        raw = gen.next_frame()
        result = validate_raw_message(raw)
        if not result.is_valid:
            log.warning("[%s] INVALID: %s", gen.hand, "; ".join(result.errors))
            continue
        for w in result.warnings:
            log.warning("[%s] %s", gen.hand, w)
        frame = parse_hand_message(raw, hand_side_hint=gen.hand)
        for w in monitors[gen.hand].update(frame.packet_counter):
            log.warning("[%s] %s", gen.hand, w)
        if frame.packet_counter % 60 == 0:
            log.info(
                "[%s] counter=%d t=%.2fs joints=%d",
                frame.hand_side, frame.packet_counter, frame.timestamp, len(frame.joints),
            )
        viewer.update_hand(frame)


viewer.run(tick, interval_ms=33)
