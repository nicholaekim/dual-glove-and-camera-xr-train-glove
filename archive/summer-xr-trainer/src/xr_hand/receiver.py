"""OSC receiver for StretchSense / XR Trainer hand messages.

The OSC server runs on its own thread. Incoming packets are pushed onto a
thread-safe queue; the main thread (where the viewer lives) drains the queue
each animation tick.

Two wire formats are supported simultaneously, so the same receiver works for
real gloves and for the mock sender:

  1. Real glove (default address `/v1/animation/kinematic/all`):
       args = [tick:int, frame:int, status:int,
               "Reality Glove (L)":str, "Reality Glove":str,
               <26 joints * 7 floats>]
     Hand side is parsed from the device-label string.

  2. Mock per-hand addresses (`/hand/left` and `/hand/right`):
       args = 187 floats
     Hand side comes from the address.

Strings in the header are replaced with 0.0 placeholders before queueing so
the downstream validator/parser only ever see numeric data.
"""
import logging
from queue import Empty, Full, Queue
from threading import Thread
from typing import List, Optional, Tuple

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from .joints import EXPECTED_VALUE_COUNT

log = logging.getLogger(__name__)

QueueItem = Tuple[str, List[float]]  # (hand_side, raw_values)


def _hand_from_device_label(label: str) -> Optional[str]:
    """Extract 'left' or 'right' from a device label like 'Reality Glove (L)'."""
    if not isinstance(label, str):
        return None
    if "(L)" in label or label.endswith(" L") or label.endswith("_L"):
        return "left"
    if "(R)" in label or label.endswith(" R") or label.endswith("_R"):
        return "right"
    return None


def _to_numeric(args) -> List[float]:
    """Convert OSC args to floats; strings become 0.0 placeholders."""
    out = []
    for a in args:
        if isinstance(a, (int, float)):
            out.append(float(a))
        else:
            out.append(0.0)  # placeholder for strings (device labels)
    return out


class OSCHandReceiver:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9002,
        kinematic_addr: str = "/v1/animation/kinematic/all",
        left_addr: str = "/hand/left",
        right_addr: str = "/hand/right",
        queue_maxsize: int = 240,
        dump: bool = False,
    ):
        self.host = host
        self.port = port
        self.kinematic_addr = kinematic_addr
        self.left_addr = left_addr
        self.right_addr = right_addr
        self.dump = dump
        self.queue: "Queue[QueueItem]" = Queue(maxsize=queue_maxsize)
        self._server: Optional[ThreadingOSCUDPServer] = None
        self._thread: Optional[Thread] = None
        self.msg_count = {"left": 0, "right": 0}
        self.dropped = 0
        self.unknown_count = 0

    # --- dispatcher callbacks -----------------------------------------
    def _on_kinematic(self, address, *args):
        """Real-glove handler: hand side embedded in payload string."""
        if len(args) != EXPECTED_VALUE_COUNT:
            log.warning(
                "[%s] unexpected arg count %d (expected %d) — skipping",
                address, len(args), EXPECTED_VALUE_COUNT,
            )
            return
        hand = _hand_from_device_label(args[3]) if len(args) > 3 else None
        if hand is None:
            log.warning(
                "[%s] could not determine hand from device label %r — skipping",
                address, args[3] if len(args) > 3 else None,
            )
            return
        values = _to_numeric(args)
        if self.dump:
            log.info("[%s] %s hand=%s args=%d", address, "OK", hand, len(values))
        self._enqueue(hand, address, values)

    def _on_left(self, address, *args):
        """Legacy mock handler: hand from address."""
        self._enqueue("left", address, _to_numeric(args))

    def _on_right(self, address, *args):
        self._enqueue("right", address, _to_numeric(args))

    def _on_unknown(self, address, *args):
        # Silenced at INFO — XR Trainer streams many addresses we don't care
        # about. Bump to WARNING only if it persists / nothing else fires.
        self.unknown_count += 1
        log.debug("ignored OSC address %s (%d args)", address, len(args))

    def _enqueue(self, hand: str, address: str, values: List[float]) -> None:
        self.msg_count[hand] += 1
        try:
            self.queue.put_nowait((hand, values))
        except Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait((hand, values))
            except (Empty, Full):
                pass
            self.dropped += 1

    # --- lifecycle ----------------------------------------------------
    def start(self) -> None:
        dispatcher = Dispatcher()
        dispatcher.map(self.kinematic_addr, self._on_kinematic)
        dispatcher.map(self.left_addr, self._on_left)
        dispatcher.map(self.right_addr, self._on_right)
        dispatcher.set_default_handler(self._on_unknown)
        self._server = ThreadingOSCUDPServer((self.host, self.port), dispatcher)
        log.info(
            "OSC server bound %s:%d (kinematic=%s, mock=%s|%s)",
            self.host, self.port,
            self.kinematic_addr, self.left_addr, self.right_addr,
        )
        self._thread = Thread(
            target=self._server.serve_forever, daemon=True, name="osc-server"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # --- consumer-side API --------------------------------------------
    def drain(self, max_items: int = 16) -> List[QueueItem]:
        items: List[QueueItem] = []
        for _ in range(max_items):
            try:
                items.append(self.queue.get_nowait())
            except Empty:
                break
        return items
