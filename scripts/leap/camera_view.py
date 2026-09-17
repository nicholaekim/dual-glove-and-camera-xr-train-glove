"""Live view of what the Ultraleap IR camera sees, with the tracked hand drawn on it.

Opens a window showing the camera's infrared image (mirrored, so your left hand
is on the left like a mirror), the skeleton the tracker has fitted, and for each
hand: tracked or not, height above the lens against a target band, and whether
the palm faces the lens. Use it to position your hand before and during any
recording. It is a second, read-only client of the tracking service, so it can
run next to any other script.

Usage:
  python camera_view.py                      # free-running viewer, q or Esc to close
  python camera_view.py --hand left          # only judge the left hand
  python camera_view.py --band 18,28         # target height band in cm

Other scripts start it for you and feed it a caption through --status-file.
"""
import argparse, math, os, sys, threading, time

import cv2
import numpy as np
import leap
from leap import enums
from leapc_cffi import ffi, libleapc

ap = argparse.ArgumentParser()
ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
ap.add_argument("--band", default="18,28", help="target palm height in cm: LOW,HIGH")
ap.add_argument("--status-file", default=None, help="text file whose first line is shown as a caption; '__quit__' closes the viewer")
ap.add_argument("--parent-pid", type=int, default=0, help="close when this process exits (set by scripts that launch the viewer)")
ap.add_argument("--seconds", type=float, default=0.0, help="close by itself after this long (0 = never)")
ap.add_argument("--snapshot", default=None, help="save the last composed frame here on exit (testing)")
ap.add_argument("--no-window", action="store_true", help="do not open a window (testing)")
args = ap.parse_args()
LOW, HIGH = (float(v) for v in args.band.split(","))

SCALE = 2                         # 384 px sensor image -> 768 px window
BASELINE_HALF_MM = 32.0           # SIR170 lenses are 64 mm apart; the left lens sits at x = +32 mm... see to_px
LEFT_CAMERA = 1                   # eLeapPerspectiveType_stereo_left
COLOURS = {"left": (255, 220, 0), "right": (60, 60, 255)}      # BGR: cyan-ish left, red right (as in the repo viewers)

lock = threading.Lock()
state = {"img": None, "hands": [], "hands_t": 0.0, "fps": 0.0, "frames": 0}


def image_to_numpy(image):
    c = image.c_data; p = c.properties
    n = p.width * p.height * p.bpp
    ptr = ffi.cast("uint8_t *", c.data) + c.offset
    a = np.frombuffer(ffi.buffer(ptr, n), dtype=np.uint8).copy()       # copy now: LeapC reuses the buffer
    return a.reshape(p.height, p.width) if p.bpp == 1 else a.reshape(p.height, p.width, p.bpp)[:, :, 0]


def vec(v):
    return (v.x, v.y, v.z)


class Listener(leap.Listener):
    def on_image_event(self, e):
        try:
            img = image_to_numpy(e.image[0])
        except Exception:
            return
        with lock:
            state["img"] = img

    def on_tracking_event(self, e):
        hands = []
        for h in e.hands:
            side = "left" if h.type == enums.HandType.Left else "right"
            chains = []
            for d in h.digits:
                b = d.bones
                chains.append([vec(b[0].prev_joint), vec(b[1].prev_joint), vec(b[2].prev_joint), vec(b[3].prev_joint), vec(b[3].next_joint)])
            n = vec(h.palm.normal); p = vec(h.palm.position)
            # viewing angle: palm normal against the ray from the palm to the camera (the origin)
            r = np.array([-p[0], -p[1], -p[2]]); r = r / (np.linalg.norm(r) or 1.0)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(np.array(n), r))))))
            hands.append(dict(side=side, id=h.id, palm=p, wrist=vec(h.arm.next_joint), elbow=vec(h.arm.prev_joint),
                              chains=chains, view_deg=ang, visible=h.visible_time / 1e6))
        with lock:
            state["hands"] = hands; state["hands_t"] = time.time()
            state["fps"] = float(e.framerate or 0.0); state["frames"] += 1


conn = leap.Connection(listeners=[Listener()])
conn.connect()
conn.set_tracking_mode(enums.TrackingMode.Desktop)
try:
    active = conn.set_policy_flags(flags_to_set=[enums.PolicyFlag.Images])
    images_on = enums.PolicyFlag.Images in active
except Exception as ex:                                   # viewer still useful without the picture
    images_on = False; print("could not enable camera images:", ex)
PTR = conn.get_connection_ptr()


def to_px(p):
    """Tracking-space millimetres -> pixel in the (unmirrored, upscaled) left image. Checked against saved IR stills."""
    x, y, z = p
    if y <= 5.0:
        return None
    v = ffi.new("LEAP_VECTOR*"); v.x = -(x - BASELINE_HALF_MM) / y; v.y = z / y; v.z = 1.0
    r = libleapc.LeapRectilinearToPixel(PTR, LEFT_CAMERA, v[0])
    if not (math.isfinite(r.x) and math.isfinite(r.y)):
        return None
    return int(r.x * SCALE), int(r.y * SCALE)


def put(frame, text, org, scale=0.6, colour=(255, 255, 255), thick=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def parent_alive():
    """True while the launching script is still running (Windows-safe: never signals the process)."""
    if not args.parent_pid:
        return True
    try:
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, args.parent_pid)
        if not h:
            return False
        alive = ctypes.windll.kernel32.WaitForSingleObject(h, 0) != 0      # 0 = signalled = exited
        ctypes.windll.kernel32.CloseHandle(h)
        return alive
    except Exception:
        return True


def read_caption():
    """First line of the status file is the caption; an optional second line
    `band=LOW,HIGH` (cm) lets the launching script move the height target."""
    global LOW, HIGH
    if not args.status_file:
        return ""
    try:
        with open(args.status_file, encoding="utf-8") as fh:
            first = fh.readline().strip()
            second = fh.readline().strip()
    except OSError:
        return ""
    if second.startswith("band="):
        try:
            lo, hi = (float(v) for v in second[5:].split(","))
            if lo < hi:
                LOW, HIGH = lo, hi
        except ValueError:
            pass
    return first


def compose():
    with lock:
        img = None if state["img"] is None else state["img"].copy()
        hands = list(state["hands"]) if time.time() - state["hands_t"] < 0.25 else []
        fps = state["fps"]; frames = state["frames"]
    size = 384 * SCALE
    if img is None:
        frame = np.zeros((size, size, 3), np.uint8)
    else:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.convertScaleAbs(img, alpha=2.2, beta=8)               # the raw IR image is dark; lift it for the eye
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # where the camera's own axis is: a ring for the sweet spot 22 cm above the lens
    c = to_px((0.0, 220.0, 0.0))
    if c:
        cv2.circle(frame, c, 70, (90, 90, 90), 1, cv2.LINE_AA); cv2.drawMarker(frame, c, (90, 90, 90), cv2.MARKER_CROSS, 16, 1)
    for h in hands:
        col = COLOURS[h["side"]]
        w = to_px(h["wrist"]); e = to_px(h["elbow"])
        if w and e:
            cv2.line(frame, e, w, col, 2, cv2.LINE_AA)
        for chain in h["chains"]:
            pts = [to_px(p) for p in chain]
            if w and pts[0]:
                cv2.line(frame, w, pts[0], col, 1, cv2.LINE_AA)
            for a, b in zip(pts, pts[1:]):
                if a and b:
                    cv2.line(frame, a, b, col, 2, cv2.LINE_AA)
            for p in pts:
                if p:
                    cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    frame = cv2.flip(frame, 1)                                           # mirror: your left hand on the left
    # ---- text, after the flip so it reads normally
    wanted = [args.hand] if args.hand != "both" else ["left", "right"]
    seen = {h["side"]: h for h in hands}
    y = 30
    caption = read_caption()
    if caption and caption != "__quit__":
        put(frame, caption, (14, y), 0.75, (0, 255, 255), 2); y += 34
    ok_all = True
    for side in wanted:
        h = seen.get(side)
        if h is None:
            ok_all = False
            other = [s for s in seen if s != side]
            msg = f"{side.upper()}: NO HAND" + (f"  (camera sees a {other[0].upper()} hand)" if other and args.hand != "both" else "")
            put(frame, msg, (14, y), 0.7, (60, 60, 255), 2); y += 30
            continue
        cm = h["palm"][1] / 10.0
        where = "OK" if LOW <= cm <= HIGH else ("TOO LOW - raise it" if cm < LOW else "TOO HIGH - lower it")
        facing = "palm to lens" if h["view_deg"] < 40 else ("tilted" if h["view_deg"] < 65 else "EDGE-ON")
        good = (LOW <= cm <= HIGH) and h["view_deg"] < 65
        ok_all = ok_all and good
        put(frame, f"{side.upper()}: tracked  {cm:4.1f} cm {where}  |  {facing} ({h['view_deg']:.0f} deg)", (14, y), 0.62,
            (80, 255, 80) if good else (0, 190, 255), 2); y += 30
    border = (80, 255, 80) if (ok_all and seen) else (60, 60, 255)
    cv2.rectangle(frame, (0, 0), (size - 1, size - 1), border, 6)
    put(frame, f"tracking {fps:4.1f} Hz   target {LOW:.0f}-{HIGH:.0f} cm   q / Esc closes", (14, size - 16), 0.5, (200, 200, 200), 1)
    if not images_on:
        put(frame, "camera images are off: enable 'Allow Images' in the Ultraleap Control Panel", (14, size - 44), 0.5, (0, 190, 255), 1)
    return frame


WIN = "Ultraleap camera view"
if not args.no_window:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 768, 768)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
t0 = time.time(); last = None
try:
    while True:
        last = compose()
        if not args.no_window:
            cv2.imshow(WIN, last)
            k = cv2.waitKey(30) & 0xFF
            if k in (27, ord("q")):
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            time.sleep(0.03)
        if args.seconds and time.time() - t0 > args.seconds:
            break
        if read_caption() == "__quit__" or not parent_alive():
            break
except KeyboardInterrupt:
    pass
finally:
    if args.snapshot and last is not None:
        cv2.imwrite(args.snapshot, last)
    try:
        conn.disconnect()
    except Exception:
        pass
    if not args.no_window:
        cv2.destroyAllWindows()
