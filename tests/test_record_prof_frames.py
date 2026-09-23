"""The batch runner for the professor's frames: which frames, which hand, what is left."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("record_prof_frames", ROOT / "scripts" / "leap" / "record_prof_frames.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _frame(root: Path, fid: str, status: str, hand: str) -> Path:
    d = root / (f"frame_{fid}_{status}" if status else f"frame_{fid}")
    d.mkdir()
    (d / f"frame_{fid}.txt").write_text(f"Frame {fid} | Hand ID 1499 ({hand})\n0 (Palm): (1, 2, 3)\n", encoding="utf-8")
    (d / f"frame_{fid}.png").write_bytes(b"png")
    return d


def test_lists_only_the_na_frames_by_default(tmp_path):
    _frame(tmp_path, "102287", "DONE", "left")
    _frame(tmp_path, "153624", "NA", "right")
    _frame(tmp_path, "156023", "NA", "left")
    (tmp_path / "notes.txt").write_text("x")
    assert [f for f, _ in mod.list_frames(tmp_path)] == ["153624", "156023"]
    assert [f for f, _ in mod.list_frames(tmp_path, include_done=True)] == ["102287", "153624", "156023"]


def test_hand_comes_from_the_professors_file(tmp_path):
    d = _frame(tmp_path, "153624", "NA", "right")
    assert mod.hand_in_reference(d, "153624") == "right"
    (d / "frame_153624.txt").write_text("Frame 153624 | Hand ID 7\n", encoding="utf-8")
    assert mod.hand_in_reference(d, "153624") is None


def test_already_recorded_looks_for_the_keypoint_file_next_to_the_take_folder(tmp_path):
    out = tmp_path / "out"
    assert not mod.already_recorded(out, "153624")
    (out / "frame_153624").mkdir(parents=True)
    (out / "frame_153624" / "frame_153624_take1_20260923_155100.jsonl").write_text("{}")
    assert not mod.already_recorded(out, "153624")      # a take alone is not done
    (out / "frame_153624" / "notes.txt").write_text("k")
    assert not mod.already_recorded(out, "153624")      # nor a text file inside it
    (out / "frame_153624_keypoints.txt").write_text("k")
    assert mod.already_recorded(out, "153624")


def test_dry_run_passes_no_hand_filter_by_default(tmp_path, capsys):
    _frame(tmp_path, "153624", "NA", "right")
    _frame(tmp_path, "156023", "NA", "left")
    rc = mod.main(["--reference", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--dry-run"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "2 to record" in text
    assert "frame 153624, keeping any hand (his file says right)" in text
    assert "--hand" not in text.replace("keeping any hand", "")
    rc = mod.main(["--reference", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--dry-run", "--hand", "any"])
    assert "--hand" not in capsys.readouterr().out.replace("keeping any hand", "") and rc == 0
    rc = mod.main(["--reference", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--dry-run", "--only", "156023", "--hand", "left"])
    text = capsys.readouterr().out
    assert "1 to record" in text and "--hand left" in text and rc == 0


def test_a_done_frame_is_skipped_on_the_next_run(tmp_path, capsys):
    _frame(tmp_path, "153624", "NA", "right")
    _frame(tmp_path, "156023", "NA", "left")
    out = tmp_path / "out"
    out.mkdir()
    (out / "frame_153624_keypoints.txt").write_text("k")
    mod.main(["--reference", str(tmp_path), "--out-dir", str(out), "--dry-run"])
    text = capsys.readouterr().out
    assert "1 to record, 1 already done" in text and "frame 156023" in text


# --- the keypoint file and --rewrite ----------------------------------------
def _take(path: Path, n_left: int, n_right: int) -> Path:
    """A real Leap take: mock hands through LeapRecorder, n frames per label."""
    from leap_hand.mock import MockLeapStream
    from leap_hand.recorder import LeapRecorder

    stream = MockLeapStream(pose="open_palm", noise_mm=0.3)
    stream.start()
    left = right = 0
    rec = LeapRecorder(pose=path.stem, take=1)
    rec.start(path)
    for side, lh in stream.generate(max(n_left, n_right) + 50):
        if side == "left" and left < n_left:
            rec.record(lh)
            left += 1
        elif side == "right" and right < n_right:
            rec.record(lh)
            right += 1
    rec.stop()
    assert (left, right) == (n_left, n_right)
    return path


def test_write_prof_file_keeps_the_label_with_the_most_frames(tmp_path):
    rf = mod.record_frame_module()
    take = _take(tmp_path / "t.jsonl", n_left=12, n_right=5)
    out = tmp_path / "frame_1_keypoints.txt"

    result = rf.write_prof_file(take, out)
    assert result["side"] == "left" and result["counts"] == {"left": 12, "right": 5}
    text = out.read_text(encoding="utf-8")
    assert text.count("Frame ") == 1 and "Hand ID: left" in text
    assert rf.kept_summary(result) == "kept 12 frames labelled left, dropped 5 labelled right"

    result = rf.write_prof_file(take, out, hand_filter="right")
    assert result["side"] == "right" and "Hand ID: right" in out.read_text(encoding="utf-8")

    only_right = _take(tmp_path / "r.jsonl", n_left=0, n_right=4)
    result = rf.write_prof_file(only_right, tmp_path / "none.txt", hand_filter="left")
    assert result["side"] is None and not (tmp_path / "none.txt").exists()
    assert rf.kept_summary(result) == "dropped 4 labelled right"


def test_rewrite_uses_the_newest_take_and_records_nothing(tmp_path, capsys):
    ref = tmp_path / "ref"
    ref.mkdir()
    _frame(ref, "153624", "NA", "right")
    _frame(ref, "156023", "NA", "left")
    out = tmp_path / "out"
    takes = out / "frame_153624"
    takes.mkdir(parents=True)
    _take(takes / "frame_153624_take1_20260923_155100.jsonl", n_left=3, n_right=9)
    newest = _take(takes / "frame_153624_take1_20260923_161003.jsonl", n_left=8, n_right=2)

    rc = mod.main(["--reference", str(ref), "--out-dir", str(out), "--rewrite"])
    text = capsys.readouterr().out
    assert rc == 1                                   # 156023 has no take
    assert f"153624  {newest.name}  kept left  (frames: left 8, right 2)" in text
    assert "156023  no take" in text
    kp = (out / "frame_153624_keypoints.txt").read_text(encoding="utf-8")
    assert kp.count("Frame ") == 1 and "Hand ID: left" in kp

    rc = mod.main(["--reference", str(ref), "--out-dir", str(out), "--rewrite",
                   "--only", "153624", "--hand", "right"])
    assert "kept right" in capsys.readouterr().out and rc == 0
    assert "Hand ID: right" in (out / "frame_153624_keypoints.txt").read_text(encoding="utf-8")
