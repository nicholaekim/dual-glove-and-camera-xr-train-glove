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


def test_already_recorded_needs_an_output_text_file(tmp_path):
    out = tmp_path / "out"
    assert not mod.already_recorded(out, "153624")
    (out / "frame_153624").mkdir(parents=True)
    assert not mod.already_recorded(out, "153624")
    (out / "frame_153624" / "frame_153624_keypoints.txt").write_text("k")
    assert mod.already_recorded(out, "153624")


def test_dry_run_lists_commands_with_the_right_hand(tmp_path, capsys):
    _frame(tmp_path, "153624", "NA", "right")
    _frame(tmp_path, "156023", "NA", "left")
    rc = mod.main(["--reference", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--dry-run"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "2 to record" in text
    assert "frame 153624, right hand" in text and "frame 156023, left hand" in text
    assert "--hand right" in text
    rc = mod.main(["--reference", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--dry-run", "--only", "156023", "--hand", "left"])
    assert "1 to record" in capsys.readouterr().out and rc == 0
