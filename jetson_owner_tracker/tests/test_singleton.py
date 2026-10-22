"""Tests for the FollowMe single-instance guard (followme_ros.singleton)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "followme_ros"))
from followme_ros.singleton import check_singleton, release_singleton  # noqa: E402


def test_fresh_lock_is_acquired(tmp_path):
    lock = str(tmp_path / "followme.lock")
    r = check_singleton(lock, own_pid=111, pid_alive=lambda p: False)
    assert r.acquired and r.other_pid is None
    assert Path(lock).read_text().strip() == "111"


def test_second_instance_is_refused_while_first_is_alive(tmp_path):
    lock = str(tmp_path / "followme.lock")
    check_singleton(lock, own_pid=111, pid_alive=lambda p: True)
    r2 = check_singleton(lock, own_pid=222, pid_alive=lambda p: p == 111)  # only 111 is alive
    assert not r2.acquired and r2.other_pid == 111
    assert Path(lock).read_text().strip() == "111"           # lock untouched, still held by 111


def test_stale_lock_from_a_dead_pid_is_reclaimed(tmp_path):
    lock = str(tmp_path / "followme.lock")
    check_singleton(lock, own_pid=111, pid_alive=lambda p: False)   # 111 "starts" then...
    r2 = check_singleton(lock, own_pid=222, pid_alive=lambda p: False)   # ...crashed: not alive anymore
    assert r2.acquired and r2.other_pid is None
    assert Path(lock).read_text().strip() == "222"


def test_same_pid_can_reacquire_its_own_lock(tmp_path):
    lock = str(tmp_path / "followme.lock")
    check_singleton(lock, own_pid=111, pid_alive=lambda p: True)
    r2 = check_singleton(lock, own_pid=111, pid_alive=lambda p: True)
    assert r2.acquired


def test_corrupt_lock_file_is_treated_as_free(tmp_path):
    lock = tmp_path / "followme.lock"
    lock.write_text("not-a-pid")
    r = check_singleton(str(lock), own_pid=999, pid_alive=lambda p: True)
    assert r.acquired
    assert lock.read_text().strip() == "999"


def test_release_only_removes_own_lock(tmp_path):
    lock = str(tmp_path / "followme.lock")
    check_singleton(lock, own_pid=111, pid_alive=lambda p: False)
    release_singleton(lock, own_pid=222)                      # not ours: must not touch it
    assert Path(lock).is_file()
    release_singleton(lock, own_pid=111)                      # ours: removed
    assert not Path(lock).is_file()


def test_release_of_missing_lock_is_a_no_op(tmp_path):
    release_singleton(str(tmp_path / "nope.lock"), own_pid=1)  # must not raise


def test_default_pid_alive_matches_real_process_life():
    """Sanity check against the real OS: our own PID is alive; PID 1 (init) is almost
    always alive too on Linux; a very unlikely-to-exist high PID should not be."""
    from followme_ros.singleton import _pid_alive_default

    assert _pid_alive_default(os.getpid())
    assert not _pid_alive_default(2 ** 30)       # extremely unlikely to be a real PID
