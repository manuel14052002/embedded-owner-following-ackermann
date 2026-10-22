"""Single-instance guard for the FollowMe node.

Two ``followme_node`` processes publishing to the same ``/cmd_vel`` under the same
node name is a real, silent failure mode: ROS lets it happen, both keep running,
and whichever one's message lands last on a given cycle wins the wheels — every
``ros2 param set``/``get`` and every physical test becomes unreliable because there
is no guarantee which process you are actually talking to or driving.

This is a plain PID-lockfile check, independent of ROS, so it is trivial to unit
test and cannot silently no-op the way the old startup-only parameter read did.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, NamedTuple, Optional

DEFAULT_LOCK_PATH = "/tmp/followme_node.lock"


def _pid_alive_default(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False                 # no such process: definitely not alive
    except PermissionError:
        return True                  # process exists, just owned by someone else: alive
    except OSError:
        return True                  # ambiguous: err toward blocking rather than allowing a duplicate
    else:
        return True


class SingletonResult(NamedTuple):
    acquired: bool
    other_pid: Optional[int]      # set when acquired is False: the PID already holding the lock


def check_singleton(lock_path: str = DEFAULT_LOCK_PATH, own_pid: Optional[int] = None,
                    pid_alive: Callable[[int], bool] = _pid_alive_default) -> SingletonResult:
    """Try to claim the lock. Stale locks (holder PID no longer alive) are reclaimed
    automatically, so a crashed/killed process never permanently blocks a restart.
    Returns ``SingletonResult(acquired=False, other_pid=N)`` if a live instance N holds it.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    path = Path(lock_path)
    if path.is_file():
        try:
            held_pid = int(path.read_text().strip())
        except (ValueError, OSError):
            held_pid = None
        if held_pid is not None and held_pid != own_pid and pid_alive(held_pid):
            return SingletonResult(False, held_pid)
    path.write_text(str(own_pid))
    return SingletonResult(True, None)


def release_singleton(lock_path: str = DEFAULT_LOCK_PATH, own_pid: Optional[int] = None) -> None:
    """Remove the lock, but only if it is still ours (never delete a newer holder's lock)."""
    own_pid = os.getpid() if own_pid is None else own_pid
    path = Path(lock_path)
    try:
        if path.is_file() and int(path.read_text().strip()) == own_pid:
            path.unlink()
    except (ValueError, OSError):
        pass
