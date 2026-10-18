"""Run a campaign command within the shared memory and reserve limits."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable is required for the campaign memory reserve')


def session_rss_bytes(session_id):
    pages = 0
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(') ',1)[1].split()
            if int(fields[3]) == session_id:
                pages += int((path/'statm').read_text().split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return pages*os.sysconf('SC_PAGE_SIZE')


def main():
    command = sys.argv[1:]
    if not command:
        raise ValueError('a campaign command is required')
    reserve = 8*1024**3
    ceiling = 48*1024**3
    if available_bytes() < reserve:
        raise RuntimeError('less than 8 GiB is available before the campaign command')
    def terminate(signum, frame):
        raise SystemExit(128+signum)

    previous_term = signal.signal(signal.SIGTERM, terminate)
    start = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    peak = 0
    try:
        while child.poll() is None:
            peak = max(peak, session_rss_bytes(child.pid))
            if peak > ceiling or available_bytes() < reserve:
                raise RuntimeError('campaign memory exceeds 48 GiB or leaves less than 8 GiB available')
            time.sleep(.2)
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        signal.signal(signal.SIGTERM, previous_term)
        print(f'Campaign resources: {time.monotonic()-start:.2f} s, '
              f'peak sampled process-session RSS {peak/1024**3:.3f} GiB', file=sys.stderr, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
