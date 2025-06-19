import cProfile
import os
import signal
from typing import Callable


def profile(func: Callable, out_file: str = "./cprofile_test.prof"):
    with cProfile.Profile() as pr:
        orig_handler = signal.getsignal(signal.SIGTERM)

        # Listen for SIGTERM signal, so we can properly clean up and terminate the child process
        def handle_sigterm(_signum, _frame):
            pr.dump_stats(out_file)
            signal.signal(signal.SIGTERM, orig_handler)
            os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, handle_sigterm)

        try:
            func()
        finally:
            pr.dump_stats(out_file)
