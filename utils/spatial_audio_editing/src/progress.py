"""A single progress bar owned and updated by the CLI's main process."""

import os
from multiprocessing import current_process

from tqdm import tqdm


def write(message: str) -> None:
    """Keep log messages above the bar; pool workers stay quiet."""
    if current_process().name == "MainProcess":
        tqdm.write(str(message))


class Progress:
    """Reuse one tqdm instance across all stages of one command."""

    def __init__(self, desc="Preparing", total=None, unit="items"):
        self._owner_pid = os.getpid()
        self._bar = None
        if current_process().name == "MainProcess":
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                position=0,
                leave=True,
            )

    def _owned_bar(self):
        return self._bar if os.getpid() == self._owner_pid else None

    def reset(self, total=None, desc="Processing", unit="items", initial=0):
        bar = self._owned_bar()
        if bar is None:
            return
        # tqdm.reset(total=None) retains the previous total, so clear it first.
        bar.total = total
        bar.unit = unit
        bar.initial = initial
        bar.set_description_str(desc, refresh=False)
        bar.set_postfix_str("", refresh=False)
        bar.reset()
        if initial:
            bar.n = bar.last_print_n = initial
            bar.refresh()

    def update(self, n=1):
        bar = self._owned_bar()
        if bar is not None:
            bar.update(n)

    def set_postfix(self, **counts):
        bar = self._owned_bar()
        if bar is not None:
            bar.set_postfix(counts, refresh=False)

    def write(self, message):
        if os.getpid() == self._owner_pid and current_process().name == "MainProcess":
            write(message)

    def close(self):
        bar = self._owned_bar()
        if bar is not None:
            bar.close()
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
