import builtins
import logging
import os

from rich.console import Console
from rich.logging import RichHandler

from .. import distributed as dist

_FORMAT = "%(message)s"
_CONFIGURED = False

orig_print = builtins.print


def print_wrapper(*args, **kwargs) -> None:
    if dist.is_main_process():
        orig_print(*args, **kwargs)


# TODO: better way to manage
def configure_logger(level: int = logging.INFO):
    global _CONFIGURED
    if dist.is_distributed():
        rank = dist.get_rank()
        level = level if dist.is_main_process() else logging.ERROR
        format = f"[Rank {rank}]: {_FORMAT}"
        # in MPI env, the width of terminal cannot be detected correctly.
        console = Console(width=int(os.getenv("CONSOLE_WIDTH", 200)))
    else:
        level = level
        format = _FORMAT
        console = Console()

    logging.basicConfig(
        level=level,
        format=format,
        handlers=[RichHandler(console=console, markup=True)],
        force=True,
    )
    builtins.print = print_wrapper
    _CONFIGURED = True


def get_logger(name: str = "uav_data_generation"):
    if not _CONFIGURED:
        configure_logger()
    return logging.getLogger(name)


def raise_error(exception: BaseException):
    logger = get_logger()
    logger.exception(exception)
    raise exception
