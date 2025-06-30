# ruff: noqa: F401, E402
import mpi4py

mpi4py.rc.initialize = False  # don't initialize automatically when import
mpi4py.rc.finalize = True

from .utils import (
    barrier,
    get_default_comm,
    get_dist_info,
    get_local_comm,
    get_local_rank,
    get_local_size,
    get_node_name,
    get_rank,
    get_world_comm,
    get_world_size,
    init_dist,
    is_distributed,
    is_main_process,
    master_only,
)

__all__ = list(globals().keys())
