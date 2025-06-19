# References from https://github.com/open-mmlab/mmengine/blob/main/mmengine/dist/utils.py

import functools
from typing import Callable, Optional
import sys
from mpi4py import MPI

_COMM_WORLD = MPI.COMM_WORLD
_COMM_NODE: Optional[MPI.Intracomm] = None


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return MPI.Is_initialized()


def _mpi_abort_excepthook(type, exception, traceback):
    if is_distributed():
        _COMM_WORLD.Abort()
    sys.__excepthook__(type, exception, traceback)


def init_dist():
    sys.excepthook = _mpi_abort_excepthook
    # the user may import MPI before calling. initializing twice will raise an error.
    if not MPI.Is_initialized():
        MPI.Init()
    rank = _COMM_WORLD.Get_rank()

    # split into communicators based on nodes
    _COMM_NODE = _COMM_WORLD.Split_type(
        MPI.COMM_TYPE_RESOURCE_GUIDED, key=rank)

    assert _COMM_NODE is not None, "Cannot create a node intra-comm.."
    assert isinstance(
        _COMM_NODE, MPI.Intracomm), "The node comm. should be an intra-comm.."


def get_world_comm() -> Optional[MPI.Intracomm]:
    return _COMM_WORLD if is_distributed() else None


def get_local_comm() -> Optional[MPI.Intracomm]:
    return _COMM_NODE if is_distributed() else None


def get_default_comm() -> Optional[MPI.Intracomm]:
    """Return default communicator."""
    return get_world_comm()


def get_world_size(comm: Optional[MPI.Comm] = None) -> int:
    """Return the number of the given communicator.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        comm (MPI.Comm, optional): The communicator to work on. If None,
            the default communicator will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given communicator if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        if comm is None:
            comm = get_default_comm()
        return comm.Get_size()
    else:
        return 1


def get_local_size() -> int:
    """Return the number of the current node.

    Returns:
        int: Return the number of processes in the current node if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        assert _COMM_NODE is not None, "You should call init_dist() first."
        return _COMM_NODE.Get_size()
    else:
        return 1


def get_rank(comm: Optional[MPI.Comm] = None) -> int:
    """Return the rank of the given communicator.

    Rank is a unique identifier assigned to each process within a distributed
    communicator. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        comm (MPI.Comm, optional): The communicator to work on. If None,
            the default communicator will be used. Defaults to None.

    Returns:
        int: Return the rank of the communicator if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        if comm is None:
            comm = get_default_comm()
        return comm.Get_rank()
    else:
        return 0


def get_local_rank() -> int:
    """Return the rank of current process in the current node.

    Returns:
        int: Return the rank of current process in the current node if in
        distributed environment, otherwise 0
    """
    if is_distributed():
        assert _COMM_NODE is not None, "You should call init_dist() first."
        return _COMM_NODE.Get_rank()
    else:
        return 0


def get_dist_info(comm: Optional[MPI.Comm] = None) -> tuple[int, int]:
    """Get distributed information of the given communicator.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        comm (MPI.Comm, optional): The communicator to work on. If None,
            the default communicator will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(comm)
    rank = get_rank(comm)
    return rank, world_size


def is_main_process(comm: Optional[MPI.Comm] = None) -> bool:
    """Whether the current rank of the given communicator is equal to 0.

    Args:
        comm (MPI.Comm, optional): The communicator to work on. If None,
            the default communicator will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given communicator is
        equal to 0, otherwise False.
    """
    return get_rank(comm) == 0


def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def barrier(comm: Optional[MPI.Comm] = None) -> None:
    """Synchronize all processes from the given communicator.

    This collective blocks processes until the whole communicator enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        comm (MPI.Comm, optional): The communicator to work on. If None,
            the default communicator will be used. Defaults to None.
    """
    if is_distributed():
        if comm is None:
            comm = get_default_comm()
        comm.Barrier()
