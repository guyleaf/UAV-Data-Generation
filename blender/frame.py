import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc
import threading
from types import TracebackType
from typing import Optional, Type

import bpy


class _FrameState(threading.local):
    """
    This class is only used in the KeyFrame class
    """

    def __init__(self):
        super().__init__()
        self.depth = 0


class Frame:
    """
    A context manager for setting the frame and subframe correctly.

    Notice: You cannot use the Frame and KeyFrame interchangeably. It may cause unexpected behavior.
    """

    state = _FrameState()

    def __init__(self, frame: Optional[int], subframe: float = 0):
        """Sets the frame number for its complete block.

        :param frame: The frame number to set. If None is given, nothing is changed.
        :param subframe: The subframe number to set.
        """
        self._frame = frame
        self._subframe = subframe
        self._prev_frame = None
        self._prev_subframe = 0

    def __enter__(self):
        Frame.state.depth += 1
        if self._frame is not None:
            self._prev_frame = bpy.context.scene.frame_current
            self._prev_subframe = bpy.context.scene.frame_subframe
            bpy.context.scene.frame_set(self._frame, subframe=self._subframe)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        Frame.state.depth -= 1
        if self._prev_frame is not None:
            bpy.context.scene.frame_set(self._prev_frame, subframe=self._prev_subframe)

    @staticmethod
    def is_any_active() -> bool:
        """Returns whether the current execution point is surrounded by a Frame context manager.

        :return: True, if there is at least one surrounding Frame context manager
        """
        return Frame.state.depth > 0


if __name__ == "__main__":
    scene = bpy.context.scene
    print(f"Depth: {Frame.state.depth}")
    print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
    with Frame(2, 0.6):
        print(f"Depth: {Frame.state.depth}")
        print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
        with Frame(3, 0.5):
            print(f"Depth: {Frame.state.depth}")
            print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
        print(f"Depth: {Frame.state.depth}")
        print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
        with Frame(4):
            print(f"Depth: {Frame.state.depth}")
            print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
        print(f"Depth: {Frame.state.depth}")
        print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
    print(f"Depth: {Frame.state.depth}")
    print(f"Current Frame: ({scene.frame_current}, {scene.frame_subframe})")
