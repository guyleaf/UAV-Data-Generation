import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc
from types import TracebackType
from typing import Optional, Type

import bpy
from blenderproc.python.utility.Utility import KeyFrame


# must inherit from KeyFrame to track the context correctly
class Frame(KeyFrame):
    """
    A content manager for setting the frame and subframe correctly.
    """

    def __init__(self, frame: int, subframe: float = 0):
        """Sets the frame number for its complete block.

        :param frame: The frame number to set. If None is given, nothing is changed.
        """
        super().__init__(frame)
        self._subframe = subframe
        self._prev_subframe = None

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


if __name__ == "__main__":
    print(Frame.state.depth, KeyFrame.state.depth)
    with Frame(1):
        print(Frame.state.depth, KeyFrame.state.depth)
        with KeyFrame(3):
            print(Frame.state.depth, KeyFrame.state.depth)
        print(Frame.state.depth, KeyFrame.state.depth)
    print(Frame.state.depth, KeyFrame.state.depth)
