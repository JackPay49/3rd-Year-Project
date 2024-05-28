import cv2
import typing
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ImageOperationKeys:
    FLIP_OPERATION_KEY: str = "flip_operation"


@dataclass
class DefaultImageOperation:
    DEFAULT_FLIP_OPERATION: int = 0


class VideoStream(ABC):
    """
    Abstract class for loading in video
    """

    def __init__(
        self,
        stream_property: typing.Union[int, str] = 0,
        operations: typing.Dict[str, typing.Any] = {},
    ):
        self.stream_property: typing.Union[int, str] = stream_property
        self.operations: typing.Dict[str, typing.Any] = operations

        self.enable_video_stream(stream_property=stream_property)

        self.__operation_mapping = self.__init_operation_mapping()

    def enable_video_stream(self, stream_property: typing.Union[int, str]):
        if stream_property is not None:
            self.stream: cv2.VideoCapture = self.__init_stream(
                stream_property=stream_property
            )
            self.fps: float = self.stream.get(cv2.CAP_PROP_FPS)
            self.num_frames: int = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.stream = None
            self.fps = 0
            self.num_frames = 0

    def __init_operation_mapping(self) -> typing.Dict[str, callable]:
        """
        Creates a mapping from each method key to the correct method within the class. Used to call the correct operations based on the input string
        """
        return {ImageOperationKeys.FLIP_OPERATION_KEY: self.flip_frame}

    def __init_stream(
        self, stream_property: typing.Union[int, str]
    ) -> cv2.VideoCapture:
        if isinstance(stream_property, int):
            return cv2.VideoCapture(index=int(stream_property))
        elif isinstance(stream_property, str):
            return cv2.VideoCapture(filename=str(stream_property))

    def stream_feed(self) -> None:
        """
        Will display the frames from the video stream until the user enters some input
        """
        end_feed = False
        while not end_feed:
            ret, frame = self.stream.read()
            if not ret:
                break
            # Will do some operation on the frame before displaying it
            frame = self.frame_op(frame=frame)

            end_feed = self.use_frame(frame=frame)

    @abstractmethod
    def use_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Will use the frame to do something. Method to be overriden. Uses each frame of the video stream. Must return False when the stream has ended
        """
        return False

    def flip_frame(self, frame: np.ndarray, param: typing.Any) -> np.ndarray:
        """
        Will flip the input frame using the frame property
        """
        flip_property: int = param
        return cv2.flip(src=frame, flipCode=flip_property)

    def frame_op(self, frame: np.ndarray) -> np.ndarray:
        """
        Will carry out each of the input base operations on the frame. Can be overriden
        """
        for x in self.operations:
            if x in self.__operation_mapping:
                frame = self.__operation_mapping[x](
                    frame=frame, param=self.operations[x]
                )
        return frame

    def __del__(self) -> None:
        """
        Destructor for the video stream object. Will close the video stream and destroy the window
        """
        self.stream.release()
