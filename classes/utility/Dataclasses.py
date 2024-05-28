from dataclasses import dataclass
import typing


@dataclass(frozen=True)
class FileTypes:
    VIDEO: str = ".mp4"
    ANNOTATION: str = ".txt"
    NPY: str = ".npy"
    CSV: str = ".csv"
    IMAGE_JPG: str = ".jpg"
    MAT: str = ".mat"
    HDF5: str = ".hdf5"
    JSON: str = ".json"
    KERAS: str = ".keras"
    OBJECT: str = ".obj"


ALL_FILE_TYPES = [
    FileTypes.VIDEO,
    FileTypes.ANNOTATION,
    FileTypes.NPY,
    FileTypes.CSV,
    FileTypes.IMAGE_JPG,
    FileTypes.MAT,
    FileTypes.HDF5,
    FileTypes.JSON,
    FileTypes.KERAS,
]


@dataclass(frozen=True)
class Region:
    LIPS: str = "LIPS"
    MOUTH: str = "MOUTH"
    LANDMARKS: str = "LANDMARKS"
    LIPS_AND_MOUTH: str = "LIPS_AND_MOUTH"
    MOUTH_AND_LANDMARKS: str = "MOUTH_AND_LANDMARKS"
    NONE: str = "NONE"


@dataclass(frozen=True)
class FeatureExtractor:
    INCEPTION_IMAGENET_KEY: str = "INCEPTION_IMAGENET"
    INCEPTION_IMAGENET_SIZE_KEY: int = 2048
    NONE_KEY: str = "NONE"


@dataclass(frozen=True)
class PredictionMethod:
    BINARY_KEY: str = "BINARY"
    ARGMAX_KEY: str = "ARGMAX"
    CTC_KEY: str = "CTC"
    NONE_KEY: str = "NONE"


@dataclass(frozen=True)
class BoundingBox:
    x_min: int  # Minimum x-coordinate
    y_min: int  # Minimum y-coordinate
    x_max: int  # Maximum x-coordinate
    y_max: int  # Maximum y-coordinate

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min

    def area(self) -> int:
        return self.width() * self.height()

    def get_min(self) -> typing.Tuple[int]:
        return (self.x_min, self.y_min)

    def get_max(self) -> typing.Tuple[int]:
        return (self.x_max, self.y_max)


@dataclass(frozen=True)
class LipState:
    """
    State of the lips, based on landmark placement
    """

    OPEN_KEY: str = "OPEN"
    CLOSED_KEY: str = "CLOSED"
    UNKNOWN_KEY: str = "UNKNOWN"


@dataclass(frozen=True)
class GUIMode:
    """
    Mode of the GUI between doing just model inference and model training
    """

    TRAIN: str = "TRAIN"
    INFERENCE: str = "INFERENCE"
