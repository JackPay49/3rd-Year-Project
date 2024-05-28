# Purpose of this script is to store general helper functions to do things
import json
import typing
import os
import re
import numpy as np
from PIL import Image
import tensorflow as tf

# import winsound

# from classes.utility.Dataclasses import FileTypes
from utility.Dataclasses import FileTypes, ALL_FILE_TYPES


def pad_image(img: np.array, padding: int = 0) -> np.array:
    """
    Takes an image and returns a version padded to teh square value input. Pads with 0 value
    """
    if padding == 0:
        return img
    image = Image.fromarray(img)
    width, height = image.size
    if width > padding or height > padding:
        print(
            f"Higher padding value is required. Image is larger than the padding width & height. Cropping image..."
        )
    elif width == padding and height == padding:
        return img
    return tf.image.resize_with_crop_or_pad(img, padding, padding).numpy()


def load_json(filename: str) -> typing.Dict[str, typing.Any]:
    """
    Just loads a json
    """
    return_json = None
    if FileTypes.JSON not in filename:
        filename += FileTypes.JSON
    with open(filename, "r") as f:
        return_json = json.loads(f.read())
    return return_json


def export_json(dict: typing.Dict[str, typing.Any], path: str, indent: int = 4) -> None:
    """
    Standard method for exporting json files
    """
    if not check_path_exists(path=os.path.dirname(path)):
        print(f"Unable to export json to path {path}. This path does not exist")
        return
    with open(file=path, mode="w+") as f:
        json.dump(obj=dict, fp=f, indent=indent)


def check_path_exists(path: str) -> bool:
    """
    Standard method to check if a file exists
    """
    return os.path.exists(path=path)


def check_file_type(path: str, allowed_file_types: typing.List[FileTypes]) -> bool:
    """
    Will check the file type of a file against that of a lsit of allowed file types, returning whether it is allowed
    """
    if allowed_file_types == []:
        return True
    return get_file_type(path=path) in allowed_file_types


def remove_file_types(path: str) -> str:
    """
    Will remove any file types from the path, leaving it without an extension
    """
    for x in ALL_FILE_TYPES:
        if x in path:
            return re.sub(x, "", path)
    return path


def get_file_type(
    path: str,
) -> FileTypes:
    """
    Will check the file type of a file and return the dataclass FileTypes
    """
    return "." + path.split(".")[-1]


# def play_sound(path: str) -> None:
#     """
#     Plays a sound out loud
#     """
#     winsound.PlaySound(path, winsound.SND_FILENAME)


def get_files(
    paths: typing.List[str], allowed_file_types: typing.List[FileTypes] = []
) -> typing.List[str]:
    """
    Takes a list of paths to files or directories. Will expand this, adding in the paths of all files within the directories specified. Will validate that directories exist
    """
    return_paths: typing.List[str] = []
    if paths:
        for path in paths:
            if os.path.isfile(path=path) and check_file_type(
                path=path, allowed_file_types=allowed_file_types
            ):
                return_paths.append(path)
                continue
            elif not check_path_exists(path=path):
                print(f"Input path of {path} was not valid. This path does not exist")
                continue
            elif os.path.isdir(s=path):
                return_paths += get_files(
                    paths=[os.path.join(path, x) for x in os.listdir(path=path)],
                    allowed_file_types=allowed_file_types,
                )
    return return_paths
