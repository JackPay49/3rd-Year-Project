from scipy.io import loadmat, savemat
import typing
from sklearn.model_selection import train_test_split
import numpy as np
import random
from copy import deepcopy

from utility.Dataclasses import FileTypes
from utility.Utilities import get_files


def split_data(
    data_dict: typing.Dict[str, str],
    train: float,
    test: float,
    val: float,
) -> typing.Tuple[typing.Dict[str, str]]:
    """
    Will split the data into train/test/val randomly
    """
    num_data_samples = len(data_dict[DataSplitter.DEFAULT_LABEL_KEY])
    random_indices = list(range(0, num_data_samples))
    random.shuffle(random_indices)

    train_dict = {}
    test_dict = {}
    val_dict = {}

    train_end = int(train * len(random_indices))
    test_end = train_end + int(test * len(random_indices))
    if val != 0.0:
        val_end = test_end + int(val * len(random_indices))

    train_indices = np.array(random_indices)[:train_end]
    test_indices = np.array(random_indices)[train_end:test_end]
    if val != 0.0:
        val_indices = np.array(random_indices)[test_end:val_end]

    for key in data_dict:
        train_dict[key] = np.array(data_dict[key])[train_indices]
        test_dict[key] = np.array(data_dict[key])[test_indices]
        if val != 0.0:
            val_dict[key] = np.array(data_dict[key])[val_indices]

    return (train_dict, test_dict, val_dict)


class DataSplitter:
    """
    Will take a series of file paths. Will load data in from these & will combine this data & labels. Will randomise this data & split into train, test, val sets before exporting
    """

    DEFAULT_DATA_KEY: str = "data"
    DEFAULT_LABEL_KEY: str = "labels"
    DEFAULT_NUM_FILES: int = 1
    IGNORE_KEYS: typing.List[str] = ["__header__", "__version__", "__globals__"]

    def __init__(
        self,
        output_path: str,
        file_paths: typing.List[str] = [],
        file_type: FileTypes = FileTypes.MAT,
        data_split: typing.Tuple[float] = [0.8, 0.1, 0.1],
        data_key: str = DEFAULT_DATA_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        verbose: bool = False,
        num_files: int = DEFAULT_NUM_FILES,
    ) -> None:
        if file_paths == []:
            print(f"No files specified. Early finishing...")

        self.verbose: bool = verbose

        self.output_path: str = output_path
        self.files: typing.List[str] = get_files(
            paths=file_paths, allowed_file_types=[file_type]
        )
        self.file_type: FileTypes = file_type

        self.data_key: str = data_key
        self.label_key: str = label_key

        self.train, self.test, self.validation = data_split

        self.num_files: int = num_files

        self.data_dict = self.load_data(file_paths=self.files, file_type=self.file_type)
        if self.verbose:
            print(f"Labels length: {len(self.data_dict[self.DEFAULT_LABEL_KEY])}")

        self.train_set, self.test_set, self.val_set = split_data(
            data_dict=self.data_dict,
            train=self.train,
            test=self.test,
            val=self.validation,
        )
        self.save_data(
            train_set=self.train_set,
            test_set=self.test_set,
            val_set=self.val_set,
            output_path=self.output_path,
            file_type=self.file_type,
            num_files=self.num_files,
        )

    def save_data(
        self,
        train_set: typing.Dict[str, str],
        test_set: typing.Dict[str, str],
        val_set: typing.Dict[str, str],
        output_path: str,
        file_type: FileTypes,
        num_files: int = DEFAULT_NUM_FILES,
    ) -> None:
        """
        Saves the data to the output path specified
        """
        if num_files >= self.DEFAULT_NUM_FILES:
            base_output_path = output_path.replace(file_type, "")
            output_paths = [
                f"{base_output_path}_{(i + 1)}{file_type}" for i in range(num_files)
            ]

        if self.verbose:
            print(f"Train length: {len(train_set[self.DEFAULT_LABEL_KEY])}")
            print(f"Test length: {len(test_set[self.DEFAULT_LABEL_KEY])}")
            print(f"Val length: {len(val_set[self.DEFAULT_LABEL_KEY])}")

            if num_files >= self.DEFAULT_NUM_FILES:
                print(f"Splitting data into {num_files} files")
                print(f"Output paths will be: {output_paths}")

        if file_type == FileTypes.MAT:
            output_dict = {}
            for key in train_set:
                output_dict[f"train_{key}"] = train_set[key]
            for key in test_set:
                output_dict[f"test_{key}"] = test_set[key]
            for key in val_set:
                output_dict[f"val_{key}"] = val_set[key]

            if num_files >= self.DEFAULT_NUM_FILES:
                subset_size = int(
                    (np.max([len(output_dict[key]) for key in output_dict])) / num_files
                )
                for i in range(num_files):
                    temp_dict = deepcopy(output_dict)
                    for key in temp_dict:
                        temp_dict[key] = temp_dict[key][
                            ((i) * subset_size) : ((i + 1) * subset_size)
                        ]

                    savemat(
                        file_name=output_paths[i],
                        mdict=temp_dict,
                    )
            else:
                savemat(
                    file_name=output_path,
                    mdict=output_dict,
                )

    def load_data(
        self, file_paths: typing.List[str], file_type: FileTypes = FileTypes.MAT
    ) -> typing.Dict[str, str]:
        """
        Loads data from the correct files and returns this
        """
        # Get first sample
        if self.verbose:
            print(f"Loading data from file {file_paths[0]}")
        data_mat = loadmat(file_name=file_paths[0])
        data_dict = {}
        for key in data_mat:
            if key in self.IGNORE_KEYS:
                continue
            data_dict[key] = data_mat[key]

        # Get others
        for file_name in file_paths[1:]:
            if self.verbose:
                print(f"Loading data from file {file_name}")
            if file_type == FileTypes.MAT:
                data_mat = loadmat(file_name=file_name)
                for key in data_mat:
                    if key in self.IGNORE_KEYS:
                        continue
                    data_dict[key] = np.concatenate(
                        (data_dict[key], data_mat[key]), axis=0
                    )
        return data_dict
