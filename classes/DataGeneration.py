from LipLocaliser import (
    LocaliserMode,
)
from ConvertDataset import ConvertDatasetLRW
from FaceMapper import FaceMapper
from utility.Dataclasses import FileTypes, Region

from utility.DataSplitter import DataSplitter

split = True
word_set = [
    "ABOUT",
    "AGAIN",
    "AHEAD",
    "ARRESTED",
    "BELIEVE",
    "BUSINESS",
    "CHANCE",
    "DESPITE",
    "ELECTION",
    "FAMILY",
    "FIGURES",
    "FOREIGN",
    "FRANCE",
    "GIVING",
    "LATER",
    "LEADERS",
    "MIGHT",
    "MINUTES",
    "NEEDS",
    "NOTHING",
    "OFFICIALS",
    "PAYING",
    "POSSIBLE",
    "POWERS",
    "RIGHTS",
    "SITUATION",
    "STREET",
    "TALKING",
    "THIRD",
    "THROUGH",
]
# word_set = ['ABOUT','BELIEVE','CHANCE', 'FAMILY','THROUGH']


if not split:
    for word in word_set:
        CD = ConvertDatasetLRW(
            output_dir="proj_data/batches_both/",
            input_paths=[
                f"proj_data/raw_data/{word}",
            ],
            filename_in_path=[-1],
            export_type=FileTypes.MAT,
            padding=100,
            confidence_threshold=0.0,
            region=Region.MOUTH,
            normalisation_method=FaceMapper.NormalisationMethods.MOUTH_LOCATION_KEY,
            word_in_position=-3,
            override_output_path=word + FileTypes.MAT,
            localiser_mode=LocaliserMode.BOTH_KEY,
        )
else:
    # Data split
    file_paths = [f"proj_data/batches_both/{word}.mat" for word in word_set]

    DS = DataSplitter(
        output_path=f"proj_data/batches_both/split_large.mat",
        file_type=FileTypes.MAT,
        file_paths=file_paths,
        num_files=3,
    )

    # file_paths = [f"proj_data/batches_both/{word}.mat" for word in word_set]

    # DS = DataSplitter(
    #     output_path=f"proj_data/batches_both/split_large_test.mat",
    #     file_type=FileTypes.MAT,
    #     file_paths=file_paths,
    # )
