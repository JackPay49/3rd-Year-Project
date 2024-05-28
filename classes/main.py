from LipLocaliser import (
    LipLocaliserShowFeed,
    LipLocaliserExportFrames,
    LipLocaliserExportKeypoints,
    LocaliserMode,
)
from ConvertDataset import ConvertDatasetLRS2, ConvertDatasetLRW
from VideoStream import ImageOperationKeys
from FaceMapper import FaceMapper
from utility.Dataclasses import FileTypes, Region, FeatureExtractor, PredictionMethod
from LipReader import LipReaderShowFeed

# from utility.Utilities import play_sound
from utility.DataSplitter import DataSplitter

# Will show using the current computer's camera, a flipped version of the view with a box drawn around the lips
LLSF = LipLocaliserShowFeed(
    stream_property=0,
    operations={ImageOperationKeys.FLIP_OPERATION_KEY: 1},
    region=Region.MOUTH,
    localiser_mode=LocaliserMode.LANDMARKS_KEY,
    normalisation_method=FaceMapper.NormalisationMethods.MOUTH_LOCATION_KEY,
)
LLSF.stream_feed()


# Will load in the video file of test_vid.mp4, apply lip localisation and then output the resulting frames together to output.mp4
# LLE = LipLocaliserExportFrames(
#     stream_property="test_vid.mp4", output_path= "output.mp4", frame_limit = 100
# )
# LLE.stream_feed()

# Will show using the current computer's camera, a flipped version of the view with a box drawn around the lips
# LLSF = LipLocaliserExportKeypoints(
#     stream_property=0, operations={ImageOperationKeys.FLIP_OPERATION_KEY: 1},output_path="out.txt",frame_limit=5
# )
# LLSF.stream_feed()

# for word in ["ABOUT", "BELIEVE", "CHANCE", "FAMILY"]:
#     CD = ConvertDatasetLRW(
#         output_dir="proj_data/batches_both/",
#         input_paths=[
#             f"proj_data/raw_data/{word}",
#         ],
#         filename_in_path=[-1],
#         export_type=FileTypes.MAT,
#         padding=100,
#         confidence_threshold=0.0,
#         region=Region.MOUTH,
#         normalisation_method=FaceMapper.NormalisationMethods.MOUTH_LOCATION_KEY,
#         word_in_position=-3,
#         override_output_path=word + FileTypes.MAT,
#         localiser_mode=LocaliserMode.BOTH_KEY,
#     )


# CD = ConvertDatasetLRW(
#     output_dir="D:/USB/data_gen_6",
#     input_paths=[
#         f"D:/USB/LRW/lipread_mp4/ABOUT/train/ABOUT_00091.mp4",
#     ],
#     filename_in_path=[-1],
#     export_type=FileTypes.MAT,
#     padding=100,
#     confidence_threshold=0.0,
#     region=Region.MOUTH,
#     normalisation_method=FaceMapper.NormalisationMethods.MOUTH_LOCATION_KEY,
#     word_in_position=-3,
#     override_output_path= "ABOUT" + FileTypes.MAT,
#     localiser_mode=LocaliserMode.BOTH_KEY,
# )

### Data generate
# CD = ConvertDatasetLRS2(
#     output_dir="proj_data/lrs2/new",
#     input_paths=[
#         "proj_data/first_100/5535415699068794046/",
#         "proj_data/first_100/5535423430009926848/",
#         "proj_data/first_100/5535496873950688380/",
#         "proj_data/first_100/5535864093654496929/",
#         "proj_data/first_100/5536038039829982468/",
#         "proj_data/first_100/5536266102593401990/",
#         "proj_data/first_100/5536760882825901738/",
#         "proj_data/first_100/5536915501648559593/",
#         "proj_data/first_100/5536968329746298779/",
#         "proj_data/first_100/5537369050195015499/",
#         "proj_data/first_100/5537514649586349811/",
#         "proj_data/first_100/5537693749722594824/",
#         "proj_data/first_100/5537751731781090844/",
#         "proj_data/first_100/5537885734760724252/",
#         "proj_data/first_100/5538013295289415430/",
#         "proj_data/first_100/5538021026230548229/",
#         "proj_data/first_100/5538256819935098692/",
#         "proj_data/first_100/5538264550876231491/",
#         "proj_data/first_100/5539444807889172133/",
#         "proj_data/first_100/5539535002202392187/",
#         "proj_data/first_100/5539702505926936192/",
#         "proj_data/first_100/5539741160632598296/",
#         "proj_data/first_100/5539826200985059108/",
#         "proj_data/first_100/5540119976748105533/",
#         "proj_data/first_100/5540197286159433545/",
#         "proj_data/first_100/5540444676275685005/",
#         "proj_data/first_100/5540483330981347168/",
#         "proj_data/first_100/5540618622451171190/",
#         "proj_data/first_100/5540815761450059421/",
#         "proj_data/first_100/5540854416155721607/",
#         "proj_data/first_100/5540862147096854408/",
#         "proj_data/first_100/5541186846624433836/",
#         "proj_data/first_100/5541225501330096046/",
#         "proj_data/first_100/5541956075267145599/",
#         "proj_data/first_100/5542026942227535168/",
#         "proj_data/first_100/5542042404109797485/",
#         "proj_data/first_100/5542044981090174822/",
#         "proj_data/first_100/5542072039384139495/",
#         "proj_data/first_100/5542132598423013227/",
#         "proj_data/first_100/5542338756853221069/",
#         "proj_data/first_100/5542346487794353870/",
#         "proj_data/first_100/5543080927201970107/",
#         "proj_data/first_100/5543208487730661326/",
#         "proj_data/first_100/5543216218671794127/",
#         "proj_data/first_100/5543459743317477351/",
#         "proj_data/first_100/5543830828491851790/",
#         "proj_data/first_100/5544538209605506454/",
#         "proj_data/first_100/5544553671487770303/",
#         "proj_data/first_100/5544628403918724157/",
#         "proj_data/first_100/5544730194643635920/",
#         "proj_data/first_100/5544944084014976732/",
#     ],
#     filename_in_path=[-2, -1],
#     export_type=FileTypes.MAT,
#     normalisation_method=FaceMapper.NormalisationMethods.MOUTH_LOCATION_KEY,
#     padding=100,
#     confidence_threshold=0.0,
#     word_subset=[],
#     frame_pad=200,
#     region=Region.MOUTH,
#     localiser_mode=LocaliserMode.BOTH_KEY,
# )

# Data split
# word_set = ["ABOUT", "BELIEVE", "CHANCE", "FAMILY", "THROUGH"]
# file_paths = [f"proj_data/batches_both/{word}.mat" for word in word_set]

# DS = DataSplitter(
#     output_path="proj_data/batches_both/split_large.mat",
#     file_type=FileTypes.MAT,
#     file_paths=file_paths,
# )
