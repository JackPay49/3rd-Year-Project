import typing
import numpy as np
import os
from dataclasses import dataclass
import json
import csv
import cv2
from copy import deepcopy
from abc import ABC, abstractmethod
from scipy.io import savemat, loadmat

from LipLocaliser import LipLocaliserExportKeypoints, LocaliserMode
from utility.Utilities import export_json, get_files, get_file_type, pad_image
from utility.Dataclasses import FileTypes, Region
from FaceMapper import FaceMapper
from PhonemesVisemes import word_to_phonemes, phonemes_to_visemes, BAD_WORD_KEY
from utility.FaceMapperKeyLists import (
    lip_landmarks_outer_keys,
    lip_landmarks_inner_keys,
)

np.set_printoptions(threshold=np.inf)

size_lip_landmarks = len(lip_landmarks_inner_keys) + len(lip_landmarks_outer_keys)
padded_lip_landmarks = np.zeros([size_lip_landmarks, 3]).tolist()


class ConvertDataset(LipLocaliserExportKeypoints, ABC):
    """
    Class for converting data from LRS2, pairing videos with annotation files. Output function can do multiple different things based on the parameter of the export data type
    """

    DEFAULT_OVERRIDE_OUTPUT_PATH: str = ""
    DEFAULT_MISSED_WORDS_FILENAME: str = "missed_words.txt"
    DEFAULT_FRAME_PAD: int = 29

    @dataclass(frozen=True)
    class LRS2Delimiters:
        TEXT_KEY: str = "Text:  "
        TIMING_KEY: str = "WORD START END ASDSCORE"

    @dataclass(frozen=True)
    class FileAnnotationKeys:
        FULL_TEXT_KEY: str = "FULL_TEXT"
        TEXT_LIST_KEY: str = "TEXT_LIST"
        WORD_KEY: str = "WORD"
        START_KEY: str = "START"
        END_KEY: str = "END"
        ASDSCORE_KEY: str = "ASDSCORE"
        VISEMES: str = "VISEMES"
        PHONEMES: str = "PHONEMES"

    def __init__(
        self,
        output_dir: str,
        input_paths: typing.List[str] = [],
        filename_in_path: typing.List[int] = [-1],
        export_type: FileTypes = FileTypes.NPY,
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        padding: int = 0,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        override_output_path: str = DEFAULT_OVERRIDE_OUTPUT_PATH,
        missed_words_filename: str = DEFAULT_MISSED_WORDS_FILENAME,
        allowed_file_types: typing.List[FileTypes] = [FileTypes.VIDEO],
        frame_pad: int = DEFAULT_FRAME_PAD,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        self.missed_words_filename: str = missed_words_filename
        self.missed_words: typing.List[str] = []
        self.frame_pad: int = frame_pad
        self.localiser_mode: LocaliserMode = localiser_mode

        self.allowed_file_types = allowed_file_types
        self.input_paths = list(
            set(
                self.get_all_files(
                    input_paths=input_paths, allowed_file_types=allowed_file_types
                )
            )
        )
        self.output_dir = output_dir
        self.filename_in_path = filename_in_path
        self.export_type = export_type
        self.padding = padding
        self.override_output_path = override_output_path

        self.current_file = ""

        os.makedirs(name=self.output_dir, exist_ok=True)

        for x in self.input_paths:
            self.process_input_path(
                input_path=x,
                normalisation_method=normalisation_method,
                confidence_threshold=confidence_threshold,
                max_num_faces=max_num_faces,
                region=region,
            )

        self.export_missed_words(missed_words=self.missed_words)

    def process_input_path(
        self,
        input_path: str,
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
    ):
        """
        Will carry out the necessary operations given an input path. Loads the video in, gets keypoints & video dicttionaries and outputs the result
        """
        self.current_file = input_path
        if FileTypes.VIDEO not in input_path:
            input_path = input_path + FileTypes.VIDEO
        super().__init__(
            stream_property=input_path,
            output_path="",
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
            localiser_mode=self.localiser_mode,
        )

        # Stream feed will kick off. This will move through each frame getting the keypoints and adding them to the export json
        self.stream_feed()

        # Will combine the annotation file and keypoints list and export this
        keypoint_dict = self.get_video_keypoints()
        video_dict = self.get_video()
        input_path = input_path.replace("\\", "/")
        self.output(
            input_path=input_path, keypoint_dict=keypoint_dict, video_dict=video_dict
        )

    @abstractmethod
    def output(
        self,
        input_path: str,
        keypoint_dict: typing.Dict[str, typing.Any],
        video_dict: typing.Dict[str, typing.Any],
    ) -> None:
        return

    @abstractmethod
    def get_video(self) -> typing.Dict[str, typing.Any]:
        return None

    @abstractmethod
    def get_video_keypoints(self) -> typing.Dict[str, typing.Any]:
        return None

    def get_output_name(
        self, input_path: str, file_type: FileTypes = FileTypes.ANNOTATION
    ):
        """
        Will return the output name for the specified input file
        """
        path_split = input_path.split(".")[0].split("/")
        filename = ""
        for x in self.filename_in_path:
            filename += path_split[x] + "_"
        return filename[:-1] + file_type

    def get_all_files(
        self, input_paths: typing.List[str], allowed_file_types: typing.List[str]
    ):
        """
        Will get all of the input paths to files, looking in directories specified. Will check these come in mp4 and txt pairs
        """
        return get_files(
            paths=input_paths,
            allowed_file_types=allowed_file_types,
        )

    def check_empty_entries(self, output_json: typing.Dict) -> typing.Dict:
        """
        Checks for word dicts with no keypoints and removes them
        """
        remove_list = []
        for word_dict in output_json[self.FileAnnotationKeys.TEXT_LIST_KEY]:
            if (
                LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                not in word_dict
            ) and (
                LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY not in word_dict
            ):
                remove_list.append(word_dict)
        for x in remove_list:
            output_json[self.FileAnnotationKeys.TEXT_LIST_KEY].remove(x)
        return output_json

    def get_word(self, annot_file: typing.Dict, time: float) -> int:
        """
        Will get the index of the word for the given time unit
        """
        for i in range(len(annot_file[self.FileAnnotationKeys.TEXT_LIST_KEY])):
            entry = annot_file[self.FileAnnotationKeys.TEXT_LIST_KEY][i]
            start = entry[self.FileAnnotationKeys.START_KEY]
            end = entry[self.FileAnnotationKeys.END_KEY]
            if time >= start and time <= end:
                return i
        return -1

    def get_list_from_frame(self, list: typing.Dict, frame: int) -> typing.Dict:
        """
        Will get keypoint sub-dict associated with the current frame. This checks if the frame number is correct
        """
        if frame < len(list):
            true_frame_num = list[frame][
                LipLocaliserExportKeypoints.DictKeywords.FRAME_ID_KEY
            ]
            if frame == true_frame_num:
                return list[frame]
        for entry in list:
            if frame == entry[LipLocaliserExportKeypoints.DictKeywords.FRAME_ID_KEY]:
                return entry
        return {}

    def __del__(self) -> None:
        return

    def export_missed_words(self, missed_words: typing.List[str]) -> None:
        """
        Will export the list of missed words, appending them to the list if it exists already
        """
        if missed_words:
            output = os.path.join(self.output_dir, self.missed_words_filename)
            if os.path.exists(output):
                with open(file=output) as f:
                    old_missed = json.load(f)
                old_missed += missed_words
                export_json(dict=list(set(old_missed)), path=output)
            else:
                export_json(dict=missed_words, path=output)

    def add_visemes(
        self, export_dict: typing.Dict[str, typing.Any]
    ) -> typing.Dict[typing.List[str], typing.Any]:
        """
        Takes the export dict and will convert the word labels to instead be lists of Visemes
        """
        for word_dict in export_dict[self.FileAnnotationKeys.TEXT_LIST_KEY]:
            old_word = word_dict[self.FileAnnotationKeys.WORD_KEY].lower()
            phonemes = word_to_phonemes(word=old_word)
            if phonemes == [BAD_WORD_KEY]:
                self.missed_words.append(old_word)
                continue
            visemes = phonemes_to_visemes(phonemes=phonemes)
            word_dict[self.FileAnnotationKeys.VISEMES] = visemes
            word_dict[self.FileAnnotationKeys.PHONEMES] = phonemes
        return export_dict

    def add_padding(
        self,
        export_dict: typing.Dict[str, typing.Any],
        padding: int,
        pad_type: LocaliserMode,
        pad_dict: bool = True,
    ) -> typing.Dict[str, typing.Any]:
        """
        Will pad the video so that the number of frames is equal to that specified
        """
        if pad_type == LocaliserMode.REGION_KEY:
            list_key = LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
            padded_frame = np.zeros(shape=(self.padding, self.padding), dtype=np.uint8)
            if pad_dict:
                padding_value = {
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_ID_KEY: -1,
                    LipLocaliserExportKeypoints.DictKeywords.CROP_KEY: padded_frame,
                }
            else:
                padding_value = padded_frame
        elif pad_type == LocaliserMode.LANDMARKS_KEY:
            list_key = LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
            if pad_dict:
                padding_value = {
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_ID_KEY: -1,
                    LipLocaliserExportKeypoints.DictKeywords.KEYPOINTS_KEY: padded_lip_landmarks,
                }
            else:
                padding_value = padded_lip_landmarks
        frame_list_len = len(export_dict[list_key])

        if frame_list_len == 0:
            print("Frame list was empty so returning None")
            return None
        add_pad = padding - frame_list_len
        if add_pad == 0:
            return export_dict
        elif add_pad < 0:
            print(
                f"Number of frames is already greater than the padding amount. It is {frame_list_len} frames long. Cropping to be {padding} frames long."
            )
            export_dict[list_key] = export_dict[list_key][:padding]
            return export_dict
        for _ in range(add_pad):
            export_dict[list_key].append(padding_value)
        return export_dict

    def output_mat(
        self,
        filename: str,
        export_dict: typing.Dict[str, typing.Any],
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ) -> None:
        """
        Will export each cropped video frame to mat file. Stores a dictionary list for each word found
        """
        print(filename)
        if export_dict is None:
            print(f"Unknown error with file {filename}")
            return

        data = {}
        if localiser_mode == LocaliserMode.REGION_KEY:
            data["data"] = [
                np.array(
                    export_dict[LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY]
                )
            ]
            word = export_dict[self.FileAnnotationKeys.WORD_KEY]
        elif localiser_mode == LocaliserMode.LANDMARKS_KEY:
            data["data"] = [
                np.array(
                    export_dict[
                        LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                    ]
                )
            ]
            word = export_dict[self.FileAnnotationKeys.WORD_KEY]
        elif localiser_mode == LocaliserMode.BOTH_KEY:
            if (
                export_dict["frame_dict"] is None
                or export_dict["keypoint_dict"] is None
            ):
                print(f"Unknown error with file {filename}")
                return
            data["data_frames"] = [
                np.array(
                    export_dict["frame_dict"][
                        LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                    ]
                )
            ]
            data["data_keypoints"] = [
                np.array(
                    export_dict["keypoint_dict"][
                        LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                    ]
                )
            ]
            word = export_dict["frame_dict"][self.FileAnnotationKeys.WORD_KEY]

        if self.override_output_path != self.DEFAULT_OVERRIDE_OUTPUT_PATH:
            filename = os.path.join(self.output_dir, self.override_output_path)
        data["labels"] = [word]

        if os.path.isfile(filename):  # Appends if it exists already
            data_mat = loadmat(file_name=filename)
            for entry in data:
                data_mat[entry] = np.append(data_mat[entry], data[entry], axis=0)
            savemat(file_name=filename, mdict=data_mat)
        else:  # Makes new file if it doesn't exist already
            savemat(file_name=filename, mdict=data)


class ConvertDatasetLRS2(ConvertDataset):
    def __init__(
        self,
        output_dir: str,
        input_paths: typing.List[str] = [],
        filename_in_path: typing.List[int] = [-1],
        export_type: FileTypes = FileTypes.NPY,
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        padding: int = 0,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        override_output_path: str = ConvertDataset.DEFAULT_OVERRIDE_OUTPUT_PATH,
        missed_words_filename: str = ConvertDataset.DEFAULT_MISSED_WORDS_FILENAME,
        word_subset: typing.List[str] = [],
        frame_pad: int = ConvertDataset.DEFAULT_FRAME_PAD,
        asdscore_threshold: float = 0.0,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        self.word_subset: typing.List[str] = [x.upper() for x in word_subset]
        self.asdscore_threshold: float = asdscore_threshold

        super().__init__(
            output_dir,
            input_paths,
            filename_in_path,
            export_type,
            normalisation_method,
            padding,
            confidence_threshold,
            max_num_faces,
            region,
            override_output_path,
            missed_words_filename,
            allowed_file_types=[FileTypes.VIDEO, FileTypes.ANNOTATION],
            frame_pad=frame_pad,
            localiser_mode=localiser_mode,
        )

    def process_input_path(
        self,
        input_path: str,
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
    ):
        """
        Will carry out the necessary operations given an input path. Loads the video in, gets keypoints & video dicttionaries and outputs the result
        """
        self.annotation_file = self.get_annotation_file(
            file_path=(input_path + FileTypes.ANNOTATION),
            word_subset=self.word_subset,
        )
        super().process_input_path(
            input_path=input_path,
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
        )

    def output(
        self,
        input_path: str,
        keypoint_dict: typing.Dict[str, typing.Any],
        video_dict: typing.Dict[str, typing.Any],
    ) -> None:
        """
        Will export:
        If annotation file will export the json paired annotations and keypoints
        If npy then will export the keypoints
        If video a separate video in folders labelled with the words
        If csv a list of the landmarks for each frame with viseme and word info
        """
        keypoint_dict = self.add_visemes(export_dict=keypoint_dict)
        filename = os.path.join(self.output_dir, f"{input_path.split('/')[-2]}.mat")

        words = ""
        frames = []
        keypoints = []
        for i in range(len(video_dict[self.FileAnnotationKeys.TEXT_LIST_KEY])):
            word = video_dict[self.FileAnnotationKeys.TEXT_LIST_KEY][i][
                self.FileAnnotationKeys.WORD_KEY
            ]

            current_frames = [
                cv2.cvtColor(
                    src=pad_image(
                        img=frame[LipLocaliserExportKeypoints.DictKeywords.CROP_KEY],
                        padding=self.padding,
                    ),
                    code=cv2.COLOR_RGB2GRAY,
                )
                for frame in video_dict[self.FileAnnotationKeys.TEXT_LIST_KEY][i][
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                ]
            ]

            current_keypoints = [
                frame[LipLocaliserExportKeypoints.DictKeywords.KEYPOINTS_KEY]
                for frame in keypoint_dict[self.FileAnnotationKeys.TEXT_LIST_KEY][i][
                    LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                ]
            ]

            words += " " + word
            frames.extend(current_frames)
            keypoints.extend(current_keypoints)

        video_dict = self.add_padding(
            export_dict={
                LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY: frames,
                self.FileAnnotationKeys.WORD_KEY: words,
            },
            padding=self.frame_pad,
            pad_type=LocaliserMode.REGION_KEY,
            pad_dict=False,
        )

        keypoint_dict = self.add_padding(
            export_dict={
                LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY: keypoints,
                self.FileAnnotationKeys.WORD_KEY: words,
            },
            padding=self.frame_pad,
            pad_type=LocaliserMode.LANDMARKS_KEY,
            pad_dict=False,
        )

        if self.export_type == FileTypes.ANNOTATION:
            self.output_annotations(filename=filename, export_dict=keypoint_dict)
        elif self.export_type == FileTypes.NPY:
            self.output_numpy(filename=filename, export_dict=keypoint_dict)
        elif self.export_type == FileTypes.CSV:
            self.output_csv(
                filename=self.override_output_path, export_dict=keypoint_dict
            )
        elif self.export_type == FileTypes.VIDEO:
            self.output_videos(filename=filename, export_dict=video_dict)
        elif self.export_type == FileTypes.MAT:
            if self.localiser_mode == LocaliserMode.REGION_KEY:
                self.output_mat(
                    filename=filename,
                    export_dict=video_dict,
                    localiser_mode=self.localiser_mode,
                )
            elif self.localiser_mode == LocaliserMode.LANDMARKS_KEY:
                self.output_mat(
                    filename=filename,
                    export_dict=keypoint_dict,
                    localiser_mode=self.localiser_mode,
                )
            elif self.localiser_mode == LocaliserMode.BOTH_KEY:
                self.output_mat(
                    filename=filename,
                    export_dict={
                        "keypoint_dict": keypoint_dict,
                        "frame_dict": video_dict,
                    },
                    localiser_mode=self.localiser_mode,
                )

        else:
            print(
                f"Input export type does not make sense. Instead exporting as annotation file."
            )
            self.output_annotations(filename=filename, export_dict=keypoint_dict)

    def output_videos(
        self, filename: str, export_dict: typing.Dict[str, typing.Any]
    ) -> None:
        """
        For each word will create a new small video using the cropped frames. Will write these to a directory of the word being spoken
        """
        for word_dict in export_dict[self.FileAnnotationKeys.TEXT_LIST_KEY]:
            word = word_dict[self.FileAnnotationKeys.WORD_KEY]
            out_folder = os.path.join(self.output_dir, word)
            os.makedirs(name=out_folder, exist_ok=True)

            next_filename = os.path.join(out_folder, filename)

            frame_size = (self.padding, self.padding)
            out = cv2.VideoWriter(
                next_filename, cv2.VideoWriter_fourcc(*"MP4V"), 1, frame_size
            )
            for frame_dict in word_dict[
                LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
            ]:
                padded_img = pad_image(
                    img=frame_dict[LipLocaliserExportKeypoints.DictKeywords.CROP_KEY],
                    padding=self.padding,
                )
                out.write(padded_img)
            out.release()

    def output_csv(
        self, filename: str, export_dict: typing.Dict[str, typing.Any]
    ) -> None:
        """
        To export as csv the landmarks for each frame are all written to one large csv file. Columns of this are the word, phonemes, visemes and then one for each frame, storing the lip landmarks for each
        """
        export_dict = self.add_padding(
            export_dict=export_dict,
            padding=self.padding,
            pad_type=LocaliserMode.LANDMARKS_KEY,
        )
        mode = "w"
        filename = os.path.join(self.output_dir, filename)
        if os.path.exists(path=filename):
            mode = "a"
        with open(file=filename, mode=mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(
                    [self.FileAnnotationKeys.WORD_KEY]
                    + [self.FileAnnotationKeys.VISEMES]
                    + [self.FileAnnotationKeys.PHONEMES]
                    + [f"Frame {i}" for i in range(1, (self.padding + 1))]
                )
            for word_dict in export_dict[self.FileAnnotationKeys.TEXT_LIST_KEY]:
                if self.FileAnnotationKeys.VISEMES not in word_dict:
                    visemes = []
                else:
                    visemes = word_dict[self.FileAnnotationKeys.VISEMES]

                if self.FileAnnotationKeys.PHONEMES not in word_dict:
                    phonemes = []
                else:
                    phonemes = word_dict[self.FileAnnotationKeys.VISEMES]

                keypoints = [
                    x[LipLocaliserExportKeypoints.DictKeywords.KEYPOINTS_KEY]
                    for x in word_dict[
                        LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                    ]
                ]

                writer.writerow(
                    [word_dict[self.FileAnnotationKeys.WORD_KEY]]
                    + [visemes]
                    + [phonemes]
                    + keypoints
                )

    def output_numpy(
        self, filename: str, export_dict: typing.Dict[str, typing.Any]
    ) -> None:
        """
        To output as a numpy file each word is written to a different directory. Within each a npy file storing a list of the landmarks is stored
        """
        for word_dict in export_dict[self.FileAnnotationKeys.TEXT_LIST_KEY]:
            word = word_dict[self.FileAnnotationKeys.WORD_KEY]

            keypoints = []
            for frame in word_dict[
                LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
            ]:
                keypoints.append(
                    frame[LipLocaliserExportKeypoints.DictKeywords.KEYPOINTS_KEY]
                )

            out_dir = os.path.join(self.output_dir, word)
            os.makedirs(name=out_dir, exist_ok=True)
            np.save(
                file=os.path.join(out_dir, filename),
                arr=keypoints,
                allow_pickle=True,
            )

    def output_annotations(
        self, filename: str, export_dict: typing.Dict[str, typing.Any]
    ) -> None:
        """
        To export annotations just the export dict dictionary is exported as a json to a txt file
        """
        export_json(
            dict=export_dict,
            path=os.path.join(self.output_dir, filename),
        )

    def get_video(self) -> typing.Dict[str, typing.Any]:
        """
        Will try to associate frame crops with the word annotations and return a dictionary of these
        """
        output_json = deepcopy(self.annotation_file)
        sec_per_frame = 1 / self.fps
        current_time = 0
        for i in range(self.num_frames):
            current_word: int = self.get_word(
                annot_file=self.annotation_file, time=current_time
            )

            if current_word != -1:
                frame_dict: typing.Dict[str, typing.Any] = self.get_list_from_frame(
                    list=self.frame_dict[
                        LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                    ],
                    frame=i,
                )

                if frame_dict == {}:
                    print(
                        f"Could not find face for video {self.current_file}, frame {i}"
                    )
                    continue
                if (
                    not LipLocaliserExportKeypoints.DictKeywords.FRAME_ID_KEY
                    in frame_dict
                ):
                    continue

                if (
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                    not in output_json[self.FileAnnotationKeys.TEXT_LIST_KEY][
                        current_word
                    ]
                ):
                    output_json[self.FileAnnotationKeys.TEXT_LIST_KEY][current_word][
                        LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                    ] = []

                output_json[self.FileAnnotationKeys.TEXT_LIST_KEY][current_word][
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                ].append(frame_dict)
            current_time += sec_per_frame
        return self.check_empty_entries(output_json=output_json)

    def get_video_keypoints(self) -> typing.Dict[str, typing.Any]:
        """
        Will try to associate frames with individual words using their timing. Will create a dictionary to output that stores each word with the lip keypoints
        """
        output_json = deepcopy(self.annotation_file)
        sec_per_frame = 1 / self.fps
        current_time = 0
        for i in range(self.num_frames):
            current_word: int = self.get_word(
                annot_file=self.annotation_file, time=current_time
            )

            if current_word != -1:
                keypoint_dict: typing.Dict[str, typing.Any] = self.get_list_from_frame(
                    list=self.keypoint_dict[
                        LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                    ],
                    frame=i,
                )

                if keypoint_dict == {}:
                    print(
                        f"Could not find keypoints for video {self.current_file}, frame {i}"
                    )
                    continue
                if (
                    not LipLocaliserExportKeypoints.DictKeywords.KEYPOINTS_KEY
                    in keypoint_dict
                ):
                    continue

                if (
                    LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                    not in output_json[self.FileAnnotationKeys.TEXT_LIST_KEY][
                        current_word
                    ]
                ):
                    output_json[self.FileAnnotationKeys.TEXT_LIST_KEY][current_word][
                        LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                    ] = []

                output_json[self.FileAnnotationKeys.TEXT_LIST_KEY][current_word][
                    LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                ].append(keypoint_dict)
            current_time += sec_per_frame
        return self.check_empty_entries(output_json=output_json)

    def get_all_files(
        self, input_paths: typing.List[str], allowed_file_types: typing.List[str]
    ):
        """
        Will get all of the input paths to files, looking in directories specified. Will check these come in mp4 and txt pairs
        """
        paths = super().get_all_files(
            input_paths=input_paths, allowed_file_types=allowed_file_types
        )
        ann_paths, vid_paths = [], []
        for path in paths:
            if get_file_type(path=path) == FileTypes.ANNOTATION:
                ann_paths.append(path.split(".")[0])
            elif get_file_type(path=path) == FileTypes.VIDEO:
                vid_paths.append(path.split(".")[0])
        intersection = list(set(ann_paths) & set(vid_paths))
        missed_files = list(set(ann_paths) - set(intersection)) + list(
            set(vid_paths) - set(intersection)
        )
        if missed_files:
            print(
                f"Missed files, due to not having both {FileTypes.ANNOTATION} and {FileTypes.VIDEO} files: {missed_files}"
            )
        return intersection

    def get_annotation_file(
        self, file_path: str, word_subset: typing.List[str]
    ) -> typing.Dict:
        """
        Will get the input annotation file and load it into a dictionary, returning this
        """
        annot_file = {}
        with open(file=file_path, mode="r") as f:
            # Get first text line
            text_line = f.readline().replace(self.LRS2Delimiters.TEXT_KEY, "")
            annot_file[self.FileAnnotationKeys.FULL_TEXT_KEY] = text_line

            # Skip blank and not used lines
            next = ""
            while next != self.LRS2Delimiters.TIMING_KEY:
                next = f.readline().strip()

            # Get timings
            next = f.readline().strip()
            annot_file[self.FileAnnotationKeys.TEXT_LIST_KEY] = []
            while next != "":
                word, start, end, asdscore = next.split(" ")
                word = word.strip()
                if (word_subset == [] or word in word_subset) and (
                    float(asdscore) > self.asdscore_threshold
                ):
                    word_dict = {
                        self.FileAnnotationKeys.WORD_KEY: word,
                        self.FileAnnotationKeys.START_KEY: float(start),
                        self.FileAnnotationKeys.END_KEY: float(end),
                        self.FileAnnotationKeys.ASDSCORE_KEY: float(asdscore),
                    }
                    annot_file[self.FileAnnotationKeys.TEXT_LIST_KEY].append(word_dict)
                next = f.readline().strip()
        return annot_file


class ConvertDatasetLRW(ConvertDataset):
    def __init__(
        self,
        output_dir: str,
        input_paths: typing.List[str] = [],
        filename_in_path: typing.List[int] = [-1],
        export_type: FileTypes = FileTypes.NPY,
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        padding: int = 0,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        override_output_path: str = ConvertDataset.DEFAULT_OVERRIDE_OUTPUT_PATH,
        missed_words_filename: str = ConvertDataset.DEFAULT_MISSED_WORDS_FILENAME,
        frame_pad: int = ConvertDataset.DEFAULT_FRAME_PAD,
        word_in_position: int = -1,
        max_num_videos: int = -1,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        self.word_in_position: int = word_in_position
        self.max_num_videos: int = max_num_videos

        super().__init__(
            output_dir=output_dir,
            input_paths=input_paths,
            filename_in_path=filename_in_path,
            export_type=export_type,
            normalisation_method=normalisation_method,
            padding=padding,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
            override_output_path=override_output_path,
            missed_words_filename=missed_words_filename,
            allowed_file_types=[FileTypes.VIDEO],
            frame_pad=frame_pad,
            localiser_mode=localiser_mode,
        )

    def output(
        self,
        input_path: str,
        keypoint_dict: typing.Dict[str, typing.Any],
        video_dict: typing.Dict[str, typing.Any],
    ) -> None:
        """
        If video will crop the video to be just the lip region. Will add padding for the lips, as specified
        If csv will crop videos and export the data to a csv file. Basically holds together data from video export in a csv file
        """
        filename = self.get_output_name(
            input_path=input_path, file_type=self.export_type
        )
        ##Prepare video dict: add word detail, pad frames and video
        video_dict = self.add_word_details(
            export_dict=video_dict, input_path=input_path
        )
        # Remove padding temporarily. Don't want to pad
        # video_dict = self.add_padding(
        #     export_dict=video_dict,
        #     padding=self.frame_pad,
        #     pad_type=LocaliserMode.REGION_KEY,
        # )
        if video_dict is not None:
            video_dict[LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY] = [
                cv2.cvtColor(
                    src=pad_image(
                        img=x[LipLocaliserExportKeypoints.DictKeywords.CROP_KEY],
                        padding=self.padding,
                    ),
                    code=cv2.COLOR_RGB2GRAY,
                )
                for x in video_dict[
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                ]
            ]
        # Return & don't export if not enough data
        if (
            len(video_dict[LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY])
            != self.frame_pad
        ):
            print(
                f"Only found {len(video_dict[LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY])} frames. Not exporting file {filename}"
            )
            return

        # Prepare landmark dict: add word detail & pad frames
        keypoint_dict = self.add_word_details(
            export_dict=keypoint_dict, input_path=input_path
        )
        # Remove padding temporarily. Don't want to pad
        # keypoint_dict = self.add_padding(
        #     export_dict=keypoint_dict,
        #     padding=self.frame_pad,
        #     pad_type=LocaliserMode.LANDMARKS_KEY,
        # )
        if keypoint_dict is not None:
            keypoint_dict[
                LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
            ] = [
                x[LipLocaliserExportKeypoints.DictKeywords.KEYPOINTS_KEY]
                for x in keypoint_dict[
                    LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                ]
            ]
        # Return & don't export if not enough data
        if (
            len(
                keypoint_dict[
                    LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY
                ]
            )
            != self.frame_pad
        ):
            print(
                f"Only found {len(keypoint_dict[LipLocaliserExportKeypoints.DictKeywords.KEYPOINT_LIST_KEY])} frames of landmarks. Not exporting file {filename}"
            )
            return

        if self.export_type == FileTypes.CSV:
            self.output_csv(filename=self.override_output_path, video_dict=video_dict)
        elif self.export_type == FileTypes.VIDEO:
            self.output_videos(filename=filename, export_dict=video_dict)
        elif self.export_type == FileTypes.MAT:
            if self.localiser_mode == LocaliserMode.REGION_KEY:
                self.output_mat(
                    filename=filename,
                    export_dict=video_dict,
                    localiser_mode=self.localiser_mode,
                )
            elif self.localiser_mode == LocaliserMode.LANDMARKS_KEY:
                self.output_mat(
                    filename=filename,
                    export_dict=keypoint_dict,
                    localiser_mode=self.localiser_mode,
                )
            elif self.localiser_mode == LocaliserMode.BOTH_KEY:
                self.output_mat(
                    filename=filename,
                    export_dict={
                        "keypoint_dict": keypoint_dict,
                        "frame_dict": video_dict,
                    },
                    localiser_mode=self.localiser_mode,
                )
        else:
            print(
                f"Input export type does not make sense. Instead exporting as a mat file."
            )
            self.output_mat(
                filename=filename,
                export_dict=video_dict,
                localiser_mode=self.localiser_mode,
            )

    def output_csv(
        self,
        filename: str,
        video_dict: typing.Dict[str, typing.Any],
    ) -> None:
        """
        Will export each cropped video frame to a csv file
        """
        mode = "w"
        filename = os.path.join(self.output_dir, filename)
        if os.path.exists(path=filename):
            mode = "a"
        with open(file=filename, mode=mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(
                    [self.FileAnnotationKeys.WORD_KEY]
                    + [self.FileAnnotationKeys.VISEMES]
                    + [self.FileAnnotationKeys.PHONEMES]
                    + [f"Frame {i}" for i in range(1, (self.frame_pad + 1))]
                )
            if self.FileAnnotationKeys.VISEMES not in video_dict:
                visemes = []
            else:
                visemes = video_dict[self.FileAnnotationKeys.VISEMES]

            if self.FileAnnotationKeys.PHONEMES not in video_dict:
                phonemes = []
            else:
                phonemes = video_dict[self.FileAnnotationKeys.VISEMES]

            frames = [
                str(x[LipLocaliserExportKeypoints.DictKeywords.CROP_KEY].tolist())
                for x in video_dict[
                    LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
                ]
            ]
            writer.writerow(
                [video_dict[self.FileAnnotationKeys.WORD_KEY]]
                + [visemes]
                + [phonemes]
                + frames
            )

    def output_videos(
        self, filename: str, export_dict: typing.Dict[str, typing.Any]
    ) -> None:
        export_dict = self.add_padding(
            export_dict=export_dict,
            padding=self.frame_pad,
            pad_type=LocaliserMode.REGION_KEY,
        )

        out_folder = os.path.join(
            self.output_dir, export_dict[self.FileAnnotationKeys.WORD_KEY]
        )
        os.makedirs(name=out_folder, exist_ok=True)

        next_filename = os.path.join(out_folder, filename)

        frame_size = (self.padding, self.padding)
        out = cv2.VideoWriter(
            next_filename, cv2.VideoWriter_fourcc(*"MP4V"), 1, frame_size
        )

        for frame_dict in export_dict[
            LipLocaliserExportKeypoints.DictKeywords.FRAME_LIST_KEY
        ]:
            padded_img = pad_image(
                img=frame_dict[LipLocaliserExportKeypoints.DictKeywords.CROP_KEY],
                padding=self.padding,
            )
            out.write(padded_img)
        out.release()

    def get_video(self) -> typing.Dict[str, typing.Any]:
        return self.frame_dict

    def get_video_keypoints(self) -> typing.Dict[str, typing.Any]:
        return self.keypoint_dict

    def get_word_utterance(self, input_path: str) -> str:
        """
        Will return the word being said from the filename
        """
        path_split = input_path.split(".")[0].split("/")
        return path_split[self.word_in_position]

    def add_word_details(
        self, export_dict: typing.Dict[str, typing.Any], input_path: str
    ) -> typing.Dict[typing.List[str], typing.Any]:
        """
        Takes the export dict and will add the word, phonemes and visemes
        """
        current_word = self.get_word_utterance(input_path=input_path)
        current_phonemes = word_to_phonemes(word=current_word)
        if current_phonemes == [BAD_WORD_KEY]:
            self.missed_words.append(current_word)
            return
        current_visemes = phonemes_to_visemes(phonemes=current_phonemes)
        export_dict[self.FileAnnotationKeys.WORD_KEY] = current_word
        export_dict[self.FileAnnotationKeys.VISEMES] = current_visemes
        export_dict[self.FileAnnotationKeys.PHONEMES] = current_phonemes

        return export_dict

    def get_all_files(
        self, input_paths: typing.List[str], allowed_file_types: typing.List[FileTypes]
    ):
        """
        Will get all files, limits the number to the maximum allowed
        """
        if self.max_num_videos != -1:
            return super().get_all_files(
                input_paths=input_paths, allowed_file_types=allowed_file_types
            )[: self.max_num_videos]
        else:
            return super().get_all_files(
                input_paths=input_paths, allowed_file_types=allowed_file_types
            )
