import cv2
import typing
import numpy as np
from dataclasses import dataclass
import typing
from copy import deepcopy
import tensorflow as tf
from PIL import Image

from VideoStream import VideoStream
from FaceMapper import FaceMapper
from utility.Utilities import export_json
from utility.Dataclasses import Region


@dataclass(frozen=True)
class LocaliserMode:
    REGION_KEY: str = "REGION"
    LANDMARKS_KEY: str = "LANDMARKS"
    BOTH_KEY: str = "BOTH"
    NONE_KEY: str = "NONE"


class LipLocaliser(VideoStream):
    """
    Class for loading video streams and drawing a box around the lips within the frames
    """

    LIP_LANDMARK_COLOUR: typing.Tuple[int] = (255, 0, 0)
    MOUTH_BBOX_COLOUR: typing.Tuple[int] = (0, 0, 255)
    LIP_BBOX_COLOUR: typing.Tuple[int] = (0, 255, 0)

    def __init__(
        self,
        stream_property: int | str = 0,
        operations: typing.Dict[str, typing.Any] = {},
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        super().__init__(stream_property=stream_property, operations=operations)
        self.confidence_threshold: float = confidence_threshold
        self.max_num_faces: int = max_num_faces
        self.face_mapper: FaceMapper = FaceMapper(
            threshold=confidence_threshold, max_num_faces=max_num_faces
        )
        self.normalisation_method: FaceMapper.NormalisationMethods = (
            normalisation_method
        )
        self.region: Region = region
        self.localiser_mode: LocaliserMode = localiser_mode

    def set_stream_property(self, stream_property: int | str):
        """
        Will set the stream property. If it is different to the previous then we also close and reopen the stream
        """
        if stream_property != self.stream_property:
            self.stream_property = stream_property
            self.stream.release()
            self.enable_video_stream(stream_property=self.stream_property)

    def set_max_num_faces(self, max_num_faces: int):
        """
        Will change the internal max_num_faces property and change it for this class's facemapper
        """
        self.max_num_faces = max_num_faces
        self.face_mapper.max_num_faces = max_num_faces

    def frame_op(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame operation for this class is to use the face map to localise on the pair of lips in frame
        """
        self.face_mapper.make_face_map(
            img=frame, normalisation_method=self.normalisation_method
        )

        self.lip_crops = self.face_mapper.crop_to_face_region(region_type=Region.LIPS)
        self.mouth_crops = self.face_mapper.crop_to_face_region(
            region_type=Region.MOUTH
        )

        bbox_list1 = []
        bbox_list2 = []
        colour1 = None
        colour2 = None

        if self.region == Region.LIPS:
            bbox_list1 = self.face_mapper.lip_bboxs
            colour1 = self.LIP_BBOX_COLOUR
        elif self.region == Region.MOUTH:
            bbox_list1 = self.face_mapper.mouth_bboxs
            colour1 = self.MOUTH_BBOX_COLOUR
        elif self.region == Region.LIPS_AND_MOUTH:
            bbox_list1 = self.face_mapper.lip_bboxs
            bbox_list2 = self.face_mapper.mouth_bboxs
            colour1 = self.LIP_BBOX_COLOUR
            colour2 = self.MOUTH_BBOX_COLOUR

        new_frame = deepcopy(self.face_mapper.display_img)
        if (
            self.localiser_mode == LocaliserMode.REGION_KEY
            or self.localiser_mode == LocaliserMode.BOTH_KEY
        ):
            for bbox in bbox_list1:
                new_frame = cv2.rectangle(
                    img=new_frame,
                    pt1=bbox.get_min(),
                    pt2=bbox.get_max(),
                    color=colour1,
                    thickness=2,
                )
            for bbox in bbox_list2:
                new_frame = cv2.rectangle(
                    img=new_frame,
                    pt1=bbox.get_min(),
                    pt2=bbox.get_max(),
                    color=colour2,
                    thickness=2,
                )
        if (
            self.localiser_mode == LocaliserMode.LANDMARKS_KEY
            or self.localiser_mode == LocaliserMode.BOTH_KEY
        ):
            # new_frame=self.face_mapper.draw_landmarks_blank_background(
            #     landmarks=self.face_mapper.normalised_lip_landmarks
            # )
            width, height = Image.fromarray(new_frame).size
            for ldmk_list in self.face_mapper.lip_landmarks_condensed:
                for ldmk in ldmk_list:
                    new_frame = cv2.circle(
                        img=new_frame,
                        center=(int(ldmk[0] * width), int(ldmk[1] * height)),
                        radius=1,
                        thickness=1,
                        color=self.LIP_LANDMARK_COLOUR,
                    )

                # ##Below will draw indexes of the landmarks instead
                # for i in range(len(ldmk_list)):
                #     ldmk = ldmk_list[i]
                #     new_frame = cv2.putText(new_frame,str(i) , (int(ldmk[0] * width), int(ldmk[1] * height)), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (self.LIP_LANDMARK_COLOUR), 1, cv2.LINE_AA)
        return super().frame_op(frame=new_frame)


class LipLocaliserExportKeypoints(LipLocaliser):
    """
    Class for loading video streams, localising to the lips and exporting the lip keypoints to an output dictionary
    """

    @dataclass(frozen=True)
    class DictKeywords:
        KEYPOINTS_KEY: str = "KEYPOINTS"
        KEYPOINT_LIST_KEY: str = "KEYPOINT_LIST"
        FRAME_ID_KEY: str = "FRAME"
        FRAME_LIST_KEY: str = "FRAME_LIST"
        CROP_KEY: str = "CROP"

    def __init__(
        self,
        output_path: str = "",
        frame_limit: int = -1,
        stream_property: typing.Union[int, str] = 0,
        operations: typing.Dict[str, typing.Any] = {},
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        super().__init__(
            stream_property=stream_property,
            operations=operations,
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
            localiser_mode=localiser_mode,
        )
        self.output_path: str = output_path
        self.keypoint_dict: typing.Dict = self.__init_keypoint_dict()
        self.frame_dict: typing.Dict = self.__init_frame_dict()
        self.frame_limit: int = frame_limit
        self.frame_num: int = 0

    def __init_keypoint_dict(self) -> typing.Dict:
        return {self.DictKeywords.KEYPOINT_LIST_KEY: []}

    def __init_frame_dict(self) -> typing.Dict:
        return {self.DictKeywords.FRAME_LIST_KEY: []}

    def use_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Will write the next frame to the video
        """
        if (
            self.face_mapper.normalised_lip_landmarks != []
            and self.face_mapper.normalised_lip_landmarks[0].tolist() != []
        ):
            self.keypoint_dict[self.DictKeywords.KEYPOINT_LIST_KEY].append(
                {
                    self.DictKeywords.FRAME_ID_KEY: self.frame_num,
                    self.DictKeywords.KEYPOINTS_KEY: self.face_mapper.normalised_lip_landmarks[
                        0
                    ].tolist()[
                        0
                    ],
                }
            )

            skip_frame = False
            if self.region == Region.LIPS:
                crop = self.lip_crops[0]
            elif self.region == Region.MOUTH:
                crop = self.mouth_crops[0]
            else:
                crop = None
                skip_frame = True
            if crop is None or crop.tolist() == []:
                skip_frame = True

            if not skip_frame:
                self.frame_dict[self.DictKeywords.FRAME_LIST_KEY].append(
                    {
                        self.DictKeywords.FRAME_ID_KEY: self.frame_num,
                        self.DictKeywords.CROP_KEY: crop,
                    }
                )

        self.frame_num += 1
        if self.frame_limit != -1 and self.frame_num > self.frame_limit:
            return True
        return False

    def __del__(self) -> None:
        """
        Destructor for the video stream object. Will close the video stream and destroy the window
        """
        super().__del__()
        export_json(dict=self.keypoint_dict, path=self.output_path)


class LipLocaliserExportFrames(LipLocaliser):
    """
    Class for loading video streams, localising to the lips and exporting the output to an mp4 output file
    """

    def __init__(
        self,
        output_path: str,
        frame_limit: int = -1,
        stream_property: typing.Union[int, str] = 0,
        operations: typing.Dict[str, typing.Any] = {},
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        super().__init__(
            stream_property=stream_property,
            operations=operations,
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
            localiser_mode=localiser_mode,
        )
        self.output_path: str = output_path
        self.output_file = self.open_export_file(output_path=self.output_path)
        self.frame_limit = frame_limit
        self.frame_num = 0

    def open_export_file(self, output_path: str):
        """
        Will open the correct output file and return it. Will raise exception if unable to open file
        """
        output_file = None
        output_file = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=self.stream.get(cv2.CAP_PROP_FPS),
            frameSize=(int(self.stream.get(3)), int(self.stream.get(4))),
            isColor=True,
        )
        if not output_file.isOpened():
            raise Exception(f"Unable to open videowriter at output path {output_path}")
        return output_file

    def use_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Will write the next frame to the video
        """
        self.output_file.write(frame)

        self.frame_num += 1
        if self.frame_limit != -1 and self.frame_num > self.frame_limit:
            return True
        return False

    def __del__(self) -> None:
        """
        Destructor for the video stream object. Will close the video stream and destroy the window
        """
        super().__del__()
        if self.output_file != None and self.output_file.isOpened():
            self.output_file.release()


class LipLocaliserShowFeed(LipLocaliser):
    """
    Class for loading video streams, localising to the lips and displaying them on the screen
    """

    def __init__(
        self,
        stream_property: typing.Union[int, str] = 0,
        window_name: str = "frame",
        stop_key: str = "q",
        operations: typing.Dict[str, typing.Any] = {},
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        region: Region = Region.LIPS,
        localiser_mode: LocaliserMode = LocaliserMode.REGION_KEY,
    ):
        super().__init__(
            stream_property=stream_property,
            operations=operations,
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
            localiser_mode=localiser_mode,
        )
        self.window_name: str = window_name
        self.stop_key: str = stop_key

    def use_frame(self, frame: np.ndarray) -> bool:
        """
        Will display the current frame to the screen, only returning false when the user wants the session to end
        """
        cv2.imshow(winname=self.window_name, mat=frame)
        if cv2.waitKey(1) & 0xFF == ord(self.stop_key):
            return True
        return False

    def __del__(self) -> None:
        """
        Destructor for the video stream object. Will close the video stream and destroy the window
        """
        super().__del__()
        if cv2.getWindowProperty(
            winname=self.window_name, prop_id=cv2.WND_PROP_VISIBLE
        ):
            cv2.destroyWindow(winname=self.window_name)
