### DEPRECIATED
import typing
import numpy as np
import cv2
from collections import deque

from FaceMapper import FaceMapper
from LipLocaliser import LipLocaliserShowFeed
from VideoStream import ImageOperationKeys
from ConvertDataset import padded_lip_landmarks
from utility.Dataclasses import Region, FeatureExtractor, FileTypes, PredictionMethod
from ModelWrapper import ModelWrapper


class LipReaderShowFeed(LipLocaliserShowFeed):
    """
    Class will apply a model to the keypoints generated for the lips, trying to interpret into a word to display on the screen

    Properties:
        model_file_name:
        stop_key:
        normalisation_method:
        confidence_threshold:
        max_num_faces:
        padding: Sequence padding: amount of whole frames to pad, how long the sequence should be
        word_dict:
        region:
        feature_extractor_type: Type of feature extractor to use or whether one should be used at all
        frame_pad: Inner frame padding: amount each frame should be padded
        model_type: File type of the model. Typically within the subset of [.keras, .hdf5]. If hdf5 then the corresponding json is also loaded in
        prediction_method: Method to interpret the results of the model
    """

    def __init__(
        self,
        model_file_name: str,
        stop_key: str = "q",
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
        max_num_faces: int = FaceMapper.DEFAULT_MAX_NUM_FACES,
        padding: int = 1,
        word_dict: typing.Dict[int, str] = {},
        region: Region = Region.LIPS,
        feature_extractor_type: FeatureExtractor = FeatureExtractor.NONE_KEY,
        frame_pad: int = 0,
        model_type: FileTypes = FileTypes.KERAS,
        prediction_method: PredictionMethod = PredictionMethod.BINARY_KEY,
    ):
        super().__init__(
            stream_property=0,
            window_name=f"Lip Reader: {model_file_name}",
            stop_key=stop_key,
            operations={ImageOperationKeys.FLIP_OPERATION_KEY: 1},
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces,
            region=region,
        )

        self.model_wrapper = ModelWrapper(
            model_file_name=model_file_name,
            word_dict=word_dict,
            feature_extractor_type=feature_extractor_type,
            model_type=model_type,
            prediction_method=prediction_method,
            frame_pad=frame_pad,
        )

        self.padding: int = padding
        self.frame_pad: int = frame_pad

        self.word_dict: typing.Dict[int, str] = word_dict
        self.frame_buffer = self.init_frame_buffer(
            padding=self.padding, region=self.region, frame_pad=self.frame_pad
        )

    def init_frame_buffer(
        self, padding: int, region: Region, frame_pad: int = -1
    ) -> typing.List[np.ndarray]:
        """
        Will create the frame buffer. This saves the frames so that they can be run through the model in a batch
        """
        if (
            self.model_wrapper.feature_extractor_type
            == FeatureExtractor.INCEPTION_IMAGENET_KEY
        ):
            pad_fv = np.zeros(
                shape=(FeatureExtractor.INCEPTION_IMAGENET_SIZE_KEY), dtype=np.uint8
            )
            return deque([pad_fv for _ in range(padding)])

        if region == Region.LANDMARKS:
            return deque([padded_lip_landmarks for _ in range(padding)])
        elif region == Region.MOUTH or region == Region.LIPS:
            pad_img = np.zeros(shape=(frame_pad, frame_pad, 3), dtype=np.uint8)
            return deque([pad_img for _ in range(padding)])

    def use_frame(self, frame: np.ndarray) -> bool:
        """
        Will display the current frame to the screen, only returning false when the user wants the session to end
        """
        data = frame

        if self.region == Region.LIPS:
            data = self.lip_crop
        elif self.region == Region.MOUTH:
            data = self.mouth_crop
        elif self.region == Region.LANDMARKS:
            data = self.face_mapper.normalised_lip_landmarks

        if data != []:
            # Don't need to pad landmarks, there's a set amount
            if self.frame_pad != -1:
                if self.region == Region.LIPS or self.region == Region.MOUTH:
                    data = self.pad_image(img=data, padding=self.frame_pad)

            ##Convert to feature vectors if a feature extractor has been specified
            if (
                self.model_wrapper.feature_extractor_type != FeatureExtractor.NONE_KEY
                and self.region != Region.LANDMARKS
            ):
                data = self.model_wrapper.convert_to_feature_vectors(data=data)
            self.frame_buffer.append(data)
            self.frame_buffer.popleft()

            # If just 1 frame pass this alone into the model
            if self.padding == 1:
                word = self.model_wrapper.use_model(data=data)
            else:  # Otherwise pass the whole frame buffer in to be used
                word = self.model_wrapper.use_model(data=self.frame_buffer)
            cv2.putText(
                img=frame,
                text=word,
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        cv2.imshow(winname=self.window_name, mat=frame)
        if cv2.waitKey(1) & 0xFF == ord(self.stop_key):
            return True
        return False
