import mediapipe as mp
import typing
import numpy as np
from dataclasses import dataclass
from PIL import Image
import cv2
from copy import deepcopy

from utility.Dataclasses import Region, BoundingBox
from utility.FaceMapperKeyLists import (
    lip_landmarks_outer_keys,
    lip_landmarks_inner_keys,
    mouth_landmarks_outer_keys,
)


class FaceMapper:
    nose_landmark_key: typing.List[int] = [3]

    DEFAULT_THRESHOLD: float = 0.5
    DEFAULT_MAX_NUM_FACES: int = 1

    @dataclass(frozen=True)
    class NormalisationMethods:
        MOUTH_LOCATION_KEY: str = "MOUTH_LOCATION"
        NOSE_KEY: str = "NOSE"
        NONE_KEY: str = "NONE"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_num_faces: int = DEFAULT_MAX_NUM_FACES,
    ) -> None:
        self.threshold: float = threshold
        self.max_num_faces: int = max_num_faces

    def make_face_map(
        self,
        img: np.array,
        normalisation_method: NormalisationMethods = NormalisationMethods.MOUTH_LOCATION_KEY,
    ) -> None:
        self.display_img = img
        self.face_masks = self.__detect_face(
            img=self.display_img,
            threshold=self.threshold,
            max_num_faces=self.max_num_faces,
        )
        self.face_found = bool(self.face_masks.multi_face_landmarks)

        self.lip_landmarks_condensed = []
        self.lip_bboxs: typing.List[BoundingBox] = []
        self.mouth_bboxs: typing.List[BoundingBox] = []
        self.normalised_lip_landmarks = np.array([])
        if not self.face_found:
            return

        # List of lists of landmarks
        self.lip_landmarks: typing.List[
            typing.List[np.array]
        ] = self.__get_landmark_lists(
            indices=(lip_landmarks_inner_keys + lip_landmarks_outer_keys)
        )

        # List of lists of landmarks
        self.mouth_landmarks: typing.List[
            typing.List[np.array]
        ] = self.__get_landmark_lists(
            indices=(
                lip_landmarks_inner_keys
                + lip_landmarks_outer_keys
                + mouth_landmarks_outer_keys
            )
        )
        self.mouth_landmarks_condensed = self.make_landmarks_condensed(
            landmark_list=self.mouth_landmarks
        )
        self.lip_landmarks_condensed = self.make_landmarks_condensed(
            landmark_list=self.lip_landmarks
        )

        self.lip_bboxs: typing.List[BoundingBox] = self.make_bboxs(
            img=self.display_img, landmark_list=self.lip_landmarks
        )
        self.mouth_bboxs: typing.List[BoundingBox] = self.make_bboxs(
            img=self.display_img, landmark_list=self.mouth_landmarks
        )

        self.normalised_lip_landmarks = self.normalise_lip_landmarks(
            normalisation_method=normalisation_method, landmark_list=self.lip_landmarks
        )

    def __get_landmark_lists(
        self, indices: typing.List[int]
    ) -> typing.List[typing.List[np.array]]:
        """
        Will return the landmark list based on the indices input
        """
        if self.face_found:
            face_meshes = []
            for x in self.face_masks.multi_face_landmarks:
                face_meshes.append(np.array(x.landmark)[indices])
            return face_meshes
        else:
            return None

    def normalise_lip_landmarks(
        self,
        landmark_list: typing.List[np.ndarray],
        normalisation_method: NormalisationMethods = NormalisationMethods.NONE_KEY,
    ) -> np.array:
        """
        Will normalise the lip landmarks using the specified normalisation method
        """
        return_list = []

        for landmarks in landmark_list:
            normalised_lip_landmarks = None
            top_corner, bottom_corner = self.get_bbox(landmarks=landmarks)

            width = abs(top_corner[0] - bottom_corner[0])
            height = abs(top_corner[1] - bottom_corner[1])

            if normalisation_method == self.NormalisationMethods.NOSE_KEY:
                nose_landmark = self.make_landmarks_condensed(
                    landmark_list=self.__get_landmark_lists(
                        indices=self.nose_landmark_key
                    )
                )[0][0]
                normalised_lip_landmarks = self.lip_landmarks_condensed - nose_landmark
            elif normalisation_method == self.NormalisationMethods.MOUTH_LOCATION_KEY:
                x, y = top_corner
                normalised_lip_landmarks = self.lip_landmarks_condensed - np.array(
                    [x, y, 0]
                )
                normalised_lip_landmarks = (
                    normalised_lip_landmarks / np.array([width, height, 1]).T
                )

            else:
                if normalisation_method != self.NormalisationMethods.NONE_KEY:
                    print(
                        f"Normalisation method not recognised. Defaulting to {self.NormalisationMethods.NONE_KEY}"
                    )
                return self.lip_landmarks_condensed
            return_list.append(normalised_lip_landmarks)
        return return_list

    def make_landmarks_condensed(
        self, landmark_list: typing.List[np.array]
    ) -> typing.List[np.array]:
        """
        Returns a list of face landmarks that are easier to read: ones that are just (x,y,z)
        """
        return_list = []
        for y in landmark_list:
            landmarks = []
            if not y is None:
                for x in y:
                    landmarks.append(np.array([x.x, x.y, x.z]))
            return_list.append(landmarks)
        return return_list

    def get_bbox(
        self, landmarks: np.ndarray
    ) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:
        """
        Will get the coords of the corners of a bounding box containing the landmarks specified. Will return them
        """
        x_coords = [l.x for l in landmarks]
        y_coords = [l.y for l in landmarks]

        bottom_corner = (
            max(x_coords),
            max(y_coords),
        )
        top_corner = (
            min(x_coords),
            min(y_coords),
        )
        return top_corner, bottom_corner

    def make_bboxs(
        self, img: mp.Image, landmark_list: typing.List[np.ndarray]
    ) -> typing.List[BoundingBox]:
        """
        Will find the bounding box around all the landmarks specified for the image, returns coordinates for the box in the image
        """
        if self.face_found:
            formatted_img = Image.fromarray(img)

            bbox_list = []
            for landmarks in landmark_list:
                top_corner, bottom_corner = self.get_bbox(landmarks=landmarks)
                bbox_list.append(
                    BoundingBox(
                        x_min=(int(top_corner[0] * formatted_img.width)),
                        y_min=(int(top_corner[1] * formatted_img.height)),
                        x_max=(int(bottom_corner[0] * formatted_img.width)),
                        y_max=(int(bottom_corner[1] * formatted_img.height)),
                    )
                )
            return bbox_list
        else:
            return []

    def crop_to_face_region(self, region_type: Region) -> typing.List[np.array]:
        """
        Crops to a region. Either the mouth or lips region
        """
        return_list = []
        if region_type == Region.LIPS:
            for bbox in self.lip_bboxs:
                return_list.append(
                    self.crop_to_region(xy1=bbox.get_min(), xy2=bbox.get_max())
                )
        elif region_type == Region.MOUTH:
            for bbox in self.mouth_bboxs:
                return_list.append(
                    self.crop_to_region(xy1=bbox.get_min(), xy2=bbox.get_max())
                )
        else:
            return [np.array([])]
        return return_list

    def crop_to_region(
        self, xy1: typing.Tuple[int, int], xy2: typing.Tuple[int, int]
    ) -> np.array:
        """
        Crops the current frame to the region specified by 2 coordinate tuples (format of [x,y])
        """
        x1, y1 = xy1
        x2, y2 = xy2
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        if (
            (right - left) <= 0
            or (bottom - top) <= 0
            or (left < 0)
            or (right < 0)
            or (top < 0)
            or (bottom < 0)
        ):
            return np.array([])
        return deepcopy(self.display_img)[top:bottom, left:right]

    def __detect_face(
        self,
        img: np.array,
        threshold: float = DEFAULT_THRESHOLD,
        max_num_faces: int = DEFAULT_MAX_NUM_FACES,
    ):
        """
        Given an image and a path to a model it will return the raw landmark map
        """
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=threshold,
        ) as face_mesh:
            results = face_mesh.process(img)
            return results

    def draw_landmarks_on_image(self, rgb_image: np.array, results):
        """
        Will take an image and detection result and will draw the detection onto the image, returning said image
        """
        if bool(results.multi_face_landmarks):
            # Create a copy of the image
            annotated_image = rgb_image.copy()
            drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                thickness=2, circle_radius=1, color=[0, 0, 255]
            )

            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            return annotated_image

    def draw_landmarks_blank_background(
        self,
        landmarks: np.array,
        size: typing.Tuple[int, int] = (500, 500),
        radius: int = 1,
        color: typing.Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
    ) -> np.array:
        """
        Will return an entirely blank image, showing a representation of where the landmarks are
        """
        width, height = size
        img = np.zeros(size)
        for landmark in landmarks:
            x = landmark[0] * width
            y = landmark[1] * height

            img = cv2.circle(
                img=img,
                center=(int(x), int(y)),
                radius=radius,
                color=color,
                thickness=thickness,
            )
        return img
