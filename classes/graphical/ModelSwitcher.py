import tkinter as tk
from tkinter import ttk
import typing
import sys
import numpy as np
from PIL import Image, ImageTk
import cv2
from pygrabber.dshow_graph import FilterGraph
import time

sys.path.append("../")
from ModelWrapper import ModelWrapper
from utility.Dataclasses import (
    FileTypes,
    FeatureExtractor,
    PredictionMethod,
    GUIMode,
)
from LipLocaliser import LipLocaliser
from FaceMapper import FaceMapper
from utility.Dataclasses import Region, LipState
from LipLocaliser import LocaliserMode
from VideoStream import ImageOperationKeys

from DisplayFrame import DisplayFrame
from ControlFrameUpper import ControlFrameUpper
from ControlFrameLower import ControlFrameLower
from ConvertDataset import padded_lip_landmarks


class ModelSwitcher(LipLocaliser):
    """
    Class is a GUI for lip reading models. There is various functionality such as swapping between models, displaying lip regions, etc

    Parameters:
        normalisation_method:   Normalisation method to use if landmark based is used
        confidence_threshold:   Confidence threshold for MediaPipe face mesh detection
    Properties:
        model_options:          Stores the different models available
        current_model:          String of the current model being used. Key to model_options

        available_cameras:      Stores the available cameras. Dictionary associates camera IDs with camera names
        default_camera:         The default input stream feed

        external_vid:           Whether the input vid stream is external or the currently selected camera
        frame_num:              The current frame number. Used to synchronise model predictions to external videos
        frame_prune:            The frame prune for videos. Calculated based on the size of the input video and the size of the model. Ensures predictions happen on the central part of the video
        vid_length:             Stores the number of frames for the current video, if it has a set, fixed length. Only used for external videos

        epochs:                 The number of epochs for finetuning new models, controlled in the display frame
        batch_size:             The batch size for finetuning new models, controlled in the display frame

    GUI Properties:
        window:                 Root window
    """

    WINDOW_MIN_WIDTH: int = 100
    WINDOW_MIN_HEIGHT: int = 100

    WINDOW_WIDTH_REL: float = 0.75
    WINDOW_HEIGHT_REL: float = 0.75

    WINDOW_NAME: str = "Lip Reader"

    DEFAULT_WINDOW_MODE: GUIMode = GUIMode.INFERENCE

    ACTIVE_BUTTON_COLOUR: str = "#c0c0c0"
    INACTIVE_BUTTON_COLOUR: str = "#f0f0f0"

    TITLE_FONT: typing.Tuple = ("Arial", 12, "bold")
    NORMAL_FONT: typing.Tuple = ("Arial", 12)
    DEFAULT_FONT_COLOUR: str = "#000000"

    DEFAULT_OUTPUT_DURATION: int = 1500

    USER_SPECIFIED_KEY: str = "User Specified"
    NO_MODEL_KEY: str = "No Model"

    # Dictionary storing each different model for the GUI to swap between
    MODEL_OPTIONS = {
        NO_MODEL_KEY: None,
        "Image Model": ModelWrapper(
            model_file_name="../../models/4th experiment/2/best_run.hdf5",
            word_dict={0: "ABOUT", 1: "BELIEVE", 2: "CHANCE", 3: "FAMILY"},
            feature_extractor_type=FeatureExtractor.INCEPTION_IMAGENET_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.ARGMAX_KEY,
            frame_pad=21,
            padding=100,
            region=Region.MOUTH,
            closed_mouth_threshold=-1,  # Not landmark based, can't use the lip state
        ),
        "Landmark Model": ModelWrapper(
            model_file_name="../../models/5th experiment/4/best_run.hdf5",
            word_dict={0: "ABOUT", 1: "BELIEVE", 2: "CHANCE", 3: "FAMILY"},
            feature_extractor_type=FeatureExtractor.NONE_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.ARGMAX_KEY,
            frame_pad=21,
            padding=100,
            region=Region.LANDMARKS,
            closed_mouth_threshold=0,  # Landmark based so use lip state
        ),
        "Combined Model": ModelWrapper(
            model_file_name="../../models/8th experiment/video_based_run.hdf5",
            word_dict={0: "ABOUT", 1: "BELIEVE", 2: "CHANCE", 3: "FAMILY"},
            feature_extractor_type=FeatureExtractor.INCEPTION_IMAGENET_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.ARGMAX_KEY,
            frame_pad=21,
            padding=100,
            region=Region.MOUTH_AND_LANDMARKS,
            closed_mouth_threshold=0,
        ),
        "Letter-Based": ModelWrapper(
            model_file_name="../../models/9th experiment/1/best_run.hdf5",
            word_dict={
                16: " ",
                1: "A",
                2: "B",
                3: "C",
                4: "E",
                5: "F",
                6: "H",
                7: "I",
                8: "L",
                9: "M",
                10: "N",
                11: "O",
                12: "T",
                13: "U",
                14: "V",
                15: "Y",
                -1: " ",
            },
            feature_extractor_type=FeatureExtractor.INCEPTION_IMAGENET_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.CTC_KEY,
            frame_pad=21,
            padding=100,
            region=Region.MOUTH_AND_LANDMARKS,
            closed_mouth_threshold=0,
        ),
        "Phoneme-Based": ModelWrapper(
            model_file_name="../../models/9th experiment/2/best_run.hdf5",
            word_dict={
                1: "AE1",
                2: "AH0",
                3: "AW1",
                4: "B",
                5: "CH",
                6: "F",
                7: "IH0",
                8: "IY0",
                9: "IY1",
                10: "L",
                11: "M",
                12: "N",
                13: "S",
                14: "T",
                15: "V",
                16: " ",
                -1: " ",
                0: " ",
            },
            feature_extractor_type=FeatureExtractor.INCEPTION_IMAGENET_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.CTC_KEY,
            frame_pad=21,
            padding=100,
            region=Region.MOUTH_AND_LANDMARKS,
            closed_mouth_threshold=0,
        ),
        "Viseme-Based": ModelWrapper(
            model_file_name="../../models/9th experiment/3/best_run.hdf5",
            word_dict={
                1: "aa",
                2: "ah",
                3: "ch",
                4: "eh",
                5: "f",
                6: "iy",
                7: "k",
                8: "p",
                9: "t",
                10: "uh",
                11: " ",
                -1: " ",
                0: " ",
            },
            feature_extractor_type=FeatureExtractor.INCEPTION_IMAGENET_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.CTC_KEY,
            frame_pad=21,
            padding=100,
            region=Region.MOUTH_AND_LANDMARKS,
            closed_mouth_threshold=0,
        ),
        "Large Model": ModelWrapper(
            model_file_name="../../models/10th experiment/4/best_run.hdf5",
            word_dict={
                0: "ARRESTED",
                1: "DESPITE",
                2: "FOREIGN",
                3: "GIVING",
                4: "LATER",
                5: "MINUTES",
                6: "OFFICIALS",
                7: "POSSIBLE",
                8: "SITUATION",
                9: "THROUGH",
            },
            feature_extractor_type=FeatureExtractor.INCEPTION_IMAGENET_KEY,
            model_type=FileTypes.HDF5,
            prediction_method=PredictionMethod.ARGMAX_KEY,
            frame_pad=29,
            padding=100,
            region=Region.LANDMARKS,
            closed_mouth_threshold=0,
        ),
        USER_SPECIFIED_KEY: None,
    }

    def __init__(
        self,
        normalisation_method: FaceMapper.NormalisationMethods = FaceMapper.NormalisationMethods.NONE_KEY,
        confidence_threshold: float = FaceMapper.DEFAULT_THRESHOLD,
    ):
        self.available_cameras: typing.Dict[str:int] = self.get_available_cameras()
        self.default_camera: int = self.available_cameras[
            list(self.available_cameras.keys())[0]
        ]  # TODO: document and use default camera more elsewhere

        super().__init__(
            stream_property=self.default_camera,
            operations={ImageOperationKeys.FLIP_OPERATION_KEY: 1},
            normalisation_method=normalisation_method,
            confidence_threshold=confidence_threshold,
            max_num_faces=1,  # Use 1 for now. Can alter this with the GUI
            region=Region.NONE,  # Use None for now. can alter this with the GUI later
            localiser_mode=LocaliserMode.NONE_KEY,  # Use None for now. can alter this with the GUI later
        )

        self.model_options: typing.Dict[str, ModelWrapper] = self.MODEL_OPTIONS
        self.current_model: str = None

        self.external_vid: bool = False
        self.frame_num: int = 0
        self.frame_prune: int = 0
        self.vid_length: int = 0

        self.epochs: int = ModelWrapper.DEFAULT_EPOCHS
        self.batch_size: int = ModelWrapper.DEFAULT_BATCH_SIZE

        self.window = self.init_window()

    # ---------------------------------------------------------------------------------#
    ###                             Window Based Functionality
    # ---------------------------------------------------------------------------------#
    def init_window(self) -> tk.Tk:
        """
        Creates the window & widgets within it
        """
        window = tk.Tk()
        window.title(self.WINDOW_NAME)
        # Bind to exit out the window
        window.bind("<Escape>", lambda _: window.quit())
        window.bind("q", lambda _: window.quit())
        window.minsize(width=self.WINDOW_MIN_WIDTH, height=self.WINDOW_MIN_HEIGHT)

        screen_geometry: str = f"{int(window.winfo_screenwidth() * self.WINDOW_WIDTH_REL)}x{int(window.winfo_screenheight() * self.WINDOW_HEIGHT_REL)}"
        window.geometry(newGeometry=screen_geometry)

        self.frame = tk.Frame(window)
        self.frame.pack(fill="both", expand=True, side=tk.BOTTOM)

        self.init_top_frame(frame=self.frame)

        self.display_frame = DisplayFrame(
            frame=self.frame, parent=self, mode=self.DEFAULT_WINDOW_MODE
        )
        self.init_control_frame(frame=self.frame)

        window.state("zoomed")
        return window

    def change_mode_graphical(self, mode: GUIMode) -> None:
        """
        Changes the mode buttons to reflect the mode change
        """
        if mode == GUIMode.TRAIN:
            self.change_button(button=self.inference_button, clicked=False)
            self.change_button(button=self.retrain_button, clicked=True)
        elif mode == GUIMode.INFERENCE:
            self.change_button(button=self.inference_button, clicked=True)
            self.change_button(button=self.retrain_button, clicked=False)

    def change_button(self, button, clicked) -> None:
        """
        Changes a button based on whether it was clicked
        """
        if clicked:
            button.configure(relief=tk.SUNKEN, bg=self.ACTIVE_BUTTON_COLOUR)
        else:
            button.configure(relief=tk.RAISED, bg=self.INACTIVE_BUTTON_COLOUR)

    def output(
        self,
        text: str,
        colour: str = DEFAULT_FONT_COLOUR,
        duration: int = DEFAULT_OUTPUT_DURATION,
    ) -> None:
        """
        Will output information on the GUI, of a specific colour, for a limited amount of time
        """
        self.output_label.config(fg=colour)
        self.output_text.set(text)
        if duration != -1:  # Keeps message viewable if duration is -1
            self.window.after(ms=duration, func=lambda: self.output_text.set(""))

    def init_top_frame(self, frame) -> None:
        """
        Will create a frame storing the two buttons controlling the window's mode: Training & inference
        """
        top_frame = tk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill="both", expand=True)

        self.output_text = tk.StringVar()
        self.output_label = tk.Label(
            top_frame, font=self.TITLE_FONT, textvariable=self.output_text
        )
        self.output_label.pack(side=tk.TOP, fill="both", expand=True)

        # Retrain Model button
        self.inference_button = tk.Button(
            top_frame,
            font=self.NORMAL_FONT,
            text="Inference",
            command=lambda: self.change_mode(GUIMode.INFERENCE),
        )
        self.inference_button.pack(side=tk.LEFT, fill="both", expand=True)

        # Retrain Model button
        self.retrain_button = tk.Button(
            top_frame,
            font=self.NORMAL_FONT,
            text="Train",
            command=lambda: self.change_mode(GUIMode.TRAIN),
        )
        self.retrain_button.pack(side=tk.LEFT, fill="both", expand=True)

        self.change_mode_graphical(mode=self.DEFAULT_WINDOW_MODE)

    def init_control_frame(self, frame) -> None:
        """
        Creates the upper and lower parts of the control frame half (right half)

        This has two halfs: upper half it the tabbed section controlling the model. The lower is to control the whole application
        """
        control_frame = tk.Frame(frame)
        control_frame.pack(side=tk.LEFT, fill="both", expand=True)

        # Create upper control frame area: model control area
        control_frame_upper = tk.Frame(control_frame)
        control_frame_upper.pack(side=tk.TOP, fill="both", expand=True)
        self.ControlFrameUpper = ControlFrameUpper(frame=control_frame, parent=self)

        # Create lower control frame area: control display area
        self.control_frame_lower = tk.Frame(
            control_frame, highlightbackground="black", highlightthickness=1
        )
        self.control_frame_lower.pack(side=tk.TOP, fill="both", expand=True)
        self.ControlFrameLower = ControlFrameLower(
            frame=self.control_frame_lower, parent=self, mode=self.DEFAULT_WINDOW_MODE
        )

    def clear_control_frame_lower(self):
        for widgets in self.control_frame_lower.winfo_children():
            widgets.destroy()

    def start(self) -> None:
        """
        Just starts the window looping
        """
        self.ControlFrameLower.camera_feed_button.invoke()
        self.ControlFrameUpper.tabControl.select(tab_id=0)
        self.window.mainloop()

    def set_current_model(self, new_model: str) -> None:
        """
        Sets the current model string as the newly selected model
        """
        self.current_model: str = new_model
        self.clear_frame_buffer()  # Clears the frame buffer incase the data is not valid for the new model
        if self.display_frame.mode == GUIMode.INFERENCE:
            if self.current_model != list(self.model_options.keys())[0]:
                if self.ControlFrameLower.use_model is not None:
                    self.ControlFrameLower.use_model.set(False)
                self.ControlFrameLower.change_model_state(
                    value=(int(self.ControlFrameLower.use_model.get()) + 1)
                )
            else:
                self.ControlFrameLower.change_model_state(value=0)
        elif self.display_frame.mode == GUIMode.TRAIN:
            self.display_frame.set_training_classes(classes=self.get_current_classes())
        self.set_frame_prune()

    def update_closed_mouth_threshold(self):
        """
        Changes the closed mouth threshold using the value in the frame
        """
        for model in self.model_options:
            if (
                self.model_options[model] is not None
                and self.model_options[model].closed_mouth_threshold != -1
            ):
                self.model_options[
                    model
                ].closed_mouth_threshold = int(
                    self.ControlFrameLower.closed_mouth_threshold.get()
                )

    # ---------------------------------------------------------------------------------#
    ###                             General functions used for the GUI
    # ---------------------------------------------------------------------------------#
    def get_current_model(self) -> str:
        """
        Just returns the current model that is selected
        """
        if self.model_options[self.current_model] is None:
            self.output(text="No model currently selected!", colour="red")
            return None
        return self.model_options[self.current_model].model

    def model_selected(self) -> bool:
        """
        Returns whether a valid model has been selected
        """
        return self.get_current_model() is not None

    def get_current_classes(self) -> typing.Dict[int, str]:
        """
        Just returns the classes for the currently selected model
        """
        if self.model_options[self.current_model] is None:
            self.output(text="No model currently selected!", colour="red")
            return None
        return self.model_options[self.current_model].word_dict

    def change_mode(self, mode: GUIMode):
        """
        Changes the mode of the window and reconfigures the child frames based on the mode
        """
        if mode == GUIMode.TRAIN:
            new_mode = GUIMode.TRAIN
        elif mode == GUIMode.INFERENCE:
            new_mode = GUIMode.INFERENCE
        self.change_mode_graphical(mode=mode)
        self.display_frame.configure_mode(mode=new_mode)
        self.ControlFrameLower.configure_control_frame(mode=new_mode)

    def get_learning_rate(self) -> float:
        """
        Will get the current LR from the control frame and return it
        """
        try:
            lr = float(self.ControlFrameLower.learning_rate.get())
            return lr
        except ValueError:
            self.output(text="Learning rate was not numeric", colour="red")
            return ModelWrapper.DEFAULT_LR

    def get_available_cameras(self) -> typing.Dict[str, int]:
        """
        Gets all available cameras attached to the current system
        """
        graph = FilterGraph()
        available_cams = graph.get_input_devices()
        return {x: available_cams.index(x) for x in available_cams}

    def change_probability_threshold(self, new_threshold: float) -> None:
        """
        Will change the prediction probability threshold for each model
        """
        for model in self.model_options:
            if self.model_options[model] is not None:
                self.model_options[model].probability_threshold = new_threshold

    def change_lip_height_threshold(self, new_threshold: float) -> None:
        """
        Will get the lip height threshold & change all those that use this feature

        If the lip height of the model wrapper is -1 then they don't support lip height thresholding and so we don't use it
        """
        for model in self.model_options:
            if self.model_options[model] is not None:
                if self.model_options[model].lip_height_threshold == -1:
                    continue
                self.model_options[model].lip_height_threshold = new_threshold

    def clear_frame_buffer(self) -> None:
        """
        Will clear the frame buffer of the current model
        """
        if self.model_options[self.current_model] is None:
            return
        self.model_options[self.current_model].reset_frame_buffer()
        self.display_frame.frame_buffer_length.set("0")

    # Control if the model is currently being used & applied to images
    def toggle_use_model(self) -> None:
        """
        Controls whether the model is currently being applied or not
        """
        if self.current_model != list(self.model_options.keys())[0]:
            self.ControlFrameLower.change_model_state(
                value=(int(self.ControlFrameLower.use_model.get()) + 1)
            )
        else:
            self.ControlFrameLower.change_model_state(value=0)

    # Toggle on and off the camera
    def toggle_camera(self) -> None:
        """
        Will either display the camera feed or turn it off
        """
        if self.ControlFrameLower.use_camera_feed.get():
            self.stream_feed()
        else:
            self.display_frame.display_area.configure(
                image=ImageTk.PhotoImage(
                    image=Image.fromarray(
                        np.zeros(
                            shape=(
                                self.display_frame.display_area.winfo_height(),
                                self.display_frame.display_area.winfo_width(),
                            )
                        )
                    )
                )
            )

    def toggle_bounding_box(self) -> None:
        """
        Will toggle on whether bounding boxes are displayed aswell as lip landmarks
        """
        if (
            self.ControlFrameLower.bounding_box_mouth.get()
            or self.ControlFrameLower.bounding_box_lips.get()
        ):
            if self.localiser_mode == LocaliserMode.LANDMARKS_KEY:
                self.localiser_mode = LocaliserMode.BOTH_KEY
            elif self.localiser_mode == LocaliserMode.NONE_KEY:
                self.localiser_mode = LocaliserMode.REGION_KEY
        else:
            if self.region == Region.NONE:
                if self.localiser_mode == LocaliserMode.BOTH_KEY:
                    self.localiser_mode = LocaliserMode.LANDMARKS_KEY
                else:
                    self.localiser_mode = LocaliserMode.NONE_KEY

    def toggle_bb_mouth_area(self) -> None:
        """
        Toggles on whether the bounding boxes are showing the mouth area, lip area or both
        """
        if self.ControlFrameLower.bounding_box_mouth.get():
            if self.region == Region.LIPS:
                self.region = Region.LIPS_AND_MOUTH
            else:
                self.region = Region.MOUTH
        else:
            if self.region == Region.LIPS_AND_MOUTH:
                self.region = Region.LIPS
            else:
                self.region = Region.NONE
        self.toggle_bounding_box()

    def toggle_bb_lip_area(self):
        """
        Toggles on whether the bounding boxes are showing the mouth area, lip area or both
        """
        if self.ControlFrameLower.bounding_box_lips.get():
            if self.region == Region.MOUTH:
                self.region = Region.LIPS_AND_MOUTH
            else:
                self.region = Region.LIPS
        else:
            if self.region == Region.LIPS_AND_MOUTH:
                self.region = Region.MOUTH
            else:
                self.region = Region.NONE
        self.toggle_bounding_box()

    def toggle_landmarks(self) -> None:
        """
        Toggles whether lip landmarks are drawn aswell as bounding boxes
        """
        if self.ControlFrameLower.lip_landmarks.get():
            if self.localiser_mode == LocaliserMode.REGION_KEY:
                self.localiser_mode = LocaliserMode.BOTH_KEY
            else:
                self.localiser_mode = LocaliserMode.LANDMARKS_KEY
        else:
            if self.localiser_mode == LocaliserMode.BOTH_KEY:
                self.localiser_mode = LocaliserMode.REGION_KEY
            else:
                self.localiser_mode = LocaliserMode.NONE_KEY

    def reset_video_stream(self) -> None:
        """
        Resets the video stream to the default
        """
        self.external_vid = False
        self.clear_frame_buffer()
        self.set_stream_property(stream_property=self.default_camera)
        self.set_video_length(length=-1)

    def set_video_length(self, length: int) -> None:
        """
        Changes the video length and the frame prune
        """
        self.vid_length = length
        self.set_frame_prune()

    def set_frame_prune(self) -> None:
        """
        Sets the frame prune based on the input video length
        """
        if self.vid_length == -1 or not self.model_selected():
            self.frame_prune = 0
            return

        frame_pad = self.model_options[self.current_model].frame_pad
        if frame_pad != self.vid_length:
            self.frame_prune = (self.vid_length - frame_pad) / 2

    def select_video(self, filename: str) -> None:
        """
        Changes the stream to be an input file
        """
        self.external_vid = True
        self.clear_frame_buffer()
        self.set_stream_property(stream_property=filename)
        self.set_video_length(length=int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)))

    # ---------------------------------------------------------------------------------#
    ###                             Model Based Functionality
    # ---------------------------------------------------------------------------------#
    def use_frame(self, frame: np.ndarray) -> None:
        """
        Will set the display area of the window as the current frame from the camera
        """
        if (
            self.current_model is not None
            and self.model_options[self.current_model] is not None
        ):
            if (
                self.display_frame.mode == GUIMode.INFERENCE
                and self.ControlFrameLower.use_model.get()
            ):
                # If it is an external video it may be unsynchronised at first with the model prediction. This resynchronises it
                if self.external_vid:
                    sync = self.frame_num
                    self.frame_num += 1
                    # Frame prune is used if the number of video frames doesn't align with the model input size
                    if self.frame_prune != -1:
                        sync -= self.frame_prune
                    if len(self.model_options[self.current_model].frame_buffer) != sync:
                        return

                # Adds frame to the frame buffer and tries to make a prediction
                self.pass_frame()
                predict, all_predictions = self.use_model_wrapper()
                if predict != None and all_predictions != []:
                    self.display_frame.current_prediction.set(predict)
                    if predict != ModelWrapper.UNKNOWN_KEY:
                        self.display_frame.all_words_detected.insert(
                            index=tk.END, chars=f" {predict}"
                        )

                    # Use all predictions
                    self.display_frame.all_predictions.set(all_predictions)

                # Will change the display to say whether the lips are open, based on landmark locations
                if (
                    self.model_options[self.current_model].region == Region.LANDMARKS
                    or self.model_options[self.current_model].region
                    == Region.MOUTH_AND_LANDMARKS
                ):
                    if self.model_options[self.current_model].lip_state:
                        self.display_frame.mouth_state.set(LipState.OPEN_KEY)
                    else:
                        self.display_frame.mouth_state.set(LipState.CLOSED_KEY)
                else:
                    self.display_frame.mouth_state.set(LipState.UNKNOWN_KEY)

                self.display_frame.frame_buffer_length.set(
                    str(
                        self.model_options[self.current_model].get_frame_buffer_length()
                    )
                )
            elif (
                self.display_frame.mode == GUIMode.TRAIN
                and self.display_frame.recording
            ):
                if (
                    len(self.model_options[self.current_model].frame_buffer)
                    < self.model_options[self.current_model].frame_pad
                ):
                    self.pass_frame()
                    self.display_frame.frame_buffer_length.set(
                        str(
                            self.model_options[
                                self.current_model
                            ].get_frame_buffer_length()
                        )
                    )
                else:
                    self.display_frame.recording = False
        photo_img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.display_frame.display_area.photo_image = photo_img
        self.display_frame.display_area.configure(image=photo_img)

    def pass_frame(self) -> None:
        """
        Takes the correct data for the current model and passes it onto the frame buffer
        """
        model = self.model_options[self.current_model]
        region = model.region
        data = None
        found_face: bool = False
        if region == Region.LIPS:
            if self.lip_crops:
                data = self.lip_crops[0]
                found_face = True
        elif region == Region.MOUTH:
            if self.mouth_crops:
                data = self.mouth_crops[0]
                found_face = True
        elif region == Region.LANDMARKS:
            if self.face_mapper.normalised_lip_landmarks:
                data = self.face_mapper.normalised_lip_landmarks[0][0]
                found_face = True
        elif region == Region.MOUTH_AND_LANDMARKS:
            if self.face_mapper.normalised_lip_landmarks and self.mouth_crops:
                data = [
                    np.array(self.mouth_crops[0]),
                    np.array(self.face_mapper.normalised_lip_landmarks[0][0]),
                ]
                found_face = True

        if data is not None and found_face:
            model.pass_frame(frame=data)

    def use_model_wrapper(self) -> typing.Tuple[str, typing.Dict[str, float]]:
        """
        Will get the crop/landmarks and pass them to the model wrapper. Will run inference to get the prediction and return this
        """
        model = self.model_options[self.current_model]
        predict = None
        all_predictions = []
        prediction_made = model.predict()
        if prediction_made:
            predict = model.get_single_result()
            all_predictions = model.get_all_result()

        return predict, all_predictions

    def stream_feed(self) -> None:
        """
        Overwrite other stream feed. Instead use recursive approach to control display
        """
        if self.ControlFrameLower.use_camera_feed.get():
            ret, frame = self.stream.read()
            if not ret:
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_num = 0
                _, frame = self.stream.read()
            frame = self.frame_op(frame=frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.use_frame(frame=frame)
            self.display_frame.display_area.after(1, self.stream_feed)
        else:
            return
