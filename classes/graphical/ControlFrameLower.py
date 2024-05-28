import tkinter as tk
from tkinter import ttk
from utility.Dataclasses import (
    GUIMode,
)
from ModelWrapper import ModelWrapper


class ControlFrameLower:
    """
    Parameters:
        frame:                  Frame to place the current frame within
        parent:                 Parent class, used to access items such as the current model
    Properties:
        bounding_box_mouth:     Whether to show the mouth bounding box on the display feed
        bounding_box_lips:      Whether to show the lip bounding box on the display feed
        lip_landmarks:          Whether to show the lip landmarks on the display feed
        num_people:             The number of people within the frame, used to configure MediaPipe
        use_camera_feed:        Whether to use the camera feed. Disabled when the camera is disabled
        camera_feed_button:     Button used to invoke and disable the camera feed
        use_model:              Controls if the model is currently being applied to the stream
        model_state:            State of the model: whether it is running, turned off or not present
        delay:                  Delay between the predictions, used when predictions are made every frame
        new_threshold:          Prediction threshold. Predictions below htis will just be marked as unknown
        lip_height_threshold:   Height between the lip segments. Below this the mouth is marked as closed
        closed_mouth_threshold: Number of frames for which the mouth can be closed, before predictions are halted & the model is paused
    """

    DEFAULT_LIP_HEIGHT_THRESHOLD: float = 0.1
    DEFAULT_PROBABILITY_THRESHOLD: float = 0.0

    def __init__(self, frame, parent, mode: GUIMode) -> None:
        """
        Establishes the lwoer part of the control frame
        """
        self.frame = frame
        self.parent = parent
        self.mode: GUIMode = mode

        self.use_model: tk.BooleanVar = None

        check_button_frame = tk.Frame(self.frame, padx=10, pady=10)
        check_button_frame.pack(side=tk.TOP, fill="both", expand=True)
        self.init_localiser_selection(frame=check_button_frame)

        camera_control_frame = tk.Frame(self.frame, padx=10, pady=10)
        camera_control_frame.pack(side=tk.TOP, fill="both", expand=True)
        self.init_camera_control(frame=camera_control_frame)

        self.control_frame_lower_lower = tk.Frame(frame)
        self.control_frame_lower_lower.pack(side=tk.TOP, fill="both", expand=True)
        self.configure_control_frame(mode=self.mode)

    def configure_control_frame(self, mode: GUIMode):
        """
        Will configure the control frame based on the current mode, whether it is training or inference time
        """
        self.clear_control_frame_lower_lower()
        if mode == GUIMode.INFERENCE:
            self.init_control_lower_inference(frame=self.control_frame_lower_lower)
            self.change_model_state(value=0)
        elif mode == GUIMode.TRAIN:
            self.init_control_lower_train(frame=self.control_frame_lower_lower)

    def clear_control_frame_lower_lower(self):
        """
        Clears the lower area of the control frame
        """
        for widgets in self.control_frame_lower_lower.winfo_children():
            widgets.destroy()

    def init_control_lower_train(self, frame) -> None:
        """
        Creates the lower half of the control frame lower frame. This section stores controls for configuring training
        """
        # Controls the number of epochs used for finetuning
        epochs_label = tk.Label(
            frame,
            text="Number of Epochs:",
            font=self.parent.TITLE_FONT,
        )
        epochs_label.grid(row=0, column=0, pady=2, padx=10)
        self.epoch_box = tk.Spinbox(
            frame, from_=1, to=200, command=self.update_number_epochs, width=25
        )
        self.epoch_box.grid(row=0, column=1, pady=2, padx=10)

        # Controls the number of epochs used for finetuning
        batch_size_label = tk.Label(
            frame,
            text="Batch Size:",
            font=self.parent.TITLE_FONT,
        )
        batch_size_label.grid(row=1, column=0, pady=2, padx=10)

        batch_size = tk.StringVar()
        batch_size.set(str(ModelWrapper.DEFAULT_BATCH_SIZE))
        self.batch_size_box = tk.Spinbox(
            frame,
            from_=1,
            to=200,
            command=self.update_batch_size,
            width=25,
            textvariable=batch_size,
        )
        self.batch_size_box.grid(row=1, column=1, pady=2, padx=10)

        # Controls the learning rate for finetuning
        self.learning_rate = tk.StringVar()
        self.learning_rate.set(str(ModelWrapper.DEFAULT_LR))
        learning_rate_label = tk.Label(
            frame,
            text="Learning Rate:",
            font=self.parent.TITLE_FONT,
        )
        learning_rate_label.grid(row=2, column=0, pady=2, padx=10)
        learning_rate_entry = tk.Entry(
            frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.learning_rate,
        )
        learning_rate_entry.grid(row=2, column=1, pady=2, padx=10)

    def init_camera_control(self, frame) -> None:
        """
        Creates the camera control area, a frame for controlling the camera feed
        """
        # Camera toggle control
        self.use_camera_feed: tk.BooleanVar() = tk.BooleanVar(value=False)
        camera_feed_label = tk.Label(
            frame, text="Toggle Camera Feed:", font=self.parent.TITLE_FONT
        )
        camera_feed_label.grid(row=0, column=0, pady=2, padx=10)
        self.camera_feed_button = tk.Checkbutton(
            frame,
            text="",
            variable=self.use_camera_feed,
            onvalue=True,
            offvalue=False,
            font=self.parent.NORMAL_FONT,
            command=self.parent.toggle_camera,
        )
        self.camera_feed_button.grid(row=0, column=1, pady=2, padx=10)

        # Camera selection control
        self.selected_camera = tk.StringVar()

        def on_select_camera(event):
            self.parent.set_stream_property(
                stream_property=self.parent.available_cameras[
                    self.selected_camera.get()
                ]
            )

        select_camera_label = tk.Label(
            frame,
            text="Select Camera Feed:",
            font=self.parent.TITLE_FONT,
        )
        select_camera_label.grid(row=1, column=0, pady=2, padx=10)
        select_camera = ttk.Combobox(
            frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.selected_camera,
            values=list(self.parent.available_cameras.keys()),
            width=25,
        )
        select_camera.bind("<<ComboboxSelected>>", on_select_camera)
        select_camera.grid(row=1, column=1, pady=2, padx=10)
        select_camera.current(newindex=0)

    def init_localiser_selection(self, frame) -> None:
        """
        Creates the localiser selection area. This stores the checkboxes that enable each of the localisation methods
        """
        self.bounding_box_mouth: tk.BooleanVar() = tk.BooleanVar(value=False)
        self.bounding_box_lips: tk.BooleanVar() = tk.BooleanVar(value=False)
        self.lip_landmarks: tk.BooleanVar() = tk.BooleanVar(value=False)

        check_button_label = tk.Label(
            frame, text="Localiser Method:", font=self.parent.TITLE_FONT
        )
        check_button_label.pack(side=tk.TOP, anchor=tk.W)

        ## Check buttons for toggling localisers

        bbm_check_button = tk.Checkbutton(
            frame,
            text="Bounding Box: Mouth Area",
            variable=self.bounding_box_mouth,
            onvalue=True,
            offvalue=False,
            font=self.parent.NORMAL_FONT,
            command=self.parent.toggle_bb_mouth_area,
        )
        bbm_check_button.pack(side=tk.TOP, anchor=tk.W)
        bbl_check_button = tk.Checkbutton(
            frame,
            text="Bounding Box: Lip Area",
            variable=self.bounding_box_lips,
            onvalue=True,
            offvalue=False,
            font=self.parent.NORMAL_FONT,
            command=self.parent.toggle_bb_lip_area,
        )
        bbl_check_button.pack(side=tk.TOP, anchor=tk.W)

        ll_check_button = tk.Checkbutton(
            frame,
            text="Lip Landmarks",
            variable=self.lip_landmarks,
            onvalue=True,
            offvalue=False,
            font=self.parent.NORMAL_FONT,
            command=self.parent.toggle_landmarks,
        )
        ll_check_button.pack(side=tk.TOP, anchor=tk.W)

    def init_control_lower_inference(self, frame) -> None:
        """
        Creates the lower half of the control frame lower frame. This section stores controls for other utilities of the GUI
        """
        # Number of people control
        num_people_label = tk.Label(
            frame, text="Number of People:", font=self.parent.TITLE_FONT
        )
        num_people_label.grid(row=0, column=0, pady=2, padx=10)
        self.num_people = tk.Spinbox(
            frame, from_=1, to=3, command=self.update_number_people, width=25
        )
        self.num_people.grid(row=0, column=1, pady=2, padx=10)

        # Use model toggle control
        self.use_model: tk.BooleanVar() = tk.BooleanVar(value=False)
        frame.pack(side=tk.TOP, fill="both", expand=True)
        use_model_label = tk.Label(
            frame, text="Use Model:", font=self.parent.TITLE_FONT
        )
        use_model_label.grid(row=1, column=0, pady=2, padx=10)
        self.use_model_button = tk.Checkbutton(
            frame,
            text="",
            variable=self.use_model,
            onvalue=True,
            offvalue=False,
            font=self.parent.NORMAL_FONT,
            command=self.parent.toggle_use_model,
        )
        self.use_model_button.grid(row=1, column=1, pady=2, padx=10)

        # State of the model
        model_state_label = tk.Label(
            frame,
            text="Model State:",
            font=self.parent.TITLE_FONT,
        )
        model_state_label.grid(row=2, column=0, pady=2, padx=10)
        self.model_state: tk.Label = tk.Label(
            frame, text="", font=self.parent.NORMAL_FONT
        )
        self.model_state.grid(row=2, column=1, pady=2, padx=10)

        # Prediction thresholds
        self.new_threshold = tk.StringVar()
        self.new_threshold.set(str(self.DEFAULT_PROBABILITY_THRESHOLD))
        threshold_label = tk.Label(
            frame,
            text="Probability Threshold:",
            font=self.parent.TITLE_FONT,
        )
        threshold_label.grid(row=4, column=0, pady=2, padx=10)
        threshold_entry = tk.Entry(
            frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.new_threshold,
        )
        threshold_entry.grid(row=4, column=1, pady=2, padx=10)
        threshold_button = tk.Button(
            frame,
            font=self.parent.NORMAL_FONT,
            text="Submit",
            command=self.submit_threshold,
        )
        threshold_button.grid(row=4, column=2, pady=2, padx=10)
        threshold_button.invoke()

        # Threshold for the lip height control
        self.lip_height_threshold = tk.StringVar()
        self.lip_height_threshold.set(str(self.DEFAULT_LIP_HEIGHT_THRESHOLD))
        lip_threshold_label = tk.Label(
            frame,
            text="Lip Height Threshold:",
            font=self.parent.TITLE_FONT,
        )
        lip_threshold_label.grid(row=5, column=0, pady=2, padx=10)
        lip_threshold_entry = tk.Entry(
            frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.lip_height_threshold,
        )
        lip_threshold_entry.grid(row=5, column=1, pady=2, padx=10)
        lip_threshold_button = tk.Button(
            frame,
            font=self.parent.NORMAL_FONT,
            text="Submit",
            command=self.submit_height_threshold,
        )
        lip_threshold_button.grid(row=5, column=2, pady=2, padx=10)
        lip_threshold_button.invoke()

        # Closed mouth threshold control
        closed_mouth_threshold_label = tk.Label(
            frame,
            text="Closed Mouth Threshold:",
            font=self.parent.TITLE_FONT,
        )
        closed_mouth_threshold_label.grid(row=6, column=0, pady=2, padx=10)

        self.closed_mouth_threshold = tk.Spinbox(
            frame,
            from_=0,
            to=29,
            command=self.parent.update_closed_mouth_threshold,
            width=25,
        )
        self.closed_mouth_threshold.grid(row=6, column=1, pady=2, padx=10)

    def update_number_epochs(self) -> None:
        """
        Updates the number of epochs to train with
        """
        self.parent.epochs = int(self.epoch_box.get())

    def update_batch_size(self) -> None:
        """
        Updates the batch size used for finetuning
        """
        self.parent.batch_size = int(self.batch_size_box.get())

    # Control number of people for localisation
    def update_number_people(self) -> None:
        """
        Will update the max number of faces for each of the model wrappers
        """
        self.parent.set_max_num_faces(max_num_faces=int(self.num_people.get()))

    def change_model_state(self, value: int) -> None:
        """
        Will configure the display to show whether a model is loaded & running
        """
        if value == 0:
            self.model_state.configure(text="No Model", fg="blue")
        elif value == 1:
            self.model_state.configure(text="Not Running", fg="red")
        elif value == 2:
            self.model_state.configure(text="Running", fg="green")

    def submit_height_threshold(self):
        """
        Will submit the lip threshold value to the parent to be updated for each model
        """
        try:
            value = float(self.lip_height_threshold.get())
            if 0.0 <= value <= 1.0:
                self.parent.change_lip_height_threshold(new_threshold=value)
        except ValueError:
            self.lip_height_threshold.set(str(self.DEFAULT_LIP_HEIGHT_THRESHOLD))

    def submit_threshold(self):
        """
        Will submit the probability threshold value to the parent to be updated for each model
        """
        try:
            value = float(self.new_threshold.get())
            if 0.0 <= value <= 1.0:
                self.parent.change_probability_threshold(new_threshold=value)
        except ValueError:
            self.new_threshold.set(str(self.DEFAULT_PROBABILITY_THRESHOLD))
