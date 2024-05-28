import tkinter as tk
from tkinter import ttk
import typing
import pickle

from utility.Dataclasses import (
    GUIMode,
    FileTypes,
)


class DisplayFrame:
    """
    Creates the frame for displaying the current feed & predictions

    Parameters:
        frame:                  Frame to place the current frame within
        parent:                 Parent class, used to access items such as the current model
        mode:                   Mode of the window, used to swap between the training & inference mode
    Properties:
        display_area:           Area to show the feed within
        current_prediction:     Shows the last prediction made from the model
        mouth_state:            Shows whether the mouth is open or closed
        all_predictions:        Shows all the predictions & their probabilities from the last model prediction
        frame_buffer_length:    Shows the length of the frame buffer currently
        all_words_detected:     Shows all of the predictions that have been made

        recording:              Boolean, whether the user is being recorded for data for training
        current_classes:        List of the current classes of the selected model
        current_class:          Index of the current class for data collection

        data_batches:           Batches of recorded data from the user. Processed for the selected model
        data_labels:            Class labels for the data. One hot encoded
    """

    def __init__(self, frame, parent, mode: GUIMode):
        self.frame = frame
        self.parent = parent
        self.mode: GUIMode = mode

        self.recording: bool = False
        self.current_classes: typing.List = []
        self.current_class: int = -1

        self.data_batches: typing.List = []
        self.data_labels: typing.List = []

        display_frame = tk.Frame(
            self.frame, highlightbackground="black", highlightthickness=1
        )
        display_frame.pack(side=tk.LEFT, fill="both", expand=True)

        self.init_camera_frame(frame=display_frame)

        self.display_frame_lower = tk.Frame(display_frame, padx=10, pady=10)
        self.display_frame_lower.pack(side=tk.TOP, fill="both", expand=True)

        self.configure_mode(mode=self.mode)

    def init_camera_frame(self, frame) -> None:
        """
        Creates the frame hosting the output feed and buttons for this
        """
        camera_frame = tk.Frame(
            frame, highlightbackground="black", highlightthickness=1
        )
        camera_frame.pack(side=tk.TOP, fill="both", expand=True)

        display_frame = tk.Frame(camera_frame)
        display_frame.pack(side=tk.LEFT, fill="both", expand=True)

        self.display_area: tk.Label = tk.Label(
            display_frame, image=None, borderwidth=10, relief="solid"
        )
        self.display_area.pack(fill="none", expand=True, side=tk.LEFT)

        button_frame = tk.Frame(camera_frame)
        button_frame.pack(side=tk.LEFT, fill="both", expand=True)
        button_frame.rowconfigure(0, weight=1)
        button_frame.rowconfigure(1, weight=1)

        select_video_btn = tk.Button(
            button_frame,
            text="Select Video",
            command=self.select_video,
            font=self.parent.NORMAL_FONT,
        )
        select_video_btn.grid(row=0, column=0)

        normal_camera_btn = tk.Button(
            button_frame,
            text="Swap to Camera",
            command=self.parent.reset_video_stream,
            font=self.parent.NORMAL_FONT,
        )
        normal_camera_btn.grid(row=1, column=0)

    def select_video(self) -> None:
        """
        Creates a dialog to select a video to use as input
        """
        filename = tk.filedialog.askopenfilename(
            defaultextension=FileTypes.VIDEO,
            filetypes=[
                ("Video File", FileTypes.VIDEO),
                ("All files", "*.*"),
            ],
        )
        self.parent.select_video(filename=filename)

    def clear_lower_frame(self) -> None:
        """
        Clears the lower part of the frame of widgets, to redraw them.

        Used for changing modes
        """
        for widgets in self.display_frame_lower.winfo_children():
            widgets.destroy()

    def configure_mode(self, mode: GUIMode) -> None:
        """
        Changes mode between training and inference, changing the layout and widgets of the window
        """
        self.mode = mode
        if self.mode == GUIMode.TRAIN:
            self.clear_lower_frame()
            self.display_frame_lower = self.init_training_lower(
                frame=self.display_frame_lower
            )
        elif self.mode == GUIMode.INFERENCE:
            self.clear_lower_frame()
            self.display_frame_lower = self.init_inference_lower(
                frame=self.display_frame_lower
            )
        self.display_frame_lower.pack(side=tk.TOP, fill="both", expand=True)

    def init_training_lower(self, frame) -> tk.Frame:
        """
        Creates widgets for training and places them within the part of the window
        """
        top_frame = tk.Frame(frame, padx=10, pady=10)
        top_frame.pack(side=tk.TOP, fill="both", expand=True)

        # Progress bar for data collection
        self.progress = tk.IntVar()
        self.training_progress = ttk.Progressbar(
            top_frame, maximum=(len(self.current_classes)), variable=self.progress
        )
        self.training_progress.pack(side=tk.TOP, fill="both", expand=True)

        mid_frame = tk.Frame(frame, padx=10, pady=10)
        mid_frame.pack(side=tk.TOP, fill="both", expand=True)

        # Current class for data collection
        self.training_class: tk.StringVar() = tk.StringVar()
        training_class_label = tk.Label(
            mid_frame,
            text="Word to Say:",
            font=self.parent.TITLE_FONT,
        )
        training_class_label.grid(row=1, column=0, pady=2, padx=10)
        training_class = tk.Label(
            mid_frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.training_class,
            width=80,
            height=1,
        )
        training_class.grid(row=1, column=1, pady=2, padx=10)

        # Frame buffer
        self.create_frame_buffer_frame(frame=mid_frame, row=2)

        button_frame = tk.Frame(frame, padx=10, pady=10)
        button_frame.pack(side=tk.TOP, fill="both", expand=True)

        # Clear frame buffer button
        clear_words_button = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Clear Last Batch",
            command=self.parent.clear_frame_buffer,
        )
        clear_words_button.grid(row=0, column=0, pady=2, padx=15)

        # Begin data collection button
        begin_recording_btn = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Begin Recording",
            command=self.begin_recording,
        )
        begin_recording_btn.grid(row=0, column=1, pady=2, padx=15)

        # Save data batch. Will store internally the data in a batch ready for training later
        save_batch = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Save Data Batch",
            command=self.save_batch,
        )
        save_batch.grid(row=0, column=2, pady=2, padx=15)

        # Button to begin finetuning the model with the data collected so
        finetune_model = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Train Model",
            command=self.finetune_model,
        )
        finetune_model.grid(row=0, column=3, pady=2, padx=15)

        self.set_training_classes(classes=self.parent.get_current_classes())

        # Save model button. Will create a dialog for file location selection for the resulting file
        save_model = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Save Model",
            command=self.save_model,
        )
        save_model.grid(row=0, column=4, pady=2, padx=15)

        # Save model wrapped class so it can easily be imported back in
        save_model_wrapper = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Save Model Wrapper",
            command=self.save_model_wrapper,
        )
        save_model_wrapper.grid(row=0, column=5, pady=2, padx=15)

        return frame

    def set_training_classes(self, classes: typing.Dict[int, str]) -> None:
        """
        Will set the property of the current training classes. Sets the current training class as the first for the model selected
        """
        if classes is not None:
            self.current_classes: typing.List[str] = list(classes.values())
            self.training_progress.config(maximum=(len(self.current_classes)))
            self.set_current_training_class(current_class=0)
        else:
            self.current_classes: typing.List[str] = []
            self.training_progress.config(maximum=0)
            self.set_current_training_class(current_class=-1)
        self.set_current_training_class(current_class=self.current_class)

    def set_current_training_class(self, current_class: int) -> None:
        """
        Will set the current training class using the input integer as an index for the list of classes
        """
        self.current_class: int = current_class
        if current_class == -1:
            self.training_class.set(str(""))
        else:
            self.training_class.set(str(self.current_classes[current_class]))

    def init_inference_lower(self, frame) -> tk.Frame:
        """
        Creates widgets for inference and places them within the part of the window
        """
        current_pred_frame = tk.Frame(frame, padx=10, pady=5)
        current_pred_frame.pack(side=tk.TOP, fill="both", expand=True)

        self.current_prediction: tk.StringVar() = tk.StringVar()
        words_detected_label = tk.Label(
            current_pred_frame,
            text="Words Detected:",
            font=self.parent.TITLE_FONT,
        )
        words_detected_label.grid(row=0, column=0, pady=2, padx=10)
        words_detected = tk.Label(
            current_pred_frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.current_prediction,
            width=80,
            height=1,
        )
        words_detected.grid(row=0, column=1, pady=2, padx=10)

        self.mouth_state: tk.StringVar() = tk.StringVar()
        mouth_state_label = tk.Label(
            current_pred_frame,
            text="Mouth State:",
            font=self.parent.TITLE_FONT,
        )
        mouth_state_label.grid(row=1, column=0, pady=2, padx=10)
        mouth_state = tk.Label(
            current_pred_frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.mouth_state,
            width=80,
            height=1,
        )
        mouth_state.grid(row=1, column=1, pady=2, padx=10)

        self.all_predictions: tk.StringVar() = tk.StringVar()
        all_predictions_label = tk.Label(
            current_pred_frame,
            text="All Predictions:",
            font=self.parent.TITLE_FONT,
        )
        all_predictions_label.grid(row=2, column=0, pady=2, padx=10)
        all_predictions = tk.Label(
            current_pred_frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.all_predictions,
            width=80,
            height=1,
        )
        all_predictions.grid(row=2, column=1, pady=2, padx=10)

        self.create_frame_buffer_frame(frame=current_pred_frame, row=3)

        all_pred_frame = tk.Frame(frame, padx=10, pady=5)
        all_pred_frame.pack(side=tk.TOP, fill="both", expand=True)

        self.all_words_detected = tk.Text(
            all_pred_frame,
            font=self.parent.NORMAL_FONT,
            width=40,
            height=1,
            wrap=tk.WORD,
        )
        scrollbar = tk.Scrollbar(all_pred_frame, command=self.all_words_detected.yview)
        self.all_words_detected.config(yscrollcommand=scrollbar.set)
        self.all_words_detected.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = tk.Frame(frame, padx=10, pady=5)
        button_frame.pack(side=tk.TOP, fill="both", expand=True)

        clear_words_button = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Clear Words Detected",
            command=self.clear_all_words_detected,
        )
        clear_words_button.pack(side=tk.LEFT, expand=True, fill=tk.Y)

        clear_buffer_button = tk.Button(
            button_frame,
            font=self.parent.NORMAL_FONT,
            text="Clear Frame Buffer",
            command=self.parent.clear_frame_buffer,
        )
        clear_buffer_button.pack(side=tk.LEFT, expand=True, fill=tk.Y)

        return frame

    def clear_all_words_detected(self) -> None:
        """
        Will delete all content from the text area self.all_words_detected
        """
        self.all_words_detected.delete("1.0", tk.END)
        self.current_prediction.set("")
        self.all_predictions.set("")
        self.mouth_state.set("")

    def create_frame_buffer_frame(self, frame, row: int) -> None:
        """
        Creates a frame buffer section. This shows the frame buffer label and the label for the frame buffer length
        """
        self.frame_buffer_length: tk.StringVar() = tk.StringVar()
        frame_buffer_length_label = tk.Label(
            frame,
            text="Frame Buffer Length:",
            font=self.parent.TITLE_FONT,
        )
        frame_buffer_length_label.grid(row=row, column=0, pady=2, padx=10)
        frame_buffer_length = tk.Label(
            frame,
            font=self.parent.NORMAL_FONT,
            textvariable=self.frame_buffer_length,
        )
        frame_buffer_length.grid(row=row, column=1, pady=2, padx=10)

    def finetune_model(self) -> None:
        """
        Will finetune the model using the data collected
        """
        if not self.parent.model_selected():
            self.parent.output(text="No model was selected!", colour="red")
            return
        if len(self.data_batches) == 0 or len(self.data_labels) == 0:
            self.parent.output(text="No data has been collected!", colour="red")
            return

        return_message = self.parent.model_options[self.parent.current_model].finetune(
            data_labels=self.data_labels,
            data_batches=self.data_batches,
            epochs=self.parent.epochs,
            batch_size=self.parent.batch_size,
            learning_rate=self.parent.get_learning_rate(),
        )
        if "Error" in return_message:
            message_colour = "red"
            duration = 2000
        else:
            message_colour = "green"
            duration = -1

            # Only clear data if training was successful
            self.progress.set(0)
            self.data_batches = []
            self.data_labels = []
        self.parent.output(
            text=return_message, colour=message_colour, duration=duration
        )

    def begin_recording(self) -> None:
        """
        Will begin recording frames to use to finetune the model
        """
        if not self.parent.model_selected():
            self.parent.output(text="No model was selected!", colour="red")
            self.recording = False
            return
        self.recording = True

    def save_model_wrapper(self) -> None:
        """
        Will save the model wrapped class for easy loading back in
        """
        if not self.parent.model_selected():
            self.parent.output(text="No model was selected!", colour="red")
            return
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=FileTypes.OBJECT,
            filetypes=[
                ("Model Wrapper File", FileTypes.OBJECT),
                ("All files", "*.*"),
            ],
        )
        try:
            if FileTypes.OBJECT in filename:
                filehandler = open(filename, "wb")
                pickle.dump(
                    self.parent.model_options[self.parent.current_model], filehandler
                )
            else:
                self.parent.output(
                    text="Unknown file extension. Try again", colour="red"
                )
        except Exception as e:
            self.parent.output(
                text=f"Error whilst saving model wrapped. {e}", colour="red"
            )

    def save_model(self) -> None:
        """
        Will open a window and save the model where it is specified. Returns success
        """
        if not self.parent.model_selected():
            self.parent.output(text="No model was selected!", colour="red")
            return
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=FileTypes.HDF5,
            filetypes=[
                ("Model File", FileTypes.HDF5),
                ("Keras File", FileTypes.KERAS),
                ("All files", "*.*"),
            ],
        )
        try:
            if FileTypes.HDF5 in filename:
                self.parent.get_current_model().save_weights(filename)
            elif FileTypes.KERAS in filename:
                self.parent.get_current_model().save(filename)
            else:
                self.parent.output(
                    text="Unknown file extension. Try again", colour="red"
                )
        except Exception as e:
            self.parent.output(
                text=f"Error whilst saving model weights. {e}", colour="red"
            )

    def save_batch(self) -> None:
        """
        Will save the data from the frame buffer, ready for finetuneing

        Returns true/false of whether any training was done
        """
        if (
            not self.parent.model_selected()
            or not self.parent.model_options[
                self.parent.current_model
            ].frame_buffer_full()
        ):
            self.parent.output(text="No model was selected!", colour="red")
            return

        # Grabs the current recording for the batch and adds it to the collection
        recording = self.parent.model_options[
            self.parent.current_model
        ].get_frame_buffer_data()
        self.data_batches.extend(recording)
        self.parent.output(
            text=f"Data received. Currently saved {len(self.data_batches)} data samples.",
            colour="green",
            duration=2000,
        )

        # Gets the label for this and adds it to the collection
        label = self.parent.model_options[self.parent.current_model].one_hot_encode(
            index=self.current_class
        )
        self.data_labels.extend(label)

        # Updates the progress bar and moves to the next class for data collection. Cycles through these classes to get data for each
        self.progress.set(self.current_class + 1)
        if self.current_class + 1 >= len(self.current_classes):
            self.set_current_training_class(current_class=0)
        else:
            self.set_current_training_class(current_class=(self.current_class + 1))
        self.parent.clear_frame_buffer()  # Resets the frame buffer ready for more data collection
