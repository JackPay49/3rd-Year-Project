import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.utils import plot_model
import typing
import pickle

from utility.Dataclasses import (
    FileTypes,
)


class ControlFrameUpper:
    """
    Creates the upper control frame. This is the frame responsible for displaying the available models & swapping between them

    Parameters:
        frame:                  Frame to place the current frame within
        parent:                 Parent class, used to access items such as the current model
    Properties:
        tabControl:             Used to control the tabbing system for the different models. Holds the tabs
    """

    MODEL_WINDOW_WIDTH_REL: float = 0.75
    MODEL_WINDOW_HEIGHT_REL: float = 0.75

    MODEL_DIAGRAM_NAME = ".model.png"

    def __init__(self, frame, parent) -> None:
        """
        Establishes the upper part of the control frame. Creates the tabbed unit & adds the content for each tab
        """
        self.frame = frame
        self.parent = parent

        # Changes the style of the tabbed interface to be vertical, rather than horizontal
        s1 = ttk.Style()
        s1.configure("TNotebook", tabposition="w")
        s1.configure("TNotebook.Tab", padding=[15, 5], font=self.parent.NORMAL_FONT)

        # self.tabControl = ttk.Notebook(self.frame)
        self.tabControl = ttk.Notebook(self.frame, style="TNotebook")

        # Adds tabs for each of the different models, to swap between them
        self.tabs: typing.Dict = {}
        for x in self.parent.model_options:
            # Tab frame
            tab = ttk.Frame(self.tabControl, style="TNotebook.Tab")

            if x == self.parent.USER_SPECIFIED_KEY:
                # Model wrapped file
                model_file = tk.StringVar()

                def select_model_wrapper_file():
                    filename = tk.filedialog.askopenfilename(
                        defaultextension=FileTypes.OBJECT,
                        filetypes=[
                            ("Model Wrapper File", FileTypes.OBJECT),
                            ("All files", "*.*"),
                        ],
                    )
                    model_file.set(filename)
                    model_wrapper = None
                    with open(filename, "rb") as f:
                        model_wrapper = pickle.load(f)
                    self.parent.model_options[self.parent.current_model] = model_wrapper
                    feature_extractor_str.set(model_wrapper.feature_extractor_type)
                    prediction_method_str.set(model_wrapper.prediction_method)
                    frame_num_str.set(model_wrapper.frame_pad)

                model_wrapper_file_btn = tk.Button(
                    tab,
                    font=self.parent.NORMAL_FONT,
                    text="Select Model Wrapper",
                    command=select_model_wrapper_file,
                )
                model_wrapper_file_btn.grid(row=0, column=0, pady=2, padx=10)
                model_wrapper_file_entry = tk.Entry(
                    tab,
                    textvariable=model_file,
                    font=self.parent.NORMAL_FONT,
                    state=tk.DISABLED,
                    disabledforeground="black",
                    disabledbackground="white",
                )
                model_wrapper_file_entry.grid(row=0, column=1, pady=2, padx=10)
            elif x != self.parent.NO_MODEL_KEY:
                # Filename information
                filename_label = tk.Label(
                    tab, text="Model Filename:", font=self.parent.TITLE_FONT
                )
                filename_label.grid(row=0, column=0, pady=2, padx=10)
                filename_str = tk.StringVar()
                filename_str.set(self.parent.model_options[x].model_file_name)
                filename_entry = tk.Entry(
                    tab,
                    textvariable=filename_str,
                    font=self.parent.NORMAL_FONT,
                    state=tk.DISABLED,
                    disabledforeground="black",
                    disabledbackground="white",
                )
                filename_entry.grid(row=0, column=1, pady=2, padx=10)
            if x != self.parent.NO_MODEL_KEY:
                # Feature extractor information
                feature_extractor_label = tk.Label(
                    tab, text="Feature Extractor:", font=self.parent.TITLE_FONT
                )
                feature_extractor_label.grid(row=1, column=0, pady=2, padx=10)
                feature_extractor_str = tk.StringVar()
                if self.parent.MODEL_OPTIONS[x] != None:
                    feature_extractor_str.set(
                        self.parent.model_options[x].feature_extractor_type
                    )
                feature_extractor_entry = tk.Entry(
                    tab,
                    textvariable=feature_extractor_str,
                    font=self.parent.NORMAL_FONT,
                    state=tk.DISABLED,
                    disabledforeground="black",
                    disabledbackground="white",
                )
                feature_extractor_entry.grid(row=1, column=1, pady=2, padx=10)

                # Prediction method information
                prediction_method_label = tk.Label(
                    tab, text="Prediction Method:", font=self.parent.TITLE_FONT
                )
                prediction_method_label.grid(row=2, column=0, pady=2, padx=10)
                prediction_method_str = tk.StringVar()
                if self.parent.MODEL_OPTIONS[x] != None:
                    prediction_method_str.set(
                        self.parent.model_options[x].prediction_method
                    )
                prediction_method_entry = tk.Entry(
                    tab,
                    textvariable=prediction_method_str,
                    font=self.parent.NORMAL_FONT,
                    state=tk.DISABLED,
                    disabledforeground="black",
                    disabledbackground="white",
                )
                prediction_method_entry.grid(row=2, column=1, pady=2, padx=10)

                # Number of frames (required for each prediction) information
                frame_num_label = tk.Label(
                    tab, text="Number of Frames:", font=self.parent.TITLE_FONT
                )
                frame_num_label.grid(row=3, column=0, pady=2, padx=10)
                frame_num_str = tk.StringVar()
                if self.parent.MODEL_OPTIONS[x] != None:
                    frame_num_str.set(str(self.parent.model_options[x].frame_pad))
                frame_num_entry = tk.Entry(
                    tab,
                    textvariable=frame_num_str,
                    font=self.parent.NORMAL_FONT,
                    state=tk.DISABLED,
                    disabledforeground="black",
                    disabledbackground="white",
                )
                frame_num_entry.grid(row=3, column=1, pady=2, padx=10)

                # Show model architecture button
                show_architecture = tk.Button(
                    tab,
                    font=self.parent.NORMAL_FONT,
                    text="Show Model Architecture",
                    command=self.show_model_architecture,
                )
                show_architecture.grid(row=4, column=0, pady=2, padx=10)

            # Operation to swap model based on the selected tab
            ##Event to store what tab we are currently on: what model we are on
            def set_current_model(event) -> None:
                """
                Sets the current model string as the newly selected model
                """
                current_tab = event.widget.tab("current")["text"]
                self.parent.set_current_model(new_model=current_tab)

            self.tabControl.add(tab, text=x)
            self.tabControl.pack(expand=1, fill="both")
            self.tabControl.bind("<<NotebookTabChanged>>", set_current_model)

            self.tabs[x] = tab

    def show_model_architecture(self):
        """
        Will create a new window, showing the model architecture of the currently selected model
        """
        model_architecture_window = tk.Tk()
        model_architecture_window.title("Model Architecture")
        model_architecture_window.bind(
            "<Escape>", lambda _: model_architecture_window.quit()
        )
        model_architecture_window.bind("q", lambda _: model_architecture_window.quit())

        frame = tk.Frame(model_architecture_window)
        frame.pack(fill="both", expand=True, side=tk.BOTTOM)

        # Keras will create a model architecture diagram, saving this
        model = plot_model(
            self.parent.get_current_model(),
            show_shapes=True,
            show_layer_names=True,
            to_file=self.MODEL_DIAGRAM_NAME,
        )
        # If the model was None then return, not showing the model architecture
        if model is None:
            return

        # Resizes image so that it is within the window size
        img = Image.open(self.MODEL_DIAGRAM_NAME)
        img = img.resize(
            (
                int(
                    (model_architecture_window.winfo_screenheight() / img.height)
                    * img.width
                    * self.MODEL_WINDOW_WIDTH_REL
                ),
                int(
                    model_architecture_window.winfo_screenheight()
                    * self.MODEL_WINDOW_HEIGHT_REL
                ),
            )
        )

        # Display image in the new window
        model_diagram = ImageTk.PhotoImage(image=img, master=model_architecture_window)
        model_show: tk.Label = tk.Label(
            frame,
            image=model_diagram,
        )
        model_show.pack(fill="none", expand=True, side=tk.LEFT)
        model_show.image = model_diagram
