import typing
from tensorflow import keras
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras.optimizers import Adam

from utility.Dataclasses import Region, FeatureExtractor, FileTypes, PredictionMethod
from utility.Utilities import load_json, remove_file_types, pad_image
from ConvertDataset import padded_lip_landmarks

lip_inner_top_landmarks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
lip_inner_bottom_landmarks = [11, 12, 13, 14, 15, 16, 17, 18, 19]


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


class ModelWrapper:
    """
    Class will load details needed for a model. Loads a model, sometimes a feature extractor based on input parameters

    Parameters:
        model_file_name:            Filename to the model being used
        word_dict:                  Translation for the word classes. Form of {index: word_class}
        feature_extractor_type:     Whether to use a feature extractor & what type
        model_type:                 Type of model to load in. Used to see if we need to load the model json architecture aswell as the weights
        prediction_method:          Prediction method used to deal with the output from the model
        frame_pad:                  External, between frame padding. The number of frames that the video classification deals with
        padding:                    Internal, inner data padding. Size of the data to be used
        region:                     Region that the model uses. Eg, lip image region or lip landmarks. Type of input data
        probability_threshold:      Probability threshold used on the output of the model. If all are below this we mark the prediction as unknown
        lip_height_threshold:       Lip height threshold, used to calculate when the lips are open
        closed_mouth_threshold:     Threshold for the number of frames in a row that can be closed before the frame buffer is emptied. -1 will cause this feature to not be considered
    """

    UNKNOWN_KEY: str = "UNKNOWN"

    DEFAULT_EPOCHS: int = 1
    DEFAULT_LR: float = 0.0001
    DEFAULT_BATCH_SIZE: int = 1

    def __init__(
        self,
        model_file_name: str,
        word_dict: typing.Dict[int, str] = {},
        feature_extractor_type: FeatureExtractor = FeatureExtractor.NONE_KEY,
        model_type: FileTypes = FileTypes.KERAS,
        prediction_method: PredictionMethod = PredictionMethod.BINARY_KEY,
        frame_pad: int = 0,
        padding: int = 100,
        region: Region = Region.MOUTH,
        probability_threshold: float = 0.0,
        lip_height_threshold: float = 0.0,
        closed_mouth_threshold: int = -1,
    ):
        self.model_file_name: str = model_file_name
        self.feature_extractor_type: FeatureExtractor = feature_extractor_type
        self.model_type: FileTypes = model_type
        self.prediction_method: PredictionMethod = prediction_method
        self.frame_pad: int = frame_pad
        self.padding: int = padding
        self.region: Region = region
        self.probability_threshold: float = probability_threshold
        self.lip_height_threshold: float = lip_height_threshold
        self.closed_mouth_threshold: int = closed_mouth_threshold

        self.feature_extractor = self.build_feature_extractor(
            img_shape=self.padding, feature_extractor_type=self.feature_extractor_type
        )

        self.model = self.load_model(
            model_type=self.model_type, model_file_name=self.model_file_name
        )

        self.word_dict: typing.Dict[int, str] = word_dict

        self.reset_frame_buffer()
        self.lip_state: bool = False
        self.num_closed: int = 0

        self.DEFAULT_ALL_PREDICTIONS: typing.Dict = [
            round(0.0, 4) for _ in range(len(self.word_dict.keys()))
        ]
        self.last_predictions: typing.Dict = self.DEFAULT_ALL_PREDICTIONS

    def convert_to_feature_vectors(self, data: np.ndarray):
        """
        Will convert data to feature vectors
        """
        return self.feature_extractor.predict_on_batch(np.expand_dims(data, axis=0))[0]

    def load_model(
        self, model_type: FileTypes, model_file_name: str
    ) -> keras.Sequential:
        """
        Will use the model type to load the correct model
        """
        if self.model_type == FileTypes.KERAS:
            if FileTypes.KERAS not in model_file_name:
                model_file_name += FileTypes.KERAS
            model = keras.models.load_model(model_file_name)
        elif self.model_type == FileTypes.HDF5:
            model_file_name = remove_file_types(path=model_file_name)
            model_json = load_json(filename=model_file_name)

            model = keras.models.model_from_json(model_json)
            model.load_weights((model_file_name + FileTypes.HDF5))
        else:
            print(
                f"Unable to load the model specified: {model_type + model_file_name} as the model type was not an expected one. Model type should be one of [{FileTypes.KERAS}, {FileTypes.HDF5}]"
            )
            model = None
        return model

    def use_model(self, data: typing.List[np.ndarray]) -> typing.List[float]:
        """
        Will use the model on the keypoint list passed in, returns the prediction probabilities
        """
        data = self.format_frame_buffer_data(data=data)
        predictions = self.model.predict(data)
        return predictions

    def format_frame_buffer_data(
        self, data: typing.List[np.ndarray]
    ) -> typing.List[float]:
        """
        Will reformat the data from the frame buffer to be used for inference or training
        """
        if self.region == Region.MOUTH_AND_LANDMARKS:
            img_data = []
            ldmk_data = []
            for frame in data:
                img_data.append(frame[0])
                ldmk_data.append(frame[1])
            img_data = np.array(img_data)
            ldmk_data = np.array(ldmk_data)

            img_data = tf.expand_dims(img_data, 0)
            ldmk_data = tf.expand_dims(ldmk_data, 0)

            return [img_data, ldmk_data]
        else:
            return np.expand_dims(np.array(data), axis=0)

    def build_feature_extractor(
        self, feature_extractor_type: FeatureExtractor, img_shape: int = 100
    ):
        if feature_extractor_type == FeatureExtractor.INCEPTION_IMAGENET_KEY:
            feature_extractor = keras.applications.InceptionV3(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(img_shape, img_shape, 3),
            )
            preprocess_input = keras.applications.inception_v3.preprocess_input

            inputs = keras.Input((img_shape, img_shape, 3))
            preprocessed = preprocess_input(inputs)

            outputs = feature_extractor(preprocessed)
            return keras.Model(inputs, outputs, name="feature_extractor")
        else:
            return None

    def reset_frame_buffer(self) -> None:
        """
        Will set the frame buffer back to be an empty list
        """
        self.frame_buffer = []

    def pass_frame(self, frame: np.ndarray) -> None:
        """
        Passes a frame to the model wrapper to store. Model wrapper will either turn this into feature vectors or normally just add to the frame buffer ready to make predictions. Will pad data too
        """
        data = frame

        if self.padding != -1:
            if self.region == Region.LIPS or self.region == Region.MOUTH:
                data = pad_image(img=data, padding=self.padding)
            elif self.region == Region.MOUTH_AND_LANDMARKS:
                data[0] = np.array(pad_image(img=data[0], padding=self.padding))

        if (
            self.feature_extractor_type != FeatureExtractor.NONE_KEY
            and self.region != Region.LANDMARKS
        ):
            if self.region == Region.MOUTH_AND_LANDMARKS:
                data[0] = np.array(self.convert_to_feature_vectors(data=data[0]))
            else:
                data = self.convert_to_feature_vectors(data=data)

        self.frame_buffer.append(data)

        if self.region == Region.LANDMARKS:
            self.lip_state: bool = self.get_lip_state(
                landmarks=data, height_threshold=self.lip_height_threshold
            )
            if self.closed_mouth_threshold != -1:
                self.use_lip_state(buffer=self.frame_buffer, lip_state=self.lip_state)
        elif self.region == Region.MOUTH_AND_LANDMARKS:
            self.lip_state: bool = self.get_lip_state(
                landmarks=data[1], height_threshold=self.lip_height_threshold
            )
            if self.closed_mouth_threshold != -1:
                self.use_lip_state(buffer=self.frame_buffer, lip_state=self.lip_state)

    def use_lip_state(self, buffer: list, lip_state: bool) -> None:
        """
        Will check the number of closed lip frames in a row. If this is above the threshold then it will clear the buffer
        """
        if not lip_state:
            self.num_closed += 1
        else:
            self.num_closed = 0
            return buffer
        if self.closed_mouth_threshold == 0:
            self.num_closed = 0
            return

        if self.num_closed >= self.closed_mouth_threshold:
            self.reset_frame_buffer()

    def get_frame_buffer_length(self) -> int:
        return len(self.frame_buffer)

    def predict(self) -> bool:
        """
        Will use data from the frame buffer to make a prediction. Returns boolean of whether the prediction was made
        """
        if self.closed_mouth_threshold > 0:
            if self.num_closed >= self.closed_mouth_threshold:
                return False

        # Will skip making predictions until we have enough frames
        if len(self.frame_buffer) != self.frame_pad:
            return False

        predictions = self.DEFAULT_ALL_PREDICTIONS
        predictions = self.use_model(data=self.frame_buffer)
        self.reset_frame_buffer()
        self.last_predictions = predictions
        return True

    def get_single_result(self) -> str:
        """
        Will use the model to get a prediction. Will return a single result, the best class give the prediction method
        """
        predictions = self.last_predictions

        if self.prediction_method == PredictionMethod.BINARY_KEY:
            # For binary usese the threshold of 0.5 to distinguish between the two classes
            score = int(predictions[0][0] > 0.5)
        elif self.prediction_method == PredictionMethod.ARGMAX_KEY:
            threshold = True
            predictions = predictions[0]
            for prob in predictions:
                if prob > self.probability_threshold:
                    threshold = False
                    break
            if threshold:
                return self.UNKNOWN_KEY
            score = int(np.argmax(predictions))
        elif self.prediction_method == PredictionMethod.CTC_KEY:
            output = self.ctc_predict(predictions=predictions)
            return output[0]
        else:
            print("No prediction method was specified!")
            score = predictions[0][0]
        pred_class = self.word_dict[score]
        self.reset_frame_buffer()
        return pred_class

    def ctc_predict(self, predictions) -> typing.List[str]:
        """
        Will process the output of a model trained with CTC loss. Uses keras' ctc decoder
        """
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        results = keras.backend.ctc_decode(
            predictions, input_length=input_len, greedy=True
        )[0][0]
        output_text = []
        for result in results:
            word = ""
            for num in result.numpy():
                word += self.word_dict[num]
            result = tf.strings.reduce_join(word).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    def get_all_result(self) -> typing.Dict[str, float]:
        """
        Will use the model to probabilities for each class, will return the classes with the associated probabiltiies
        """
        if self.prediction_method == PredictionMethod.CTC_KEY:
            return [""]
        predictions = self.last_predictions[0]
        self.reset_frame_buffer()
        return {
            self.word_dict[i]: round(predictions[i], 4)
            for i in list(self.word_dict.keys())
        }

    def get_lip_state(self, landmarks: np.ndarray, height_threshold: float) -> bool:
        """
        Returns the lip state: True if open, False if closed
        """
        return self.calc_diff_lip_landmarks(landmarks=landmarks) >= height_threshold

    def calc_diff_lip_landmarks(self, landmarks) -> float:
        """
        Gets the mean height difference between corresponding lip landmarks
        """
        height_vector = np.abs(
            landmarks[lip_inner_top_landmarks].T[1]
            - landmarks[lip_inner_bottom_landmarks].T[1]
        )
        return np.mean(a=height_vector)

    def frame_buffer_full(self) -> bool:
        """
        Returns whether the frame buffer is empty
        """
        return len(self.frame_buffer) == self.frame_pad

    def get_frame_buffer_data(self) -> np.array:
        return self.format_frame_buffer_data(data=self.frame_buffer)

    def one_hot_encode(self, index: int) -> np.array:
        """
        Takes an index and using the word dictionary will return a one-hot encoding representing the selected class
        """
        Y = np.zeros((1, len(self.word_dict)))
        Y[0][index] = 1
        return Y

    def finetune(
        self,
        data_labels: np.array,
        data_batches: np.array,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> str:
        """
        Will finetune the currently selected model, using the label passed in and the data currently in the frame buffer

        Returns string of the accuracy and loss of the training, or an error message
        """
        # Set batch size
        if len(data_labels) % batch_size != 0:
            return "Error: Batch size does not match the amount of data"

        # Select a loss metric based on the prediction method for the model being finetuned
        CATEGORICAL_CE_KEY: str = "categorical_crossentropy"
        if self.prediction_method == PredictionMethod.ARGMAX_KEY:
            loss = CATEGORICAL_CE_KEY
        elif self.prediction_method == PredictionMethod.CTC_KEY:
            loss = CTCLoss
        else:
            loss = CATEGORICAL_CE_KEY

        # Setup the model for training
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"]
        )

        # Batch together data for training
        if self.region == Region.MOUTH_AND_LANDMARKS:
            # Unpack data if it is a model with multiple inputs
            img_data, ldmk_data = [], []
            for i in range(len(data_batches)):
                if i % 2 == 0:
                    img_data.extend(data_batches[i])
                else:
                    ldmk_data.extend(data_batches[i])
            train_ds = tf.data.Dataset.from_tensor_slices(
                ((img_data, ldmk_data), data_labels)
            )
            train_ds = train_ds.padded_batch(
                batch_size,
                padded_shapes=(
                    (np.shape(img_data[0]), np.shape(ldmk_data[0])),
                    np.shape(data_labels[0]),
                ),
            )
        else:
            train_ds = tf.data.Dataset.from_tensor_slices((data_batches, data_labels))
            train_ds = train_ds.padded_batch(
                batch_size,
                padded_shapes=(np.shape(data_batches[0]), np.shape(data_labels[0])),
            )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Train
        history = self.model.fit(train_ds, epochs=epochs)
        #   Gets the accuracy & loss metrics from training to return this for outputting
        return f"Training Complete  Accuracy: {history.history.get('accuracy')[-1]}    Loss: {history.history.get('loss')[-1]}"
