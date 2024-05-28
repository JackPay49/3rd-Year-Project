# Will create images of the various different models, given an experiment number and a path to export to
import keras
import tensorflow as tf
import numpy as np
import sys
import visualkeras

from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.utils import plot_model

landmark_shape = (21, 40, 3)
visual_shape = (21, 2048)
old_visual_shape = (29, 2048)

model_num = float(sys.argv[1])
out_path = sys.argv[2]


def make_model(model_num):
    """
    Creates the base model based on the experiment number
    """
    if model_num < 9:
        num_classes = 4
    elif model_num == 9.1:
        num_classes = 15
    elif model_num == 9.2:
        num_classes = 15
    elif model_num == 9.3:
        num_classes = 10

    if model_num == 1:
        model = Sequential(
            [
                layers.InputLayer(input_shape=old_visual_shape, name="Input"),
                layers.Conv1D(
                    256, 3, padding="same", activation="relu", name="Convolution_1"
                ),
                layers.Conv1D(
                    128, 3, padding="same", activation="relu", name="Convolution_2"
                ),
                layers.LSTM(
                    128, activation="relu", return_sequences=True, name="LSTM_1"
                ),
                layers.LSTM(
                    128, activation="relu", return_sequences=True, name="LSTM_2"
                ),
                layers.LSTM(
                    128, activation="relu", return_sequences=True, name="LSTM_3"
                ),
                layers.Dense(64, activation="relu", name="Dense_1"),
                layers.Dropout(0.2, name="Dropout_1"),
                layers.Dense(32, activation="relu", name="Dense_2"),
                layers.Dropout(0.2, name="Dropout_2"),
                layers.Dense(16, activation="relu", name="Dense_3"),
                layers.Flatten(name="Flatten"),
                layers.Dense(
                    num_classes, activation="softmax", name="Prediction_Layer"
                ),
            ]
        )
    if model_num == 2:
        model = Sequential(
            [
                layers.InputLayer(input_shape=old_visual_shape, name="Input"),
                layers.Conv1D(
                    256, 3, padding="same", activation="relu", name="Convolution_1"
                ),
                layers.Conv1D(
                    128, 3, padding="same", activation="relu", name="Convolution_2"
                ),
                layers.LSTM(
                    128, activation="relu", return_sequences=True, name="LSTM_1"
                ),
                layers.LSTM(
                    128, activation="relu", return_sequences=True, name="LSTM_2"
                ),
                layers.LSTM(
                    128, activation="relu", return_sequences=False, name="LSTM_3"
                ),
                layers.Dense(64, activation="relu", name="Dense_1"),
                layers.Dropout(0.2, name="Dropout_1"),
                layers.Dense(32, activation="relu", name="Dense_2"),
                layers.Dropout(0.2, name="Dropout_2"),
                layers.Dense(16, activation="relu", name="Dense_3"),
                layers.Dense(
                    num_classes, activation="softmax", name="Prediction_Layer"
                ),
            ]
        )
    elif 3 <= model_num <= 6:
        if model_num == 3 or model_num == 4:
            input_shape = visual_shape
        else:
            input_shape = landmark_shape
        model = Sequential(
            [
                layers.InputLayer(input_shape=input_shape, name="Input"),
                layers.TimeDistributed(layers.Flatten(), name="Flatten"),
                layers.Bidirectional(
                    layers.LSTM(128, activation="relu", return_sequences=True),
                    name="Bi-LSTM_1",
                ),
                layers.Dropout(0.2, name="Dropout_1"),
                layers.Bidirectional(
                    layers.LSTM(128, activation="relu", return_sequences=True),
                    name="Bi-LSTM_2",
                ),
                layers.Dropout(0.2, name="Dropout_2"),
                layers.Bidirectional(
                    layers.LSTM(128, activation="relu", return_sequences=False),
                    name="Bi-LSTM_3",
                ),
                layers.Dense(64, activation="relu", name="Dense_1"),
                layers.Dropout(0.2, name="Dropout_3"),
                layers.Dense(32, activation="relu", name="Dense_2"),
                layers.Dropout(0.2, name="Dropout_4"),
                layers.Dense(16, activation="relu", name="Dense_3"),
                layers.Dropout(0.2, name="Dropout_5"),
                layers.Dense(
                    num_classes, activation="softmax", name="Prediction_Layer"
                ),
            ]
        )
    elif model_num == 7:
        sequence_length = landmark_shape[0]
        d_model = 120
        num_encoder_layers = 3
        num_heads = 3
        dff = 120
        epsilon = 1e-6

        input_layer = layers.Input(shape=landmark_shape, name="Input")
        flatten_layer1 = layers.TimeDistributed(layers.Flatten(), name="Flatten_1")(
            input_layer
        )

        positional_encoding = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=d_model, name="Positional_Encoding"
        )(tf.range(start=0, limit=sequence_length, delta=1))

        add_layer = flatten_layer1 + positional_encoding

        for i in range(num_encoder_layers):
            x = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model,
                name=f"Multi-Head_Attention_{(i+1)}",
            )(add_layer, add_layer)
            x = layers.LayerNormalization(epsilon=epsilon, name=f"AddNorm_{(i+1)}a")(
                x + add_layer
            )

            ffn = keras.Sequential(
                [layers.Dense(dff, activation="relu"), layers.Dense(d_model)],
                name=f"Feed-forward_{(i+1)}",
            )
            y = ffn(x)
            add_layer = layers.LayerNormalization(
                epsilon=epsilon, name=f"AddNorm_{(i+1)}b"
            )(y + x)

        flatten_layer2 = layers.Flatten(name="Flatten_2")(x)
        output_layer = layers.Dense(
            num_classes, activation="softmax", name="Prediction_Layer"
        )(flatten_layer2)

        model = keras.Model(input_layer, output_layer)
    elif model_num >= 8:
        epsilon = 1e-6

        input_layer_frames = layers.Input(shape=visual_shape, name="Input_Visual")
        flatten_layer_frames = layers.TimeDistributed(
            layers.Flatten(), name="Flatten_Visual"
        )(input_layer_frames)
        attention_layer_frames = layers.Attention(
            use_scale=True, name="Self-Attention_Visual"
        )([flatten_layer_frames, flatten_layer_frames])
        norm_layer_frames = layers.LayerNormalization(
            epsilon=epsilon, name="AddNorm_Visual"
        )(flatten_layer_frames + attention_layer_frames)

        input_layer_landmarks = layers.Input(
            shape=landmark_shape, name="Input_Landmarks"
        )
        flatten_layer_landmarks = layers.TimeDistributed(
            layers.Flatten(), name="Flatten_Landmarks"
        )(input_layer_landmarks)
        dense_scale_layer = layers.Dense(visual_shape[-1], name="Scaling_Layer")(
            flatten_layer_landmarks
        )
        attention_layer_landmarks = layers.Attention(
            use_scale=True, name="Self-Attention_Landmarks"
        )([dense_scale_layer, dense_scale_layer])
        norm_layer_landmarks = layers.LayerNormalization(
            epsilon=epsilon, name="AddNorm_Landmarks"
        )(dense_scale_layer + attention_layer_landmarks)

        cross_attention_layer1 = layers.Attention(
            use_scale=True, name="Cross-Attention_a"
        )([norm_layer_landmarks, norm_layer_frames])
        cross_norm_layer1 = layers.LayerNormalization(
            epsilon=epsilon, name="Cross_AddNorm_a"
        )(cross_attention_layer1 + norm_layer_landmarks + norm_layer_frames)

        cross_attention_layer2 = layers.Attention(
            use_scale=True, name="Cross-Attention_b"
        )([norm_layer_frames, norm_layer_landmarks])
        cross_norm_layer2 = layers.LayerNormalization(
            epsilon=epsilon, name="Cross_AddNorm_b"
        )(cross_attention_layer2 + norm_layer_landmarks + norm_layer_frames)

        concatenate_layer = layers.Concatenate(axis=-1, name="Concatenate")(
            [cross_norm_layer1, cross_norm_layer2]
        )

        conv_layer1 = layers.Conv1D(
            1024, 3, padding="same", activation="relu", name="Convolution_1"
        )(concatenate_layer)
        dropout1 = layers.Dropout(0.1, name="Dropout_1")(conv_layer1)
        max_pooling1 = layers.MaxPooling1D(name="Max-Pool_1")(dropout1)

        conv_layer2 = layers.Conv1D(
            256, 3, padding="same", activation="relu", name="Convolution_2"
        )(max_pooling1)
        dropout2 = layers.Dropout(0.1, name="Dropout_2")(conv_layer2)
        max_pooling2 = layers.MaxPooling1D(name="Max-Pool_2")(dropout2)

        conv_layer3 = layers.Conv1D(
            64, 3, padding="same", activation="relu", name="Convolution_3"
        )(max_pooling2)
        dropout3 = layers.Dropout(0.1, name="Dropout_3")(conv_layer3)
        max_pooling3 = layers.MaxPooling1D(name="Max-Pool_3")(dropout3)

        conv_layer4 = layers.Conv1D(
            32, 3, padding="same", activation="relu", name="Convolution_4"
        )(max_pooling3)
        max_pooling4 = layers.MaxPooling1D(name="Max-Pool_4")(conv_layer4)

        flatten_layer_final = layers.Flatten(name="Flatten_Final")(max_pooling4)

        output_layer = layers.Dense(
            num_classes, activation="softmax", name="Prediction_Layer"
        )(flatten_layer_final)

        model = Model(
            inputs=[input_layer_frames, input_layer_landmarks], outputs=output_layer
        )
    return model


model = make_model(model_num=model_num)
plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    to_file="models/2nd experiment/2 architecture.png",
)