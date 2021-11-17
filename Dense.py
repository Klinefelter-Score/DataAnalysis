from tensorflow.keras import Model
from tensorflow.keras import layers


def get_builder(raw_input_shape):
    def _build():
        # create model
        # Define the input as a tensor with shape input_shape
        X_input = layers.Input(raw_input_shape, name="Input")

        # A possible second Input Layer

        X = layers.Dense(10, name="10-ReLU")(X_input)

        # output layer
        X = layers.Dropout(0.5, name="0.2")(X)
        X = layers.Dense(5, activation='relu', name="15-ReLU")(X)

        X = layers.Dense(1, activation='sigmoid', name="1-Sigmoid")(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='Basic-DNN')

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    return _build


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    plot_model(get_builder((11, 1024), (5))(), show_layer_names=True, show_shapes=True, to_file="model_dense_full.png")
