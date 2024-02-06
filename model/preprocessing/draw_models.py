import keras.utils

from common.constants import IMAGES_PATH, OPTIMIZER_SGD, OPTIMIZER_ADAM
from model.network.model import build_mlp_model2, build_mlp_model, build_mlp_encodings_model


def draw_model(model, file_name):
    keras.utils.plot_model(
        model,
        to_file=file_name,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=False
    )


file_name = IMAGES_PATH + 'build_mlp_encodings_model(8_outputs)'
# model = build_mlp_model(optimizer=OPTIMIZER_SGD)
# model = build_mlp_model2(optimizer=OPTIMIZER_SGD)
model = build_mlp_encodings_model(optimizer=OPTIMIZER_ADAM, input_size=5)
draw_model(model, file_name)