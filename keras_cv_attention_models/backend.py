import os

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, callbacks
from tensorflow.keras.utils import register_keras_serializable, get_file
from . import tf_functional as functional


def backend():
    return tf.keras.backend.backend()

def image_data_format():
    return tf.keras.backend.image_data_format()

__is_channels_last__ = image_data_format() == "channels_last"


def is_channels_last():
    return __is_channels_last__


def align_input_shape_by_image_data_format(input_shape):
    """Regard input_shape as force using original shape if len(input_shape) == 4,
    else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format,
    or if dynamic shape liek `(None, None, 3)`, will regard the known shape 3 as channel dimentsion.

    Examples:
    >>> os.environ['KECAM_BACKEND'] = "torch"
    >>> from keras_cv_attention_models import backend
    >>> print(f"{backend.image_data_format() = }")
    >>> # backend.image_data_format() = 'channels_first'
    >>> print(backend.align_input_shape_by_image_data_format([224, 224, 3]))
    >>> # [3, 224, 224]
    >>> print(backend.align_input_shape_by_image_data_format([3, 224, 224]))
    >>> # [3, 224, 224]
    >>> print(backend.align_input_shape_by_image_data_format([None, None, 3]))
    >>> # [3, None, None]
    >>> print(backend.align_input_shape_by_image_data_format([None, 224, 224, 3]))
    >>> # [224, 224, 3]
    """
    # channel_axis = if image_data_format() == "channels_last" else
    if len(input_shape) == 4:  # Regard this as force using original shape
        return input_shape[1:]

    # Assume channel dimention is the one with min value in input_shape
    if None in input_shape:
        orign_channel_axis, channel_dim = min(enumerate(input_shape), key=lambda xx: xx[1] is None)
    else:
        orign_channel_axis, channel_dim = min(enumerate(input_shape), key=lambda xx: xx[1])
    orign_block_shape = [dim for axis, dim in enumerate(input_shape) if axis != orign_channel_axis]
    aligned = [*orign_block_shape, channel_dim] if image_data_format() == "channels_last" else [channel_dim, *orign_block_shape]
    if aligned[1] != input_shape[1] or aligned[-1] != input_shape[-1]:
        print(">>>> Aligned input_shape:", aligned)
    return aligned


def in_train_phase(train_phase, eval_phase, training=None):
    return tf.keras.backend.in_train_phase(train_phase, eval_phase, training=training)

def numpy_image_resize(inputs, target_shape, method="bilinear", is_source_channels_last=True):
    ndims = len(inputs.shape)
    if ndims < 2 or ndims > 4:
        raise ValueError("inputs with shape={}, ndims={} not supported, ndims must in [2, 4]".format(ipnuts.shape, ndims))

    if is_source_channels_last:
        inputs = inputs if ndims == 4 else (inputs[None] if ndims == 3 else inputs[None, :, :, None])
        inputs = inputs if image_data_format() == "channels_last" else inputs.transpose([0, 3, 1, 2])

        inputs = functional.resize(inputs, target_shape, method=method)
        inputs = inputs.detach().numpy() if hasattr(inputs, "detach") else inputs.numpy()

        inputs = inputs if image_data_format() == "channels_last" else inputs.transpose([0, 2, 3, 1])
        inputs = inputs if ndims == 4 else (inputs[0] if ndims == 3 else inputs[0, :, :, 0])
    else:
        inputs = inputs if ndims == 4 else (inputs[None] if ndims == 3 else inputs[None, None, :, :])
        inputs = inputs.transpose([0, 2, 3, 1]) if image_data_format() == "channels_last" else inputs

        inputs = functional.resize(inputs, target_shape, method=method)
        inputs = inputs.detach().numpy() if hasattr(inputs, "detach") else inputs.numpy()

        inputs = inputs.transpose([0, 3, 1, 2]) if image_data_format() == "channels_last" else inputs
        inputs = inputs if ndims == 4 else (inputs[0] if ndims == 3 else inputs[0, 0])
    return inputs