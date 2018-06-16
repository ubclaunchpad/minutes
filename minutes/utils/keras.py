import keras


def copy_model(model):
    """Returns a copy of the model.

    Arguments:
        model {keras.Sequential} -- A model for copying.
    """
    # https://github.com/keras-team/keras/issues/1765
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    return model_copy
