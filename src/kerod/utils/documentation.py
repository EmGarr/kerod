import tensorflow as tf
from tensorflow.python.keras.engine.training import Model


def remove_unwanted_doc(_class, pdoc: dict):
    """Remove unwanted documentation from tensorflow keras inheritance."""
    if issubclass(_class, Model):
        doc_class_to_remove = Model
    elif issubclass(_class, tf.keras.layers.Layer):
        doc_class_to_remove = tf.keras.layers.Layer

    for k in doc_class_to_remove.__dict__:
        if k not in ['call', '__doc__', '__module__', '__init__', '__name__']:
            pdoc[f'{_class.__name__}.{k}'] = None
    pdoc[f'{_class.__name__}.with_name_scope'] = None
