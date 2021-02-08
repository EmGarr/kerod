import pytest
import tensorflow as tf
from kerod.layers.positional_encoding import PositionEmbeddingLearned, PositionEmbeddingSine


def test_positional_embedding_learned():
    pos_embed = PositionEmbeddingLearned(128)
    out = tf.random.uniform((3, 45, 50, 4))
    batch_emb = pos_embed(out)
    assert batch_emb.shape == (3, 45, 50, 128)


def test_catch_value_error():
    with pytest.raises(ValueError):
        pos_embed = PositionEmbeddingLearned(3)
    with pytest.raises(ValueError):
        pos_embed = PositionEmbeddingSine(3)


def test_positional_embedding_sine():
    pos_embed = PositionEmbeddingSine(12)
    out = tf.constant([
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
            [False, False, False, False],
        ],
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
    ],
                      dtype=tf.bool)
    batch_emb = pos_embed(out)
    assert batch_emb.shape == (2, 4, 4, 12)
