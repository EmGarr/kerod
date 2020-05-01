import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils

from kerod.core.learning_rate_schedule import (ManualStepping, LearningRateScheduler)


def test_manual_stepping_without_warmup():

    manual_step = ManualStepping(boundaries=[2, 3, 7], rates=[1.0, 2.0, 3.0, 4.0], warmup=False)
    output_rates = [manual_step([np.array(i).astype(np.int64)]) for i in range(10)]
    exp_rates = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
    np.testing.assert_allclose(output_rates, exp_rates)


def test_manual_stepping_with_warmup():
    manual_step = ManualStepping(boundaries=[4, 6, 8], rates=[0.02, 0.10, 0.01, 0.001], warmup=True)
    output_rates = [manual_step([np.array(i).astype(np.int64)]) for i in range(9)]
    exp_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.10, 0.01, 0.01, 0.001]
    np.testing.assert_allclose(output_rates, exp_rates)


def test_manual_stepping_without_boundaries():
    manual_step = ManualStepping(boundaries=[], rates=[0.01], warmup=False)
    output_rates = [manual_step([np.array(i).astype(np.int64)]) for i in range(4)]
    exp_rates = [0.01] * 4
    np.testing.assert_allclose(output_rates, exp_rates)


class KerasCallbacksTest(keras_parameterized.TestCase):

    def test_LearningRateScheduler(self):
        with self.cached_session():
            np.random.seed(1337)
            batch_size = 5
            num_classes = 2
            input_dim = 3

            (x_train, y_train), (x_test,
                                 y_test) = testing_utils.get_test_data(train_samples=10,
                                                                       test_samples=10,
                                                                       input_shape=(input_dim,),
                                                                       num_classes=num_classes)
            y_test = tf.keras.utils.to_categorical(y_test)
            y_train = tf.keras.utils.to_categorical(y_train)
            model = testing_utils.get_small_sequential_mlp(num_hidden=5,
                                                           num_classes=num_classes,
                                                           input_dim=input_dim)

            epochs = [1, 2]
            callback = LearningRateScheduler(1, 1, epochs, num_warmup_steps=1)
            assert callback.slope == 1 - 1e-2 / 3

            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            model.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test),
                      callbacks=[callback],
                      epochs=4,
                      verbose=0)

            self.assertAllClose(tf.keras.backend.get_value(model.optimizer.lr), 0.01)

            # Here the epochs scheduling won't apply because the warmup hasn't been done
            num_warmup_steps = 16
            init_lr = 1e-2 / 3
            callback = LearningRateScheduler(1, 16, epochs, num_warmup_steps=num_warmup_steps)
            expected_slope = (1 - init_lr * 0.5) / num_warmup_steps
            assert callback.slope == expected_slope

            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            model.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test),
                      callbacks=[callback],
                      epochs=2,
                      verbose=0)

            # There are 2 epochs wich will two a total of 4 steps (train_samples = 10 with batch_size 5)
            expected_lr = init_lr * 0.5 + expected_slope * 3
            self.assertAllClose(tf.keras.backend.get_value(model.optimizer.lr), expected_lr)

            callback = LearningRateScheduler(1, 16, epochs, num_warmup_steps=num_warmup_steps, use_warmup=False)
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            model.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test),
                      callbacks=[callback],
                      epochs=2,
                      verbose=0)
            self.assertAllClose(tf.keras.backend.get_value(model.optimizer.lr), 0.01)
