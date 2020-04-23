"""
Manual Stepping is designed to integrate the computation graph and compute the learning_rate at
each step.

However, the WarmupLearningRateScheduler is Callback handle by the fit in keras.
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.keras.callbacks import Callback


# TODO clean it up it is a bit dirty
class WarmupLearningRateScheduler(Callback):
    """Warmup Learning rate scheduler. It will perform at the beginning of the training
    a linear warmup from `init_lr` to `base_lr`. The learning rate is decreased by 10 according
    to the schedule provided by `epochs`.

    Arguments:

    - *base_lr*: The target learning rate value after the linear warmup
    - *num_gpus*: Number of gpus used during the training
    - *epochs*: A list of epoch on which the learning rate should be reduce.
    - *init_lr*: Learning rate value from which the warmup will start.
    - *num_warmup_steps*: Number of training step on which the warmup will be performed.
    """

    def __init__(self,
                 base_lr: float,
                 num_gpus: int,
                 epochs: List[int],
                 init_lr: float = 1e-2 / 3,
                 num_warmup_steps: int = 1000):
        super().__init__()
        self._init_lr = init_lr * min(8 / num_gpus, 1)
        self.slope = (base_lr - self._init_lr) / num_warmup_steps
        self._epochs_to_lr = {epoch: base_lr * 1 / 10**(i + 1) for i, epoch in enumerate(epochs)}
        self._epochs = epochs
        self._num_gpus = num_gpus
        self._num_warmup_steps = num_warmup_steps

    def on_train_batch_begin(self, batch, logs=None):
        global_step = K.get_value((self.model.optimizer.iterations))
        if global_step <= self._num_warmup_steps and global_step != 0:
            lr = self._init_lr + global_step * self.slope
            K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'iterations'):
            raise ValueError('Optimizer must have an "iterations" attribute.')

        global_step = K.get_value(self.model.optimizer.iterations)

        target_epochs = [
            e for e in self._epochs if epoch >= e and global_step > self._num_warmup_steps
        ]
        if target_epochs:
            lr = self._epochs_to_lr[max(target_epochs)]
            K.set_value(self.model.optimizer.lr, lr)


class ManualStepping(LearningRateSchedule):
    """Manually stepped learning rate schedule. (Taken and modified from Google object detection)

    This function provides fine grained control over learning rates.  One must
    specify a sequence of learning rates as well as a set of integer steps
    at which the current learning rate must transition to the next.  For example,
    if boundaries = [5, 10] and rates = [.1, .01, .001], then the learning
    rate returned by this function is .1 for step=0,...,4, .01 for
    step=5...9, and .001 for step=10 and onward.

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.

    ```python
    lr_schedule = tf.keras.optimizers.schedules.ManualStepping(
        boundaries=[5, 10],
        rates=[.1, .01, .001],
        warmup=True)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Arguments:

    - *boundaries*: A List of scalar `int32` or `int64` or a `Tensor`. It is a
    list of global steps at which to switch learning
    rates.  This list is assumed to consist of increasing positive integers.
    - *rates*: a list of (float) learning rates corresponding to intervals between
    the boundaries.  The length of this list must be exactly
    len(boundaries) + 1.
    - *warmup*: Whether to linearly interpolate learning rate for steps in
    [0, boundaries[0]].
    - *name*: String.  Optional name of the operation.  Defaults to
        'ExponentialDecay'.

    Return:

    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `rates`.
    """

    def __init__(self, boundaries, rates, warmup=False, name=None):
        super().__init__()
        self.name = name
        if warmup and boundaries:
            slope = (rates[1] - rates[0]) * 1.0 / boundaries[0]
            warmup_steps = list(range(boundaries[0]))
            warmup_rates = [rates[0] + slope * step for step in warmup_steps]
            boundaries = warmup_steps + boundaries
            rates = warmup_rates + rates[1:]
        else:
            boundaries = [0] + boundaries
        self.warmup = warmup
        self.rates = rates
        self.boundaries = boundaries
        self.num_boundaries = len(boundaries)
        self.dtype = tf.convert_to_tensor(rates[0]).dtype


    def __call__(self, step):
        with tf.name_scope(self.name or "ManualStepping"):
            boundaries = tf.convert_to_tensor(self.boundaries, self.dtype)
            rates = tf.convert_to_tensor(self.rates, self.dtype)
            step = tf.convert_to_tensor(step, self.dtype)
            rate_index = tf.reduce_max(
                tf.where(tf.greater_equal(step, boundaries), list(range(self.num_boundaries)),
                         [0] * self.num_boundaries))
            return tf.reduce_sum(rates * tf.one_hot(rate_index, depth=self.num_boundaries))

    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "rates": self.rates,
            "warmup": self.warmup,
            "name": self.name
        }
