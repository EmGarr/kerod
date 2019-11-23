import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


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

    name: String.  Optional name of the operation.  Defaults to
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
