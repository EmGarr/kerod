import numpy as np
from od.learning_rate_schedule import ManualStepping


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
