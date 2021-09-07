import numpy as np

from ParametricCircuit import *
import scipy


def cost_function(probability_vector: np.ndarray) -> float:
    """

    :param probability_vector:
    :return:
    """
    bilinear_form = np.array([[-3, 2, 2, 2, 2],
                              [0, -3, 2, 2, 2],
                              [0, 0, -3, 2, 2],
                              [0, 0, 0, -3, 2],
                              [0, 0, 0, 0, -3]])
    return probability_vector @ bilinear_form @ probability_vector + 4


def cost_function_of_angular_argument(rotation_parameters: list) -> float:
    return cost_function(ParametricCircuit(5, rotation_parameters).probability_vector)


def minimize_cost_function() -> list:
    starting_point = np.zeros(10)
    lower_bound = -2 * np.pi * np.ones(10)
    upper_bound = -lower_bound
    bounds = [(low, high) for low, high in zip(lower_bound, upper_bound)]
    minimizer_kwargs = dict(bounds=bounds)
    description = scipy.optimize.basinhopping(cost_function_of_angular_argument, starting_point,
                                              minimizer_kwargs=minimizer_kwargs, niter=250)
    return description


print(minimize_cost_function())
