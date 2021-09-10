import numpy as np

from ParametricCircuit import *
import scipy


def cost_function(probability_vector: np.ndarray) -> float:
    """

    :param probability_vector: vector of probabilities for 5 qubits
    :return: the cost value
    """
    bilinear_form = np.array([[-3, 2, 2, 2, 2],
                              [0, -3, 2, 2, 2],
                              [0, 0, -3, 2, 2],
                              [0, 0, 0, -3, 2],
                              [0, 0, 0, 0, -3]])
    return probability_vector @ bilinear_form @ probability_vector + 4


def cost_function_of_angular_argument(rotation_parameters: list) -> float:
    """

    :param rotation_parameters: angles of rotation for 5 qubits
    :return: the cost value (using class ParametricCircuit)
    """
    return cost_function(ParametricCircuit(5, rotation_parameters).probability_vector)


def minimize_cost_function() -> list:
    """

    :return: result of applying function from scipy.optimize to the cost function
    """
    starting_point = np.zeros(10)
    lower_bound = -2 * np.pi * np.ones(10)
    upper_bound = -lower_bound
    bounds = [(low, high) for low, high in zip(lower_bound, upper_bound)]
    # minimizer_kwargs = dict(bounds=bounds)
    description = scipy.optimize.dual_annealing(cost_function_of_angular_argument, bounds=bounds)
    return description


print(minimize_cost_function())
