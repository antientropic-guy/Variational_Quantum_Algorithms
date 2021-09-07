from __future__ import annotations
import cirq
import numpy
import numpy as np


class ParametricCircuit:
    def __init__(self, qubit_circuit_length: int, rotation_parameters: list):
        """

        :param qubit_circuit_length: amount of qubits
        :param rotation_parameters: list of 2 * qubit_circuit_length rotation angles
        """
        self.qubit_circuit_length = qubit_circuit_length
        self.rotation_parameters = rotation_parameters

    @property
    def generate_circuit(self) -> cirq.Circuit:
        """

        :return: cirq.Circuit object
        """
        qubit_circuit = cirq.LineQubit.range(self.qubit_circuit_length)
        new_circuit = cirq.Circuit()
        for i in range(len(qubit_circuit)):
            new_circuit.append(cirq.Rx(rads=self.rotation_parameters[i])(qubit_circuit[i]))
        for i in range(len(qubit_circuit) - 1):
            new_circuit.append(cirq.CNOT(qubit_circuit[i], qubit_circuit[i + 1]))
        for i in range(len(qubit_circuit)):
            new_circuit.append(cirq.Rx(rads=self.rotation_parameters[i + len(qubit_circuit)])(qubit_circuit[i]))
        for i in range(len(qubit_circuit) - 1):
            new_circuit.append(cirq.CNOT(qubit_circuit[i], qubit_circuit[i + 1]))
        return new_circuit

    @property
    def probability_vector(self) -> numpy.ndarray:
        """

        :return: the list of probability to collapse in |0> state after z-basic measure for each qubit
        """
        simulator = cirq.Simulator()
        result = simulator.simulate(self.generate_circuit)
        probability_list = []
        for i in range(self.qubit_circuit_length):
            probability_list.append(1 / 2 * (1 + result.bloch_vector_of(cirq.LineQubit(i))[2]))
        probability_list = np.array(probability_list)
        return probability_list
