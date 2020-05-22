import numpy as np

from typing import Union, Set

H = 1. / np.sqrt(2) * np.array(
    [
        [1, 1],
        [1, -1]
    ]
)


def scale_gate_to_n_qubits(num_qubits: int, gate: np.ndarray, qubit_applied: Union[int, Set[int]]):
    """Applies a series of kronecker products to generate a `2^num_qubits` version of a gate.

    Args:
        num_qubits (int): Number of qubits in a system
        gate (np.ndarray): An 2x2 numpy array representing unitary operation.
        qubit_applied (Union[int, Set[int]]): Index(es) of a qubit(s) at which that gate is applied.
    """
    if isinstance(qubit_applied, int):
        qubit_applied = set([qubit_applied])

    result = None

    for idx in range(num_qubits):
        if idx == qubit_applied:
            gate_idx = gate
        else:
            gate_idx = np.identity(2)
        result = np.kron(result, gate_idx)
    return result


def control_gate(control_qubits: int, apply_qubits: int, total_qubits: int):
    map_to_apply = {}

    for i in range(2**control_qubits, 2**total_qubits):
        i_bytes = "{0:{total_qubits}b}".format(i, total_qubits=total_qubits)

        # Determine whether an index is valid to be controlled
        is_valid = True
        if (i >> control_qubits) % 2 != 1:  # Controlled qubit is not 1
            is_valid = False

        # if is_valid:
        # print(i >> control_qubits)
        # print(np.array(i, dtype=bool))
        print(i)
        print(i_bytes)
        print(is_valid)
        # print(i_bytes[0, 1])
        pass

control_gate(2, 4, 4)
