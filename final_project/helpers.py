import numpy as np

from typing import Union, Set

H = 1. / np.sqrt(2) * np.array(
    [
        [1, 1],
        [1, -1]
    ]
)

def apply_to_all(num_qubits: int, gate: np.ndarray):
    r"""Applies the same gate over all of the qubits

    Args:
        num_qubits (int): Number of qubits in a state
        gate (np.ndarray): A gate to apply for each qubit (Has to be 2 x 2)

    Returns:
        np.ndarray: A matrix of size `($$2^{num\_qubits}, 2^{num\_qubits}$$),
            which corresponds to the provided `gate` applied simultaneously on all qubits.
    """
    result = None
    for _ in num_qubits:
        if result is None:
            result = gate
        else:
            result = np.kron(result, gate)
    return result


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
        if idx in qubit_applied:
            gate_at_idx = gate
        else:
            gate_at_idx = np.identity(2)
        if result is None:
            result = gate_at_idx
        else:
            result = np.kron(result, gate_at_idx)
    return result


def control_gate(control_qubits: Union[Set[int], int], apply_qubits: Union[Set[int], int], total_qubits: int, gate_to_apply: np.ndarray):
    if not isinstance(control_qubits, set):
        if isinstance(control_qubits, int):
            control_qubits = set([control_qubits])
        else:
            control_qubits = set(control_qubits)
    if not isinstance(apply_qubits, set):
        if isinstance(apply_qubits, int):
            apply_qubits = set([apply_qubits])
        else:
            apply_qubits = set(apply_qubits)
    if not isinstance(gate_to_apply, np.ndarray):
        gate_to_apply = np.array(gate_to_apply)

    remaining_qubits = []
    for i in range(total_qubits):
        if i not in control_qubits and i not in apply_qubits:
            remaining_qubits.append(i)
    remaining_qubits = set(remaining_qubits)

    control_qubits_value = sum([2**control_qubit for control_qubit in control_qubits])

    # Map from placeholders, to indexes where to apply the gate
    map_to_apply = {}

    for i in range(control_qubits_value, 2**total_qubits):
        # Determine whether an index is valid to be controlled
        is_valid = True
        for control_qubit in control_qubits:
            if (i >> control_qubit) % 2 != 1:  # Controlled qubit is not 1
                is_valid = False
                break

        if is_valid:
            map_array_idx = 0
            apply_qubits_value = 0
            for idx, apply_qubit in enumerate(apply_qubits):
                if (i >> apply_qubit) % 2 == 1:
                    map_array_idx += 2 ** idx
                    apply_qubits_value += 2 ** apply_qubit

            map_key = i - control_qubits_value - apply_qubits_value

            if map_key not in map_to_apply:
                map_to_apply[map_key] = np.zeros(len(gate_to_apply), dtype=int)


            map_to_apply[map_key][map_array_idx] = i

    result = np.identity(2**total_qubits, dtype=complex)
    for _, idxs in map_to_apply.items():
        result[np.ix_(idxs, idxs)] = gate_to_apply
    return result


def r_k(k: int, pow: int = 1):
    i = np.complex(0, 1)

    return [
        [1, 0],
        [0, np.exp(2 * np.pi * i * pow / 2**k)]
    ]


# print(control_gate(0, 2, 3, [[0, 1], [1, 0]]))
