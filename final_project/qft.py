import numpy as np

from helpers import apply_to_all, scale_gate_to_n_qubits, control_gate, H, r_k

from typing import List, Union

import timeit

i = np.complex(0, 1)

QFTs = {}

def qft1(num_qubits: int):
    """

    Args:
        num_qubits (int): [description]

    Returns:
        [type]: [description]
    """
    # ! Failed attempt at implementing a QFT from the circuit presented in class.
    # ! There seems to be an error of order -i in some cases.
    result = np.identity(2**num_qubits)
    for idx in range(num_qubits):
        print(f'idx: {idx}')
        # print(f'scaled: {scale_gate_to_n_qubits(num_qubits, H, idx)}')
        result = np.matmul(scale_gate_to_n_qubits(num_qubits, H, idx), result)
        for jdx in range(idx + 1, num_qubits):
            result = np.matmul(control_gate(jdx, idx, num_qubits, r_k(jdx - idx + 1)), result)

    # Swap Operator
    swap_gate = np.zeros((2**num_qubits, 2**num_qubits))
    for idx in range(2**num_qubits):
        new_idx = 0
        for jdx in range(num_qubits):
            if (idx >> jdx) % 2 == 1:
                new_idx += 2 ** (num_qubits - jdx - 1)
        swap_gate[idx, new_idx] = 1

    return swap_gate @ result


def qft2(input_state: np.ndarray, start: int = 0, end: int = None):
    global QFTs

    num_qubits_total = int(np.log2(len(input_state)))

    if end is None:
        end = num_qubits_total

    size_qft = end - start

    if size_qft not in QFTs:
        # Such QFT does not exist (yet)
        result = np.zeros((2 ** size_qft, 2 ** size_qft), dtype=np.complex)
        for idx in range(2 ** size_qft):
            for jdx in range(2 ** size_qft):
                result[idx, jdx] = np.exp(i * idx * jdx * 2 * np.pi / 2 ** size_qft)
        QFTs[size_qft] = result / np.sqrt(2 ** size_qft)

    else:
        # This QFT was already calculated, do nothing
        pass

    if size_qft == num_qubits_total:
        return np.matmul(QFTs[size_qft], input_state)
    else:
        result = None
        idx = 0
        while idx < num_qubits_total:
            if idx == start:
                gate_at_idx = QFTs[size_qft]
                idx = end
            else:
                gate_at_idx = np.identity(2)
                idx += 1
            if result is None:
                result = gate_at_idx
            else:
                result = np.kron(result, gate_at_idx)

        return np.matmul(
            result,
            input_state
        )

# print(qft1(3) @ np.array([0, 0, 0, 1, 0, 0, 0, 0]))
print(qft2([0, 0, 1, 0, 0, 0, 0, 0]))
print(qft2([0, 0, 0, 0, 0, 0, 0, 0], start=0, end=2))

