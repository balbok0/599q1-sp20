import numpy as np

from helpers import apply_to_all, scale_gate_to_n_qubits, control_gate, H, r_k, to_Gn

from typing import List, Union

import timeit

i = np.complex(0, 1)

QFTs = {}
QFTs_inv = {}

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


def qft2(input_state: np.ndarray, start: int = 0, end: int = None, inverse: bool = False):
    """
    For Bit Oracle Alg. (Original)
    """
    global QFTs, QFTs_inv

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

    if not inverse:
        result_op = QFTs[size_qft]
    else:
        if size_qft not in QFTs_inv:
            QFTs_inv[size_qft] = QFTs[size_qft].conj().T
        result_op = QFTs_inv[size_qft]

    if size_qft == num_qubits_total:
        return np.matmul(result_op, input_state)
    else:
        result = None
        idx = 0
        while idx < num_qubits_total:
            if idx == start:
                gate_at_idx = result_op
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


def qft_gn(input_state: np.ndarray, start: int = 0, end: int = None, inverse: bool = False):
    """For Phase Oracle Alg.
    """
    num_qubits_total = int(np.log2(len(input_state)))

    if end is None:
        end = num_qubits_total

    size_qft = end - start

    mult = 1.
    if inverse:
        mult = -1.

    result = np.zeros((2 ** size_qft, 2 ** size_qft), dtype=np.complex)
    for idx in range(2 ** size_qft):
        for jdx in range(2 ** size_qft):
            i_arg = to_Gn(idx, size_qft)
            j_arg = to_Gn(jdx, size_qft)
            result[idx, jdx] = np.exp(i * i_arg * j_arg * 2 * np.pi * (2 ** size_qft) * mult)

    result_op = np.copy(result) / np.sqrt(2 ** size_qft)
    result = None
    idx = 0
    while idx < num_qubits_total:
        if idx == start:
            gate_at_idx = result_op
            idx = end
        else:
            gate_at_idx = np.identity(2)
            idx += 1
        if result is None:
            result = gate_at_idx
        else:
            result = np.kron(result, gate_at_idx)

    return result @ input_state
