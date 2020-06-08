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


def bitstring_to_num(arr):
    """Little Endian (i.e. 001 = 1 and 100 = 4)
    """
    result = 0
    for idx, elem in enumerate(arr[::-1]):
        result += elem * 2**idx
    return result


def num_to_bitstring(num, size):
    """Little Endian (i.e. 001 = 1 and 100 = 4)
    """
    num = num % (2**size)
    result = np.zeros(size)
    for idx in range(size):
        if (num >> idx) % 2 == 1:
            result[size - idx - 1] = 1
    return result


def add_bit_strings(*args):
    if len(args) == 0:
        raise ValueError('No Argument given.')
    elif len(args) == 1:
        return args[0]
    else:
        max_len = max([len(x) for x in args])
        result = 0
        for x in args:
            result += bitstring_to_num(x)
        result = int(result) % 2**max_len
        return num_to_bitstring(result, max_len)


def square_a_state(idx_of_state: int, size_of_out, size_of_dim, start, end, size_of_state, displacements):
    inp = num_to_bitstring(idx_of_state, size_of_state)

    result = 0
    relative_end = end - start
    for disp_idx, idx in enumerate(range(0, relative_end - size_of_out, size_of_dim)):
        result_idx = bitstring_to_num(inp[idx:idx + size_of_dim]) * 1.0 / 2**size_of_dim - displacements[disp_idx]
        result += result_idx ** 2

    clean_result = num_to_bitstring(int(result * 2**size_of_out), size_of_out)
    inp[relative_end - size_of_out:relative_end] = add_bit_strings(inp[relative_end - size_of_out:relative_end], clean_result)

    result = bitstring_to_num(inp)
    return result


SUM_SQUARES = {}

def sum_squares(input_state: np.ndarray, size_out: int, dim_num: int = 1,  start: int = 0, end: int = None, displacements: Union[float, Set[float]] = 0.0):
    global SUM_SQUARES
    num_qubits_total = int(np.log2(len(input_state)))

    if end is None:
        end = num_qubits_total

    size_op = end - start

    size_in = size_op - size_out

    assert size_in % dim_num == 0

    size_of_dim = size_in // dim_num

    if not hasattr(displacements, 'len'):
        displacements = [displacements] * dim_num
    else:
        displacements = list(displacements)
        assert len(displacements) == dim_num

    key_to_global_dict = f'{size_in}_{size_out}_{size_of_dim}_{displacements}'
    if key_to_global_dict not in SUM_SQUARES:
        result_op = np.zeros((2**size_op, 2**size_op), dtype=complex)
        power_diff = num_qubits_total - end
        for i in range(2**size_op):
            idx = int(square_a_state(
                idx_of_state=i,
                size_of_out=size_out,
                size_of_dim=size_of_dim,
                start=start,
                end=end,
                size_of_state=size_op,
                displacements=displacements))
            result_op[idx, i] = 1
        SUM_SQUARES[key_to_global_dict] = result_op
    else:
        result_op = SUM_SQUARES[key_to_global_dict]

    if size_op == num_qubits_total:
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


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    input_state = np.zeros(2**9)  # 5 qubits
    input_state[480 + 16] = 1
    result = sum_squares(
        input_state=input_state,
        size_out=2,
        dim_num=2,
        start=2,
        end=8
    )
