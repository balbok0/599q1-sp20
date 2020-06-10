import numpy as np

if __name__ == "__main__":
    from helpers import scale_gate_to_n_qubits, sum_squares, H
    from qft import qft2
else:
    from .helpers import scale_gate_to_n_qubits, sum_squares, H
    from .qft import qft2


def main(d: int, n: int, n_o: int, l: float, displacements = 0.0):
    num_qubits = d * n + n_o
    num_input_qubits = d * n

    # Prepare a state
    input_state = np.zeros(2 ** num_qubits, dtype=complex)

    # Make first qubit 1
    input_state[1] = 1

    # Apply Hadamard to input_qubits
    prep_input_state = scale_gate_to_n_qubits(num_qubits, H, list(range(num_input_qubits))) @ input_state

    # Perform an inverse qft on the output registers
    prep_output_state = qft2(prep_input_state, start=num_input_qubits, inverse=True)

    # Use blackbox
    blackbox_state = sum_squares(
        input_state=prep_output_state,
        size_out=n_o,
        dim_num=d,
        displacements=displacements,
        l=l
    )

    # QFT on each register
    post_qft = np.copy(blackbox_state)
    for idx in range(d):
        start_idx = idx * n
        post_qft = qft2(post_qft, start=start_idx, end=start_idx + n)
    # post_qft = qft2(blackbox_state, end=num_input_qubits)

    # print(post_qft)
    return post_qft

if __name__ == "__main__":
    l = 0.125
    d = 2
    n = 2
    n_o = 4

    # Result 0
    run_defs = {
        0: 0,
        0.5: 3,
        1: 4,
        2: 4,
    }
    for p, target in run_defs.items():
        print('')
        print(f'p: {p}')
        for n_o in range(3, 8):
            result = main(d, n, n_o, l, p)

            reshape_results = np.reshape(result, (-1, 2**n_o))
            probabilities = [sum([float(x * x.conj()) for x in y]) for y in reshape_results]

            print(n_o)
            print(probabilities[target])
            print('')


