import numpy as np

if __name__ == "__main__":
    from helpers import scale_gate_to_n_qubits, sum_squares, H, apply_to_all, sum_squares_phase
    from qft import qft2, qft_gn
else:
    from .helpers import scale_gate_to_n_qubits, sum_squares, H, apply_to_all, sum_squares_phase
    from .qft import qft2, qft_gn


def __main_algorithm(d: int, n: int, n_o: int, l: float, displacements = 0.0):
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


def __main_algorithm_phase(d: int, n: int, displacements = 0.0):
    num_qubits = d * n

    # Prepare a state
    input_state = np.zeros(2 ** num_qubits, dtype=complex)
    input_state[0] = 1

    # Apply Hadamard to input_qubits
    prep_input_state = apply_to_all(num_qubits, H) @ input_state

    # Oracle Call
    post_oracle_state = sum_squares_phase(prep_input_state, d, displacements)

    # QFT Gn on each register
    post_qft = np.copy(post_oracle_state)
    for idx in range(d):
        start_idx = idx * n
        post_qft = qft_gn(post_qft, start_idx, start_idx + n, inverse=True)

    return post_qft


def main_paper_exp():
    l = 0.125
    d = 2
    n = 2
    n_o = 4

    # Result 0
    run_defs = {
        0: [0],
        0.5: [0, 3],
        1: [0, 4],
        2: [0, 4],
    }
    for p, targets in run_defs.items():
        for target in targets:
            print('')
            print(f'p: {p}')
            print(f'target: {target}')
            for n_o in range(3, 8):
                result = __main_algorithm(d, n, n_o, l, p)

                reshape_results = np.reshape(result, (-1, 2**n_o))
                probabilities = [sum([float(x * x.conj()) for x in y]) for y in reshape_results]

                print(n_o)
                print(probabilities[target])
                print('')


def main_to_play_around():
    epsilon = 0.125
    d = 2
    n_o = 6

    l = epsilon / (2 * np.sqrt(d))
    N = 1 / epsilon
    n = int(np.ceil(np.log2(N)))

    # Result 0
    run_defs = {
        # 0: [0],
        0.5: [3],
        1: [4],
        2: [4],
    }
    for p, targets in run_defs.items():
        for target in targets:
            print('')
            print(f'p: {p}')
            print(f'target: {target}')
            for n_o in range(5, 8):
                result = __main_algorithm(d, n, n_o, l, p)

                reshape_results = np.reshape(result, (-1, 2**n_o))
                probabilities = [sum([float(x * x.conj()) for x in y]) for y in reshape_results]

                print(n_o)
                idx = 2**n * target + target
                print(probabilities)
                print('')


def main_phase():
    d = 2
    n = 4
    result = [float(x * x.conj()) for x in __main_algorithm_phase(d, n)]
    print(result)


if __name__ == "__main__":
    main_paper_exp()
    # main_to_play_around()
    main_phase()

