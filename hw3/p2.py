import numpy as np

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 200



cnot = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
)
x = np.array(
    [[0, 1],
     [1, 0]]
)

def crx(l):
    result = np.identity(4) * np.complex(1, 0)

    result[-2:, -2:] = rx(l)

    result[np.abs(result) < 10e-10] = 0

    return result


def rx(l):
    result = np.identity(2) * np.complex(1, 0)

    # result[0, 0] = np.complex(0, np.cos(np.pi * l))
    # result[0, 1] = np.complex(np.sin(np.pi * l), 0)
    # result[1, 0] = np.complex(np.sin(np.pi * l), 0)
    # result[1, 1] = np.complex(0, np.cos(np.pi * l))

    result[0, 0] = np.complex(np.cos(np.pi * l), 0)
    result[0, 1] = np.complex(0, -np.sin(np.pi * l))
    result[1, 0] = np.complex(0, -np.sin(np.pi * l))
    result[1, 1] = np.complex(np.cos(np.pi * l), 0)

    result[np.abs(result) < 10e-10] = 0

    return result


# def universal_gate(l):
#     result = np.identity(8) * np.complex(1, 0)
#     result[6, 6] = np.complex(0, np.cos(np.pi * l))
#     result[6, 7] = np.complex(np.sin(np.pi * l), 0)
#     result[7, 6] = np.complex(np.sin(np.pi * l), 0)
#     result[7, 7] = np.complex(0, np.cos(np.pi * l))

#     result[np.abs(result) < 10**-10] = 0

#     return result
# print(rx(0.25))

rxi = lambda l: np.kron(rx(l), np.identity(2))

xi = np.kron(x, np.identity(2))
xx = np.kron(x, x)

zero_plus_state = np.array([1, 1, 0, 0]) / np.sqrt(2)
one_plus_state = np.array([0, 0, 1, 1]) / np.sqrt(2)


circuit = rxi(-0.25) @ cnot @ rxi(0.25)
# circuit = lambda l: np.matmul(xi, np.matmul(crx(l), np.matmul(xi, crx(-l))))

# print(np.matmul(circuit(0.5), zero_plus_state))

# print(rxi(0.25))
print(circuit)
# exit(0)

# input_state = np.array([1, 0, 0, 0])

# first = np.kron(rx(0.25), np.identity(2))
# second = crx(0.5)
# third = np.kron(rx(-0.25), np.identity(2))

# r = np.matmul(third, np.matmul(second, np.matmul(first, input_state)))

# print(r)

# Measuring the first qubit
# tmp = np.zeros(4) * complex(1, 0)
# r = r[-2:] * np.sqrt(2) * complex(0, 1)
# tmp[1] = r[0]
# tmp[3] = r[1]

# print(tmp)

# print(crx(1/2))


exit(0)


rx0i1 = lambda l: np.kron(rx(l), np.identity(2))

circuit = lambda l: np.matmul(rx0i1(l), np.matmul(crx(-l), rx0i1(l)))

print(circuit(0.25))
exit(0)

print()

prep = np.array([0, 0, 0, 0, 0, 0, 1, 0]) * complex(1, 0)
prep = np.matmul(universal_gate(0.25), prep)


last_one = np.zeros(8) * complex(1, 0)
last_one[5] = prep[6]
last_one[7] = prep[7]

last_zero = np.zeros(8) * complex(1, 0)
last_zero[4] = prep[6]
last_zero[6] = prep[7]

last_one = np.matmul(last_one, universal_gate(0.5))
print(last_one)