import numpy as np
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 2000


i = np.complex(0, 1)


u = 0.5 * np.array(
    [
        [1,  1,  1,  1],
        [1,  i, -1, -i],
        [1, -1,  1, -1],
        [1, -i, -1,  i],
    ]
)

def part_a():

    def get_u(u, i, j):
        assert j > i >= 0
        a = u[i, i]
        b = u[j, i]

        result = np.identity(len(u)) * np.complex(1, 0)

        if b == 0:
            result[i, i] = a.conjugate()
            return result

        normalization = np.linalg.norm([a, b])

        result[i, i] = a.conjugate() / normalization
        result[j, i] = b.conjugate() / normalization
        result[i, j] = b / normalization
        result[j, j] = -a / normalization

        return result

    u1 = get_u(u, 0, 1)

    u1u = np.matmul(u1, u)
    u2 = get_u(u1u, 0, 2)

    u2u1u = np.matmul(u2, u1u)
    u3 = get_u(u2u1u, 0, 3)

    u3u2u1u = np.matmul(u3, u2u1u)

    u3u2u1u = np.matmul(u3, u2u1u)


    # Correct for numerical errors
    u3u2u1u[np.abs(u3u2u1u) < 10**-15] = 0

    print(u3)

    print()
    # for x in u3u2u1u:
    #     print(x)
    # print(u3u2u1u)


    u4 = u3u2u1u.copy()

    u4c = np.conjugate(u4)
    u3c = np.conjugate(u3)
    u2c = np.conjugate(u2)
    u1c = np.conjugate(u1)

    uc = np.matmul(u1c, np.matmul(u2c, np.matmul(u3c, u4c)))

    result = np.matmul(uc, u)
    result[np.abs(result) < 10**-15] = 0
    # print(result)

    # print(np.linalg.norm([np.complex(1, -1), np.complex(1, 0)]))
    # print(u3u2u1u)
    # print(get_u(u, 0, 1))

    # print(np.matmul(u1, u))


def part_b():

    """
    Problem 1b
    """


    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    H = np.array([[1,1],[1,-1]])
    I = np.array([[1,0],[0,1]])
    R2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1j],])
    H1 = np.kron(H, I)
    H2 = np.kron(I, H)
    res = H1 @ R2 @ H2 @ SWAP
    print(res)
    exit(0)

    s = np.array([[1, 0], [0, i]])
    h = complex(1, 0) / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    # a = np.kron(h, s)


    first = [
        [1,                  0, 0,                  0],
        [0,  complex(0.5, 0.5), 0, complex(0.5, -0.5)],
        [0,                  0, 1,                  0],
        [0, complex(0.5, -0.5), 0,  complex(0.5, 0.5)],
    ]

    swap = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]

    h2 = np.kron(h, h)

    result = np.matmul(swap, np.matmul(first, swap))

    print(result)

    # a = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]) * complex(0.5, 0)


    # print(np.matmul(u, a.conjugate()))


if __name__ == "__main__":
    # part_a()
    part_b()